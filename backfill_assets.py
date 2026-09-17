# -*- coding: utf-8 -*-
"""补登记：把 metadata.json 里已有、但 DB assets 里缺的帧批量补进数据库。

背景：抽帧的分块向量化分支一度漏掉 sync_metadata_paths，导致元数据/磁盘上有几十万帧、
库里只有几千条 —— 明细表/数据仓库读库，所以"看不到数据"。

为什么不用 db_service.sync_asset_records_to_db（抽帧正在用的那条路径，本脚本不动它）：
  该函数在一个写事务里对每条记录做 Source 查询 + exists/getsize + PIL.open，
  全是逐帧网盘往返，直到函数末尾才 commit —— 实测一个 2000 帧的批次能把
  SQLite 写锁连续占住好几分钟。于是它自己补不出数据（跑 50 分钟 0 条），
  还会把抽帧自己的写库（ensure_sync_hook 吞异常只打日志）一起挡掉。
本脚本先把已有 vector_id / source 一次性读进内存，再按批做纯 DB 写：
每个批次只开一个事务、只 commit 一次，写锁只占毫秒级；
撞上抽帧的长事务就退避重试（busy_timeout + 多轮退避），磨到间隙为止。
代价：width/height/file_size 记 0、file_hash 留空，后续扫描/YOLO 任务可补。

口径：唯一事实来源是 metadata.json（= app 的帧全集）。
  /api/image/{project}/{id} 用 image_id 直接下标查 metadata，FAISS 向量数也与
  metadata 记录数逐一对应；metadata 里没有的帧（如改动前的老扁平帧
  v<序号>_<时间戳>_<顺序号>.jpg）既不能显示也不能检索，不在补登记范围。

用法：
  /venv/bin/python -u backfill_assets.py            # 补全部项目
  /venv/bin/python -u backfill_assets.py G91        # 只补 G91
  /venv/bin/python -u backfill_assets.py --verify   # 只核对不写
环境变量：
  AD_BACKFILL_CHUNK=500       每批条数（越小越容易抢到锁）
  AD_BACKFILL_BUSY_MS=60000   单次等锁上限
  AD_BACKFILL_TRIES=20        每批最多重试轮数
"""
import json
import os
import sys
import time
import uuid

sys.path.insert(0, "/opt/ad_mining")
os.chdir("/opt/ad_mining")

from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from db_service import get_db_session
from models import Asset, AssetStatus, Project, Source, SourceType

CHUNK = max(100, int(os.environ.get("AD_BACKFILL_CHUNK", "500")))
IN_MAX = 500          # SQLite 变量上限保守值
SCAN_STATUS = "synced"
BUSY_MS = int(os.environ.get("AD_BACKFILL_BUSY_MS", "60000"))
TRIES = max(1, int(os.environ.get("AD_BACKFILL_TRIES", "20")))


def _log(msg):
    print(msg, flush=True)


def _root():
    cfg = json.load(open(os.path.join(os.getcwd(), "data_config.json"), encoding="utf-8"))
    return cfg["data_root"]


def _lookup_sources(db, pid, bases):
    """按 relative_path 批量取回 source id（分小批，避开 SQLite 变量上限）。"""
    out = {}
    bases = list(bases)
    for i in range(0, len(bases), IN_MAX):
        part = bases[i:i + IN_MAX]
        for rel, sid in db.execute(
            select(Source.relative_path, Source.id).where(
                Source.project_id == pid, Source.relative_path.in_(part)
            )
        ):
            out[rel] = sid
    return out


def _load_records(meta_path):
    """读 metadata.json，按 id 去重，返回 [(vid, path, rec), ...]（按 id 升序）。"""
    meta = json.load(open(meta_path, encoding="utf-8"))
    seen = set()
    recs = []
    for m in meta or []:
        if not isinstance(m, dict):
            continue
        p = m.get("path")
        if not p:
            continue
        try:
            vid = int(m.get("id"))
        except (TypeError, ValueError):
            continue
        if vid in seen:
            continue
        seen.add(vid)
        recs.append((vid, str(p), m))
    recs.sort(key=lambda x: x[0])
    return recs


def _src_row(pid, base, path, rec):
    fname = rec.get("filename") or base
    src_dir = (rec.get("src_dir") or "").strip()
    root = src_dir if (src_dir and os.path.isabs(src_dir)) else (os.path.dirname(path) or ".")
    vinfo = rec.get("video_source") or (rec.get("extra") or {}).get("video_source")
    return {
        "source_id": str(uuid.uuid4())[:16],
        "project_id": pid,
        "source_type": SourceType.IMAGE,
        "source_root": root,
        "relative_path": base,
        "directory_chain": [c for c in (rec.get("directory_chain") or []) if isinstance(c, str)],
        "file_name": fname,
        "extension": os.path.splitext(fname)[1].lower(),
        "file_size": 0,
        "scan_status": SCAN_STATUS,
        "meta": {"video_source": vinfo} if isinstance(vinfo, dict) else {},
    }


def _asset_row(pid, sid, path, vid, rec):
    try:
        fi = int(rec.get("frame_index") or 0)
    except (TypeError, ValueError):
        fi = 0
    try:
        ts = float(rec.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    vinfo = rec.get("video_source") or (rec.get("extra") or {}).get("video_source")
    return {
        "asset_id": str(uuid.uuid4())[:16],
        "project_id": pid,
        "source_id": sid,
        "source_type": SourceType.IMAGE,
        "image_path": path,
        "frame_index": fi,
        "timestamp": ts,
        "total_frames": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "asset_metadata": {"video_source": vinfo} if isinstance(vinfo, dict) else None,
        "vector_id": vid,
        "status": AssetStatus.EXTRACTED,
    }


def backfill_project(db, proj, root, verify_only=False):
    meta_path = os.path.join(root, "projects", proj.name, "metadata.json")
    if not os.path.exists(meta_path):
        _log("  %-16s 无 metadata.json，跳过" % proj.name)
        return None
    t0 = time.time()
    recs = _load_records(meta_path)
    existing = set(
        v for (v,) in db.execute(select(Asset.vector_id).where(Asset.project_id == proj.id))
    )
    have_before = len(existing)
    todo = [r for r in recs if r[0] not in existing]
    _log("  %-16s metadata=%-7d 库内已有=%-7d 待补=%-7d" % (proj.name, len(recs), have_before, len(todo)))
    if verify_only or not todo:
        return {"name": proj.name, "meta": len(recs), "before": have_before, "added": 0, "sec": 0.0}

    src_map = dict(
        (r[0], r[1])
        for r in db.execute(
            select(Source.relative_path, Source.id).where(Source.project_id == proj.id)
        )
    )
    added = 0
    pending = set()       # 已插入但还没取回 id 的 basename
    for i in range(0, len(todo), CHUNK):
        part = todo[i:i + CHUNK]
        rows = []
        for attempt in range(TRIES):
            new_bases = []
            new_srcs = []
            for _vid, path, rec in part:
                base = os.path.basename(path)
                if base in src_map or base in pending:
                    continue
                new_bases.append(base)
                pending.add(base)
                new_srcs.append(_src_row(proj.id, base, path, rec))
            try:
                if new_srcs:
                    db.execute(Source.__table__.insert().prefix_with("OR IGNORE"), new_srcs)
                    # 同一事务内即可读回刚插入的 id（不另开提交，写锁只占这一次）
                    for base, sid in _lookup_sources(db, proj.id, list(pending)).items():
                        src_map[base] = sid
                        pending.discard(base)
                rows = []
                for _vid, path, rec in part:
                    sid = src_map.get(os.path.basename(path))
                    if sid is not None:
                        rows.append(_asset_row(proj.id, sid, path, _vid, rec))
                if rows:
                    db.execute(Asset.__table__.insert().prefix_with("OR IGNORE"), rows)
                db.commit()
                added += len(rows)
                break
            except OperationalError as e:
                db.rollback()
                # 回滚后本批新建的 source 已消失，内存映射必须一起撤掉，下轮重建
                for b in new_bases:
                    src_map.pop(b, None)
                    pending.discard(b)
                rows = []
                if "locked" not in str(e) and "busy" not in str(e):
                    raise
                if attempt == TRIES - 1:
                    raise RuntimeError(
                        "连续 %d 轮抢不到写锁（抽帧的长事务一直占着），本批 %d 条未写入" % (TRIES, len(part))
                    )
                wait = min(10 * (attempt + 1), 60)
                _log("     ...%s 写锁被抽帧占着，%d 秒后重试 (%d/%d)" % (proj.name, wait, attempt + 1, TRIES))
                time.sleep(wait)
        done = min(i + CHUNK, len(todo))
        if done == len(todo) or (i // CHUNK) % 10 == 0:
            el = max(time.time() - t0, 0.001)
            _log("     %-16s %d/%d 已补 %d 条 (%.0fs, %.0f 条/秒)"
                 % (proj.name, done, len(todo), added, el, added / el))
    return {"name": proj.name, "meta": len(recs), "before": have_before, "added": added,
            "sec": time.time() - t0}


def main():
    args = [a for a in sys.argv[1:]]
    verify_only = "--verify" in args
    only = [a for a in args if not a.startswith("-")]
    root = _root()
    db = get_db_session()
    try:
        db.execute(text("PRAGMA busy_timeout=%d" % BUSY_MS))
        projects = db.execute(select(Project).order_by(Project.id)).scalars().all()
        if only:
            projects = [p for p in projects if p.name in only]
        _log("补登记%s，项目目录: %s" % ("核对（只读）" if verify_only else "开始", os.path.join(root, "projects")))
        results = []
        for p in projects:
            try:
                r = backfill_project(db, p, root, verify_only=verify_only)
            except Exception as e:
                db.rollback()
                _log("  %-16s 补登记失败: %s" % (p.name, e))
                r = None
            if r:
                results.append(r)
        _log("")
        _log("=== 汇总（口径：metadata.json = 帧全集）===")
        tot_m = tot_a = 0
        for r in results:
            tot_m += r["meta"]
            tot_a += r["before"] + r["added"]
            _log("  %-16s metadata=%-7d 库内 %d -> %d  (%.0fs)"
                 % (r["name"], r["meta"], r["before"], r["before"] + r["added"], r["sec"]))
        gap = tot_m - tot_a
        _log("  合计 metadata=%d, 库内=%d, 落差=%d" % (tot_m, tot_a, gap))
        _log("全部完成" if not verify_only else "核对完成")
    finally:
        db.close()


if __name__ == "__main__":
    main()
