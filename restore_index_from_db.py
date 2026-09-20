# -*- coding: utf-8 -*-
"""重建 Oversea_欧洲 的 metadata.json + index.faiss：按 DB assets.vector_id 原始顺序逐帧重编码。

事故由来
--------
CIFS 不允许 rename 覆盖 -> faiss.write_index 只能直接写目标文件（非原子）。写盘中途被重启打断
就留下半截 index.faiss；`_load_project_context_cold` 里 faiss.read_index 失败时会**连 metadata
一起清空**，于是随后的"在线向量化"从空 metadata 重建，帧 id 与 DB 的 assets.vector_id 错位
（库里标签/检测会挂到别的帧上）。

为什么不用普通向量化恢复
------------------------
普通链路按 os.walk 顺序补帧、并且会用 basename 去重，出来的顺序与 vector_id 不一致，还是错位。
唯一可信快照是 DB：50,684 行 (image_path, frame_index, timestamp, video_source) 且 vector_id 连续
0..50683。按 vector_id 升序重编码 -> 新建 IndexFlatIP 顺序 add -> id 天然等于 vector_id。

编码路径
--------
直接复用 app.py 的 `SigLIPImageDataset` + 生产函数 `extract_and_index_project`（预处理 384/BICUBIC/
0.5 归一化与线上完全一致），模型自己加载后塞进 app 的全局变量，绕开 GPU Manager（避免在另一个
进程里跟在线服务抢显存管理）。

用法（服务器上，先确认该项目没有在跑的向量化/抽帧）
  /venv/bin/python -u /tmp/_restore_eu.py check
  /venv/bin/python -u /tmp/_restore_eu.py bench --n 300 --workers 4
  /venv/bin/python -u /tmp/_restore_eu.py run --chunk 4000
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, "/opt/ad_mining")   # 必须：/tmp 下同名文件会盖住项目模块

PROJECT = os.environ.get("REC_PROJECT", "Oversea_欧洲")
DB_PATH = os.environ.get("REC_DB", "/opt/ad_mining/workspace/mining.db")
PROJ_DIR = "/mnt/Data_Platform/zhj_datamining/test/projects/" + PROJECT
META_PATH = os.path.join(PROJ_DIR, "metadata.json")
IDX_PATH = os.path.join(PROJ_DIR, "index.faiss")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def db_rows():
    """DB 里的权威顺序：vector_id 升序。返回 [(vid, path, frame_index, timestamp, video_source)]"""
    con = sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True)
    try:
        rows = con.execute(
            "select a.vector_id, a.image_path, a.frame_index, a.timestamp, a.asset_metadata "
            "from assets a join projects p on p.id = a.project_id "
            "where p.name = ? order by a.vector_id",
            (PROJECT,),
        ).fetchall()
    finally:
        con.close()
    out = []
    for vid, path, fi, ts, am in rows:
        vs = None
        if am:
            try:
                vs = (json.loads(am) or {}).get("video_source")
            except Exception:
                vs = None
        if vs:
            # 原记录就是这么存的：video_source 整个 dict 含 frame_index/timestamp/文件名/路径
            out.append((int(vid), path, fi, ts, vs))
        else:
            out.append((int(vid), path, fi, ts, None))
    return out


def frame_meta_of(rows):
    """只给"有 video_source"的帧做 frame_meta —— 生产函数对没有条目的帧会写 video_source=None，
    正好等于原记录；若给它一个只有 frame_index/timestamp 的 dict，反而会写成一个非空
    video_source（与库里不一致）。已核对：所有无 video_source 的帧 frame_index/timestamp 均为 0。"""
    fm = {}
    for _vid, path, _fi, _ts, vs in rows:
        if vs:
            fm[path] = dict(vs)
    return fm


def load_siglip():
    """加载 SigLIP 并塞进 app 的全局变量，使生产函数直接可用（不经过 GPU Manager）。"""
    import torch
    import app as A
    from transformers import AutoProcessor, AutoModel
    if A.siglip_model is not None:
        log("SigLIP 已在 app 全局")
        return A.siglip_model, A.siglip_processor
    log("加载 SigLIP (%s) ..." % A.SIGLIP_MODEL_NAME)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(A.SIGLIP_MODEL_NAME, local_files_only=True)
    mdl = AutoModel.from_pretrained(
        A.SIGLIP_MODEL_NAME,
        torch_dtype=torch.float16 if A.DEVICE == "cuda" else torch.float32,
        local_files_only=True,
    ).to(A.DEVICE)
    mdl.eval()
    A.siglip_model, A.siglip_processor = mdl, proc
    log("SigLIP 就绪，用时 %.1fs" % (time.time() - t0))
    return mdl, proc


def mem_mb():
    try:
        with open("/proc/meminfo") as f:
            d = dict(l.split(":", 1) for l in f if ":" in l)
        return int(d["MemAvailable"].split()[0]) // 1024
    except Exception:
        return -1


def cmd_check(args):
    import app as A  # noqa: F401  （只为拿到 FEAT_DIM 等常量）
    rows = db_rows()
    log("DB 行数 %d，vector_id %d..%d" % (len(rows), rows[0][0], rows[-1][0]))
    gap = [i for i, r in enumerate(rows) if r[0] != i]
    log("vector_id 连续性: %s" % ("连续 0..N-1 ✓" if not gap else "断档 %d 处，例 %s" % (len(gap), gap[:5])))
    with_vs = sum(1 for r in rows if r[4])
    log("带 video_source 的帧 %d，不带的 %d" % (with_vs, len(rows) - with_vs))
    t0 = time.time()
    missing = [r for r in rows if not os.path.exists(r[1])]
    log("存在性检查耗时 %.1fs：缺失 %d 帧" % (time.time() - t0, len(missing)))
    if missing:
        with open("/tmp/eu_missing.txt", "w", encoding="utf-8") as f:
            for r in missing:
                f.write("%d\t%s\n" % (r[0], r[1]))
        log("缺失清单已写 /tmp/eu_missing.txt（前 5 条：%s）" % [(m[0], os.path.basename(m[1])) for m in missing[:5]])
    log("当前 metadata.json 条数 %s / index 向量数 %s" % (_meta_len(), _idx_ntotal()))
    log("FEAT_DIM=%d" % A.FEAT_DIM)


def _meta_len():
    try:
        with open(META_PATH, encoding="utf-8") as f:
            return len(json.load(f))
    except Exception as e:
        return "读取失败(%s)" % e


def _idx_ntotal():
    try:
        import faiss
        return faiss.read_index(IDX_PATH).ntotal
    except Exception as e:
        return "读取失败(%s)" % e


def _fresh_ctx():
    import faiss
    import app as A
    return {
        "name": A.get_project_paths(PROJECT)[0],
        "dir": PROJ_DIR,
        "img_dir": os.path.join(PROJ_DIR, "images"),
        "idx_path": IDX_PATH,
        "meta_path": META_PATH,
        "index": faiss.IndexFlatIP(A.FEAT_DIM),
        "metadata": [],
    }


def cmd_bench(args):
    import app as A
    rows = db_rows()
    step = max(1, len(rows) // args.n)
    sample = rows[::step][: args.n]
    paths = [r[1] for r in sample]
    fm = frame_meta_of(sample)
    ctx = _fresh_ctx()
    # ⚠️ bench 的落盘路径改成 /tmp：生产函数每批都会 save_project_context，
    # 不换路径就会把样本写进真实的项目目录，把待恢复的文件又覆盖一次。
    ctx["idx_path"] = "/tmp/_bench_eu.faiss"
    ctx["meta_path"] = "/tmp/_bench_eu.json"
    log("bench: %d 帧，%s（内存 %d MB）" % (len(paths), set_workers(args.workers), mem_mb()))
    load_siglip()
    t0 = time.time()
    n = A.extract_and_index_project(ctx, paths, fm)
    dt = time.time() - t0
    log("bench 完成：入索引 %d 帧，用时 %.1fs → %.2f 帧/秒，可用内存 %d MB"
        % (n, dt, n / max(dt, 1e-6), mem_mb()))
    log("index.ntotal=%d len(metadata)=%d（写在 /tmp/_bench_eu.*，未碰项目目录）"
        % (ctx["index"].ntotal, len(ctx["metadata"])))


def set_workers(w):
    """让生产函数用指定解码并行度：走自适应档位，再用上限把 batch/workers 夹到要的值。
    （自适应被上限压过时 prefetch 会自动压回 1，与线上口径一致）"""
    if w is None:
        return "默认 32/2/1"
    os.environ["AD_VEC_ADAPTIVE"] = "1"
    os.environ["AD_VEC_BATCH_MAX"] = "32"
    os.environ["AD_VEC_WORKERS_MAX"] = str(int(w))
    return "adaptive 上限 workers=%s（batch 上限 32）" % w


def cmd_run(args):
    import faiss
    import app as A
    rows = db_rows()
    total = len(rows)
    fm = frame_meta_of(rows)

    have = 0
    ctx = _fresh_ctx()
    if args.resume:
        try:
            with open(META_PATH, encoding="utf-8") as f:
                md = json.load(f)
            idx = faiss.read_index(IDX_PATH)
            # 只认"前缀完全对齐"的部分，且索引向量数不少于记录数
            k = 0
            while k < min(len(md), total) and md[k].get("path") == rows[k][1]:
                k += 1
            if k and idx.ntotal >= k:
                if idx.ntotal > k:
                    idx.remove_ids(faiss.IDSelectorRange(k, idx.ntotal))
                ctx["metadata"], ctx["index"], have = md[:k], idx, k
                log("断点续跑：已有 %d 帧与前缀对齐，从第 %d 帧继续（index.ntotal=%d）"
                    % (k, k, idx.ntotal))
            else:
                log("未发现可用断点（累进 %d，index.ntotal=%d），从头开始" % (k, idx.ntotal))
        except Exception as e:
            log("读取断点失败(%s)，从头开始" % e)

    if not args.no_backup and os.path.exists(META_PATH):
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (META_PATH, IDX_PATH):
            if os.path.exists(p):
                dst = "%s.bak_%s" % (p, ts)
                shutil.copy2(p, dst)
                log("已备份 -> %s" % dst)

    _w = set_workers(args.workers)   # 必须在 import app / 调用生产函数前设好环境变量
    load_siglip()
    log("开始重建：共 %d 帧，chunk=%d，%s，起始可用内存 %d MB"
        % (total, args.chunk, _w, mem_mb()))
    t0 = time.time()
    for i in range(have, total, args.chunk):
        chunk = rows[i:i + args.chunk]
        paths = [r[1] for r in chunk]
        got = A.extract_and_index_project(ctx, paths, fm)
        if got != len(chunk):
            # 生产函数会静默跳过错帧（文件坏了/读不出）；这会打断 id 对齐，必须在对应位置补零向量
            log("⚠️ 本批入索引 %d != %d，逐帧定位补零向量" % (got, len(chunk)))
            _pad(ctx, chunk, fm)
        if ctx["index"].ntotal != len(ctx["metadata"]):
            raise SystemExit("❌ 索引 %d != metadata %d，对齐已破，停手" % (ctx["index"].ntotal, len(ctx["metadata"])))
        el = time.time() - t0
        done = i + len(chunk)
        log("进度 %d/%d（%.1f%%）用时 %.0fs（%.2f 帧/秒），ETA %.0f 分钟，内存 %d MB"
            % (done, total, 100.0 * done / total, el, done / max(el, 1e-6),
               (total - done) / max(done / max(el, 1e-6), 1e-6) / 60.0, mem_mb()))
    A.save_project_context(ctx)
    log("✅ 落盘完成：index.ntotal=%d metadata=%d" % (ctx["index"].ntotal, len(ctx["metadata"])))
    _verify(rows)


def _pad(ctx, chunk, fm):
    """逐帧复查这一批：生产函数漏掉的帧（读不出/不存在）补零向量，保证 id 与 vector_id 一一对应。"""
    import numpy as np
    import app as A
    have_paths = set(m["path"] for m in ctx["metadata"])
    for vid, path, _fi, _ts, _vs in chunk:
        if path in have_paths:
            continue
        log("  ↳ 补零向量 vid=%d %s" % (vid, path))
        ctx["index"].add(np.zeros((1, A.FEAT_DIM), dtype="float32"))
        cur = len(ctx["metadata"])
        ctx["metadata"].append({
            "id": cur,
            "filename": os.path.basename(path),
            "path": path,
            "url": "/api/image/%s/%d" % (ctx["name"], cur),
            "frame_index": (fm.get(path) or {}).get("frame_index", 0),
            "timestamp": (fm.get(path) or {}).get("timestamp", 0.0),
            "video_source": fm.get(path) or None,
        })
        have_paths.add(path)


def _verify(rows):
    import faiss
    with open(META_PATH, encoding="utf-8") as f:
        md = json.load(f)
    idx = faiss.read_index(IDX_PATH)
    bad = [i for i in range(min(len(md), len(rows))) if md[i].get("path") != rows[i][1]]
    log("校验：DB %d 条 / metadata %d 条 / index.ntotal %d，路径错位 %d 处"
        % (len(rows), len(md), idx.ntotal, len(bad)))
    if bad:
        log("❗ 错位示例 %s" % [(i, md[i].get("filename"), os.path.basename(rows[i][1])) for i in bad[:5]])
    else:
        log("✅ 逐帧对齐：metadata[i] == DB vector_id i")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--workers", type=int, default=None)
    r = sub.add_parser("run")
    r.add_argument("--chunk", type=int, default=4000)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    {"check": cmd_check, "bench": cmd_bench, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
