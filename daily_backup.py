# -*- coding: utf-8 -*-
"""每日备份 + 一致性对账（cron 03:30 跑，见 /etc/cron.d/ad_mining_backup）。

1) mining.db 每日 VACUUM INTO 一份（在线安全，WAL 模式可边跑边备），保留最近 7 份；
2) 索引/metadata 快照：index_store -> backups/index_store（rsync 增量，保留最新一份；
   索引可由帧重编码再生，无需多版本）；
3) 三方一致性对账：FAISS ntotal == metadata 条数 == DB assets 数，不一致写告警日志。

崩溃恢复口径：mining.db 用 backups/db 最新一份；index_store 用 backups/index_store 或
NAS 上的旧版（代码在冷加载时会自动回迁）。
"""
import datetime as _dt
import json
import os
import shutil
import sqlite3
import subprocess
import sys

sys.path.insert(0, "/opt/ad_mining")
os.chdir("/opt/ad_mining")
import faiss  # noqa: E402
from settings import db_path as _db_path  # noqa: E402
import app as _A  # noqa: E402  （复用 FEAT_DIM / LOCAL_INDEX_STORE，保持单一来源）

BASE = "/opt/ad_mining/backups"
DB_BAK = os.path.join(BASE, "db")
IDX_BAK = os.path.join(BASE, "index_store")
KEEP = 7
FEAT_DIM = _A.FEAT_DIM


def log(m):
    print("[%s] %s" % (_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), m), flush=True)


def backup_db():
    src = _db_path()
    os.makedirs(DB_BAK, exist_ok=True)
    dst = os.path.join(DB_BAK, "mining_%s.db" % _dt.date.today().strftime("%Y%m%d"))
    t0 = _dt.datetime.now()
    con = sqlite3.connect(src)
    try:
        con.execute("VACUUM INTO ?", (dst,))
    finally:
        con.close()
    log("DB 备份 -> %s (%.0f MB, %.1fs)" % (dst, os.path.getsize(dst) / 1048576,
                                           (_dt.datetime.now() - t0).total_seconds()))
    olds = sorted(os.listdir(DB_BAK))[:-KEEP]
    for f in olds:
        os.remove(os.path.join(DB_BAK, f))
        log("清理旧备份 %s" % f)


def snapshot_index_store():
    t0 = _dt.datetime.now()
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", "--delete",
                        _A.LOCAL_INDEX_STORE.rstrip("/") + "/", IDX_BAK + "/"], check=True)
    else:
        if os.path.exists(IDX_BAK):
            shutil.rmtree(IDX_BAK)
        shutil.copytree(_A.LOCAL_INDEX_STORE, IDX_BAK)
    n = sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(IDX_BAK) for f in fs)
    log("索引快照 -> %s (%.1f GB, %.0fs)" % (IDX_BAK, n / 1e9,
                                            (_dt.datetime.now() - t0).total_seconds()))


def integrity_check():
    """三方对账：ntotal == len(metadata) == DB assets 数。任何不一致都写 ⚠️ 行。"""
    import sqlite3 as _sq
    db = _sq.connect("file:%s?mode=ro" % _db_path(), uri=True)
    db_names = dict(db.execute("select name, id from projects").fetchall())
    problems = []
    for name in sorted(os.listdir(_A.LOCAL_INDEX_STORE)):
        idx_p = os.path.join(_A.LOCAL_INDEX_STORE, name, "index.faiss")
        meta_p = os.path.join(_A.LOCAL_INDEX_STORE, name, "metadata.json")
        if not (os.path.exists(idx_p) and os.path.exists(meta_p)):
            continue
        try:
            nt = faiss.read_index(idx_p).ntotal
            nm = len(json.load(open(meta_p, encoding="utf-8")))
            nd = 0
            if name in db_names:
                nd = db.execute("select count(*) from assets where project_id=?",
                                (db_names[name],)).fetchone()[0]
            tag = "OK" if nt == nm == nd else "⚠️"
            if tag != "OK":
                problems.append((name, nt, nm, nd))
            log("%s %-20s ntotal=%d metadata=%d db_assets=%d" % (tag, name, nt, nm, nd))
        except Exception as e:
            problems.append((name, -1, -1, -1))
            log("⚠️ %-20s 校验异常: %s" % (name, e))
    db.close()
    if problems:
        log("⚠️⚠️ 一致性对账发现 %d 处不一致: %s —— 处理前先看 AGENTS.md「索引损坏」章节，"
            "禁用向量化并人工介入" % (len(problems), problems))
    else:
        log("✅ 全部项目三方一致")


if __name__ == "__main__":
    log("===== 每日备份开始 =====")
    try:
        backup_db()
    except Exception as e:
        log("❌ DB 备份失败: %s" % e)
    try:
        snapshot_index_store()
    except Exception as e:
        log("❌ 索引快照失败: %s" % e)
    try:
        integrity_check()
    except Exception as e:
        log("❌ 对账失败: %s" % e)
    log("===== 每日备份结束 =====")
