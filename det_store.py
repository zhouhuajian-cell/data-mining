# -*- coding: utf-8 -*-
r"""目标检测记录存储层：SQLite 落盘 + 内存 LRU（生产级「千万帧」改造 P0）。

为什么：原实现在 `app.detections_cache` 里把**所有项目所有帧**的检测框全量常驻内存
（2026-10-03 实测：90.2 万帧 + 全项目 144 万条 ≈ 4 GB，且每检测一批就往上加），
加上向量索引 3.9G 后服务 RSS 到 22.8G、机器 31G 只剩 5G 可用。本模块把冷帧挪到本地 NVMe，
内存只留最近用过的 N 帧 —— 内存不再随帧数线性增长。

对外接口刻意做成"像原来的 dict-of-dict"（`store[项目]` / `.get(帧号)` / `[帧号]=记录`），
这样 app.py 的调用点几乎不用改。SQLite 用 WAL、写入批量提交（避免每帧一次 fsync）。

开关：环境变量 `AD_DET_STORE=sqlite|json`（默认 json = 老行为，零风险回滚）。
      `AD_DET_LRU` 控制内存里留多少帧（默认 50000）。
"""
import json
import os
import sqlite3
import threading
from collections import OrderedDict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS det (
  project TEXT NOT NULL,
  fid     TEXT NOT NULL,
  rec     TEXT NOT NULL,
  PRIMARY KEY (project, fid)
);
CREATE INDEX IF NOT EXISTS idx_det_project ON det(project);
"""


class _ProjView:
    """单个项目的检测记录视图：对外表现得像个 dict（str(帧号) -> 记录 dict）。"""

    __slots__ = ("_st", "_name")

    def __init__(self, store, name):
        self._st = store
        self._name = name

    # ---- 读 ----
    def get(self, fid, default=None):
        return self._st._get(self._name, fid, default)

    def __getitem__(self, fid):
        v = self._st._get(self._name, fid, _MISS)
        if v is _MISS:
            raise KeyError(fid)
        return v

    def __contains__(self, fid):
        return self._st._has(self._name, fid)

    # ---- 写 ----
    def __setitem__(self, fid, rec):
        self._st._put(self._name, fid, rec)

    def setdefault(self, fid, default=None):
        cur = self._st._get(self._name, fid, _MISS)
        if cur is _MISS:
            self._st._put(self._name, fid, default)
            return default
        return cur

    def update(self, other):
        for k, v in dict(other).items():
            self._st._put(self._name, k, v)

    # ---- 遍历（注意：会全量物化，大项目慎用）----
    def items(self):
        for fid, rec in self._st._iter(self._name):
            yield fid, rec

    def keys(self):
        for fid, _ in self._st._iter(self._name):
            yield fid

    def values(self):
        for _, rec in self._st._iter(self._name):
            yield rec

    def __iter__(self):
        for fid, _ in self._st._iter(self._name):
            yield fid

    def __len__(self):
        return self._st._count(self._name)

    def to_dict(self):
        """物化成普通 dict（仅用于 /api/get_all_detections 这类全量接口；大项目会占内存）。"""
        return {fid: rec for fid, rec in self._st._iter(self._name)}


_MISS = object()


class DetStore:
    """检测记录存储：SQLite（本地 NVMe）+ 内存 LRU。"""

    def __init__(self, db_path, lru_max=50000):
        self._path = db_path
        self._lru_max = max(1000, int(lru_max))
        self._lru = OrderedDict()          # (project, fid) -> rec
        self._lock = threading.RLock()
        self._pending = 0
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.executescript(_SCHEMA)
        self._con.commit()
        self._views = {}

    # ---- 项目视图 ----
    def get(self, project, default=None):
        return self._views.setdefault(project, _ProjView(self, project))

    def setdefault(self, project, default=None):
        return self.get(project)

    def __getitem__(self, project):
        return self.get(project)

    def __contains__(self, project):
        return self._count(project) > 0

    def values(self):
        return [self.get(p) for p in self._projects()]

    def items(self):
        return [(p, self.get(p)) for p in self._projects()]

    def _projects(self):
        with self._lock:
            cur = self._con.execute("SELECT DISTINCT project FROM det")
            return [r[0] for r in cur.fetchall()]

    def __len__(self):
        return len(self._projects())

    # ---- 内部读写 ----
    def _get(self, project, fid, default):
        key = (project, str(fid))
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
            cur = self._con.execute("SELECT rec FROM det WHERE project=? AND fid=?",
                                    (project, str(fid)))
            row = cur.fetchone()
            if row is None:
                return default
            try:
                rec = json.loads(row[0])
            except Exception:
                return default
            self._touch(key, rec)
            return rec

    def _has(self, project, fid):
        return self._get(project, fid, _MISS) is not _MISS

    def _put(self, project, fid, rec):
        key = (project, str(fid))
        with self._lock:
            self._touch(key, rec)
            self._con.execute("INSERT OR REPLACE INTO det(project,fid,rec) VALUES(?,?,?)",
                              (project, str(fid), json.dumps(rec, ensure_ascii=False)))
            self._pending += 1
            if self._pending >= 200:       # 批量提交，避免每帧一次 fsync
                self._con.commit()
                self._pending = 0

    def _touch(self, key, rec):
        self._lru[key] = rec
        self._lru.move_to_end(key)
        while len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)

    def _iter(self, project):
        with self._lock:
            cur = self._con.execute("SELECT fid, rec FROM det WHERE project=?", (project,))
            rows = cur.fetchall()
        for fid, rec in rows:
            try:
                yield fid, json.loads(rec)
            except Exception:
                continue

    def _count(self, project):
        with self._lock:
            cur = self._con.execute("SELECT COUNT(*) FROM det WHERE project=?", (project,))
            return int(cur.fetchone()[0])

    def flush(self):
        with self._lock:
            self._con.commit()
            self._pending = 0

    def lru_size(self):
        return len(self._lru)

    def lru_max(self):
        return self._lru_max


def migrate_from_json(store, json_path):
    """把老的 detections_cache.json 一次性灌进 SQLite（幂等；已存在的不覆盖）。"""
    if not os.path.exists(json_path):
        return 0
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0
    n = 0
    if isinstance(data, dict):
        for proj, sub in data.items():
            if not isinstance(sub, dict):
                continue
            rows = [(proj, str(k), json.dumps(v, ensure_ascii=False)) for k, v in sub.items()]
            with store._lock:
                store._con.executemany(
                    "INSERT OR IGNORE INTO det(project,fid,rec) VALUES(?,?,?)", rows)
                store._con.commit()
            n += len(rows)
    return n
