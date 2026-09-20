# -*- coding: utf-8 -*-
# clip_service.py  —— 放在 app.py 同级目录
import os, json, re, threading
import numpy as np

_clips_lock = threading.Lock()

# ----------------- 图片连续帧的序列识别（直导帧没有 video_source，也能按"段"切） -----------------
_V_RE = re.compile(r"^(v\d+)_(\d+)_(\d+)\.jpg$", re.I)          # Clip 链路抽帧：v<N>_<毫秒>_<序号>
_MS_RE = re.compile(r"^(.+)_(\d{13})(_.*)?\.jpg$", re.I)        # 相机连拍：100002_1783714217483_13_14.jpg
_IDX_RE = re.compile(r"^(.+?)(\d{3,})\.jpg$")                   # 尾部序号：IMG_000123.jpg

def seq_key(path_or_name):
    """从文件名提取序列线索。返回 (run_id, mode, value) 或 None。

      **分桶视频帧**：images/<视频名>/<视频名>_<源帧号>.jpg   -> ("bucket:<目录>", "seq", (0, 帧号))
        文件名前缀 == 所在目录名 → 该目录就是一个视频，同目录帧天然连续（序号是源帧号）。
        这条**必须放在最前**：否则会落到下面的通用序号规则，而源帧号按抽帧步长
        （step=5/10）递增，被"序号差>2 就切"判成不连续，28.6 万帧会切成 28.6 万个单帧段
        （2026-09-20 实测：南美 clips 膨胀到 30 万段、内存 11.7G）。
      v<N>_<毫秒>_<序号>.jpg   -> ("v:N", "seq", (毫秒, 序号))
      <帧id>_<13位毫秒>_<后缀>  -> ("ms:<目录>:<后缀>", "ms", 毫秒)   run 内按毫秒排，间隔 > gap 切段
      <前缀><≥3位序号>.jpg     -> ("idx:<目录>:<前缀>", "idx", 序号)  run 内序号差 > idx_gap 切段

    纯数字文件名（123.jpg，可能是帧 id）和完全无数字线索的不算序列 —— 宁可不分段，
    也不把一批无关图片硬拼成一段（拼了就会共享一份段级标签）。"""
    _fn = os.path.basename((path_or_name or "").replace("\\", "/"))
    _d = os.path.dirname((path_or_name or "").replace("\\", "/"))
    # ① 分桶视频帧：basename 去掉最后的 _<数字>.jpg 后与目录名相同
    _base = _fn[:-4] if _fn.lower().endswith(".jpg") else _fn
    _dname = os.path.basename(_d)
    if _dname and "_" in _base:
        _stem, _tail = _base.rsplit("_", 1)
        if _stem == _dname and _tail.isdigit():
            return ("bucket:" + _d, "seq", (0, int(_tail)))
    m = _V_RE.match(_fn)
    if m:
        return ("v:%s" % m.group(1), "seq", (int(m.group(2)), int(m.group(3))))
    m = _MS_RE.match(_fn)
    if m:
        return ("ms:%s:%s" % (_d, m.group(3) or ""), "ms", int(m.group(2)))
    m = _IDX_RE.match(_fn)
    if m:
        return ("idx:%s:%s" % (_d, m.group(1)), "idx", int(m.group(2)))
    return None

def split_seq_runs(rows, gap_ms=None, idx_gap=None):
    """rows: [(run_id, mode, value, payload)] → 按连续性切成段，返回 [[payload,...], ...]。

    同一 run_id 内按 value 排序：
      seq 模式：序号回退即切（v<N> 是任务内自编的，跨任务重名，回退=新一段视频）；
      ms  模式：相邻间隔 > AD_SEQ_GAP_MS（默认 10000ms）即切；
      idx 模式：序号差 > AD_SEQ_IDX_GAP（默认 2）即切。"""
    if gap_ms is None:
        gap_ms = int(os.environ.get("AD_SEQ_GAP_MS", "10000"))
    if idx_gap is None:
        idx_gap = int(os.environ.get("AD_SEQ_IDX_GAP", "2"))
    buckets = {}
    for rid, mode, val, payload in rows:
        buckets.setdefault((rid, mode), []).append((val, payload))
    runs = []
    for (rid, mode), items in buckets.items():
        items.sort(key=lambda x: x[0])
        run, prev = [], None
        for val, payload in items:
            cut = False
            if prev is not None and run:
                if mode == "seq":
                    cut = val[1] <= prev[1]        # 序号回退 = 新一段
                elif mode == "ms":
                    cut = (val - prev) > gap_ms
                else:
                    cut = (val - prev) > idx_gap
            if cut:
                runs.append(run)
                run = []
            run.append(payload)
            prev = val
        if run:
            runs.append(run)
    return runs

def _slug(s):
    return re.sub(r"[^0-9A-Za-z_-]", "_", s or "")[-48:]

def build_seq_clips(metadata, clip_size=30):
    """无 video_source 的连续帧 → seq_key 聚类 → 连续 run → clip_size 窗口。
    返回与 build_clips_from_metadata 相同结构的 clip 列表（段内共享一份判定）。"""
    rows = []
    for m in metadata:
        if m.get("video_source"):
            continue
        sk = seq_key(m.get("path") or m.get("filename") or "")
        if sk:
            rows.append((sk[0], sk[1], sk[2], m))
    clips = []
    for run in split_seq_runs(rows):
        if not run:
            continue
        head = run[0]
        tag = _slug(head.get("path") or head.get("filename") or "")
        # 单帧/双帧的"段"没有段级语义（场景/事件判断需要帧间变化），不生成 Clip ——
        # 否则零散残帧会各占一次 VLM 判定（实测南美出现过 28.6 万个这样的段）
        _min_len = int(os.environ.get("AD_SEQ_MIN_FRAMES", "5"))
        if len(run) < _min_len:
            continue
        for i in range(0, len(run), clip_size):
            window = run[i:i + clip_size]
            clips.append({
                "clip_id": "seq_%s_%05d" % (tag, i // clip_size),
                "video_id": "seq:" + (os.path.dirname((head.get("path") or "").replace("\\", "/")) or "images"),
                "clip_index": i // clip_size,
                "start_timestamp": None,
                "end_timestamp": None,
                "frame_ids": [f["id"] for f in window],
                "frame_paths": [f["path"] for f in window],
                "vlm_result": None,
                "human_tags": {},
                "final_tags": {},
                "decision": None,
            })
    return clips

def _mk_clip(cid, vid, idx, t0, t1, window):
    return {
        "clip_id": cid,
        "video_id": vid,
        "clip_index": idx,
        "start_timestamp": t0,
        "end_timestamp": t1,
        "frame_ids": [f["id"] for f in window],
        "frame_paths": [f["path"] for f in window],
        "vlm_result": None,      # 分析结果回填到这里
        "human_tags": {},
        "final_tags": {},
        "decision": None,        # AUTO_PASS / REVIEW / APPROVED / FILTERED
    }

def build_clips_from_metadata(ctx, clip_size=30):
    """把底库中已抽帧的图片按 video_source + 时间戳分组成 30 帧窗口（Clip 统一 30 帧）。
    重建时保留同 clip_id 的已有结果(vlm_result/decision/human_tags)，否则重跑 scan 会冲掉
    全部 AI 判定与人工审核结果。

    2026-09-19 起：**没有 video_source 的直导图片连续帧也参与分段**（与视频同一逻辑：
    30 帧一段、段内均匀抽 5 帧送 VLM）—— 相机连拍 <帧id>_<13位毫秒>_<后缀>.jpg、
    尾部序号 IMG_000123.jpg、Clip 链路抽帧 v<N>_<毫秒>_<序号>.jpg 都认；完全无数字
    线索的帧仍不参与（保持单帧管线），宁缺毋滥。"""
    try:
        prev = {c.get("clip_id"): c for c in (load_clips(ctx) or []) if c.get("clip_id")}
    except Exception:
        prev = {}  # ctx 无 dir 等精简调用场景：无历史可沿用
    by_video = {}
    for m in ctx["metadata"]:
        vs = m.get("video_source") or {}
        vid = vs.get("video_path")          # 视频抽的帧才有
        if vid:
            by_video.setdefault(vid, []).append(m)   # 无 video_source 的由 build_seq_clips 处理

    clips = []
    for vid, frames in by_video.items():
        frames.sort(key=lambda x: (x.get("timestamp") or 0, x.get("frame_index") or 0))
        for i in range(0, len(frames), clip_size):
            window = frames[i:i + clip_size]
            cid = f"{os.path.splitext(os.path.basename(vid))[0]}_{i // clip_size:05d}"
            clips.append(_mk_clip(cid, vid, i // clip_size,
                                  window[0].get("timestamp"), window[-1].get("timestamp"), window))
    # 直导图片连续帧：与视频同一逻辑（30 帧一段，段内共享一份判定）
    for c in build_seq_clips(ctx["metadata"], clip_size):
        clips.append(c)
    # 沿用旧判定：仅当帧集合完全一致才继承（改了 clip_size 会生成同名但内容不同的 Clip）
    out = []
    for clip in clips:
        old = prev.get(clip["clip_id"])
        if old and (old.get("frame_ids") or []) == clip["frame_ids"]:
            for k in ("vlm_result", "human_tags", "final_tags", "decision"):
                if k in old:
                    clip[k] = old[k]
        out.append(clip)
    return out


def _clips_path(ctx):
    """clips.json 落**本地 NVMe**（与 index.faiss/metadata.json 同处）。

    ⚠️ 不能放 NAS：① 段数可达数万、文件几十~几百 MB，写 CIFS 极慢；
    ② CIFS 不允许 rename，做不到原子写；③ 网盘删不掉文件，一旦写出膨胀文件
    就永久占空间（2026-09-20 实测：南美一度生成 30 万段 / 253MB 的 clips.json，
    删不掉也没法原子替换）。"""
    _store = os.environ.get("AD_INDEX_STORE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_store"))
    d = os.path.join(_store, ctx.get("name") or "default")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "clips.json")

def _atomic_write_json(path, obj):
    """tmp + os.replace 原子写（本地 NVMe 支持）。多 MB 的大 JSON 直接覆盖写，
    一旦被重启/并发打断就留下截断文件，下次 load 直接解析失败 —— 实测发生过两次
    （NAS 上 243MB 膨胀版、本地 136MB 正常版）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def save_clips(ctx, clips):
    os.makedirs(ctx["dir"], exist_ok=True)  # 项目目录可能不存在(DB 建项目不建目录)，先补齐
    with _clips_lock:
        p = _clips_path(ctx)
        # 写新内容前把上一版留一份 .bak：判定结果（人工+模型）是最贵的数据，
        # 截断/误写时能立刻回退（网盘读慢但**本地拷贝很快**，值得）
        try:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                import shutil as _sh
                _sh.copy2(p, p + ".bak")
        except Exception:
            pass
        _atomic_write_json(p, clips)

_clips_cache = {"key": None, "data": None}   # (路径, mtime, size) -> 解析结果
def load_clips(ctx):
    """读 clips.json，带 mtime 缓存：数万段时这是上百 MB 的大 JSON，
    列表页每次请求全量 parse 会卡前端；文件没变就直接复用上次解析结果。
    损坏时自动回退 .bak（绝不静默返回空 —— 那等于把全部判定结果"弄丢"）。"""
    p = _clips_path(ctx)
    if not os.path.exists(p):
        return []
    try:
        st = os.stat(p)
        key = (p, st.st_mtime_ns, st.st_size)
        if _clips_cache["key"] == key:
            return _clips_cache["data"]
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        _clips_cache["key"] = key
        _clips_cache["data"] = data
        return data
    except Exception as e:
        print(f"[clips] ⚠️ {p} 解析失败({e})，尝试 .bak 回退", flush=True)
        try:
            bak = p + ".bak"
            if os.path.exists(bak):
                with open(bak, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[clips] 已从 .bak 恢复（{len(data)} 段）", flush=True)
                _clips_cache["key"] = None
                return data
        except Exception as e2:
            print(f"[clips] .bak 也不可用: {e2}", flush=True)
        return []

def update_clip_result(ctx, clip_id, vlm_result, decision=None):
    """单clip结果回写（带锁，防止并发写坏）"""
    os.makedirs(ctx["dir"], exist_ok=True)
    with _clips_lock:
        clips = load_clips(ctx)
        for c in clips:
            if c["clip_id"] == clip_id:
                c["vlm_result"] = vlm_result
                if decision:
                    c["decision"] = decision
                break
        _atomic_write_json(_clips_path(ctx), clips)

def apply_human_tags(ctx, clip_id, human_tags, decision="APPROVED"):
    """人工标注落盘：human_tags = 人工值，final_tags 一并设为人工值（最终标签以人工为准）。
    ⚠️ 不动 vlm_result —— 它是对比评测的依据（模型当初判了什么必须可追溯）。

    返回 True/False（找到并写入）。"""
    os.makedirs(ctx["dir"], exist_ok=True)
    with _clips_lock:
        clips = load_clips(ctx)
        hit = False
        for c in clips:
            if c["clip_id"] == clip_id:
                c["human_tags"] = {k: list(v) for k, v in (human_tags or {}).items()}
                c["final_tags"] = dict(c["human_tags"])
                if decision:
                    c["decision"] = decision
                hit = True
                break
        if hit:
            _atomic_write_json(_clips_path(ctx), clips)
            _clips_cache["key"] = None      # 让 mtime 缓存失效，下次读取拿到新版本
    return hit


def set_clip_decision(ctx, clip_id, decision, human=None):
    """人工审核回写：只改 decision（可选记录人工标记），不动 VLM 结果。找到返回 True。"""
    os.makedirs(ctx["dir"], exist_ok=True)
    found = False
    with _clips_lock:
        clips = load_clips(ctx)
        for c in clips:
            if c["clip_id"] == clip_id:
                c["decision"] = decision
                if human is not None:
                    c["human_tags"] = human
                found = True
                break
        if found:
            _atomic_write_json(_clips_path(ctx), clips)
    return found
