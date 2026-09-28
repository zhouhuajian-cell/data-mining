# -*- coding: utf-8 -*-
# clip_service.py  —— 放在 app.py 同级目录
import os, json, re, threading, time
import numpy as np

_clips_lock = threading.Lock()

# ----------------- 图片连续帧的序列识别（直导帧没有 video_source，也能按"段"切） -----------------
_V_RE = re.compile(r"^(v\d+)_(\d+)_(\d+)\.jpg$", re.I)          # Clip 链路抽帧：v<N>_<毫秒>_<序号>
_MS_RE = re.compile(r"^(.+)_(\d{13})(_.*)?\.jpg$", re.I)        # 相机连拍：100002_1783714217483_13_14.jpg
_IDX_RE = re.compile(r"^(.+?)(\d{3,})\.jpg$")                   # 尾部序号：IMG_000123.jpg
_SECMS_RE = re.compile(r"^(\d{9,})\.(\d{1,3})\.jpg$")           # 直导图片：<unix秒>.<毫秒>.jpg

def seq_key(path_or_name):
    """从文件名提取序列线索。返回 (run_id, mode, value) 或 None。

      **分桶视频帧**：images/<视频名>/<视频名>_<源帧号>.jpg   -> ("bucket:<目录>", "seq", (0, 帧号))
        文件名前缀 == 所在目录名 → 该目录就是一个视频，同目录帧天然连续（序号是源帧号）。
        这条**必须放在最前**：否则会落到下面的通用序号规则，而源帧号按抽帧步长
        （step=5/10）递增，被"序号差>2 就切"判成不连续，28.6 万帧会切成 28.6 万个单帧段
        （2026-09-20 实测：南美 clips 膨胀到 30 万段、内存 11.7G）。
      v<N>_<毫秒>_<序号>.jpg   -> ("v:N", "seq", (毫秒, 序号))
      <帧id>_<13位毫秒>_<后缀>  -> ("ms:<目录>:<后缀>", "ms", 毫秒)   run 内按毫秒排，间隔 > gap 切段
      <unix秒>.<毫秒>.jpg       -> ("secms:<目录>", "ms", 毫秒时刻)  同上；**必须排在 idx 之前**
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
    # ② 直导图片 <unix秒>.<毫秒>.jpg（东南亚 3 万张相机 dump 就是这个形态）。
    # ⚠️ 必须排在 _IDX_RE 之前：否则 "1787120247.455" 会被尾部序号规则整个吃掉
    # （(.+?) 吃 "1787120247."、(\d{3,}) 吃 "455"），每张图各得一个 run_id →
    # split_seq_runs 出来全是长度 1 的 run → 被"<5 帧不成段"过滤 → **整个项目 0 个 Clip**
    # （2026-09-21 实测：东南亚 30776 张、clips total=0）。小数位按毫秒左补零（".45"=450ms）。
    m = _SECMS_RE.match(_fn)
    if m:
        return ("secms:%s" % _d, "ms", int(m.group(1)) * 1000 + int(m.group(2).ljust(3, "0")))
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

def _frame_ms(path_or_name):
    """帧文件名里的**逐帧**时间戳（毫秒）；拿不到返回 None。

    认这三种（与 seq_key 同一套规则）：
      <unix秒>.<毫秒>.jpg            —— 直导图片，时间戳逐帧递增
      <帧id>_<13位毫秒>_<后缀>.jpg    —— 相机连拍，时间戳逐帧递增
      v<N>_<毫秒>_<序号>.jpg         —— Clip 链路抽帧；**毫秒也是逐帧的**
        （2026-09-21 实测南美 1,130 个 v<N> 组：125 帧对应 125 个不同毫秒值，
         全都逐帧递增，可当时间用。别再以为是"任务级常量"。）

    分桶目录帧（<视频名>_<源帧号>.jpg）只有源帧号、没有时间戳 → 返回 None，
    调用方退回帧数切段（这些形态的真实节奏本来就够长，30 帧 ≈ 14.5 秒）。"""
    _fn = os.path.basename((path_or_name or "").replace("\\", "/"))
    m = _SECMS_RE.match(_fn)
    if m:
        return int(m.group(1)) * 1000 + int(m.group(2).ljust(3, "0"))
    m = _V_RE.match(_fn)
    if m:
        return int(m.group(2))
    m = _MS_RE.match(_fn)
    if m:
        return int(m.group(2))
    return None

_CLIP_SEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "clip_seconds.json")

def clip_seconds(project=None):
    """段的时间窗秒数（0=关，按帧数切）。按项目配置，改完**不用重启**（下次 scan 生效）。

    配置来源（先环境变量、后文件）：
      ① 环境变量 AD_CLIP_SECONDS_<项目名> / AD_CLIP_SECONDS（全局兜底）
      ② `workspace/clip_seconds.json` 的 {"<项目名>": 秒, "*": 全局兜底}
         —— 中文项目名走环境变量在 systemd 里很脆（变量名编码/合法性问题），所以主推文件。

    **为什么需要它**：段长按"帧数"定，真实时间就随帧率漂移 —— 同样 30 帧，
    欧洲相机 2fps = 15 秒，东南亚直导图 10fps 只有 3 秒。2026-09-21 实测：
    东南亚 30 帧 = 2.9 秒（中位），要和其他项目对齐到 10~15 秒必须按时间切。
    注意**不能靠调大帧数解决**：同一项目里 100ms 和 200ms 两种节奏混着（82.8%/17.2%），
    120 帧窗口的真实跨度从 11.8 秒到 23.8 秒，17% 的段会超过 15 秒。"""
    for k in ("AD_CLIP_SECONDS_" + (project or ""), "AD_CLIP_SECONDS"):
        v = os.environ.get(k)
        if v:
            try:
                s = float(v)
                if s > 0:
                    return s
            except Exception:
                pass
    try:
        with open(_CLIP_SEC_PATH, encoding="utf-8") as f:
            cfg = json.load(f) or {}
        v = cfg.get(project)
        if v is None:
            v = cfg.get("*")
        if v:
            return float(v)
    except Exception:
        pass
    return 0.0

def _cut_windows(items, ms, sec, floor_frames, min_tail):
    """把已按时间有序的 items 切成窗口：**至少 floor_frames 帧、且至少 sec 秒**。

    为什么要"帧数下限"而不是纯按秒切：各形态帧率差很大（欧洲 2fps / 南美分桶 0.5s /
    南美平铺 0.15s / 东南亚 33~200ms）。纯按秒切会把**本来就够长**的段改短并重排
    clip_id —— 南美分桶与视频帧两类（占 76%）30 帧已经是 14.5 秒，加了下限它们原样保留，
    已有判定与人工审核结果全部沿用；只有真正短的（南美平铺 30 帧=4.5 秒、东南亚 30 帧=3 秒）
    才被拉长到 sec 秒。

    ms 是与 items 一一对应的毫秒时刻；任取不到就返回 None（调用方退回纯帧数切）。
    尾窗不足 min_tail 帧时不另立一段，而是**并进上一段**（避免丢帧）。"""
    if ms is None or len(ms) != len(items) or any(x is None for x in ms) or ms[0] == ms[-1]:
        # 末条相等 = 时间戳全都一样（如分桶帧的 metadata.timestamp 恒为 0）→ 时间不可用
        return None
    lim = sec * 1000.0
    out, cur, t0 = [], [], None
    for f, t in zip(items, ms):
        if cur and len(cur) >= floor_frames and (t - t0) >= lim:
            out.append(cur)
            cur, t0 = [], None
        if not cur:
            t0 = t
        cur.append(f)
    if cur:
        if len(cur) < min_tail and out:
            out[-1].extend(cur)
        elif len(cur) >= min_tail:
            out.append(cur)
    return out

def _windows_by_time(run, sec, floor_frames, min_tail):
    """seq 形态：逐帧时刻取自文件名（见 _frame_ms）。"""
    return _cut_windows(run, [_frame_ms(f.get("path") or f.get("filename")) for f in run],
                        sec, floor_frames, min_tail)

def build_seq_clips(metadata, clip_size=30, project=None):
    """无 video_source 的连续帧 → seq_key 聚类 → 连续 run → 切段。
    返回与 build_clips_from_metadata 相同结构的 clip 列表（段内共享一份判定）。

    切段方式二选一：设了 AD_CLIP_SECONDS[_<项目>] 就按**真实时间**切（跨项目对齐 10~15 秒），
    否则按帧数切（历史行为，30 帧一段）。"""
    _sec = clip_seconds(project)
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
        wins = _windows_by_time(run, _sec, clip_size, _min_len) if _sec > 0 else None
        if wins is None:
            wins = [run[i:i + clip_size] for i in range(0, len(run), clip_size)]
        for idx, window in enumerate(wins):
            clips.append({
                "clip_id": "seq_%s_%05d" % (tag, idx),
                "video_id": "seq:" + (os.path.dirname((head.get("path") or "").replace("\\", "/")) or "images"),
                "clip_index": idx,
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

def mark_clip_error(ctx, clip_id, msg, max_fails=3):
    """判定**失败**时落盘（2026-09-21 加，修一个会毁数据的坑）。

    原来失败走的是 `update_clip_result(ctx, clip_id, {"error": ...})`，后果两个：
      ① **覆盖掉原有的好结果** —— 一次瞬时 OOM 就把该段已判好的标签毁掉（实测东南亚出现过 2 段，
         vlm_result 变成 {"error": "CUDA out of memory..."}）；
      ② 失败段被计成"已判定"（decision 被写成 REVIEW），**永远不会重试**。
    现在：失败只记到独立的 `vlm_error` 字段（排查可见，界面不会当成标签读），
    **绝不碰 vlm_result**；并且只有"本来就没有有效结果"时才在连续失败 `max_fails` 次后
    置 REVIEW —— 既能被瞬时故障重试救回，又不会无限重试。"""
    os.makedirs(ctx["dir"], exist_ok=True)
    flush_pending()          # 别和缓冲里的判定抢同一份文件
    hit = False
    with _clips_lock:
        clips = load_clips(ctx)
        for c in clips:
            if c["clip_id"] != clip_id:
                continue
            _n = int(((c.get("vlm_error") or {}).get("n")) or 0) + 1
            c["vlm_error"] = {"msg": str(msg)[:400], "ts": time.time(), "n": _n}
            _vr = c.get("vlm_result")
            _valid = isinstance(_vr, dict) and "error" not in _vr and any(
                _vr.get(k) for k in ("scene", "events", "objects", "traffic_sign", "risk"))
            if (not _valid) and _n >= max_fails:
                c["decision"] = "REVIEW"      # 连续失败：交人工，别无限重试
            hit = True
            break
        if hit:
            _atomic_write_json(_clips_path(ctx), clips)
            _clips_cache["key"] = None
    return hit


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

    _min_len = int(os.environ.get("AD_SEQ_MIN_FRAMES", "5"))
    _sec = clip_seconds(ctx.get("name"))
    clips = []
    for vid, frames in by_video.items():
        frames.sort(key=lambda x: (x.get("timestamp") or 0, x.get("frame_index") or 0))
        # 视频帧的逐帧时刻就是**视频内位置**（metadata.timestamp，秒）—— 2026-09-21 实测
        # 南美 40.6 万视频帧 100% 有该字段且逐帧递增，可当时间用（分桶帧的该字段恒为 0，
        # 已被 _cut_windows 的"全相等"判定挡掉，退回帧数切）。
        wins = None
        if _sec > 0:
            wins = _cut_windows(frames, [int((f.get("timestamp") or 0) * 1000) for f in frames],
                                _sec, clip_size, _min_len)
        if wins is None:
            wins = [frames[i:i + clip_size] for i in range(0, len(frames), clip_size)]
        for idx, window in enumerate(wins):
            cid = f"{os.path.splitext(os.path.basename(vid))[0]}_{idx:05d}"
            clips.append(_mk_clip(cid, vid, idx,
                                  window[0].get("timestamp"), window[-1].get("timestamp"), window))
    # 直导图片连续帧：与视频同一逻辑（切段方式见 build_seq_clips：设了 AD_CLIP_SECONDS 就按时间切）
    for c in build_seq_clips(ctx["metadata"], clip_size, ctx.get("name")):
        clips.append(c)
    # 沿用旧判定：仅当帧集合完全一致才继承（改了 clip_size 会生成同名但内容不同的 Clip）
    out = []
    for clip in clips:
        old = prev.get(clip["clip_id"])
        if old and (old.get("frame_ids") or []) == clip["frame_ids"]:
            # human_rejected 必须一起继承：它是人工"点掉模型标签"的否决名单，
            # 漏了的话每次 scan 重建都会把人工的否决悄悄丢掉（2026-09-20 发现）
            for k in ("vlm_result", "human_tags", "final_tags", "human_rejected", "decision"):
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

# 判定结果批量落盘缓冲：{clips.json 路径: {"ctx":…, "items":{clip_id:(res,dec)}, "t0":…}}
_PEND = {}
_PEND_MAX = int(os.environ.get("AD_PEND_MAX", "5"))        # 攒够几段落一次盘
_PEND_SEC = float(os.environ.get("AD_PEND_SEC", "20"))     # 或最多拖多少秒
def flush_pending():
    """把缓冲的判定结果落盘。收尾、人工打标、scan 之前都要先调一次，避免被后写覆盖。"""
    with _clips_lock:
        for _p in list(_PEND.keys()):
            _flush_one(_p)
def _flush_one(path):
    """（调用方需持有 _clips_lock）"""
    rec = _PEND.get(path)
    if not rec or not rec["items"]:
        return 0
    clips = load_clips(rec["ctx"])
    hit = 0
    for c in clips:
        it = rec["items"].get(c.get("clip_id"))
        if it is not None:
            c["vlm_result"] = it[0]
            if it[1]:
                c["decision"] = it[1]
            hit += 1
    if hit:
        _atomic_write_json(path, clips)
        _clips_cache["key"] = None      # 失效 mtime 缓存，下次读拿到新版本
    rec["items"] = {}
    rec["t0"] = time.time()
    return hit
# 兜底定时落盘（2026-09-21 加）：
# `update_clip_result` 里"距上次落盘超过 AD_PEND_SEC 秒就落盘"这个判断**只在被调用时才执行**，
# 所以一批判定的最后 ≤AD_PEND_MAX-1 段在空闲下来之后永远留在内存里 —— 此时重启/崩溃就丢。
# 定时器把"已经超时"的缓冲落盘，判定条件与调用路径**完全一致**（不会让落盘变得更频繁）。
_FLUSH_TICK = float(os.environ.get("AD_PEND_TICK", "5"))     # 定时器扫描间隔（秒）
def _flush_timer():
    while True:
        time.sleep(_FLUSH_TICK)
        try:
            _now = time.time()
            with _clips_lock:
                for _p in [_p for _p, _r in _PEND.items()
                           if _r["items"] and (_now - _r["t0"]) >= _PEND_SEC]:
                    _flush_one(_p)
        except Exception as _e:
            print(f"[clips] 定时落盘失败: {_e}", flush=True)
if os.environ.get("AD_PEND_TIMER", "1") != "0":
    threading.Thread(target=_flush_timer, daemon=True, name="clips-flush-timer").start()
def update_clip_result(ctx, clip_id, vlm_result, decision=None, defer=True):
    """单 clip 结果回写。

    ⚠️ 不再每段全量重写 clips.json（南美 144MB，一段一写纯属浪费）：先入内存缓冲，
    攒够 AD_PEND_MAX 段或超过 AD_PEND_SEC 秒才落盘。defer=False 立即落盘。"""
    os.makedirs(ctx["dir"], exist_ok=True)
    path = _clips_path(ctx)
    with _clips_lock:
        rec = _PEND.get(path)
        if rec is None:
            rec = _PEND[path] = {"ctx": ctx, "items": {}, "t0": time.time()}
        rec["items"][clip_id] = (vlm_result, decision)
        if (not defer) or len(rec["items"]) >= _PEND_MAX or (time.time() - rec["t0"]) >= _PEND_SEC:
            _flush_one(path)

def apply_human_tags(ctx, clip_id, human_tags, final_tags=None, rejected=None, decision="APPROVED"):
    """人工标注落盘。

    标签语义（用户 2026-09-20 明确）：
      human_tags = 人工标注值
      final_tags = **人工 ∪ 模型（互补并集）** —— 人工补上模型漏掉的、模型保留人工没标的，
                   两者互补而非覆盖（同维度取并集、去重保序）
      vlm_result = 保持不动 —— 它是对比评测的依据（模型当初判了什么必须可追溯）

    final_tags 由调用方算好传入（模型标签要从 vlm_result 提取，那部分在 app 层）；
    未传则退回"仅人工"。返回 True/False（找到并写入）。"""
    flush_pending()      # 人工结果不能和缓冲里的判定抢同一份文件
    os.makedirs(ctx["dir"], exist_ok=True)
    with _clips_lock:
        clips = load_clips(ctx)
        hit = False
        for c in clips:
            if c["clip_id"] == clip_id:
                _h = {k: list(v) for k, v in (human_tags or {}).items()}
                c["human_tags"] = _h
                # 人工"取消"的模型标签（否决名单）：final 计算时要从模型标签里剔除，
                # 这样人工既有增（human_tags）也有删（rejected），是真正的可编辑
                c["human_rejected"] = {k: list(v) for k, v in (rejected or {}).items()}
                if final_tags is not None:
                    c["final_tags"] = {k: list(v) for k, v in (final_tags or {}).items()}
                else:
                    c["final_tags"] = dict(_h)
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
