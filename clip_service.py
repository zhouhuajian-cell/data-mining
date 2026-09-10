# -*- coding: utf-8 -*-
# clip_service.py  —— 放在 app.py 同级目录
import os, json, threading
import numpy as np

_clips_lock = threading.Lock()

def build_clips_from_metadata(ctx, clip_size=50):
    """把底库中已抽帧的图片按 video_source + 时间戳分组成 50 帧窗口。
    重建时保留同 clip_id 的已有结果(vlm_result/decision/human_tags)，否则重跑 scan 会冲掉
    全部 AI 判定与人工审核结果。"""
    try:
        prev = {c.get("clip_id"): c for c in (load_clips(ctx) or []) if c.get("clip_id")}
    except Exception:
        prev = {}  # ctx 无 dir 等精简调用场景：无历史可沿用
    by_video = {}
    for m in ctx["metadata"]:
        vs = m.get("video_source") or {}
        vid = vs.get("video_path")          # 视频抽的帧才有
        if not vid:
            continue                        # 纯图片直导的帧跳过（走原单帧管线）
        by_video.setdefault(vid, []).append(m)

    clips = []
    for vid, frames in by_video.items():
        frames.sort(key=lambda x: (x.get("timestamp") or 0, x.get("frame_index") or 0))
        for i in range(0, len(frames), clip_size):
            window = frames[i:i + clip_size]
            cid = f"{os.path.splitext(os.path.basename(vid))[0]}_{i // clip_size:05d}"
            clip = {
                "clip_id": cid,
                "video_id": vid,
                "clip_index": i // clip_size,
                "start_timestamp": window[0].get("timestamp"),
                "end_timestamp": window[-1].get("timestamp"),
                "frame_ids": [f["id"] for f in window],
                "frame_paths": [f["path"] for f in window],
                "vlm_result": None,      # 分析结果回填到这里
                "human_tags": {},
                "final_tags": {},
                "decision": None,        # AUTO_PASS / REVIEW / APPROVED / FILTERED
            }
            old = prev.get(cid)
            # 仅当帧集合完全一致才沿用：改了 clip_size 会生成同名但内容不同的 Clip，不能继承旧结果
            if old and (old.get("frame_ids") or []) == clip["frame_ids"]:
                for k in ("vlm_result", "human_tags", "final_tags", "decision"):
                    if k in old:
                        clip[k] = old[k]
            clips.append(clip)
    return clips


def _clips_path(ctx):
    return os.path.join(ctx["dir"], "clips.json")

def save_clips(ctx, clips):
    os.makedirs(ctx["dir"], exist_ok=True)  # 项目目录可能不存在(DB 建项目不建目录)，先补齐
    with _clips_lock:
        with open(_clips_path(ctx), "w", encoding="utf-8") as f:
            json.dump(clips, f, ensure_ascii=False, indent=1)

def load_clips(ctx):
    p = _clips_path(ctx)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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
        with open(_clips_path(ctx), "w", encoding="utf-8") as f:
            json.dump(clips, f, ensure_ascii=False, indent=1)

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
            with open(_clips_path(ctx), "w", encoding="utf-8") as f:
                json.dump(clips, f, ensure_ascii=False, indent=1)
    return found
