# -*- coding: utf-8 -*-
"""统一配置加载：从 config.json 读取目录配置，可用环境变量覆盖。

优先级：环境变量 > config.json > 内置默认值
环境变量：
    MINE_WORKSPACE   工作目录
    MINE_VALID_DIR   交付图片导出目录
"""
import os
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG = None


def _load():
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    _CONFIG = {}
    cfg_path = os.path.join(PROJECT_DIR, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                _CONFIG = json.load(f) or {}
        except Exception as e:
            print(f"[警告] 读取 config.json 失败，使用默认配置: {e}")
    return _CONFIG


def resolve(env_key, cfg_key, default):
    """优先级: 环境变量 > config.json > 默认值。相对路径基于项目目录。"""
    v = os.environ.get(env_key)
    if v:
        return v
    v = _load().get(cfg_key)
    if v:
        return v if os.path.isabs(v) else os.path.join(PROJECT_DIR, v)
    return default


def workspace():
    return resolve("MINE_WORKSPACE", "workspace", os.path.join(PROJECT_DIR, "workspace"))


def valid_dir():
    return resolve("MINE_VALID_DIR", "valid_dir", os.path.join(PROJECT_DIR, "有效", "Clip初筛"))


def get(key, default=None):
    """直接读取 config.json 中的某项（不含路径解析）"""
    v = _load().get(key)
    return v if v is not None else default


def server_host():
    return os.environ.get("MINE_SERVER_HOST", get("server_host", "0.0.0.0"))


def server_port():
    try:
        return int(os.environ.get("MINE_SERVER_PORT", get("server_port", 8009)))
    except (TypeError, ValueError):
        return 8009


def clip_model():
    return get("clip_model", "openai/clip-vit-large-patch14")


def dino_model():
    return get("dino_model", "IDEA-Research/grounding-dino-tiny")


def yolo_model():
    return get("yolo_model", "yolov8x.pt")


def batch_size():
    try:
        return int(get("batch_size", 8))
    except (TypeError, ValueError):
        return 8


def num_workers():
    try:
        return int(get("num_workers", 2))
    except (TypeError, ValueError):
        return 2


def device():
    return get("device", "auto")
