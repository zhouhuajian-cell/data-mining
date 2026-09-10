# -*- coding: utf-8 -*-
"""
存储布局管理 - PRD 47 网盘目录规范
默认根：<workspace>/AutodriveData（本地）或 AD_DATA_ROOT 指向 NAS 挂载点
    AutodriveData/
    ├── raw/
    ├── processed/frames, thumbnails
    ├── inference/{siglip,yolo,dino,vlm,fusion}
    ├── metadata/{jsonl,parquet,manifest}
    ├── indexes/faiss/
    ├── benchmark/
    └── exports/
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace", "AutodriveData")


def _default_root() -> str:
    """AutodriveData 树默认落在数据根（网盘）下：<data_root>/AutodriveData"""
    try:
        from settings import data_root
        return os.path.join(data_root(), "AutodriveData")
    except Exception:
        return DEFAULT_ROOT


def resolve_root() -> str:
    """数据根：AD_DATA_ROOT 环境变量 > 本地 workspace/AutodriveData"""
    return os.environ.get("AD_DATA_ROOT", "").strip() or _default_root()


_LAYOUT = {
    "raw": "raw",
    "frames": "processed/frames",
    "thumbnails": "processed/thumbnails",
    "siglip": "inference/siglip",
    "yolo": "inference/yolo",
    "dino": "inference/dino",
    "vlm": "inference/vlm",
    "fusion": "inference/fusion",
    "jsonl": "metadata/jsonl",
    "parquet": "metadata/parquet",
    "manifest": "metadata/manifest",
    "faiss": "indexes/faiss",
    "benchmark": "benchmark",
    "exports": "exports",
}


def ensure_layout(root: str = None) -> dict:
    """创建目录树并返回 {key: abs_path}"""
    root = root or resolve_root()
    paths = {}
    for key, rel in _LAYOUT.items():
        p = os.path.join(root, rel)
        os.makedirs(p, exist_ok=True)
        paths[key] = p
    return paths


def layout_info(root: str = None) -> dict:
    """返回布局信息：根、各子目录、可写性"""
    root = root or resolve_root()
    paths = ensure_layout(root)
    info = {"root": root, "dirs": {}}
    for key, p in paths.items():
        info["dirs"][key] = {
            "path": p,
            "exists": os.path.isdir(p),
            "writable": os.access(p, os.W_OK) if os.path.isdir(p) else False,
        }
    return info


def project_dirs(project: str, root: str = None) -> dict:
    """某项目在布局内的专属目录（按项目隔离）"""
    base = ensure_layout(root)
    return {
        "frames": os.path.join(base["frames"], project or "default"),
        "thumbnails": os.path.join(base["thumbnails"], project or "default"),
        "jsonl": os.path.join(base["jsonl"], project or "default"),
        "parquet": os.path.join(base["parquet"], project or "default"),
        "manifest": os.path.join(base["manifest"], project or "default"),
        "faiss": os.path.join(base["faiss"], project or "default"),
        "exports": os.path.join(base["exports"], project or "default"),
        "benchmark": os.path.join(base["benchmark"], project or "default"),
    }


if __name__ == "__main__":
    info = layout_info()
    print("root:", info["root"])
    for k, v in info["dirs"].items():
        print(f"  {k:12s} {v['path']}")
    print("all writable:", all(v["writable"] for v in info["dirs"].values()))
