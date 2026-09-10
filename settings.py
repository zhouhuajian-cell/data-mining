# -*- coding: utf-8 -*-
"""
数据根目录配置（支持 NAS / 网盘挂载点）
优先级：环境变量 > 项目根 data_config.json > 默认 <项目>/workspace

  AD_DATA_ROOT  - 媒体/项目仓根目录（原图、抽帧帧、向量 json/faiss、AutodriveData 导出树）
  AD_DB_PATH    - SQLite 数据库文件路径（建议保留本地 SSD；网盘 SMB 上 SQLite 锁性能差）

config.json（重启后端生效）:
  { "data_root": "Z:/AutodriveData", "db_path": null }
"""

import os
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_DIR, "data_config.json")
DEFAULT_ROOT = os.path.join(PROJECT_DIR, "workspace")


def load_config() -> dict:
    """读取项目根 data_config.json（不存在返回 {}）"""
    try:
        if os.path.isfile(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception:
        pass
    return {}


def save_config(data_root: str = None, db_path: str = None) -> dict:
    """持久化配置到 data_config.json（可只更新单字段，None 表示不动）"""
    cfg = load_config()
    if data_root is not None:
        cfg["data_root"] = data_root
    if db_path is not None:
        cfg["db_path"] = db_path or None
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def data_root() -> str:
    """媒体/项目仓根：AD_DATA_ROOT env > config.data_root > <项目>/workspace"""
    env = os.environ.get("AD_DATA_ROOT", "").strip()
    if env:
        return env
    cfg = load_config()
    if isinstance(cfg, dict) and cfg.get("data_root"):
        return str(cfg["data_root"])
    return DEFAULT_ROOT


def db_path() -> str:
    """SQLite 库路径：AD_DB_PATH env > config.db_path > <data_root>/mining.db"""
    env = os.environ.get("AD_DB_PATH", "").strip()
    if env:
        return env
    cfg = load_config()
    if isinstance(cfg, dict) and cfg.get("db_path"):
        return str(cfg["db_path"])
    return os.path.join(data_root(), "mining.db")


def config_info() -> dict:
    """当前生效配置摘要（供 /api/storage/config 与诊断）"""
    cfg = load_config()
    env_root = os.environ.get("AD_DATA_ROOT", "").strip()
    env_db = os.environ.get("AD_DB_PATH", "").strip()
    if env_root:
        src = "env:AD_DATA_ROOT"
    elif cfg.get("data_root"):
        src = "data_config.json"
    else:
        src = "default"
    db_src = "env:AD_DB_PATH" if env_db else ("data_config.json" if cfg.get("db_path") else "跟随 data_root")
    return {
        "config_file": CONFIG_FILE,
        "data_root": data_root(),
        "data_root_source": src,
        "db_path": db_path(),
        "db_path_source": db_src,
        "env": {"AD_DATA_ROOT": env_root or None, "AD_DB_PATH": env_db or None},
        "config_json": cfg,
    }


if __name__ == "__main__":
    import json as _j
    info = config_info()
    print(_j.dumps(info, ensure_ascii=False, indent=2))
