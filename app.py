# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
import sys
import json
import shutil
import zipfile
import re
import secrets
import contextlib
import time
import gzip
import glob
import threading
from typing import List, Optional
import numpy as np
from PIL import Image
try:
    import cv2
except Exception:
    cv2 = None  # 未安装 opencv-python 时，视频接口会友好提示
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Query,
    HTTPException,
    Request,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
# ===== 深度学习依赖（可选）：缺失时进入轻量模式，仅用于鉴权/界面测试 =====
LITE_MODE = False
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import faiss
    from transformers import (
        AutoProcessor,
        AutoModel,
        AutoModelForZeroShotObjectDetection,
    )
    from ultralytics import YOLO
except Exception as _dep_err:
    torch = Dataset = DataLoader = transforms = faiss = None
    AutoProcessor = AutoModel = AutoModelForZeroShotObjectDetection = YOLO = None
    LITE_MODE = True
    print(f"[i] 轻量模式：模型依赖缺失，已跳过模型功能 ({_dep_err})。")
# ===== 数据库与 GPU Manager 补丁 =====
try:
    from db_patch import (
        apply_db_patches,
        list_all_projects,
        get_project_db_stats,
        create_project_db,
        rename_project_db,
        delete_project_db,
        get_db_session,
    )
    from gpu_patch import (
        init_gpu_managers,
        ensure_siglip,
        free_siglip,
        ensure_yolo,
        free_yolo,
        ensure_dino,
        free_dino,
        ensure_vlm,
        free_vlm,
        print_vram_status,
    )
    from dedup import get_dedup_engine
    from decision_engine import create_decision_engine, DecisionStatus
    from tag_system import get_tag_manager, TagSource
    DB_PATCH_AVAILABLE = True
except Exception as _patch_err:
    DB_PATCH_AVAILABLE = False
    print(f"[!] 数据库/GPU补丁加载失败，回退原有逻辑: {_patch_err}")
# ===== DINO 中英词典（离线方案）：支持直接输入中文 =====
# 服务器无法访问外网，故不使用在线翻译。
# 优先读取外部 JSON 文件 dino_dict.json（可随时编辑增词，无需改代码），
# 缺失或未覆盖时回退到内置默认词典。
_DEFAULT_DINO_DICT = {
    "卡车": "truck",
    "货车": "truck",
    "大货车": "heavy truck",
    "汽车": "car",
    "小车": "car",
    "越野车": "SUV",
    "行人": "pedestrian",
    "人": "person",
    "小孩": "child",
    "锥桶": "traffic cone",
    "反光锥": "traffic cone",
    "水马": "water-filled barrier",
    "路标": "traffic sign",
    "指示牌": "traffic sign",
    "限速牌": "speed limit sign",
    "红绿灯": "traffic light",
    "交通灯": "traffic light",
    "车道线": "lane marker",
    "斑马线": "crosswalk",
    "停止线": "stop line",
    "路灯": "streetlight",
    "树木": "tree",
    "道路": "road",
    "自行车": "bicycle",
    "摩托车": "motorcycle",
    "三轮车": "tricycle",
}
def _load_dino_dict():
    """合并外部词典与内置默认词典，返回 dict"""
    dino_dict_path = os.path.join(PROJECT_DIR, "dino_dict.json")
    d = dict(_DEFAULT_DINO_DICT)
    if os.path.exists(dino_dict_path):
        try:
            with open(dino_dict_path, "r", encoding="utf-8") as f:
                user = json.load(f)
            if isinstance(user, dict):
                d.update(user)
        except Exception as _de:
            print(f"[!] 读取 dino_dict.json 失败，使用内置词典: {_de}")
    return d
# ----------------- 路径与环境配置 -----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据根：AD_DATA_ROOT / data_config.json 可指向 NAS/网盘挂载点（媒体仓与 DB 全链跟随）
from settings import data_root as _data_root, config_info as _cfg_info
WORKSPACE = _data_root()
PROJECTS_ROOT = os.path.join(WORKSPACE, "projects")
os.makedirs(PROJECTS_ROOT, exist_ok=True)
LOG_DIR = os.environ.get("AD_LOG_DIR", os.path.join(PROJECT_DIR, "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_RETENTION_DAYS = 30
# 交付打包时，伴生文件(.bin/.xml/_raw.jpg)的来源根目录（可多个，逗号/分号分隔）。
# 图片入库时若记录了 src_dir 会优先精确查找；历史已入库数据用这里配置的根目录递归按同名匹配。
SOURCE_ROOTS = [
    p.strip()
    for p in re.split(r"[,;，；]", os.environ.get("AD_SOURCE_ROOTS", ""))
    if p.strip()
]
def _today_log_path():
    day = time.strftime("%Y%m%d")
    return os.path.join(LOG_DIR, f"app_{day}.log")
def _log(msg):
    """把关键信息写入当天的日志文件 logs/app_YYYYMMDD.log"""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(_today_log_path(), "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass
def _compress_old_logs(days: int = LOG_RETENTION_DAYS):
    """把超过 days 天的 .log 压缩成 .log.gz 并删除原文件"""
    try:
        cutoff = time.time() - days * 86400
        for path in glob.glob(os.path.join(LOG_DIR, "app_*.log")):
            try:
                if os.path.getmtime(path) < cutoff:
                    gz = path + ".gz"
                    with open(path, "rb") as f_in, gzip.open(gz, "wb") as f_out:
                        f_out.write(f_in.read())
                    os.remove(path)
                    _log(
                        f"已压缩过期日志: {os.path.basename(path)} -> {os.path.basename(gz)}"
                    )
            except Exception:
                pass
    except Exception:
        pass
def _compression_loop():
    while True:
        try:
            _compress_old_logs()
        except Exception:
            pass
        time.sleep(3600)  # 每小时检查一次
def _start_compression():
    threading.Thread(target=_compression_loop, daemon=True).start()
# ===== 目标检测持久化缓存（按项目隔离：项目 -> {str(image_id): 检测结果}） =====
DETECTIONS_CACHE_FILE = os.path.join(WORKSPACE, "detections_cache.json")
detections_cache = {}
_detections_lock = threading.Lock()
if os.path.exists(DETECTIONS_CACHE_FILE):
    try:
        with open(DETECTIONS_CACHE_FILE, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
        detections_cache = _loaded if isinstance(_loaded, dict) else {}
        _total = sum(len(v) for v in detections_cache.values() if isinstance(v, dict))
        print(f"✅ 成功恢复 {_total} 帧历史目标检测记录")
    except Exception:
        detections_cache = {}
def _persist_detections():
    with open(DETECTIONS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(detections_cache, f, ensure_ascii=False)
def save_detection_record(project: str, image_id: int, result: dict):
    """把单张图片的检测结果按项目写入缓存并落盘（image_id 各项目独立，故按项目隔离存储）。"""
    global detections_cache
    proj = project or "default"
    with _detections_lock:
        sub = detections_cache.setdefault(proj, {})
        sub[str(image_id)] = result
        _persist_detections()
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
# ===== 简单登录鉴权（管理员 / 标注员）=====
# 口令不再硬编码：只从环境变量 AD_PASSWORD / ANNOTATOR_PASSWORD 或 auth_config.json 读，
# 未配置则为空（登录接口会直接拒绝，不会出现"空密码即可登录"）。鉴权中间件当前也是停用状态。
ADMIN_PASSWORD = os.environ.get("AD_PASSWORD", "")  # 管理员密码
ANNOTATOR_PASSWORD = os.environ.get("ANNOTATOR_PASSWORD", "")  # 标注员密码
AUTH_TOKENS = {}  # token -> 角色 (admin / annotator)
AUTH_CONFIG_PATH = os.path.join(PROJECT_DIR, "auth_config.json")  # 密码持久化文件
def _load_passwords():
    """启动时读取 auth_config.json 里的密码（若存在），否则用默认/环境变量值"""
    global ADMIN_PASSWORD, ANNOTATOR_PASSWORD
    if os.path.exists(AUTH_CONFIG_PATH):
        try:
            with open(AUTH_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("admin"):
                ADMIN_PASSWORD = cfg["admin"]
            if cfg.get("annotator"):
                ANNOTATOR_PASSWORD = cfg["annotator"]
        except Exception:
            pass
def _save_passwords():
    with open(AUTH_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"admin": ADMIN_PASSWORD, "annotator": ANNOTATOR_PASSWORD},
            f,
            ensure_ascii=False,
        )
# ===== 模型加载开关 =====
# 默认策略（生产优先，兼顾本地测试）：
#   - 依赖齐全（非轻量模式）→ 默认启用模型加载：本地有权重就加载；
#     本地没有则联网下载（走 HF_ENDPOINT 镜像，适配生产模型不在本机的情况）；
#     若下载也失败则自动跳过，不影响服务启动。
#   - 纯轻量测试（完全不加载/不下载模型）时，设置环境变量 AD_LOAD_MODELS=0。
LOAD_MODELS = (os.environ.get("AD_LOAD_MODELS", "1" if not LITE_MODE else "0")) == "1"
# 旗舰高配模型设置
SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
DINO_MODEL_NAME = "IDEA-Research/grounding-dino-base"
YOLO_MODEL_NAME = os.path.join(PROJECT_DIR, "yolov8x.pt")
# VLM 综合场景判定模型：默认新一代 Qwen2.5-VL-3B（可用 AD_VLM_MODEL 覆盖）；
# 需服务器 transformers>=4.57 且 pip install qwen-vl-utils；加载失败会自动回退。
# 回退链：Qwen2.5-VL → AD_VLM_FALLBACK(默认 Qwen2-VL-7B，12G 卡走 4bit) → Qwen2-VL-2B(fp16)
VLM_MODEL_NAME = os.environ.get("AD_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLM_FALLBACK_MODEL = os.environ.get("AD_VLM_FALLBACK", "Qwen/Qwen2-VL-7B-Instruct")
VLM_4BIT = os.environ.get("AD_VLM_4BIT", "1") == "1"   # 12G 显存跑 7B(fp16≈15G) 必须量化
VLM_MAX_PX = int(os.environ.get("AD_VLM_MAX_PX", "448"))  # 每帧分辨率上限(28的倍数)，调大可提升小目标识别
FEAT_DIM = 1152  # SigLIP-SO400M 原生特征维度
app = FastAPI(title="Maxieye High-Precision Scene Mining Terminal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ===== 简单登录鉴权：中间件 + 登录接口 =====
# ===== 登录鉴权已注释禁用 =====
# 如需启用鉴权，取消下面整段注释即可。
#
# 仅管理员可操作的接口（增删改项目、永久粉碎冗余）
# ADMIN_ONLY_PATHS = (
#     "/api/projects/create",
#     "/api/projects/rename",
#     "/api/projects/delete",
#     "/api/delete_and_sync",
#     "/api/reset_password",
# )
#
# @app.middleware("http")
# async def auth_middleware(request, call_next):
#     path = request.url.path
#     # 放行：登录接口、图片服务、以及所有非 /api 的路径（首页/静态资源）
#     if path == "/api/login" or path.startswith("/api/image") or not path.startswith("/api/"):
#         return await call_next(request)
#     token = request.headers.get("x-auth-token") or ""
#     role = AUTH_TOKENS.get(token)
#     if role is None:
#         return JSONResponse(status_code=401, content={"msg": "未登录或登录已失效"})
#     if path in ADMIN_ONLY_PATHS and role != "admin":
#         return JSONResponse(status_code=403, content={"msg": "无权限：仅管理员可执行此操作"})
#     return await call_next(request)
@app.post("/api/login")
def login(password: str = Form(...)):
    # 未配置任何口令时直接拒绝：避免"把口令留空 -> 空密码即可登录"这种自伤
    if not ADMIN_PASSWORD and not ANNOTATOR_PASSWORD:
        return JSONResponse(
            status_code=403,
            content={"msg": "未配置登录口令（请设置 AD_PASSWORD / ANNOTATOR_PASSWORD，或调用 /api/reset_password）"},
        )
    if password and password == ADMIN_PASSWORD:
        role = "admin"
    elif password and password == ANNOTATOR_PASSWORD:
        role = "annotator"
    else:
        return JSONResponse(status_code=401, content={"msg": "密码错误"})
    tok = secrets.token_hex(16)
    AUTH_TOKENS[tok] = role
    return {"code": 200, "msg": f"登录成功（{role}）", "token": tok, "role": role}
@app.post("/api/change_password")
def change_password(
    request: Request, old_password: str = Form(...), new_password: str = Form(...)
):
    global ADMIN_PASSWORD, ANNOTATOR_PASSWORD
    token = request.headers.get("x-auth-token") or ""
    role = AUTH_TOKENS.get(token)
    if role == "admin":
        if old_password != ADMIN_PASSWORD:
            return JSONResponse(status_code=400, content={"msg": "原密码错误"})
        ADMIN_PASSWORD = new_password
    elif role == "annotator":
        if old_password != ANNOTATOR_PASSWORD:
            return JSONResponse(status_code=400, content={"msg": "原密码错误"})
        ANNOTATOR_PASSWORD = new_password
    else:
        return JSONResponse(status_code=401, content={"msg": "未登录"})
    _save_passwords()
    return {"code": 200, "msg": "密码修改成功"}
@app.post("/api/reset_password")
def reset_password(target: str = Form(...), new_password: str = Form(...)):
    """管理员重置任意角色密码（无需原密码）"""
    global ADMIN_PASSWORD, ANNOTATOR_PASSWORD
    if target not in ("admin", "annotator"):
        return JSONResponse(status_code=400, content={"msg": "目标角色无效"})
    if not new_password or len(new_password) < 4:
        return JSONResponse(status_code=400, content={"msg": "新密码长度至少 4 位"})
    if target == "admin":
        ADMIN_PASSWORD = new_password
        label = "管理员"
    else:
        ANNOTATOR_PASSWORD = new_password
        label = "标注员"
    _save_passwords()
    return {"code": 200, "msg": f"已重置{label}密码"}
# 挂载静态目录
if os.path.exists(PROJECT_DIR):
    app.mount("/static_root", StaticFiles(directory=PROJECT_DIR), name="static_root")
# 全局模型与项目索引缓存池
siglip_model = None
siglip_processor = None
dino_model = None
dino_processor = None
yolo_model = None
project_cache = {}
# ---- YOLO 按需懒加载：用得少就让它不常驻，把显存让给 DINO 等主力模型 ----
def _ensure_yolo():
    """按需加载 YOLO，返回是否就绪。"""
    global yolo_model
    if yolo_model is not None:
        return True
    if not LOAD_MODELS:
        return False
    # 使用 GPU Manager
    if DB_PATCH_AVAILABLE:
        yolo_model = ensure_yolo()
        return yolo_model is not None
    # 回退原有逻辑
    if YOLO is None:
        return False
    try:
        yolo_model = YOLO(YOLO_MODEL_NAME)  # 本地有则用，没有则联网下载
        print("[i] YOLO 已按需加载。")
        return True
    except Exception as e:
        print(f"[!] YOLO 按需加载失败: {e}")
        yolo_model = None
        return False
def _free_yolo():
    """释放 YOLO 占用的显存，让给 DINO/SigLIP。"""
    global yolo_model
    if DB_PATCH_AVAILABLE:
        free_yolo()
        yolo_model = None
        return
    if yolo_model is not None:
        try:
            del yolo_model
        except Exception:
            pass
        yolo_model = None
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
def _ensure_dino():
    """按需加载 Grounding DINO（FP16，省显存）。返回是否就绪。"""
    global dino_model, dino_processor
    if dino_model is not None and dino_processor is not None:
        return True
    if not LOAD_MODELS:
        return False
    # 使用 GPU Manager
    if DB_PATCH_AVAILABLE:
        dino_model, dino_processor = ensure_dino()
        return dino_model is not None
    # 回退原有逻辑
    if AutoModelForZeroShotObjectDetection is None or AutoProcessor is None:
        return False
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    dino_processor = dino_model = None
    try:
        dino_processor = AutoProcessor.from_pretrained(
            DINO_MODEL_NAME, local_files_only=True
        )
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_MODEL_NAME, torch_dtype=dtype, local_files_only=True
        ).to(DEVICE)
    except Exception:
        try:
            print("[i] 本地无 DINO 权重，尝试联网下载...")
            dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_MODEL_NAME, torch_dtype=dtype
            ).to(DEVICE)
        except Exception as _e:
            dino_processor = dino_model = None
            print(f"[!] DINO 加载失败: {_e}")
            return False
    if dino_model is not None:
        dino_model.eval()
        print("[i] Grounding DINO 已按需加载 (FP16)。")
        return True
    return False
def _free_dino():
    """释放 DINO 占用的显存，让给 VLM/YOLO 等其它模型。"""
    global dino_model, dino_processor
    if DB_PATCH_AVAILABLE:
        free_dino()
        dino_model = dino_processor = None
        return
    if dino_model is not None:
        try:
            del dino_model
        except Exception:
            pass
        dino_model = None
    dino_processor = None
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
# 12G 卡上大模型无法共存：加载/推理串行化，避免 A 请求正在卸载 VLM 时 B 请求又在加载 SigLIP 而 OOM
_GPU_MODEL_LOCK = threading.RLock()
# GPU 串行闸门：同一时刻只允许一路 AI 推理用卡，后来者排队等待（不并发、不拒绝）。
# 两路负载并发（如批量任务的帧级 VLM 撞上审核中心的 Clip 判定）会触发
# "CUDA error: device-side assert triggered" 并污染整个 CUDA 上下文，
# 之后所有推理都失败、只能重启进程。等待超时（默认 30 分钟）才放弃。
_GPU_WAIT_TIMEOUT = float(os.environ.get("AD_GPU_WAIT_TIMEOUT", "1800"))
@contextlib.contextmanager
def _gpu_slot(what: str):
    got = _GPU_MODEL_LOCK.acquire(blocking=False)
    if not got:
        _log(f"[GPU] {what} 排队等待中（另一路 AI 任务正在用卡）…")
        got = _GPU_MODEL_LOCK.acquire(timeout=_GPU_WAIT_TIMEOUT)
        if got:
            _log(f"[GPU] {what} 已排到，开始执行")
    if not got:
        raise RuntimeError(f"GPU 排队等待超时（{int(_GPU_WAIT_TIMEOUT)}s）：{what}")
    try:
        yield
    finally:
        _GPU_MODEL_LOCK.release()
def _vram_free_gb():
    """当前可用显存(GB)；非 CUDA 返回 None"""
    if DEVICE != "cuda" or torch is None:
        return None
    try:
        free, _ = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:
        return None
def _vram_log(tag):
    g = _vram_free_gb()
    if g is not None:
        print(f"[VRAM] {tag}: 可用 {g:.2f} GB", flush=True)
def _free_siglip():
    """真正释放 SigLIP。GPUManager 只持有它自己那份引用，app 的全局不置空则显存不归还
    （这正是之前 make_room_for “腾不动”、SigLIP 与 7B 并存占满 10.2G 的原因）。"""
    global siglip_model, siglip_processor
    if DB_PATCH_AVAILABLE:
        try:
            free_siglip()
        except Exception:
            pass
    siglip_model = siglip_processor = None
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
def _ensure_siglip():
    """取用 SigLIP：不在则先腾空间加载，失败返回 (None, None)。直接使用 siglip_model 的
    接口都必须先走这里，否则 VLM 占显存期间会因 SigLIP 被卸载而 AttributeError。"""
    global siglip_model, siglip_processor
    if siglip_model is not None:
        return siglip_model, siglip_processor
    if DB_PATCH_AVAILABLE:
        try:
            with _GPU_MODEL_LOCK:
                make_room_for("siglip", 2.5)
                siglip_model, siglip_processor = ensure_siglip()
                _vram_log("SigLIP 就绪")
        except Exception as _e:
            print(f"[SigLIP] 加载失败: {_e}", flush=True)
            siglip_model = siglip_processor = None
    return siglip_model, siglip_processor
def make_room_for(model: str, need_gb: float = 2.0):
    """显存互斥调度：加载 model 前把与之不能共存的模型全部释放并归还显存。
    12G 卡上 VLM(7B-4bit 约 8G) 与 SigLIP/DINO/YOLO 尽量不共存。
    返回腾完后的可用显存(GB)。"""
    keep = {"vlm": ("siglip", "dino", "yolo"), "siglip": ("vlm", "dino", "yolo"),
            "dino": ("vlm", "siglip", "yolo"), "yolo": ("vlm", "siglip", "dino")}.get(model, ("vlm", "siglip", "dino", "yolo"))
    if "vlm" in keep:
        _free_vlm()
    if "siglip" in keep:
        _free_siglip()
    for _m in ("dino", "yolo"):
        if _m in keep:
            fn = globals().get("_free_" + _m)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()   # 必须归还显存，否则碎片仍会让下一次加载 OOM
        except Exception:
            pass
    g = _vram_free_gb()
    if g is not None:
        print(f"[VRAM] 加载 {model} 前腾出空间: 可用 {g:.2f} GB (需约 {need_gb} GB)", flush=True)
    return g
def _free_vlm():
    """释放 VLM 占用的显存（Qwen2-VL 与 DINO/YOLO 不共存）。"""
    global vlm_model, vlm_processor, vlm_loaded_name
    if DB_PATCH_AVAILABLE:
        free_vlm()
        vlm_model = vlm_processor = None
        vlm_loaded_name = ""
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()   # 不归还显存的话后续加载仍会 OOM
            except Exception:
                pass
        return
    if "vlm_model" in globals() and vlm_model is not None:
        try:
            del vlm_model
        except Exception:
            pass
        vlm_model = None
    if "vlm_processor" in globals():
        vlm_processor = None
    vlm_loaded_name = ""
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
def init_vlm_local():
    global vlm_model, vlm_processor, vlm_loaded_name
    global siglip_model, siglip_processor
    # 显存互斥(12G): VLM 加载前无条件卸载 SigLIP, 否则 CUDA OOM; 检索/向量化时 ensure_siglip 自动重载
    # SigLIP 实际驻留 GPUManager(mgr._models), app 全局可能为 None, 必须走 free_siglip 真卸载
    try:
        from gpu_patch import free_siglip
        free_siglip()
    except Exception:
        pass
    if siglip_model is not None:
        try:
            siglip_model = siglip_model.cpu()
        except Exception:
            pass
        siglip_model = None
        siglip_processor = None
    if not LOAD_MODELS:
        raise RuntimeError("VLM 未启用：请将 AD_LOAD_MODELS 置为 1 后再调用")
    if vlm_model is None:
        # 使用 GPU Manager
        if DB_PATCH_AVAILABLE:
            vlm_model, vlm_processor = ensure_vlm()
            if vlm_model is not None:
                vlm_loaded_name = "Qwen2.5-VL (via GPUManager)"
                vlm_model.eval()
                _log(f"[VLM] 模型加载完成: {vlm_loaded_name}")
                print(f"[✓] VLM 场景判定引擎就绪 ({vlm_loaded_name})")
            return
        # 回退原有逻辑
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        print(f"[*] 正在加载 VLM 场景判定模型: {VLM_MODEL_NAME} ...")
        vlm_model, vlm_processor, used = _try_load_vlm(VLM_MODEL_NAME, dtype)
        vlm_loaded_name = used
        vlm_model.eval()
        _log(f"[VLM] 模型加载完成: {used}")
        print(f"[✓] VLM 场景判定引擎就绪 ({used})")
# ---- 项目级写锁：同一项目并发写底库(metadata/faiss/images 记录)时串行化，跨项目仍可并行 ----
# ---- 项目级写锁：同一项目并发写底库(metadata/faiss/images 记录)时串行化，跨项目仍可并行 ----
project_locks = {}
_project_locks_guard = threading.Lock()
def _project_lock(name):
    """取/建某项目的可重入写锁（RLock），供向量化/入库/打包快照等按项目串行。"""
    with _project_locks_guard:
        if name not in project_locks:
            project_locks[name] = threading.RLock()
        return project_locks[name]
# ----------------- SigLIP DataLoader -----------------
class SigLIPImageDataset(
    Dataset if (not LITE_MODE and Dataset is not None) else object
):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (384, 384), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
                tensor = self.transform(img)
            return tensor, path, 1
        except Exception:
            return torch.zeros((3, 384, 384)), path, 0
# ----------------- 项目空间管理器 -----------------
def sanitize_project_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fa5]", "_", (name or "").strip())
    return clean if clean else "default"
def get_project_paths(project_name: str):
    p_name = sanitize_project_name(project_name)
    p_dir = os.path.join(PROJECTS_ROOT, p_name)
    img_dir = os.path.join(p_dir, "images")
    idx_path = os.path.join(p_dir, "index.faiss")
    meta_path = os.path.join(p_dir, "metadata.json")
    # 注意：这里【不再】自动创建目录。只有显式新建项目(create_project)/启动初始化才建目录，
    # 否则读取旧项目名的请求会把“已改名/已删除”的项目重新建出来(空项目复活)。
    return p_name, p_dir, img_dir, idx_path, meta_path
def _project_exists_on_disk(project_name: str) -> bool:
    """项目是否真实存在——只认磁盘/网盘上的目录（default 除外：系统保留，永远算存在）。

    项目列表接口也是扫目录得来的，所以目录就是"项目存不存在"的唯一事实来源。
    已删除的项目在任何读路径上都不能再被建出来，否则前端 localStorage 里残留的项目名
    会在每次轮询时把 DB 记录 + 网盘目录一起复活（现象：网盘里删掉的项目又自己跳出来）。"""
    p_name, p_dir, _img, _idx, _meta = get_project_paths(project_name)
    if p_name == "default":
        return True
    return bool(p_name) and os.path.isdir(p_dir)
def _find_file_by_name(root: str, filename: str) -> str:
    """在 root 目录下递归查找 basename 等于 filename 的文件（兼容 images 下子目录结构）。

    命中返回其绝对路径，未命中返回空字符串。"""
    if not filename:
        return ""
    if os.path.exists(os.path.join(root, filename)):
        return os.path.join(root, filename)
    for _dirpath, _dirnames, _filenames in os.walk(root):
        if filename in _filenames:
            return os.path.join(_dirpath, filename)
    return ""
class _FakeIndex:
    """轻量模式下替代 faiss 索引的占位对象（仅提供接口，不做真实检索）"""
    def __init__(self, dim):
        self.d = dim
        self.ntotal = 0
    def add(self, feats):
        try:
            self.ntotal += int(feats.shape[0])
        except Exception:
            pass
    def search(self, q, k):
        import numpy as _np
        n = getattr(q, "shape", [1])[0]
        return (
            _np.zeros((n, k), dtype=_np.float32),
            _np.zeros((n, k), dtype=_np.int64),
        )
    def reconstruct_n(self, a, b):
        return None
def load_project_context(project_name: str):
    p_name, p_dir, img_dir, idx_path, meta_path = get_project_paths(project_name)
    if p_name in project_cache:
        return project_cache[p_name]
    metadata = []
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = []
    if LITE_MODE or faiss is None:
        index = _FakeIndex(FEAT_DIM)
    else:
        try:
            index = faiss.read_index(idx_path)
            if index.d != FEAT_DIM:
                index = faiss.IndexFlatIP(FEAT_DIM)
                metadata = []
        except Exception:
            index = faiss.IndexFlatIP(FEAT_DIM)
            metadata = []
    ctx = {
        "name": p_name,
        "dir": p_dir,
        "img_dir": img_dir,
        "idx_path": idx_path,
        "meta_path": meta_path,
        "index": index,
        "metadata": metadata,
    }
    project_cache[p_name] = ctx
    return ctx
def save_project_context(ctx):
    with _project_lock(ctx["name"]):
        if not LITE_MODE and faiss is not None:
            try:
                faiss.write_index(ctx["index"], ctx["idx_path"])
            except Exception:
                pass
        with open(ctx["meta_path"], "w", encoding="utf-8") as f:
            json.dump(ctx["metadata"], f, ensure_ascii=False)
# ----------------- 模型加载 -----------------
@app.on_event("startup")
def startup_event():
    global siglip_model, siglip_processor, dino_model, dino_processor, yolo_model, vlm_model, vlm_processor, vlm_loaded_name
    print(f"[*] 启动 Maxieye 旗舰 AI 挖掘平台，运行设备: {DEVICE}")
    _log(f"[*] 服务启动，日志目录: {LOG_DIR}")
    _compress_old_logs()  # 启动时先压缩一次过期日志
    _start_compression()  # 后台线程每小时压缩
    _load_passwords()  # 读取持久化密码（若有）
    _cleanup_stale_jobs()  # 重启后清理僵尸 RUNNING/PENDING（诚实标 FAILED 可续跑）
    # 初始化数据库补丁
    if DB_PATCH_AVAILABLE:
        apply_db_patches()
        # 初始化 GPU Manager
        init_gpu_managers(device=DEVICE, max_vram_gb=11.0)
        print_vram_status()
    else:
        # 显式初始化默认项目目录（仅启动/新建时创建目录，读路径不建）
        _dp = get_project_paths("default")
        os.makedirs(_dp[2], exist_ok=True)  # images 目录
        load_project_context("default")
    if not LOAD_MODELS:
        print(
            "[i] 测试模式：已跳过模型加载 (AD_LOAD_MODELS=0)。部署时请将 AD_LOAD_MODELS 置为 1 再启用。"
        )
        return
    # ===== 模型加载：生产优先（本地有就加载，没有则联网下载），失败自动跳过 =====
    print(f"[*] [1/3] 装载 Google SigLIP: {SIGLIP_MODEL_NAME} ...")
    siglip_processor = siglip_model = None
    try:
        siglip_processor = AutoProcessor.from_pretrained(
            SIGLIP_MODEL_NAME, local_files_only=True
        )
        siglip_model = AutoModel.from_pretrained(
            SIGLIP_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            local_files_only=True,
        ).to(DEVICE)
    except Exception:
        try:
            print("[i] 本地无 SigLIP 权重，尝试联网下载...")
            siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            siglip_model = AutoModel.from_pretrained(
                SIGLIP_MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE)
        except Exception:
            siglip_processor = siglip_model = None
            print("[!] SigLIP 加载/下载失败，已跳过。")
    if siglip_model is not None:
        siglip_model.eval()
        print("[i] SigLIP 已就绪。")
    print(
        "[*] Grounding DINO / Qwen2-VL 已改为【按需懒加载 + FP16】：不再常驻，"
        "仅在首次调用目标检测/VLM 时加载；DINO 与 VLM 互斥，切换时会自动互相释放，避免 12G 显存 OOM。"
    )
    print("🚀 启动完成（生产模式：本地有则加载，没有则下载，失败自动跳过）")
# ----------------- API 路由 -----------------
@app.get("/")
def read_index():
    # 前端唯一入口：front.html（改前端只改这一个文件，无需再同步副本）
    index_html = os.path.join(PROJECT_DIR, "front.html")
    # 禁缓存：前端是热替换的（改了不用重启），但浏览器一旦缓存住旧页面，用户点了半天
    # 还是旧交互（"改了怎么没生效"多半是这个）。这里显式告诉浏览器每次都回源取。
    _nocache = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if os.path.exists(index_html):
        return FileResponse(index_html, headers=_nocache)
    # 兼容旧部署：历史上服务器上的文件名是 前端.html
    legacy = os.path.join(PROJECT_DIR, "前端.html")
    if os.path.exists(legacy):
        return FileResponse(legacy, headers=_nocache)
    return {"msg": "front.html 不存在"}
@app.get("/api/projects")
def list_projects():
    if DB_PATCH_AVAILABLE:
        names = list_all_projects()
        disp = {}
        try:
            _db = get_db_session()
            from models import Project as _Pr
            disp = {p.name: (p.display_name or p.name) for p in _db.query(_Pr).all()}
            _db.close()
        except Exception:
            pass
        return {
            "projects": [{"name": n, "display_name": disp.get(n, n)} for n in names]
        }
    # 回退原有逻辑
    if not os.path.exists(PROJECTS_ROOT):
        os.makedirs(PROJECTS_ROOT, exist_ok=True)
    items = [
        d
        for d in os.listdir(PROJECTS_ROOT)
        if os.path.isdir(os.path.join(PROJECTS_ROOT, d))
    ]
    # 永远保留 default 项目，避免下拉列表里消失
    if "default" not in items:
        items.insert(0, "default")
    return {"projects": [{"name": d, "display_name": d} for d in sorted(items)]}
@app.post("/api/projects/create")
def create_project(project_name: str = Form(...)):
    clean_name = sanitize_project_name(project_name)
    _log(f"[项目] 新建项目请求: {project_name} -> {clean_name}")
    if not clean_name:
        raise HTTPException(status_code=400, detail="项目名不合法")
    if DB_PATCH_AVAILABLE:
        try:
            name = create_project_db(clean_name)
            _cp = get_project_paths(clean_name)
            os.makedirs(_cp[2], exist_ok=True)
            load_project_context(clean_name)
            return {
                "code": 200,
                "msg": f"项目 [{clean_name}] 创建成功",
                "project": clean_name,
            }
        except Exception as e:
            return {"code": 500, "msg": f"创建失败: {str(e)}"}
    # 回退原有逻辑
    _cp = get_project_paths(clean_name)
    os.makedirs(_cp[2], exist_ok=True)  # 新建时创建 images 目录
    load_project_context(clean_name)
    return {"code": 200, "msg": f"项目 [{clean_name}] 创建成功", "project": clean_name}
@app.post("/api/projects/rename")
@app.post("/api/rename_project")
def rename_project(old_name: str = Form(...), new_name: str = Form(...)):
    """重命名项目 = 只改 DB 显示名 display_name；数据目录/物理路径按原名不变。
    旧实现曾 shutil.move 网盘目录(CIFS 无移动权限必失败)且漏改 DB name → 表现为"改名变新建项目"。"""
    clean_old = sanitize_project_name(old_name)
    clean_new = sanitize_project_name(new_name)
    if not clean_old or not clean_new:
        return {"code": 400, "msg": "项目名无效"}
    try:
        if not DB_PATCH_AVAILABLE:
            return {"code": 500, "msg": "数据库未就绪"}
        from db_patch import rename_project_db
        rename_project_db(clean_old, clean_new)
        return {
            "code": 200,
            "msg": f"项目显示名已改为 [{clean_new}]（数据目录与路径不变）",
            "project": clean_old,
        }
    except ValueError as ve:
        return {"code": 400, "msg": str(ve)}
    except Exception as e:
        return {"code": 500, "msg": f"重命名失败: {str(e)}"}
@app.post("/api/projects/delete")
def delete_project(project_name: str = Form(...)):
    clean_name = sanitize_project_name(project_name)
    _log(f"[项目] 删除项目请求: {clean_name}")
    if clean_name == "default":
        return {
            "code": 400,
            "msg": "default 是系统保留的默认项目，无法删除，您可以通过清理数据来清空它。",
        }
    p_dir = os.path.join(PROJECTS_ROOT, clean_name)
    # 删除 = 仅清理数据库(级联 assets/sources/jobs 等); 磁盘/网盘文件一律不动
    # 红线: 文件目录由用户手动删除, 平台不自动删任何文件
    if DB_PATCH_AVAILABLE:
        try:
            from db_patch import delete_project_db
            delete_project_db(clean_name)
        except ValueError as ve:
            return {"code": 400, "msg": str(ve)}
        except Exception as e:
            return {"code": 500, "msg": f"数据库清理失败: {str(e)}"}
    if clean_name in project_cache:
        del project_cache[clean_name]
    return {
        "code": 200,
        "msg": f"项目 [{clean_name}] 数据库已清理。数据文件目录保留(未删除), 请手动删除",
    }
@app.get("/api/image/{project}/{image_id}")
def get_image(project: str, image_id: int):
    ctx = load_project_context(project)
    meta = ctx["metadata"]
    if 0 <= image_id < len(meta):
        path = meta[image_id].get("path", "")
        # 优先读取原路径
        if path and os.path.exists(path):
            return FileResponse(path)
        # 🔑 自愈：原路径失效（如项目改名后 metadata 未同步）时，
        #    自动在当前项目 images 目录里【递归按同名】寻找（兼容子目录结构）
        filename = meta[image_id].get("filename", "")
        if filename:
            self_heal_path = _find_file_by_name(ctx["img_dir"], filename)
            if self_heal_path:
                return FileResponse(self_heal_path)
    return JSONResponse(status_code=404, content={"msg": "图片不存在"})
@app.get("/api/db_stats")
def get_db_stats(project: str = Query("default")):
    ctx = load_project_context(project)
    if not _project_exists_on_disk(ctx["name"]):
        # 网盘目录已被删除的项目：绝不能顺手 makedirs 把目录建回来——这个接口是前端
        # 进入工作台/切换项目时必调的，一建就会把删掉的项目在网盘里重新"跳出来"。
        project_cache.pop(ctx["name"], None)  # 顺带丢掉残留缓存，避免显示旧统计
        return {
            "project": ctx["name"],
            "raw_count": 0,
            "processed_count": 0,
            "pending_count": 0,
            "missing": True,
        }
    os.makedirs(ctx["img_dir"], exist_ok=True)  # 防网盘根缺目录 FileNotFoundError(500)
    raw_images = [
        f
        for f in os.listdir(ctx["img_dir"])
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    raw_count = len(raw_images)
    processed_count = ctx["index"].ntotal if ctx["index"] is not None else 0
    return {
        "project": ctx["name"],
        "raw_count": raw_count,
        "processed_count": processed_count,
        "pending_count": max(0, raw_count - processed_count),
    }
_vectorize_state = {"status": ""}  # "": 正常；unavailable: 取不到 SigLIP；evicted: 中途被卸载
def extract_and_index_project(ctx, image_paths: List[str], frame_meta: dict = None):
    # SigLIP 可能被 VLM 卸载(显存互斥)，向量化前按需重载；加载失败则跳过本批
    _vectorize_state["status"] = ""
    if _ensure_siglip()[0] is None:
        _vectorize_state["status"] = "unavailable"
        print("[SigLIP] 模型不可用, 跳过本批向量化(仅入库不建索引)", flush=True)
        return 0
    """带项目级写锁的向量化入库：同一项目并发写底库时串行化，避免 metadata/faiss 竞争；跨项目并行。"""
    with _project_lock(ctx["name"]):
        return _extract_and_index_unlocked(ctx, image_paths, frame_meta)
def _extract_and_index_unlocked(ctx, image_paths: List[str], frame_meta: dict = None):
    if LITE_MODE or torch is None or siglip_model is None or not image_paths:
        return 0
    # 固定 384 分辨率，让 cuDNN 自动挑选最优卷积内核（同形状重复前向显著提速）
    if DEVICE == "cuda" and torch is not None:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    # 提高 DataLoader 并行解码/预处理线程数，让 CPU 预处理持续喂饱 GPU
    n_workers = 8 if DEVICE == "cuda" else 0
    dl_kwargs = dict(batch_size=128, shuffle=False, pin_memory=True)
    if n_workers > 0:
        dl_kwargs["num_workers"] = n_workers
        dl_kwargs["prefetch_factor"] = 2
    dataset = SigLIPImageDataset(image_paths)
    dataloader = DataLoader(dataset, **dl_kwargs)
    extracted_feats = []
    extracted_records = []
    with torch.no_grad():
        for tensors, paths, valids in dataloader:
            mask = valids == 1
            if not mask.any():
                continue
            valid_tensors = tensors[mask].to(DEVICE, non_blocking=True)
            with _gpu_slot("SigLIP 特征提取"):
                if siglip_model is None:
                    # 锁在批与批之间是放开的：别的 AI 任务（如检测加载 YOLO 时）会趁隙把
                    # SigLIP 卸掉腾显存。这里拿到 None 再硬跑就是 500，所以本轮提前收尾；
                    # 已完成的批次在函数末尾照常入库，重跑即可续做。
                    _vectorize_state["status"] = "evicted"
                    _log("[向量化] SigLIP 中途被其它 AI 任务卸载，本轮提前结束（已完成批次已入库，可重跑续做）")
                    break
                if DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        feats = siglip_model.get_image_features(pixel_values=valid_tensors)
                else:
                    feats = siglip_model.get_image_features(pixel_values=valid_tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            extracted_feats.append(feats.cpu().numpy().astype(np.float32))
            valid_paths = [paths[i] for i in range(len(paths)) if valids[i] == 1]
            for p in valid_paths:
                current_id = len(ctx["metadata"]) + len(extracted_records)
                extracted_records.append(
                    {
                        "id": current_id,
                        "filename": os.path.basename(p),
                        "path": p,
                        "url": f"/api/image/{ctx['name']}/{current_id}",
                        "frame_index": (frame_meta or {})
                        .get(p, {})
                        .get("frame_index", 0),
                        "timestamp": (frame_meta or {})
                        .get(p, {})
                        .get("timestamp", 0.0),
                        "video_source": (frame_meta or {}).get(p) or None,
                    }
                )
    if extracted_feats:
        all_new_feats = np.vstack(extracted_feats)
        if ctx["index"] is None or ctx["index"].d != FEAT_DIM:
            ctx["index"] = faiss.IndexFlatIP(FEAT_DIM)
        ctx["index"].add(all_new_feats)
        ctx["metadata"].extend(extracted_records)
        save_project_context(ctx)
    return len(extracted_records)
# ----------------- 数据导入与上传路由 -----------------
# 🌟 直导(import) 后台任务状态（单槽，独立于打包/抽帧，可与其它任务并行）
task_status = {
    "is_running": False,
    "is_cancelled": False,
    "current_path": "",
    "processed_count": 0,
    "total_count": 0,
    "task_type": "",
    "project": "",
    "msg": "闲置中",
}
# 🌟 打包(delivery) 后台任务状态（独立单槽，可与直导/抽帧并行，各忙各的）
delivery_status = {
    "is_running": False,
    "is_cancelled": False,
    "current_path": "",
    "processed_count": 0,
    "total_count": 0,
    "task_type": "",
    "project": "",
    "msg": "闲置中",
}
def background_full_pipeline_worker(project: str, source_path: str):
    """后台线程：递归扫描目录（支持多路径逗号分隔）+ PIL 校验 + 登记入库 + 自动提取特征"""
    global task_status
    task_status["is_running"] = True
    task_status["is_cancelled"] = False
    task_status["task_type"] = "import"
    task_status["project"] = project
    task_status["current_path"] = source_path
    task_status["processed_count"] = 0
    task_status["total_count"] = 0
    task_status["msg"] = "正在扫描目录并登记文件..."
    _log("▶ 直导任务开始: " + source_path)
    try:
        ctx = load_project_context(project)
        existing_filenames = set(
            [
                os.path.basename(m.get("path", ""))
                for m in ctx["metadata"]
                if "path" in m
            ]
        )
        new_saved_paths = []
        # 兼容处理单路径或多个逗号分隔的路径
        paths = [p.strip() for p in source_path.split(",") if p.strip()]
        src_map = {}  # 文件名 -> (原始源目录, 相对父目录链)
        cancelled = False
        videos = []  # 发现的视频(直导后自动排队抽帧)
        for p in paths:
            if not os.path.exists(p):
                task_status["msg"] = f"服务器端未找到路径: {p}"
                continue
            # 1. 自动递归扫描路径并登记到原始数据池（无后缀/时间戳后缀也强行用 PIL 试探）
            for root, dirs, files in os.walk(p):
                for f in files:
                    # 🛑 刹车点：直导(拷贝登记)过程中可被急停
                    if task_status.get("is_cancelled"):
                        cancelled = True
                        break
                    file_path = os.path.join(root, f)
                    ext = os.path.splitext(f)[1].lower()
                    # 允许常规图片或数字时间戳后缀（如 .224, .225）
                    if ext in VIDEO_EXTS:
                        if not task_status.get("is_cancelled"):
                            videos.append(file_path)
                        continue
                    elif (
                        ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
                        or ext.replace(".", "").isdigit()
                    ):
                        if f in existing_filenames:
                            continue
                        try:
                            # 强行用 PIL 试探一下它到底是不是图，打得开就收编入库！
                            with Image.open(file_path) as img:
                                img.verify()
                            # 登记到原始数据池：复制进项目图片目录（文件名即唯一标识）
                            dst_file = os.path.join(ctx["img_dir"], f)
                            shutil.copy2(file_path, dst_file)
                            new_saved_paths.append(dst_file)
                            src_map[f] = (
                                root,
                                os.path.relpath(root, p),
                            )  # (源目录, 相对父目录链)
                            existing_filenames.add(f)
                        except Exception:
                            # 打不开说明真不是图片，直接跳过
                            continue
                if cancelled:
                    break
            if cancelled:
                break
        # 2. 扫描登记完成后，无缝自动触发特征提取（在线向量化，点燃 RTX 4070 Ti）
        task_status["total_count"] = len(new_saved_paths)
        if cancelled:
            task_status["msg"] = (
                f"⚠️ 直导已手动中止！已登记 {len(new_saved_paths)} 张未向量化图片。"
            )
            return
        task_status["msg"] = "目录扫描完毕，正在唤醒 RTX 4070 Ti 自动提取特征..."
        processed_count = 0
        if new_saved_paths:
            with _project_lock(ctx["name"]):
                processed_count = extract_and_index_project(ctx, new_saved_paths)
                # 给这批新图记录 src_dir（锁内完成，切片准确，供交付打包定位伴生文件）
                if processed_count:
                    for rec in ctx["metadata"][-processed_count:]:
                        fn = rec.get("filename")
                        if fn and fn in src_map:
                            sd, rel = src_map[fn]
                            rec["src_dir"] = sd
                            if rel and rel != ".":
                                rec["directory_chain"] = [
                                    c for c in rel.split(os.sep) if c
                                ]
                    save_project_context(ctx)
        task_status["processed_count"] = processed_count
        if processed_count and DB_PATCH_AVAILABLE:
            from db_service import ensure_sync_hook, sync_metadata_paths
            sync_metadata_paths(project, ctx["metadata"], new_saved_paths)
        # 3. 发现的视频自动排队抽帧(同项目, 默认时间间隔5秒一条; 前端大盘可见)
        if videos:
            uniq_videos = list(dict.fromkeys(videos))
            try:
                _launch_video_extract_job(project, uniq_videos)
                task_status["msg"] = (
                    f"全自动导入完成！向量化 {processed_count} 张新图；另发现 {len(uniq_videos)} 个视频已自动加入抽帧任务(时间间隔5s/帧)"
                    f"，库内共 {ctx['index'].ntotal} 张"
                )
            except Exception as _ve:
                _log(f"✖ 视频自动抽帧排队失败: {_ve}")
                task_status["msg"] = (
                    f"全自动导入完成！向量化 {processed_count} 张新图；但 {len(uniq_videos)} 个视频抽帧排队失败: {_ve}"
                    f"，库内共 {ctx['index'].ntotal} 张"
                )
        else:
            task_status["msg"] = (
                f"全自动导入与特征向量化已全部完成！本次向量化 {processed_count} 张新图，库内共 {ctx['index'].ntotal} 张"
            )
    except Exception as e:
        task_status["msg"] = f"后台处理异常: {str(e)}"
        _log("✖ 直导异常: " + str(e))
    finally:
        task_status["is_running"] = False
        _log("✔ 直导结束: " + task_status["msg"])
# 🌟 极速直导接口：秒回响应，绝不卡死前端
def _launch_video_extract_job(
    project: str, video_paths, frame_step: int = 5, unit: str = "time"
):
    """直导发现视频后自动排队抽帧：与批量抽帧同链路(EXTRACT job + 后台调度)。返回 job_pk 或 None"""
    if not video_paths:
        return None
    import threading as _th
    from models import JobType
    from db_service import create_job
    jobs = [(vp, project) for vp in video_paths]
    for vp, _pj in jobs:
        try:
            _extract_entry(_to_nas_local(vp), _pj)["is_running"] = True
        except Exception:
            pass
    job_pk = None
    db = get_db_session()
    try:
        p0 = _ensure_db_project(db, project)
        if p0 is not None:
            job = create_job(
                db,
                p0.id,
                JobType.EXTRACT,
                {
                    "paths": [vp for vp in video_paths],
                    "step": frame_step,
                    "unit": unit,
                    "vectorize": True,
                    "task": "frame_extract",
                },
            )
            job_pk = job.id
            db.commit()
    finally:
        db.close()
    _log(f"[直导] 自动排队抽帧 {len(video_paths)} 个视频 job={job_pk}")
    _th.Thread(
        target=background_extract_manager,
        args=(jobs, frame_step, True, unit),
        kwargs={"job_pk": job_pk, "payload": {"paths": [vp for vp in video_paths]}},
        daemon=True,
    ).start()
    return job_pk
@app.post("/api/import_from_path")
async def import_from_path(
    project: str = Form("default"),
    source_path: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    global task_status
    if task_status["is_running"]:
        return {"code": 400, "msg": "后台已有批量任务正在狂飙中，请稍候..."}
    # 丢进后台线程，立刻给前端返回成功
    background_tasks.add_task(background_full_pipeline_worker, project, source_path)
    _log(f"[直导] 收到直导请求 project={project} path={source_path}")
    return {"code": 200, "msg": "路径已接收，后台正在静默扫描并自动向量化！"}
# 🌟 前端轮询状态查询接口
@app.get("/api/import_status")
async def get_import_status():
    """前端随时打听后台导入进度"""
    return task_status
# 🌟 全局并发任务大盘（供前端 /api/all_tasks_status 轮询）
@app.get("/api/all_tasks_status")
async def get_all_tasks_status(project: str = Query(None)):
    """返回全局任务池：抽帧并发池 + 单任务(导入/打包)状态的合并视图。

    结构: { "<目录路径|类型:路径>": {project, is_running, task_type, msg, ...} }。

    传入 project 时只返回该项目的任务（后端强制项目隔离，杜绝 A 项目看到 B 项目进度）；

    不传则返回全部（便于调试）。"""
    pool = {}
    # 1) 抽帧并发任务（按目录索引）——仅保留属于当前项目的
    for key, e in extract_task_pool.items():
        if project and e.get("project") != project:
            continue
        pool[key] = {
            "project": e.get("project", ""),
            "task_type": e.get("task_type", "extract"),
            "is_running": e.get("is_running", False),
            "is_cancelled": e.get("is_cancelled", False),
            "msg": e.get("msg", ""),
            "current_path": e.get("current_path", key),
            "processed_count": e.get("processed_count", 0),
            "total_count": e.get("total_count", 0),
        }
    # 2) 直导(import) 与 打包(delivery) 各自独立的单槽状态，并入大盘；同样按项目过滤
    for _st in (task_status, delivery_status):
        if _st.get("is_running") or _st.get("msg"):
            if (not project) or _st.get("project") == project:
                key = f"{_st.get('task_type', 'task')}:{_st.get('current_path') or _st.get('project') or '当前任务'}"
                pool[key] = {
                    "project": _st.get("project", ""),
                    "task_type": _st.get("task_type", ""),
                    "is_running": _st.get("is_running", False),
                    "is_cancelled": _st.get("is_cancelled", False),
                    "msg": _st.get("msg", ""),
                    "current_path": _st.get("current_path", ""),
                }
    return pool
@app.post("/api/all_tasks/clear")
async def clear_finished_tasks():
    """清除全局任务池中所有已完成记录（is_running=False），供前端'🧹 清除已完成记录'按钮调用。"""
    global task_status
    with _extract_lock:
        finished = [k for k, e in extract_task_pool.items() if not e.get("is_running")]
        for k in finished:
            extract_task_pool.pop(k, None)
    # 直导/打包单槽若未在运行，也清掉残留的完成提示，避免大盘常驻"已完成"
    for _st in (task_status, delivery_status):
        if not _st.get("is_running"):
            _st["msg"] = ""
            _st["project"] = ""
            _st["current_path"] = ""
    running = sum(1 for e in extract_task_pool.values() if e.get("is_running"))
    _log(f"[任务] 已清除 {len(finished)} 条已完成记录，剩余运行中 {running} 个")
    return {
        "code": 200,
        "msg": f"已清除 {len(finished)} 条已完成任务记录，剩余运行中 {running} 个",
    }
# ================= 日志查看接口（实时 / 历史 / 压缩） =================
@app.get("/api/logs/live")
def logs_live(lines: int = Query(300, ge=1, le=5000)):
    """实时查看当天日志末尾 lines 行"""
    path = _today_log_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = f.read().splitlines()
        tail = arr[-lines:]
        return {"code": 200, "file": os.path.basename(path), "lines": tail}
    except Exception as e:
        return {"code": 500, "msg": str(e)}
@app.get("/api/logs/list")
def logs_list():
    """列出所有历史日志文件（含已压缩 .gz）"""
    files = []
    for p in sorted(glob.glob(os.path.join(LOG_DIR, "app_*"))):
        files.append(
            {
                "name": os.path.basename(p),
                "size": os.path.getsize(p),
                "mtime": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))
                ),
            }
        )
    return {"code": 200, "files": files}
@app.get("/api/logs/read")
def logs_read(file: str = Query(...), lines: int = Query(0, ge=0)):
    """读取指定历史日志（支持 .gz 自动解压；lines>0 时只返回末尾 lines 行）"""
    safe = os.path.basename(file)
    path = os.path.join(LOG_DIR, safe)
    if not os.path.exists(path):
        return {"code": 404, "msg": "文件不存在"}
    try:
        open_fn = gzip.open if safe.endswith(".gz") else open
        with open_fn(path, "rt", encoding="utf-8") as f:
            arr = f.read().splitlines()
        out = arr[-lines:] if lines > 0 else arr
        return {"code": 200, "file": safe, "lines": out}
    except Exception as e:
        return {"code": 500, "msg": str(e)}
# ================= 稀疏采样抽帧（后台任务） =================
def _parse_ts(name):
    """从文件名提取时间戳数值，如 '1782865441.424.jpg' -> 1782865441.424"""
    base = os.path.splitext(name)[0]
    m = re.match(r"^(\d+(?:\.\d+)?)", base)
    return float(m.group(1)) if m else None
def _to_nas_local(p):
    """把 Windows UNC 路径（\\\\host\\\\share\\\\... 或 //host/share/...）转成服务器本地挂载路径（/mnt/<share>/...）。

    服务器 10.2.248.34 把 NAS 的 Data_Platform 挂载在 /mnt/Data_Platform。"""
    if not p:
        return p
    p = p.replace("\\", "/").strip()
    if p.startswith("//"):
        parts = p.strip("/").split("/", 1)
        if len(parts) == 2 and parts[1]:
            return "/mnt/" + parts[1].lstrip("/")
    return p
def _parse_extract_jobs(folder_path: str, default_project: str):
    """folder_path 可含多条路径（换行 / 逗号 / 分号分隔）。

    每条路径可写成  `项目名::/绝对路径`  来指定归属不同项目；未指定的归到 default_project。"""
    jobs = []
    for raw in re.split(r"[\r\n;,，；]+", folder_path or ""):
        s = (raw or "").strip().strip('"')
        if not s:
            continue
        if "::" in s:
            proj, p = s.split("::", 1)
            jobs.append((p.strip().strip('"'), proj.strip() or default_project))
        else:
            jobs.append((s, default_project))
    return jobs
# ===== 抽帧并发任务池（不同目录允许并行，同一目录禁止重复提交） =====
# key = 目录路径(local)  ->  {project, is_running, task_type, msg, processed_count, total_count}
extract_task_pool = {}
_extract_lock = threading.Lock()
def _extract_entry(path, project):
    """取/建某目录对应的抽帧任务记录，并确保带 project 字段（供大盘按项目过滤）。"""
    with _extract_lock:
        entry = extract_task_pool.setdefault(
            path,
            {
                "project": project,
                "task_type": "extract",
                "is_running": False,
                "is_cancelled": False,  # 刹车信号
                "msg": "",
                "current_path": path,
                "processed_count": 0,
                "total_count": 0,
            },
        )
        if not entry.get("project"):
            entry["project"] = project
        entry.setdefault("is_cancelled", False)  # 兼容历史记录
        return entry
def _video_entry(video_path, project):
    """取/建某视频对应的处理任务记录（并入抽帧并发池，便于大盘展示/急停/并发）。"""
    with _extract_lock:
        entry = extract_task_pool.setdefault(
            video_path,
            {
                "project": project,
                "task_type": "video",
                "is_running": False,
                "is_cancelled": False,
                "msg": "",
                "current_path": video_path,
                "processed_count": 0,
                "total_count": 0,
            },
        )
        entry["task_type"] = "video"
        if not entry.get("project"):
            entry["project"] = project
        entry.setdefault("is_cancelled", False)
        return entry
def _parallel_copy_files(jobs, entry=None, workers=8, chunk=64):
    """线程池并发拷贝 [(src,dst), ...]：dst 已存在自动跳过；按 chunk 检查急停。

    返回成功写入的 dst 列表。entry 可选，用于写进度(processed_count/msg)。"""
    from concurrent.futures import ThreadPoolExecutor
    ok_dsts = []
    done = 0
    total = len(jobs)
    for c0 in range(0, total, chunk):
        if entry is not None and entry.get("is_cancelled"):
            break
        part = jobs[c0 : c0 + chunk]
        def _one(j):
            src, dst = j
            try:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    return dst
            except Exception:
                pass
            return None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_one, part):
                if r:
                    ok_dsts.append(r)
        done += len(part)
        if entry is not None:
            entry["processed_count"] = done
            if done % 200 == 0:
                entry["msg"] = f"拷贝/入库中... 已处理 {done}/{total}"
    return ok_dsts
def _run_extract_one(
    folder_path: str, frame_step: int, project: str, vectorize: bool, unit: str
):
    """执行单条路径的稀疏采样抽帧，返回结果字符串。进度写入该路径对应的并发池记录。"""
    folder_path = _to_nas_local(folder_path)  # 自动把 UNC 路径转成服务器本地挂载路径
    entry = _extract_entry(folder_path, project)
    entry["current_path"] = folder_path
    entry["processed_count"] = 0
    entry["total_count"] = 0
    entry["is_cancelled"] = False  # 新任务开始前清除急停信号
    entry["msg"] = f"正在对目录进行稀疏采样抽帧 (步长: {frame_step}, 单位: {unit})..."
    try:
        if not os.path.isdir(folder_path):
            return f"❌ 文件夹不存在: {folder_path}"
        # 1) 递归收集所有 jpg（跳过输出目录 sampled_frames）
        jpg_files = []
        for root, dirs, files in os.walk(folder_path):
            if os.path.basename(root).lower() == "sampled_frames":
                continue
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg")):
                    jpg_files.append(os.path.join(root, f))
        if not jpg_files:
            return "❌ 未找到任何 jpg 文件"
        # 2) 按时间戳排序（能解析时间戳的按时间戳，解析不了的按文件名）
        def _sort_key(p):
            ts = _parse_ts(os.path.basename(p))
            return ts if ts is not None else float("inf")
        jpg_files.sort(key=_sort_key)
        # 3) 抽取：支持按帧数间隔(count) 或 按时间戳间隔(time)
        if frame_step <= 0:
            frame_step = 1
        selected = []
        if unit == "count":
            # 按帧数间隔：每隔 frame_step 抽 1 张（第 0 张起），
            # 与视频抽帧 frame_interval=step 的语义保持一致（旧实现是 step+1，比标签说的稀一档）
            selected = jpg_files[::frame_step]
        else:
            # 按时间戳间隔（秒）
            last_ts = None
            for p in jpg_files:
                ts = _parse_ts(os.path.basename(p))
                if ts is None:
                    continue  # 无法解析时间戳的帧不参与时间戳间隔抽取
                if last_ts is None or (ts - last_ts) >= frame_step:
                    selected.append(p)
                    last_ts = ts
        if not selected:
            return "❌ 未能按间隔选出任何帧"
        # 4) 已抽帧直接入底库向量化（不再在源目录建 sampled_frames 预览文件夹）
        entry["total_count"] = len(selected)
        # 5) 自动入底库向量化（已抽帧拷贝进项目库 images/ 入库/交付）
        if not vectorize:
            # 仅返回抽取统计（不建预览目录、不落库）
            if entry.get("is_cancelled"):
                entry["msg"] = "⚠️ 任务已手动中止"
                return "⚠️ 任务已手动中止"
            if unit == "count":
                return f"✅ [{project}] 共 {len(jpg_files)} 帧，按每 {frame_step} 帧抽 1 帧，共抽 {len(selected)} 帧（未入库）"
            return f"✅ [{project}] 共 {len(jpg_files)} 帧，按 {frame_step}s 时间戳间隔抽取 {len(selected)} 帧（未入库）"
        if vectorize:
            if entry.get("is_cancelled"):
                entry["msg"] = "⚠️ 任务已手动中止！已入底库的图片保留。"
                return "⚠️ 任务已手动中止！已入底库的图片保留。"
            ctx = load_project_context(project)
            existing_filenames = set(
                [
                    os.path.basename(m.get("path", ""))
                    for m in ctx["metadata"]
                    if "path" in m
                ]
            )
            jobs = []
            src_map = {}
            for p in selected:
                fn = os.path.basename(p)
                if fn in existing_filenames:
                    continue
                dst = os.path.join(ctx["img_dir"], fn)
                jobs.append((p, dst))
                src_map[fn] = os.path.dirname(p)  # 伴生文件与源 jpg 同目录
                existing_filenames.add(fn)
            new_saved_paths = _parallel_copy_files(jobs, entry)
            if entry.get("is_cancelled"):
                entry["msg"] = (
                    f"⚠️ 任务已手动中止！保留已入 {len(new_saved_paths)} 张图片。"
                )
                return f"⚠️ 任务已手动中止！保留已入 {len(new_saved_paths)} 张图片。"
            vec = 0
            if new_saved_paths:
                with _project_lock(ctx["name"]):
                    vec = extract_and_index_project(ctx, new_saved_paths)
                    # 给这批新图记录 src_dir（锁内完成，切片准确，供交付打包定位伴生文件）
                    if vec:
                        for rec in ctx["metadata"][-vec:]:
                            fn = rec.get("filename")
                            if fn and fn in src_map:
                                rec["src_dir"] = src_map[fn]
                        save_project_context(ctx)
                        if DB_PATCH_AVAILABLE:
                            from db_service import ensure_sync_hook, sync_metadata_paths
                            sync_metadata_paths(
                                project, ctx["metadata"], new_saved_paths
                            )
        return f"✅ [{project}] 已抽 {len(selected)} 帧并入库向量化 {vec} 张（库内共 {ctx['index'].ntotal} 张）"
    except Exception as e:
        _log("✖ 抽帧异常: " + str(e))
        return f"❌ 抽帧异常: {str(e)}"
def background_extract_manager(
    jobs,
    frame_step: int,
    vectorize: bool,
    unit: str,
    job_pk: int = None,
    payload: dict = None,
):
    """后台抽帧总调度：各条路径写入独立并发池记录，不同目录可并行（同一目录由接口层拦截重复）。

    进度以 extract_task_pool 各条目为准（前端大盘读取）；若传入 job_pk 则同步镜像 DB Job

    （PENDING->RUNNING->SUCCESS/FAILED/CANCELLED），实现任务系统统一管理与断点可见。"""
    from models import JobStatus
    from datetime import datetime as _dt
    _log(
        f"▶ 抽帧任务开始: 共 {len(jobs)} 条路径 (间隔:{frame_step},{unit}) job={job_pk}"
    )
    if job_pk:
        _job_set(
            job_pk,
            status=JobStatus.RUNNING,
            progress=0.0,
            current_stage=f"扫描抽帧 {len(jobs)} 条路径",
        )
    results = []
    cancelled_any = False
    n = len(jobs)
    try:
        for idx, (path, proj) in enumerate(jobs, 1):
            local = _to_nas_local(path)
            entry = _extract_entry(local, proj)
            entry["is_running"] = True
            entry["msg"] = f"正在抽帧 第{idx}/{len(jobs)}条 [{proj}] {path}..."
            _log(f"[抽帧] 批次 {idx}/{len(jobs)} 项目={proj} 路径={path}")
            try:
                res = _run_extract_one(path, frame_step, proj, vectorize, unit)
            except Exception as e:
                res = f"❌ 抽帧异常: {str(e)}"
                _log("✖ 抽帧异常: " + str(e))
            results.append({"path": path, "project": proj, "result": res})
            entry["msg"] = res
            entry["is_running"] = False
            if entry.get("is_cancelled"):
                cancelled_any = True
            if job_pk:
                _job_set(
                    job_pk,
                    progress=round(idx / n * 100, 1),
                    current_stage=f"[{idx}/{n}] {path}",
                )
        if job_pk:
            if cancelled_any:
                _job_set(
                    job_pk,
                    status=JobStatus.CANCELLED,
                    progress=100.0,
                    current_stage="已中止",
                    result={"paths": results},
                )
            else:
                _job_set(
                    job_pk,
                    status=JobStatus.SUCCESS,
                    progress=100.0,
                    current_stage="全部完成",
                    result={"paths": results},
                    completed_at=_dt.utcnow(),
                )
    except Exception as e:
        _log("✖ 抽帧任务异常: " + str(e))
        if job_pk:
            _job_set(job_pk, status=JobStatus.FAILED, error=str(e))
    _log("✔ 抽帧结束: " + " | ".join(r["result"] for r in results))
# ==========================================
# 急停接口：对正在运行的抽帧任务发送刹车信号
# ==========================================
@app.post("/api/cancel_task")
async def cancel_task(folder_path: str = Form(...)):
    """对正在运行的后台任务发送急停信号（通用：抽帧 / 直导 / 打包）。

    任务会在当前步骤落盘后优雅停止，已产生的中间结果保留。"""
    global task_status
    raw = (folder_path or "").strip()
    key = _to_nas_local(raw)
    # 1) 抽帧并发池（按目录索引）
    with _extract_lock:
        entry = extract_task_pool.get(key)
        if entry is None and raw != key:
            entry = extract_task_pool.get(raw)  # 原始键容错
    if entry and entry.get("is_running"):
        entry["is_cancelled"] = True
        entry["msg"] = "正在中止任务，等待当前帧落盘..."
        _log(f"[任务] 已发送抽帧中止信号: {key}")
        return {"code": 200, "msg": "中止信号已发送"}
    # 2) 直导(import)：按类型前缀/路径/项目匹配即视为命中
    if task_status.get("is_running") and task_status.get("task_type") == "import":
        cur = task_status.get("current_path") or task_status.get("project") or ""
        if (
            (raw.startswith("import:"))
            or raw == cur
            or raw == task_status.get("project", "")
        ):
            task_status["is_cancelled"] = True
            task_status["msg"] = "正在中止任务，等待当前步骤落盘..."
            _log(f"[任务] 已发送直导中止信号: {raw}")
            return {"code": 200, "msg": "中止信号已发送"}
    # 3) 打包(delivery)：独立单槽单独判定
    if (
        delivery_status.get("is_running")
        and delivery_status.get("task_type") == "delivery"
    ):
        cur = (
            delivery_status.get("current_path") or delivery_status.get("project") or ""
        )
        if (
            (raw.startswith("delivery:"))
            or raw == cur
            or raw == delivery_status.get("project", "")
        ):
            delivery_status["is_cancelled"] = True
            delivery_status["msg"] = "正在中止任务，等待当前步骤落盘..."
            _log(f"[任务] 已发送打包中止信号: {raw}")
            return {"code": 200, "msg": "中止信号已发送"}
    return {"code": 400, "msg": "任务不存在或已结束"}
# ================= 视频抽帧（后台任务，并入并发池） =================
# 支持的视频格式（与图片类似，可按目录递归扫描）
VIDEO_EXTS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ts",
    ".m2ts",
    ".rmvb",
    ".rm",
    ".vob",
)
def _is_video_file(name: str) -> bool:
    return name.lower().endswith(VIDEO_EXTS)
def _collect_videos(root_path: str):
    """递归收集 root 下所有视频文件，返回绝对路径列表。"""
    vids = []
    for root, _dirs, files in os.walk(root_path):
        for f in files:
            if _is_video_file(f):
                vids.append(os.path.join(root, f))
    vids.sort()
    return vids
def _probe_video_total_frames(cap) -> int:
    """探测视频总帧数；CAP_PROP_FRAME_COUNT 不可靠时用 grab() 快读计数（不解码）。"""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        return total
    n = 0
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return n
def _extract_video_frames(
    ctx,
    video_path: str,
    step: int,
    unit: str,
    entry,
    existing_filenames,
    new_saved_paths,
    name_prefix: str,
    mode: str = "interval",
    n_frames: int = None,
    ratio: float = None,
    adapt_threshold: float = 28.0,
    frame_meta: dict = None,
):
    """对单个视频按四种模式抽帧（PRD 6.1），写入项目底库目录，返回实际保存帧数。

    模式：

      interval    - 固定间隔：unit=='count'→每隔 step 帧抽 1；unit=='time'→每隔 step 秒抽 1（向后兼容默认）

      fixed_count - 固定数量：整个视频【均匀】抽 n_frames 张（PRD 6.2 默认模式）

      ratio       - 比例抽帧：按视频总帧数的 ratio（如 0.1 = 10%）均匀抽取

      adaptive    - 自适应：帧间 HSV 直方图差异 > adapt_threshold 视为场景切换才抽；>3s 无变化强制兜底抽 1 张

    急停由 entry['is_cancelled'] 触发；命名带 name_prefix+时间戳 避免跨视频/与底库冲突。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(
            "无法打开视频文件（可能不是有效视频或编码不受支持）: "
            + os.path.basename(video_path)
        )
    # 确保输出目录存在：DB 建项目不建目录，网盘根下项目目录缺失时 cv2.imwrite 会静默失败(返回False)
    try:
        os.makedirs(ctx["img_dir"], exist_ok=True)
    except Exception as e:
        raise Exception(f"无法创建项目图片目录 {ctx['img_dir']}: {e}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    # ---- 解析模式参数（PRD 6.1 四种模式）----
    mode = (mode or "interval").lower()
    targets = None  # fixed_count / ratio 的目标帧号集合（均匀采样）
    if mode in ("fixed_count", "ratio"):
        if mode == "fixed_count":
            target_count = max(1, int(n_frames or 5))
        else:
            r = min(max(float(ratio or 0.1), 0.0), 1.0)
            target_count = None  # 由总帧数换算
        total_frames = _probe_video_total_frames(cap)
        if total_frames > 0:
            if target_count is None:
                target_count = max(1, int(round(total_frames * r)))
            # PRD 6.2：均匀采样，如 50 帧抽 5 帧 → 位置 0/12/24/37/49
            targets = set(
                np.linspace(0, total_frames - 1, target_count)
                .round()
                .astype(int)
                .tolist()
            )
        else:
            # 无法探测总帧数（如流式源）→ 退化为固定间隔，沿用调用方 step/unit
            mode = "interval"
            if not step or step <= 0:
                step = 1
                unit = "time"
    # 固定间隔换算
    frame_interval = 1
    if mode == "interval":
        if step is None or step <= 0:
            step = 1
        if unit == "time":
            frame_interval = max(1, int(round(step * video_fps)))
        else:
            frame_interval = max(1, step)
    # 自适应模式：HSV 64-bin 直方图（比 YUV 更能区分场景/光照变化）
    def _hist_hsv(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    saved = 0
    count = 0
    last_hist = None
    since_last_save = 0  # 距上次保存的帧数（自适应兜底）
    force_interval = max(1, int(round(video_fps * 3)))  # 3 秒无变化强制抽
    first_frame_saved = False
    def _save_frame(frame):
        nonlocal saved, last_hist, since_last_save, first_frame_saved
        fn = f"{name_prefix}_{int(time.time() * 1000)}_{saved:06d}.jpg"
        if fn in existing_filenames:
            return False
        dst = os.path.join(ctx["img_dir"], fn)
        ok = False
        try:
            ok = bool(cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]))
        except Exception as e:
            raise Exception(f"帧写入异常 {dst}: {e}")
        if not ok:
            # 目录不可写/磁盘满/网盘掉线：必须报错，否则统计为0帧却提示成功(静默丢数据)
            raise Exception(
                f"帧写入失败（目录不可写或网盘未挂载）: {ctx['img_dir']}"
            )
        if os.path.exists(dst):
            new_saved_paths.append(dst)
            existing_filenames.add(fn)
            saved += 1
            if frame_meta is not None:
                frame_meta[dst] = {
                    "frame_index": count,
                    "timestamp": round(count / video_fps, 3),
                    "video_filename": os.path.basename(video_path),
                    "video_path": video_path,
                }
            last_hist = _hist_hsv(frame)
            since_last_save = 0
            first_frame_saved = True
            return True
        return False
    while cap.isOpened():
        # 🛑 刹车点：可被大盘急停
        if entry.get("is_cancelled"):
            break
        ret, frame = cap.read()
        if not ret:
            break
        take = False
        if mode == "interval":
            take = count % frame_interval == 0
        elif targets is not None:
            take = count in targets
        elif mode == "adaptive":
            hist = _hist_hsv(frame)
            if not first_frame_saved:
                take = True
            elif last_hist is not None:
                diff = cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                take = (diff * 100.0) >= float(adapt_threshold or 28.0)
            # 场景长期无变化也要兜底保存，避免全静止视频抽 0 张
            if not take and since_last_save >= force_interval:
                take = True
        if take:
            _save_frame(frame)
        count += 1
        since_last_save += 1
    # adaptive 最后兜底：一个场景都没变化也没保存 → 存最后一帧
    if mode == "adaptive" and saved == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
        ret, frame = cap.read()
        if ret and frame is not None:
            _save_frame(frame)
    cap.release()
    return saved
def _video_interval_desc(
    step: int,
    unit: str,
    mode: str = "interval",
    n_frames: int = None,
    ratio: float = None,
) -> str:
    """把抽帧策略转成可读描述（PRD 6.1 四种模式）"""
    mode = (mode or "interval").lower()
    if mode == "fixed_count":
        return f"每视频均匀抽 {max(1, int(n_frames or 5))} 帧"
    if mode == "ratio":
        r = float(ratio or 0.1)
        return f"按视频总帧数 {r * 100:.0f}% 均匀抽帧"
    if mode == "adaptive":
        return "自适应场景变化抽帧"
    if unit == "time":
        return f"每隔 {step} 秒抽 1 帧"
    return f"每隔 {step} 帧抽 1 帧"
def _run_video_list(
    entry,
    project: str,
    videos,
    step: int,
    unit: str,
    kind: str,
    mode: str = "interval",
    n_frames: int = None,
    ratio: float = None,
    adapt_threshold: float = 28.0,
):
    """对一组视频（单文件或目录递归收集）逐个按四种模式抽帧（PRD 6.1），统一写入项目底库并自动向量化。

    entry 需已由调用方置为 is_running。"""
    if not videos:
        entry["msg"] = "未找到任何视频文件"
        return
    ctx = load_project_context(project)
    existing_filenames = set(
        os.path.basename(m.get("path", "")) for m in ctx["metadata"] if m.get("path")
    )
    new_saved_paths = []
    frame_meta = {}
    entry["total_count"] = len(videos)
    total_saved = 0
    cancelled = False
    errors = []
    for idx, vp in enumerate(videos, 1):
        if entry.get("is_cancelled"):
            cancelled = True
            break
        entry["current_path"] = vp
        entry["msg"] = f"[{idx}/{len(videos)}] 正在抽帧: {os.path.basename(vp)}"
        try:
            saved = _extract_video_frames(
                ctx,
                vp,
                step,
                unit,
                entry,
                existing_filenames,
                new_saved_paths,
                f"v{idx}",
                frame_meta=frame_meta,
                mode=mode,
                n_frames=n_frames,
                ratio=ratio,
                adapt_threshold=adapt_threshold,
            )
            total_saved += saved
        except Exception as e:
            errors.append(f"{os.path.basename(vp)}: {e}")
            _log(f"✖ 单个视频抽帧失败 {vp}: {str(e)}")
            entry["msg"] = (
                f"[{idx}/{len(videos)}] {os.path.basename(vp)} 抽帧失败: {str(e)}"
            )
        entry["processed_count"] = idx
    vec = 0
    if new_saved_paths:
        entry["msg"] = (
            f"共从 {len(videos)} 个视频中抽帧 {total_saved} 张，开始执行 GPU 向量化..."
        )
        vec = extract_and_index_project(ctx, new_saved_paths, frame_meta=frame_meta)
        save_project_context(ctx)
        if vec and DB_PATCH_AVAILABLE:
            from db_service import ensure_sync_hook, sync_metadata_paths
            sync_metadata_paths(project, ctx["metadata"], new_saved_paths)
    if cancelled:
        entry["msg"] = (
            f"⚠️ 任务已手动中止！已处理 {entry['processed_count']}/{len(videos)} 个视频，抽帧 {total_saved} 张，向量化入库 {vec} 张。"
        )
    elif errors and not total_saved:
        # 全部失败：必须明确报错，否则"0 张 + 成功"会掩盖真实故障(如签名不匹配/目录不可写)
        raise Exception(
            f"视频抽帧失败（{len(errors)}/{len(videos)} 个视频）: " + " | ".join(errors[:3])
        )
    elif errors:
        entry["msg"] = (
            f"⚠️ 部分视频抽帧失败（成功 {len(videos) - len(errors)}/{len(videos)}）：抽帧 {total_saved} 张，向量化 {vec} 张。"
            f"失败原因: " + " | ".join(errors[:3])
        )
    else:
        entry["msg"] = (
            f"✅ 视频抽帧与特征入库完成：{kind}共扫描 {len(videos)} 个视频，抽帧 {total_saved} 张，向量化 {vec} 张，库内共 {ctx['index'].ntotal} 张"
        )
def _run_video_job(
    entry,
    project: str,
    local: str,
    step: int,
    unit: str,
    mode: str = "interval",
    n_frames: int = None,
    ratio: float = None,
    adapt_threshold: float = 28.0,
):
    """对一条提交路径处理：目录→递归收集全部视频；单个视频文件→仅该文件。"""
    if cv2 is None:
        entry["msg"] = "视频处理失败：服务器未安装 opencv-python (cv2)"
        return
    desc = _video_interval_desc(step, unit, mode, n_frames, ratio)
    if os.path.isdir(local):
        entry["msg"] = f"正在递归扫描目录内视频文件 ({desc})..."
        _run_video_list(
            entry,
            project,
            _collect_videos(local),
            step,
            unit,
            "目录",
            mode=mode,
            n_frames=n_frames,
            ratio=ratio,
            adapt_threshold=adapt_threshold,
        )
    elif os.path.isfile(local) and _is_video_file(local):
        entry["msg"] = f"正在解析视频并抽帧 ({desc})..."
        _run_video_list(
            entry,
            project,
            [local],
            step,
            unit,
            "单视频",
            mode=mode,
            n_frames=n_frames,
            ratio=ratio,
            adapt_threshold=adapt_threshold,
        )
    else:
        entry["msg"] = f"❌ 路径无效（不是目录也不是视频文件）: {local}"
def background_video_manager(
    jobs,
    step: int,
    unit: str,
    mode: str = "interval",
    n_frames: int = None,
    ratio: float = None,
    adapt_threshold: float = 28.0,
    job_pk: int = None,
    payload: dict = None,
):
    """后台视频抽帧总调度：像图片抽帧一样支持多路径（可含 项目名::/路径 归入不同项目），

    各条路径写入独立并发池记录；每条路径可被大盘【⏹️ 强制中止】。

    传入 job_pk 时同步镜像 DB Job 状态（任务系统统一管理）。"""
    from models import JobStatus
    from datetime import datetime as _dt
    _log(f"▶ 视频抽帧任务开始: 共 {len(jobs)} 条路径 job={job_pk}")
    if job_pk:
        _job_set(
            job_pk,
            status=JobStatus.RUNNING,
            progress=0.0,
            current_stage=f"视频抽帧 {len(jobs)} 条路径",
        )
    results = []
    cancelled_any = False
    errors_any = []
    n = len(jobs)
    try:
        for idx, (path, proj) in enumerate(jobs, 1):
            local = _to_nas_local(path)
            entry = _video_entry(local, proj)
            entry["is_running"] = True
            entry["is_cancelled"] = False
            entry["processed_count"] = 0
            entry["total_count"] = 0
            entry["current_path"] = local
            entry["msg"] = f"[{idx}/{len(jobs)}] 正在处理 第{idx}条 [{proj}] {path}..."
            _log(
                f"[视频] 批次 {idx}/{len(jobs)} 项目={proj} 路径={path} mode={mode} step={step} unit={unit} n_frames={n_frames} ratio={ratio}"
            )
            try:
                _run_video_job(
                    entry,
                    proj,
                    local,
                    step,
                    unit,
                    mode=mode,
                    n_frames=n_frames,
                    ratio=ratio,
                    adapt_threshold=adapt_threshold,
                )
            except Exception as e:
                entry["msg"] = f"视频处理出错: {str(e)}"
                entry["failed"] = True
                errors_any.append(f"{path}: {e}")
                _log("✖ 视频处理异常: " + str(e))
            finally:
                entry["is_running"] = False
            if entry.get("is_cancelled"):
                cancelled_any = True
            results.append({"path": path, "project": proj, "msg": entry.get("msg", "")})
            if job_pk:
                _job_set(
                    job_pk,
                    progress=round(idx / n * 100, 1),
                    current_stage=f"[{idx}/{n}] {path}",
                )
        if job_pk:
            if cancelled_any:
                _job_set(
                    job_pk,
                    status=JobStatus.CANCELLED,
                    progress=100.0,
                    current_stage="已中止",
                    result={"paths": results},
                )
            elif errors_any and len(errors_any) >= n:
                # 全部路径都失败：标记 FAILED，别再以 SUCCESS 掩盖故障
                _job_set(
                    job_pk,
                    status=JobStatus.FAILED,
                    error=" | ".join(errors_any[:3]),
                    result={"paths": results},
                    completed_at=_dt.utcnow(),
                )
            else:
                _job_set(
                    job_pk,
                    status=JobStatus.SUCCESS,
                    progress=100.0,
                    current_stage="全部完成",
                    result={"paths": results},
                    completed_at=_dt.utcnow(),
                )
    except Exception as e:
        _log("✖ 视频抽帧任务异常: " + str(e))
        if job_pk:
            _job_set(job_pk, status=JobStatus.FAILED, error=str(e))
@app.post("/api/process_video")
async def process_video(
    project: str = Form(default="default"),
    video_path: str = Form(...),
    frame_step: int = Form(None),
    unit: str = Form(default="time"),
    mode: str = Form(default="interval"),  # interval / fixed_count / ratio / adaptive
    n_frames: int = Form(None),  # fixed_count: 每视频抽几帧
    ratio: float = Form(None),  # ratio: 抽取比例 0.1 = 10%
    adapt_threshold: float = Form(None),  # adaptive: 直方图差异阈值（0-100，默认 28）
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    if cv2 is None:
        return {"code": 400, "msg": "服务器未安装 opencv-python (cv2)，无法处理视频"}
    # 归一化可选参数（兼容 FastAPI 与直接调用）
    mode = (mode if isinstance(mode, str) else "interval") or "interval"
    mode = mode.lower()
    if mode not in ("interval", "fixed_count", "ratio", "adaptive"):
        return {
            "code": 400,
            "msg": f"未知抽帧模式: {mode}，可选 interval/fixed_count/ratio/adaptive",
        }
    frame_step = frame_step if isinstance(frame_step, int) else None
    n_frames = n_frames if isinstance(n_frames, int) else None
    ratio = (
        ratio
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
        else None
    )
    adapt_threshold = (
        adapt_threshold
        if isinstance(adapt_threshold, (int, float))
        and not isinstance(adapt_threshold, bool)
        else None
    )
    # 与图片抽帧一致：video_path 可含多条路径（换行/逗号/分号分隔，可用 项目名::/路径 指定归属项目）
    jobs = _parse_extract_jobs(video_path, project)
    if not jobs:
        return {"code": 400, "msg": "未解析到有效的视频路径"}
    # 并发控制：同一路径正在处理时拦截；不同路径可并发
    for path, proj in jobs:
        entry = _video_entry(_to_nas_local(path), proj)
        if entry.get("is_running"):
            return {
                "code": 400,
                "msg": f"该路径的视频任务正在运行中，请勿重复提交: {path}",
            }
    for path, proj in jobs:
        _video_entry(_to_nas_local(path), proj)["is_running"] = True
    # 创建 DB Job（EXTRACT 类型，任务系统统一管理）
    job_pk = None
    from models import JobType
    from db_service import create_job
    db = get_db_session()
    try:
        _proj0 = jobs[0][1] or "default"
        p0 = _ensure_db_project(db, _proj0)
        if p0 is not None:
            job = create_job(
                db,
                p0.id,
                JobType.EXTRACT,
                {
                    "paths": [j[0] for j in jobs],
                    "step": frame_step or 1,
                    "unit": unit,
                    "mode": mode,
                    "n_frames": n_frames,
                    "ratio": ratio,
                    "adapt_threshold": adapt_threshold or 28.0,
                    "task": "video_extract",
                },
            )
            job_pk = job.id
            db.commit()
    finally:
        db.close()
    background_tasks.add_task(
        background_video_manager,
        jobs,
        frame_step or 1,
        unit,
        mode,
        n_frames,
        ratio,
        adapt_threshold or 28.0,
        job_pk=job_pk,
    )
    desc = _video_interval_desc(frame_step or 1, unit, mode, n_frames, ratio)
    _log(
        f"[视频] 收到视频抽帧请求 路径数={len(jobs)} mode={mode} step={frame_step} unit={unit} n_frames={n_frames} ratio={ratio} 默认项目={project} job={job_pk}"
    )
    return {
        "code": 200,
        "msg": f"视频抽帧任务已启动：共 {len(jobs)} 条路径，{desc}，帧将抽入对应项目",
        "job_id": str(job_pk) if job_pk else None,
    }
# ================= 三件套交付打包（后台任务） =================
class ExportDeliveryRequest(BaseModel):
    project: str
    selected_image_names: List[str]
    extensions: List[str] = [".bin", ".xml"]
    export_dir_name: str = "delivery_package"
    output_dir: Optional[str] = (
        None  # 用户指定的输出目录（可填服务器/NAS路径，支持 UNC）
    )
    source_roots: Optional[List[str]] = None  # 手动指定伴生文件源根目录(历史数据兜底)
def background_delivery_package(req: ExportDeliveryRequest):
    """把选中的图片及其同名伴生文件（JPG/BIN/XML 等）打包到指定输出目录。

    伴生文件(.bin/.xml/_raw.jpg)查找顺序：

      1) 该图入库时记录的原始源目录 src_dir（抽帧/直导自动记录，最快）；

      2) 打包请求携带的 source_roots / 环境变量 AD_SOURCE_ROOTS 里的根目录（递归按同名匹配，适合历史已入库数据）；

    全找不到才计缺失。"""
    global delivery_status
    delivery_status["is_running"] = True
    delivery_status["is_cancelled"] = False
    delivery_status["task_type"] = "delivery"
    delivery_status["project"] = req.project
    delivery_status["current_path"] = req.project
    delivery_status["processed_count"] = 0
    delivery_status["total_count"] = 0
    delivery_status["msg"] = "正在匹配并打包关联交付文件 (JPG + BIN + XML)..."
    _log("▶ 打包任务开始: project=" + req.project)
    try:
        ctx = load_project_context(req.project)
        img_dir = ctx["img_dir"]
        # 输出目录：用户指定了就用用户指定的(UNC自动转服务器挂载)，否则默认放项目目录下
        if req.output_dir and req.output_dir.strip():
            output_dir = _to_nas_local(req.output_dir.strip())
        else:
            output_dir = os.path.join(ctx["dir"], req.export_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        # 在项目写锁下对元数据做一次快照，避免与同项目入库/向量化并发读到"列表正在变化"
        with _project_lock(ctx["name"]):
            meta_snapshot = list(ctx["metadata"])
        meta_by_fn = {}
        for m in meta_snapshot:
            meta_by_fn.setdefault(os.path.basename(m.get("path", "")), m)
        # 1) 组装每条选中图片：jpg 名 + 需找的伴生文件名 + 该图原始源目录
        items = []
        needed = {}  # 伴生文件名 -> 找到的绝对路径(未找到为 None)
        for img_name in req.selected_image_names:
            base = os.path.splitext(img_name)[0]
            rec = meta_by_fn.get(img_name) or {}
            src_dir = rec.get("src_dir")
            # 若文件名带 _raw，先去 _raw 再找关联文件（与挑图工具一致）
            match_base = (
                base.replace("_raw", "")
                if "_raw" in os.path.basename(base).lower()
                else base
            )
            ext_names = [f"{match_base}{ext}" for ext in req.extensions]
            items.append(
                {"img_name": img_name, "src_dir": src_dir, "ext_names": ext_names}
            )
            for n in ext_names:
                needed.setdefault(n, None)
        # 2) 先在每条图入库时记录的原始源目录里精确找
        for it in items:
            if it["src_dir"]:
                for n in it["ext_names"]:
                    p = os.path.join(it["src_dir"], n)
                    if needed.get(n) is None and os.path.exists(p):
                        needed[n] = p
        # 3) 仍有缺失时，去 source_roots / AD_SOURCE_ROOTS 递归扫一遍（只按需要的文件名匹配，一次遍历）
        roots = []
        if req.source_roots:
            roots += [r for r in req.source_roots if r and r.strip()]
        roots += SOURCE_ROOTS
        if roots and any(p is None for p in needed.values()):
            needset = set(n for n, p in needed.items() if p is None)
            for root in roots:
                root = _to_nas_local(root)
                if not os.path.isdir(root):
                    continue
                for dirpath, _, files in os.walk(root):
                    for f in files:
                        if f in needset and needed.get(f) is None:
                            needed[f] = os.path.join(dirpath, f)
                    if not any(p is None for p in needed.values()):
                        break
                if not any(p is None for p in needed.values()):
                    break
        # 4) 拷贝到输出目录
        copied_count = 0
        missing_count = 0
        total = len(req.selected_image_names)
        cancelled = False
        for idx, it in enumerate(items, 1):
            # 🛑 刹车点：打包交付过程中可被急停
            if delivery_status.get("is_cancelled"):
                cancelled = True
                break
            src_jpg = os.path.join(img_dir, it["img_name"])
            if os.path.exists(src_jpg):
                shutil.copy2(src_jpg, os.path.join(output_dir, it["img_name"]))
                copied_count += 1
            else:
                missing_count += 1
            for n in it["ext_names"]:
                p = needed.get(n)
                if p:
                    shutil.copy2(p, os.path.join(output_dir, os.path.basename(n)))
                    copied_count += 1
                else:
                    missing_count += 1
            delivery_status["processed_count"] = idx
            delivery_status["total_count"] = total
        if cancelled:
            delivery_status["msg"] = (
                f"⚠️ 打包已手动中止！已拷贝 {copied_count} 个文件，缺失 {missing_count} 个。"
            )
            _log("✔ 打包被手动中止: " + delivery_status["msg"])
        else:
            delivery_status["msg"] = (
                f"✅ 交付包打包完成！共拷贝 {copied_count} 个文件，缺失 {missing_count} 个，输出: {output_dir}"
            )
            _log("✔ 打包结束: " + delivery_status["msg"])
            # 生成交付清单 deliver_meta.json (filename/source/annotations/final_tags)
            try:
                _write_deliver_meta(project, output_dir, items, img_dir)
            except Exception as _dme:
                _log(f"✖ 交付清单生成失败: {_dme}")
    except Exception as e:
        delivery_status["msg"] = f"打包异常: {str(e)}"
        _log("✖ 打包异常: " + str(e))
    finally:
        delivery_status["is_running"] = False
# ================= 稀疏抽帧接口 =================
@app.post("/api/extract_frames")
async def extract_frames(
    folder_path: str = Form(...),
    frame_step: int = Form(default=5),
    project: str = Form(default="default"),
    vectorize: int = Form(default=1),
    unit: str = Form(default="time"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    jobs = _parse_extract_jobs(folder_path, project)
    if not jobs:
        return {"code": 400, "msg": "未解析到有效的抽帧路径"}
    # 并发控制：仅当【同一个目录】正在抽帧时才拦截；不同目录允许直接并发
    for path, proj in jobs:
        entry = _extract_entry(_to_nas_local(path), proj)
        if entry.get("is_running"):
            return {
                "code": 400,
                "msg": f"该目录的抽帧任务正在运行中，请勿重复提交: {path}",
            }
    # 先同步置为运行中（立即拦截重复提交），再丢入后台线程池
    for path, proj in jobs:
        _extract_entry(_to_nas_local(path), proj)["is_running"] = True
    # 创建 DB Job（任务系统统一管理；EXTRACT 类型）
    job_pk = None
    from models import JobType
    from db_service import create_job
    db = get_db_session()
    try:
        _proj0 = jobs[0][1] or "default"
        p0 = _ensure_db_project(db, _proj0)
        if p0 is not None:
            job = create_job(
                db,
                p0.id,
                JobType.EXTRACT,
                {
                    "paths": [j[0] for j in jobs],
                    "step": frame_step,
                    "unit": unit,
                    "vectorize": bool(vectorize),
                    "task": "frame_extract",
                },
            )
            job_pk = job.id
            db.commit()
    finally:
        db.close()
    background_tasks.add_task(
        background_extract_manager,
        jobs,
        frame_step,
        bool(vectorize),
        unit,
        job_pk=job_pk,
        payload={"paths": [j[0] for j in jobs]},
    )
    _log(
        f"[抽帧] 收到抽帧请求 路径数={len(jobs)} step={frame_step} unit={unit} vectorize={vectorize} 默认项目={project} job={job_pk}"
    )
    return {
        "code": 200,
        "msg": f"稀疏抽帧任务已启动：共 {len(jobs)} 条路径，间隔单位: {'时间戳(秒)' if unit == 'time' else '帧数'}",
        "job_id": str(job_pk) if job_pk else None,
    }
# ================= 三件套交付打包接口 =================
@app.post("/api/export_delivery")
async def export_delivery(
    req: ExportDeliveryRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    global delivery_status
    # 只检查打包自身的单槽，不再受直导/抽帧影响 → 各任务独立并行
    if delivery_status.get("is_running"):
        return {"code": 400, "msg": "已有打包任务在运行，请稍候..."}
    background_tasks.add_task(background_delivery_package, req)
    _log(
        f"[打包] 收到打包请求 project={req.project} frames={len(req.selected_image_names)} exts={req.extensions}"
    )
    return {
        "code": 200,
        "msg": f"正在后台自动打包 {len(req.selected_image_names)} 帧数据及其伴生文件！",
    }
@app.post("/api/upload_batch")
async def upload_batch(
    project: str = Form("default"), files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="未接收到文件")
    ctx = load_project_context(project)
    os.makedirs(ctx["img_dir"], exist_ok=True)  # 网盘根下项目目录可能不存在, 确保可写
    existing_filenames = set(
        [os.path.basename(m.get("path", "")) for m in ctx["metadata"] if "path" in m]
    )
    new_saved_paths = []
    for file in files:
        # 关键修复：文件名可能带有子目录相对路径（如 "夜晚/xxx.jpg" 或 "夜晚\\xxx.jpg"），
        # 直接 os.path.join(ctx["img_dir"], file.filename) 会导致 images/夜晚/ 子目录
        # 不存在而抛 FileNotFoundError(500)。这里统一取 basename，兼容 "/" 与 "\\" 分隔符。
        raw_name = (file.filename or "").replace("\\", "/")
        safe_name = os.path.basename(raw_name)
        file_path = os.path.join(ctx["img_dir"], safe_name)
        if safe_name.lower().endswith(".zip"):
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                        ) and not member.startswith("__MACOSX"):
                            fn = os.path.basename(member)
                            if fn and fn not in existing_filenames:
                                ext_path = os.path.join(ctx["img_dir"], fn)
                                with zip_ref.open(member) as s, open(
                                    ext_path, "wb"
                                ) as t:
                                    shutil.copyfileobj(s, t)
                                new_saved_paths.append(ext_path)
                                existing_filenames.add(fn)
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        elif safe_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            if safe_name not in existing_filenames:
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                new_saved_paths.append(file_path)
                existing_filenames.add(safe_name)
    processed_count = 0
    if new_saved_paths:
        processed_count = extract_and_index_project(ctx, new_saved_paths)
    if processed_count and DB_PATCH_AVAILABLE:
        from db_service import ensure_sync_hook, sync_metadata_paths
        sync_metadata_paths(project, ctx["metadata"], new_saved_paths)
    _log(
        f"[上传] 上传批次完成，新增并向量化 {processed_count} 张，库内 {ctx['index'].ntotal} 张"
    )
    return {
        "code": 200,
        "msg": f"本次新增并向量化 {processed_count} 张图片",
        "processed_count": ctx["index"].ntotal,
    }
@app.post("/api/build_index_online")
async def build_index_online(project: str = Query("default")):
    _log(f"[向量化] 触发在线向量化 project={project}")
    ctx = load_project_context(project)
    if not _project_exists_on_disk(ctx["name"]):
        return {
            "code": 404,
            "msg": f"项目 [{ctx['name']}] 的目录不存在（已删除的项目不会被重建），无法向量化",
        }
    all_imgs = [
        os.path.join(ctx["img_dir"], f)
        for f in os.listdir(ctx["img_dir"])
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    indexed_paths = set([m["path"] for m in ctx["metadata"] if "path" in m])
    unprocessed_paths = [p for p in all_imgs if p not in indexed_paths]
    if not unprocessed_paths:
        return {"code": 200, "msg": "该项目所有图片均已完成向量化"}
    processed_count = extract_and_index_project(ctx, unprocessed_paths)
    if processed_count and DB_PATCH_AVAILABLE:
        from db_service import ensure_sync_hook, sync_metadata_paths
        sync_metadata_paths(project, ctx["metadata"], unprocessed_paths)
    if _vectorize_state["status"] == "unavailable":
        return {
            "code": 503,
            "msg": "向量化未执行：SigLIP 取不到（显存被 AI 任务占用或模型缺失），请等任务跑完再试",
        }
    if _vectorize_state["status"] == "evicted":
        return {
            "code": 200,
            "msg": f"已向量化 {processed_count} 张；中途 SigLIP 被其它 AI 任务卸载腾显存，"
                   f"本轮提前结束（已完成部分已入库），稍后再点一次即可续做",
        }
    return {"code": 200, "msg": f"成功向量化 {processed_count} 张图片！"}
# ----------------- 检索路由 -----------------
@app.get("/api/search")
def search_text(
    project: str = Query("default"),
    query: str = Query(..., min_length=1),
    top_k: int = 500,
):
    _log(f"[检索] 文本检索 project={project} query={query} top_k={top_k}")
    ctx = load_project_context(project)
    if ctx["index"] is None or ctx["index"].ntotal == 0:
        return {"code": 200, "results": [], "total": 0}
    if siglip_model is None and _running_tag_jobs(project):
        # AI 任务在跑：重载 SigLIP 会腾显存把正在用的 VLM 挤掉，任务反复重载 -> 明确提示而不是硬抢
        return {"code": 503, "msg": "语义检索暂不可用：AI 分析任务正在运行（避免抢显存中断任务），请稍后重试或用文件名检索"}
    if _ensure_siglip()[0] is None:   # VLM 可能占着显存把 SigLIP 卸了，这里按需重载
        return {"code": 500, "msg": "SigLIP 未就绪（显存不足或模型缺失），无法做文本检索"}
    with _gpu_slot("SigLIP 文本检索"), torch.no_grad():
        if siglip_model is None or siglip_processor is None:
            # 排队期间被别的 AI 任务卸掉了：降级返回，不能硬用 None 前向（会 500）
            return {"code": 503, "msg": "语义检索暂不可用：显存被 AI 任务占用（SigLIP 刚被卸载），请稍后重试"}
        inputs = siglip_processor(
            text=[query], return_tensors="pt", padding="max_length", max_length=64
        ).to(DEVICE)
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                text_feat = siglip_model.get_text_features(**inputs)
        else:
            text_feat = siglip_model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_np = text_feat.cpu().numpy().astype(np.float32)
    actual_k = min(top_k, ctx["index"].ntotal)
    scores, indices = ctx["index"].search(text_np, actual_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(ctx["metadata"]):
            item = ctx["metadata"][idx].copy()
            item["score"] = float(f"{score:.4f}")
            results.append(item)
    return {"code": 200, "results": results}
@app.post("/api/search_by_filename")
def search_by_filename(project: str = Form("default"), filename_query: str = Form(...)):
    _log(f"[检索] 文件名检索 project={project} query={filename_query}")
    ctx = load_project_context(project)
    results = [
        m
        for m in ctx["metadata"]
        if filename_query.lower() in m.get("filename", "").lower()
    ]
    return {"code": 200, "results": results}
@app.post("/api/search_by_external_image")
async def search_by_external_image(
    project: str = Form("default"),
    files: List[UploadFile] = File(...),
    top_k: int = 500,
):
    _log(f"[检索] 以图搜图 project={project} 参考图={len(files)} 张")
    ctx = load_project_context(project)
    if ctx["index"] is None or ctx["index"].ntotal == 0 or not files:
        return {"code": 200, "results": []}
    if siglip_model is None and _running_tag_jobs(project):
        return {"code": 503, "msg": "以图搜图暂不可用：AI 分析任务正在运行（避免抢显存中断任务），请稍后重试"}
    if _ensure_siglip()[0] is None:   # 同上：VLM 占显存时按需重载 SigLIP
        return {"code": 500, "msg": "SigLIP 未就绪（显存不足或模型缺失），无法以图搜图"}
    pil_imgs = []
    for f in files:
        try:
            pil_imgs.append(Image.open(f.file).convert("RGB"))
        except Exception:
            continue
    if not pil_imgs:
        return {"code": 200, "results": []}
    with _gpu_slot("SigLIP 以图搜图"), torch.no_grad():
        if siglip_model is None or siglip_processor is None:
            return {"code": 503, "msg": "以图搜图暂不可用：显存被 AI 任务占用（SigLIP 刚被卸载），请稍后重试"}
        inputs = siglip_processor(images=pil_imgs, return_tensors="pt").to(DEVICE)
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                img_feats = siglip_model.get_image_features(**inputs)
        else:
            img_feats = siglip_model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        img_np = img_feats.cpu().numpy().astype(np.float32)
    actual_k = min(top_k, ctx["index"].ntotal)
    scores, indices = ctx["index"].search(img_np, actual_k)
    max_scores = {}
    for q_idx in range(len(pil_imgs)):
        for score, idx in zip(scores[q_idx], indices[q_idx]):
            if 0 <= idx < len(ctx["metadata"]):
                max_scores[idx] = max(max_scores.get(idx, -1.0), float(score))
    sorted_indices = sorted(
        max_scores.keys(), key=lambda k: max_scores[k], reverse=True
    )
    results = []
    for idx in sorted_indices:
        item = ctx["metadata"][idx].copy()
        item["score"] = float(f"{max_scores[idx]:.4f}")
        results.append(item)
    return {"code": 200, "results": results}
@app.get("/api/list_all")
def list_all(project: str = Query("default"), page: int = 1, size: int = 30):
    ctx = load_project_context(project)
    total = len(ctx["metadata"])
    start = (page - 1) * size
    end = start + size
    items = ctx["metadata"][start:end] if start < total else []
    total_pages = (total + size - 1) // size if total > 0 else 1
    return {
        "code": 200,
        "total": total,
        "total_pages": total_pages,
        "current_page": page,
        "items": items,
    }
# ----------------- 数据清洗与标注推理 -----------------
@app.post("/api/dedup_stats")
def dedup_stats(project: str = Form("default"), sim_threshold: float = Form(0.95)):
    _log(f"[降重] 三级去重扫描 project={project} threshold={sim_threshold}")
    if not DB_PATCH_AVAILABLE:
        # 回退原有逻辑
        ctx = load_project_context(project)
        total = len(ctx["metadata"])
        if total < 2 or ctx["index"] is None or LITE_MODE:
            return {
                "total_images": total,
                "unique_images": total,
                "duplicate_count": 0,
                "dedup_rate": "0.0%",
                "clusters": [],
            }
        feats = ctx["index"].reconstruct_n(0, total)
        sim_matrix = np.dot(feats, feats.T)
        np.fill_diagonal(sim_matrix, 0)
        visited = set()
        clusters = []
        duplicate_count = 0
        for i in range(total):
            if i in visited:
                continue
            sim_indices = np.where(sim_matrix[i] >= sim_threshold)[0]
            cluster_items = [ctx["metadata"][i]]
            for idx in sim_indices:
                if idx not in visited:
                    visited.add(idx)
                    cluster_items.append(ctx["metadata"][idx])
                    duplicate_count += 1
            if len(cluster_items) > 1:
                clusters.append({"items": cluster_items})
        unique_images = total - duplicate_count
        rate = f"{(duplicate_count / total * 100):.2f}%" if total > 0 else "0.0%"
        return {
            "total_images": total,
            "unique_images": unique_images,
            "duplicate_count": duplicate_count,
            "dedup_rate": rate,
            "clusters": clusters,
        }
    # 使用新的三级去重引擎
    dedup_engine = get_dedup_engine()
    dedup_engine.siglip_threshold = sim_threshold
    db = get_db_session()
    try:
        from db_service import get_project
        proj = get_project(db, project)
        if not proj:
            return {"code": 404, "msg": "项目不存在"}
        # 获取项目上下文用于 SigLIP 索引
        ctx = load_project_context(project)
        # 执行全量去重扫描
        result = dedup_engine.scan_full_dedup(
            proj.id,
            siglip_index=(
                ctx["index"]
                if ctx["index"] and hasattr(ctx["index"], "ntotal")
                else None
            ),
            siglip_metadata=ctx["metadata"] if ctx["metadata"] else None,
        )
        # 保存 duplicate_groups 到数据库
        from db_service import Asset
        for cluster in result.get("clusters", []):
            group_id = (
                cluster.get("type", "unknown")
                + "_"
                + str(hash(str(cluster.get("items", []))))[:8]
            )
            for item in cluster.get("items", []):
                asset_id = item.get("asset_id") or item.get("id")
                if asset_id:
                    asset = (
                        db.query(Asset)
                        .filter(Asset.asset_id == asset_id, Asset.project_id == proj.id)
                        .first()
                    )
                    if asset:
                        meta = asset.asset_metadata or {}
                        dup_groups = meta.get("duplicate_groups", [])
                        if group_id not in dup_groups:
                            dup_groups.append(group_id)
                        meta["duplicate_groups"] = dup_groups
                        meta["duplicate_level"] = (
                            3
                            if cluster.get("type") == "siglip"
                            else (1 if cluster.get("type") == "exact" else 2)
                        )
                        asset.asset_metadata = meta
                db.commit()
        # 兼容前端: 三级去重引擎 item 只有 asset_id/path, 无 metadata id ——
        # 按 path basename 映射回 ctx metadata 序号, 否则前端删除按钮收到空列表(全逗号)删不掉文件
        _bn2id = {}
        for _mi, _m in enumerate(ctx.get("metadata") or []):
            _bn2id.setdefault(
                os.path.basename(str(_m.get("path", ""))), _m.get("id", _mi)
            )
        for _cl in result.get("clusters", []):
            for _it in _cl.get("items", []):
                _it.setdefault(
                    "id", _bn2id.get(os.path.basename(str(_it.get("path", ""))))
                )
        return {"code": 200, **result}
    finally:
        db.close()
@app.post("/api/delete_and_sync")
def delete_and_sync(project: str = Form("default"), image_ids: str = Form(...)):
    _log(f"[降重] 永久粉碎冗余 project={project} ids={image_ids}")
    ctx = load_project_context(project)
    del_ids = set([int(x) for x in image_ids.split(",") if x.strip().isdigit()])
    if not del_ids:
        return {"deleted_count": 0}
    remaining_paths = []
    deleted_count = 0
    failed_files = []  # 删除失败(如网盘挂载账号无删除权限)的文件: 保留记录不降级待处理
    for m in ctx["metadata"]:
        if m["id"] in del_ids:
            removed = False
            if os.path.exists(m["path"]):
                try:
                    os.remove(m["path"])
                    removed = True
                except Exception as _e:
                    _log(f"[降重] 文件删除失败(权限?): {m['path']} -> {_e}")
                    failed_files.append(m["path"])
            else:
                removed = True  # 文件已不在磁盘, 视为删除成功
            if removed:
                deleted_count += 1
            else:
                remaining_paths.append(m["path"])  # 文件还在 -> 保留其已处理记录
        else:
            remaining_paths.append(m["path"])
    with _project_lock(ctx["name"]):
        ctx["metadata"] = []
        ctx["index"] = (
            faiss.IndexFlatIP(FEAT_DIM)
            if (not LITE_MODE and faiss is not None)
            else _FakeIndex(FEAT_DIM)
        )
        extract_and_index_project(ctx, remaining_paths)
    # DB 联动: metadata/faiss 重建后 vector_id 已重排(0..n) ——
    # 重排 DB 资产 vector_id; 被删(含手动删)文件对应资产连带删除(仅清DB, 不碰文件)
    try:
        from db_service import get_db_session
        from db_service import resync_asset_vector_ids
        _db = get_db_session()
        try:
            _r = resync_asset_vector_ids(
                _db, project, [m["path"] for m in ctx["metadata"]]
            )
            _log(
                f"[降重] 粉碎后DB联动 更新vector_id={_r['updated']} 连带删除资产={_r['removed']}"
            )
        finally:
            _db.close()
    except Exception as _e:
        _log(f"[降重] DB联动失败(不影响文件删除): {_e}")
        return {
            "code": 200,
            "deleted_count": deleted_count,
            "failed_count": len(failed_files),
            "failed_files": failed_files[:20],
            "msg": (
                f'有 {len(failed_files)} 个文件删除失败(多为网盘挂载账号无删除权限)，已保留其记录，请手动删除文件后点"清理失效资产"'
                if failed_files
                else "粉碎成功"
            ),
        }
@app.post("/api/ground_detect")
def ground_detect(
    project: str = Form("default"),
    image_id: int = Form(...),
    text_prompt: str = Form(...),
):
    """Grounding DINO 真实目标级开放词汇推理（支持中英文自动转译）"""
    _log(f"[检测] DINO project={project} image_id={image_id} prompt={text_prompt}")
    global dino_model, dino_processor
    ctx = load_project_context(project)
    # DINO 与 VLM/YOLO 不共存：先腾显存，再按需加载(FP16)
    _free_vlm()
    _free_yolo()
    if not _ensure_dino():
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}
    if not (0 <= image_id < len(ctx["metadata"])):
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}
    img_path = ctx["metadata"][image_id]["path"]
    try:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        # --- DINO 中英词典映射（支持直接输入中文；词典可编辑 dino_dict.json） ---
        dino_dict = _load_dino_dict()
        prompt = text_prompt.strip()
        # 离线词典替换：把命中的中文词替换成英文（长词优先，避免单字误替换）
        sorted_keys = sorted(dino_dict.keys(), key=len, reverse=True)
        for zh in sorted_keys:
            prompt = prompt.replace(zh, dino_dict[zh])
        # DINO 要求 prompt 必须以英文句号结尾
        if not prompt.endswith("."):
            prompt += "."
        print(f"[*] DINO 原始提示词: [{text_prompt}] ---> 处理后: [{prompt}]")
        inputs = dino_processor(images=image, text=prompt, return_tensors="pt").to(
            DEVICE
        )
        with torch.no_grad():
            outputs = dino_model(**inputs)
        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(h, w)],
        )[0]
        # 将英文标签翻译回中文（词典只维护后端这一份）
        dino_reverse = {en: zh for zh, en in dino_dict.items()}
        labels = [dino_reverse.get(lbl, lbl) for lbl in results["labels"]]
        res = {
            "image_id": image_id,
            "engine": "dino",
            "prompt": (
                text_prompt or ""
            ).strip(),  # 记录所用提示词，供前端判断缓存是否需失效
            "scores": results["scores"].cpu().numpy().tolist(),
            "labels": labels,
            "boxes": results["boxes"].cpu().numpy().tolist(),
            "width": w,
            "height": h,
        }
        save_detection_record(project, image_id, res)  # 自动持久化，刷新/重开仍在
        return res
    except Exception as e:
        print(f"[!] DINO 检测异常: {e}")
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}
@app.post("/api/yolo_detect")
def yolo_detect(project: str = Form("default"), image_id: int = Form(...)):
    _log(f"[检测] YOLO project={project} image_id={image_id}")
    global yolo_model
    ctx = load_project_context(project)
    if not (0 <= image_id < len(ctx["metadata"])):
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}
    # YOLO 不与 DINO/VLM 共存：先释放它们，腾出 12G 显存再按需加载
    _free_dino()
    _free_vlm()
    if not _ensure_yolo():  # 按需懒加载
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}
    img_path = ctx["metadata"][image_id]["path"]
    try:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        results = yolo_model(image, verbose=False, device=DEVICE)[0]
        cls_indices = results.boxes.cls.cpu().numpy().astype(int).tolist()
        res = {
            "image_id": image_id,
            "engine": "yolo",
            "scores": results.boxes.conf.cpu().numpy().tolist(),
            "labels": [results.names[idx] for idx in cls_indices],
            "boxes": results.boxes.xyxy.cpu().numpy().tolist(),
            "width": w,
            "height": h,
        }
        save_detection_record(project, image_id, res)  # 自动持久化，刷新/重开仍在
        return res
    except Exception as e:
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}
def _preload_images(paths, rgb=True, workers=8):
    """线程池并行解码一批图片为 ndarray（RGB 或 BGR）。返回 list，元素为 (arr, w, h)，失败为 None。

    并行解码避免 CPU 串行读图成为 GPU 瓶颈；解码后直接喂 GPU，批量推理更易把占用顶高。"""
    from concurrent.futures import ThreadPoolExecutor
    def load(p):
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img)
            if rgb:
                arr = np.ascontiguousarray(arr)
            else:
                arr = np.ascontiguousarray(arr[:, :, ::-1])  # RGB -> BGR
            return (arr, img.width, img.height)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(load, paths))
def _preload_bgr(paths, workers=8):
    return _preload_images(paths, rgb=False, workers=workers)
def _preload_rgb(paths, workers=8):
    return _preload_images(paths, rgb=True, workers=workers)
@app.post("/api/yolo_detect_batch")
def yolo_detect_batch(
    project: str = Form("default"),
    image_ids: str = Form(...),
    batch_size: int = Form(16),
    fp16: str = Form("1"),
):
    """加锁包装：整个批量检测期间与 VLM/SigLIP/DINO 的加载和推理互斥。
    12G 卡上这些模型无法共存，不互斥就会出现"一边跑检测、一边加载 7B"把显存挤爆。"""
    with _GPU_MODEL_LOCK:
        return _yolo_detect_batch_locked(project, image_ids, batch_size, fp16)
def _yolo_detect_batch_locked(
    project: str,
    image_ids: str,
    batch_size: int,
    fp16: str,
):
    """YOLO 批量推理：一次提交一批图片，由 ultralytics 自动分批切到 GPU，榨干显存/算力。

    单张单发只有 ~10% 占用，瓶颈在『喂太慢』；批量后占用率与吞吐都会大幅提升。

    返回 {str(image_id): res}，并一次性落盘持久化。"""
    _log(f"[检测] YOLO-BATCH project={project} ids={image_ids}")
    global yolo_model
    ctx = load_project_context(project)
    # YOLO 不与 DINO/VLM/SigLIP 共存：先腾空间再按需加载。
    # 必须走 make_room_for(含 SigLIP)：否则并发加载的 7B VLM(约 8.8G) 会让 YOLO
    # 批量推理 OOM（曾导致整轮检测返回空、2034 帧被静默判为“无目标”）。
    make_room_for("yolo", 6.0)
    if not _ensure_yolo():  # 按需懒加载
        return {"code": 500, "msg": "YOLO 模型加载失败/未启用 (AD_LOAD_MODELS=0?)"}
    # 批大小按剩余显存自适应：1080p 下批量太大极易 OOM（曾整轮检测因此失败）
    try:
        _free_gb = _vram_free_gb()
        if _free_gb is not None:
            _cap = 16 if _free_gb >= 8 else (8 if _free_gb >= 5 else (4 if _free_gb >= 3 else 2))
            if batch_size > _cap:
                print(f"[检测] 显存剩余 {_free_gb:.1f}G，YOLO 批大小 {batch_size} -> {_cap}", flush=True)
                batch_size = _cap
    except Exception:
        pass
    ids = []
    for s in image_ids.replace(" ", "").split(","):
        if s == "":
            continue
        try:
            ids.append(int(s))
        except Exception:
            pass
    # 只保留底库里真实存在的帧，避免越界
    valid = [
        (i, ctx["metadata"][i]["path"]) for i in ids if 0 <= i < len(ctx["metadata"])
    ]
    if not valid:
        return {"code": 200, "results": {}, "count": 0}
    paths = [p for _, p in valid]
    idxs = [i for i, _ in valid]
    bs = max(1, min(int(batch_size) or 16, len(idxs)))
    # 固定 imgsz 下让 cuDNN 自动挑选最优卷积算法（同分辨率批量时提速明显）
    if DEVICE == "cuda" and torch is not None:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    out = {}
    with _detections_lock:
        # 必须自己分块：ultralytics 收到"内存数组列表"时会忽略 batch 参数，
        # 把整份列表堆成单个张量再上 GPU（2034 张 640x640 ≈ 10G，fp16 转换峰值 ≈ 18.6G），
        # 之前整轮检测就是这样 OOM 失败的。分块后显存/内存都被 bs 限制住。
        for _s in range(0, len(paths), bs):
            _cp = paths[_s:_s + bs]
            _ci = idxs[_s:_s + bs]
            _items = _preload_bgr(_cp, workers=min(8, max(1, len(_cp))))
            _keep = [k for k in range(len(_ci)) if _items[k] is not None]
            if not _keep:
                continue
            _arr = [_items[k][0] for k in _keep]
            _sizes = {_ci[k]: (_items[k][1], _items[k][2]) for k in _keep}
            try:
                _res = yolo_model(
                    _arr,
                    imgsz=640,  # 固定分辨率 → 触发 cudnn.benchmark 最优内核
                    device=DEVICE,
                    verbose=False,
                    half=((str(fp16) == "1") and DEVICE == "cuda"),  # FP16 约 2x(仅 CUDA)
                )
            except Exception as e:
                print(f"[!] YOLO 批量推理异常: {e}", flush=True)
                return {"code": 500, "msg": f"YOLO 批量推理异常: {e}"}
            for _i, _k in enumerate(_keep):
                _idx = _ci[_k]
                try:
                    r = _res[_i]
                    if r is None or r.boxes is None:
                        continue
                    w, h = _sizes.get(_idx, (1920, 1080))
                    cls_indices = r.boxes.cls.cpu().numpy().astype(int).tolist()
                    res = {
                        "image_id": _idx,
                        "engine": "yolo",
                        "scores": r.boxes.conf.cpu().numpy().tolist(),
                        "labels": [r.names[c] for c in cls_indices],
                        "boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                        "width": w,
                        "height": h,
                    }
                    sub = detections_cache.setdefault(project or "default", {})
                    sub[str(_idx)] = res
                    out[str(_idx)] = res
                except Exception:
                    continue
            del _arr, _items      # 及时释放这一块的 BGR 数组（每张 1080p 约 6MB）
        _persist_detections()  # 全部块跑完一次落盘，避免逐张反复写文件
    return {"code": 200, "results": out, "count": len(out)}
@app.post("/api/dino_detect_batch")
def dino_detect_batch(
    project: str = Form("default"),
    image_ids: str = Form(...),
    text_prompt: str = Form(...),
):
    """加锁包装：整个 DINO 批量检测期间与 VLM/Clip 推理排队互斥（12G 卡不能并发用卡）"""
    with _gpu_slot("DINO 开放词检测"):
        return _dino_detect_batch_locked(project, image_ids, text_prompt)
def _dino_detect_batch_locked(
    project: str,
    image_ids: str,
    text_prompt: str,
):
    """Grounding DINO 满载批处理：一次喂多张图 + 单条提示词，FP16 加速，中英自动互译并回译。

    单张单发极慢，批量后可显著提升 GPU 占用与吞吐。按项目隔离并逐帧落盘。"""
    _log(f"[检测] DINO-BATCH project={project} ids={image_ids} prompt={text_prompt}")
    global dino_model, dino_processor
    ctx = load_project_context(project)
    # DINO 与 VLM/YOLO 不共存：先腾显存，再按需加载(FP16)
    _free_vlm()
    _free_yolo()
    if not _ensure_dino():
        return {"code": 500, "msg": "Grounding DINO 模型未就绪", "results": {}}
    ids = []
    for s in image_ids.replace(" ", "").split(","):
        if s == "":
            continue
        try:
            ids.append(int(s))
        except Exception:
            pass
    # 中→英：加载共享词典，长词优先替换，避免单字误替换（与单帧 ground_detect 一致）
    dino_dict = _load_dino_dict()
    prompt = (text_prompt or "").strip()
    dino_prompt_key = prompt  # 记录原始提示词，供前端判断缓存是否需失效
    sorted_keys = sorted(dino_dict.keys(), key=len, reverse=True)
    for zh in sorted_keys:
        prompt = prompt.replace(zh, dino_dict[zh])
    if not prompt.endswith("."):
        prompt += "."
    dino_reverse = {en: zh for zh, en in dino_dict.items()}
    candidates = []
    for img_id in ids:
        if 0 <= img_id < len(ctx["metadata"]):
            p = ctx["metadata"][img_id]["path"]
            if os.path.exists(p):
                candidates.append((img_id, p))
    if not candidates:
        return {"code": 200, "results": {}, "count": 0}
    # 并行解码，避免 CPU 串行读图成为 GPU 瓶颈
    items = _preload_rgb(
        [p for _, p in candidates], workers=8
    )  # [(rgb,w,h) | None, ...]
    rgb_imgs = []
    target_sizes = []  # 用「原图」尺寸，把模型输出框映射回原分辨率
    valid = []  # (image_id, orig_w, orig_h)
    for k, it in enumerate(items):
        if it is None:
            continue
        arr, w, h = it
        rgb_imgs.append(arr)
        target_sizes.append((h, w))
        valid.append((candidates[k][0], w, h))
    if not rgb_imgs:
        return {"code": 200, "results": {}, "count": 0}
    # 下采样以控显存：DINO 是大模型，全分辨率多图极易 OOM（12G 卡尤甚）。
    # 保持长宽比缩到最长边 <= DINO_MAX_SIDE 再喂；框经 target_sizes 映射回原图。
    def _downscale(arr, max_side=1280):
        hh, ww = arr.shape[:2]
        if max(hh, ww) <= max_side:
            return arr
        im = Image.fromarray(arr)
        im.thumbnail((max_side, max_side))
        return np.asarray(im)
    DINO_MAX_SIDE = 1280
    feed_imgs = [_downscale(a, DINO_MAX_SIDE) for a in rgb_imgs]
    def _post_save(pr, img_id, w, h):
        labels = [dino_reverse.get(lbl, lbl) for lbl in pr["labels"]]
        res = {
            "image_id": img_id,
            "engine": "dino",
            "prompt": dino_prompt_key,  # 记录所用提示词，供前端判断缓存是否需失效
            "scores": pr["scores"].cpu().numpy().tolist(),
            "labels": labels,
            "boxes": pr["boxes"].cpu().numpy().tolist(),
            "width": w,
            "height": h,
        }
        save_detection_record(project, img_id, res)
        return res
    def _run_batch(batch_idx):
        """跑一个子批，返回写入的 out；OOM 返回 None（调用方回退单张）。"""
        chunk_rgb = [feed_imgs[i] for i in batch_idx]
        chunk_ts = [target_sizes[i] for i in batch_idx]
        inputs = dino_processor(
            images=chunk_rgb, text=[prompt] * len(chunk_rgb), return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = dino_model(**inputs)
            else:
                outputs = dino_model(**inputs)
        prs = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.20,
            text_threshold=0.20,
            target_sizes=chunk_ts,
        )
        for pos, i in enumerate(batch_idx):
            img_id, w, h = valid[i]
            out[str(img_id)] = _post_save(prs[pos], img_id, w, h)
    def _run_single(i):
        """单张兜底；仍 OOM/出错则跳过该张。"""
        try:
            _run_batch([i])
            return True
        except Exception:
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return False
    out = {}
    DINO_SUB_BATCH = 4  # 12G 卡谨慎：先试 4，OOM 自动逐张退
    try:
        for c0 in range(0, len(feed_imgs), DINO_SUB_BATCH):
            batch_idx = list(range(c0, min(c0 + DINO_SUB_BATCH, len(feed_imgs))))
            try:
                _run_batch(batch_idx)
            except torch.cuda.OutOfMemoryError:
                if DEVICE == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                # 整批放不下 → 逐张退，实在放不下的跳过（不会整批失败）
                for i in batch_idx:
                    _run_single(i)
    except Exception as e:
        print(f"[!] DINO 批量推理异常: {e}")
        return {"code": 500, "msg": f"DINO 批量推理异常: {e}", "results": {}}
    return {"code": 200, "results": out, "count": len(out)}
@app.get("/api/get_all_detections")
def get_all_detections(project: str = Query("default")):
    """返回某项目下已持久化的所有目标检测结果，供前端加载页面时全量同步。

    仅返回当前项目，保证项目隔离。"""
    global detections_cache
    proj = project or "default"
    sub = detections_cache.get(proj, {}) if isinstance(detections_cache, dict) else {}
    return {"code": 200, "project": proj, "detections": sub}
# ----------------- [可选扩展] Qwen2-VL 综合场景判定 -----------------
vlm_model = None
vlm_processor = None
vlm_loaded_name = ""  # 实际加载成功的 VLM 模型名（用于确认是否 3B / 回退 2B）
def _bnb_4bit():
    """4bit 量化配置（12G 卡跑 7B 必须）。视觉塔不量化：VLM 的视觉编码器对量化很敏感，
    量化后远处小目标(行人/非机动车)识别会明显变差。不可用时返回 None(退回 fp16)。"""
    if DEVICE != "cuda":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception as _e:
        print(f"[!] BitsAndBytesConfig 不可用({_e})，改用 fp16")
        return None
    base = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    try:
        return BitsAndBytesConfig(**base, llm_int8_skip_modules=["visual", "vision_model"])
    except TypeError:
        return BitsAndBytesConfig(**base)
def _try_load_vlm(name: str, dtype):
    """按服务器 transformers 版本兼容加载 Qwen2.5-VL；失败自动回退 Qwen2-VL-2B。"""
    from transformers import AutoProcessor
    # 1) 优先 Qwen2.5-VL（需要 transformers>=4.57）
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
        model = _ModelCls.from_pretrained(name, torch_dtype=dtype, device_map="auto")
        try:
            proc = AutoProcessor.from_pretrained(
                name, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
            )
        except Exception:
            proc = AutoProcessor.from_pretrained(name)
        return model, proc, name
    except Exception as _e:
        print(f"[!] Qwen2.5-VL 加载失败({_e})，回退 {VLM_FALLBACK_MODEL}...")
    # 2) 回退链：大模型(默认 7B，走 4bit) → 小模型 2B(fp16)
    quant = _bnb_4bit() if VLM_4BIT else None
    free_gb = make_room_for("vlm", 10.5)
    if quant is not None and free_gb is not None and free_gb < 9.5:
        # 二次强制清理后再判断：上一轮检测 OOM 的异常栈会持有大批帧/张量引用，
        # 只靠 empty_cache 回收不了，必须 gc + ipc_collect 才能把显存真正还回去。
        import gc
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        free_gb = make_room_for("vlm", 10.5)
        print(f"[VRAM] 二次清理后可用 {free_gb:.2f} GB（gc+empty_cache+ipc_collect）", flush=True)
    for _mid, _q in ((VLM_FALLBACK_MODEL, quant), ("Qwen/Qwen2-VL-2B-Instruct", None)):
        # 7B-4bit 实测常驻约 10.5G：装不下就别硬上，直接换 2B，避免 OOM 把服务搞脏
        if _q is not None and free_gb is not None and free_gb < 9.5:
            print(f"[!] 可用显存 {free_gb:.2f}G 不足以加载 {_mid}(4bit 约需 10.5G)，跳过改用 2B", flush=True)
            continue
        try:
            from transformers import Qwen2VLForConditionalGeneration
            kw = {"device_map": "auto"}
            if _q is not None:
                kw["quantization_config"] = _q
            else:
                kw["torch_dtype"] = dtype
            print(f"[*] 加载 VLM: {_mid} {'(4bit)' if _q is not None else '(fp16)'} ...", flush=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(_mid, **kw)
            try:
                proc = AutoProcessor.from_pretrained(
                    _mid, min_pixels=256 * 28 * 28, max_pixels=VLM_MAX_PX * 28 * 28
                )
            except Exception:
                proc = AutoProcessor.from_pretrained(_mid)
            _vram_log(f"{_mid} 就绪")
            return model, proc, _mid
        except Exception as _e2:
            print(f"[!] {_mid} 加载失败({_e2})", flush=True)
            try:  # 失败可能残留显存，清掉再试下一个
                globals()["vlm_model"] = None
                torch.cuda.empty_cache()
            except Exception:
                pass
    raise RuntimeError("VLM 候选模型全部加载失败")
def init_vlm_local():
    global vlm_model, vlm_processor, vlm_loaded_name
    if not LOAD_MODELS:
        raise RuntimeError("VLM 未启用：请将 AD_LOAD_MODELS 置为 1 后再调用")
    if vlm_model is None:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        print(f"[*] 正在加载 VLM 场景判定模型: {VLM_MODEL_NAME} ...")
        vlm_model, vlm_processor, used = _try_load_vlm(VLM_MODEL_NAME, dtype)
        vlm_loaded_name = used
        vlm_model.eval()
        _log(f"[VLM] 模型加载完成: {used}")
        # 自证规模：used 只是"请求的模型名"，这里打印真实加载对象的配置，
        # 便于一眼确认到底跑的是 7B 还是被降级成 2B（7B hidden=3584，2B hidden=1536）
        try:
            _cfg = getattr(vlm_model, "config", None)
            _lay = getattr(_cfg, "num_hidden_layers", "?")
            _np = sum(p.numel() for p in vlm_model.parameters())
            _size = "hidden=%s layers=%s 参数≈%.2fB" % (getattr(_cfg, "hidden_size", "?"), _lay, _np / 1e9)
            _log("[VLM] 实际规模: %s (config=%s)" % (_size, getattr(_cfg, "_name_or_path", "?")))
            print("[VLM] 实际规模: " + _size, flush=True)
        except Exception:
            pass
        print(f"[✓] VLM 场景判定引擎就绪 ({used})")
@app.post("/api/vlm_analyze")
def vlm_analyze_scene(project: str = Form("default"), image_id: int = Form(...)):
    """加锁包装：单帧 VLM 判定与批量任务/Clip 判定排队互斥"""
    with _gpu_slot("单帧 VLM 判定"):
        return _vlm_analyze_scene_locked(project, image_id)
def _vlm_analyze_scene_locked(project: str, image_id: int):
    """

    第五层 VLM 综合场景判定：对指定图片进行多模态理解，并自动生成驾驶场景标签

    """
    global vlm_model, vlm_processor, vlm_loaded_name
    ctx = load_project_context(project)
    _log(
        f"[VLM] 场景判定 model={vlm_loaded_name or '(未加载)'} project={project} image_id={image_id}"
    )
    if not (0 <= image_id < len(ctx["metadata"])):
        return {"code": 400, "msg": "图片ID不存在"}
    # DINO/YOLO 不与 VLM 共存：加载 VLM 前先把它们释放，腾出 12G 显存
    _free_dino()
    _free_yolo()
    # 确保 VLM 已加载
    if vlm_model is None:
        try:
            init_vlm_local()
        except Exception as e:
            return {"code": 500, "msg": f"VLM 模型加载失败: {str(e)}"}
    img_path = ctx["metadata"][image_id]["path"]
    try:
        image = Image.open(img_path).convert("RGB")
        # 构造多模态对话 Prompt（图片以占位符标记，实际图片通过 images= 参数传入 processor）
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "你是一名专业的自动驾驶场景分析专家。请仔细观察这张车载前视摄像头画面，只输出场景标签词，不要任何解释、编号或多余标点。\n"
                        "请尽量从以下维度判断：\n"
                        "1) 天气与光照：白天/夜晚/黄昏/雨天/雪天/雾天/阴天/强逆光\n"
                        "2) 道路与环境：高速公路/城市道路/乡村/隧道/高架桥/匝道/施工区\n"
                        "3) 路面状况：干燥/湿滑/积水/结冰/破损\n"
                        "4) 关键目标：前方大货车/小轿车/行人/锥桶/水马/护栏/红绿灯/限速牌/路灯\n"
                        "5) 风险与驾驶状态：近距离跟车/变道/拥堵/危险/畅通\n"
                        "综合输出 3~6 个词，用中文逗号分隔。例如：夜晚,高速公路,干燥,前方大货车,近距离跟车",
                    },
                ],
            }
        ]
        text_prompt = vlm_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm_processor(
            text=[text_prompt], images=[image], return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            generated_ids = vlm_model.generate(
                **inputs, max_new_tokens=80, do_sample=False
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # 解析 VLM 返回的标签字符串并自动写入该图片的标签库
        tags = [
            t.strip() for t in output_text.replace("，", ",").split(",") if t.strip()
        ]
        item = ctx["metadata"][image_id]
        if "userTags" not in item or item["userTags"] is None:
            item["userTags"] = []
        for t in tags:
            tag = f"AI:{t}"  # 标记为 VLM 自动生成的标签
            if tag not in item["userTags"]:
                item["userTags"].append(tag)
        save_project_context(ctx)
        return {
            "code": 200,
            "image_id": image_id,
            "model": vlm_loaded_name,
            "vlm_raw_output": output_text,
            "tags": item["userTags"],
        }
    except Exception as e:
        return {"code": 500, "msg": f"VLM 推理异常: {str(e)}"}
# ============================================================
# ===== V2: 结构化 VLM 输出 + AI Pipeline + 审核中心 =====
# ============================================================
# ---- 结构化 JSON 输出 Prompt（PRD 56：VLM 必须输出 Structured JSON）----
_SCENE_CAT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scene_categories.json"
)
def _load_scene_categories():
    """场景关注分类：统一从本体(scene.json)读取；本体没有时回退旧 scene_categories.json。
    以前直接读那个文件，而该文件只在服务器上有、本地缺失 -> 换机部署会静默退化成空列表。"""
    try:
        from ontology import scene_categories as _sc
        cats = _sc()
        if cats:
            return cats
    except Exception:
        pass
    try:
        import json as _j
        return _j.load(open(_SCENE_CAT_FILE, encoding="utf-8")).get(
            "scene_categories", []
        )
    except Exception:
        return []
def _scene_prompt_enum():
    parts = []
    for c in _load_scene_categories():
        parts.append("%s(%s)" % (c.get("category", ""), "|".join(c.get("details", []))))
    parts.append("普通道路(一般公路/街区/高速主线,不属于上述任何地点)")
    parts.append("无法判断(画面不清或不确定)")
    return ", ".join(x for x in parts if x)
def _vlm_structured_prompt() -> str:
    """帧级结构化提示词：枚举全部取自统一本体(scene.json)，维度名与 Clip 链路一致。
    road = 道路形态(直路/弯道/十字路口…)，scene = 地点场景(城市道路/高速道路/隧道…)——
    原实现把地点塞进了 road，导致帧级与 Clip 级无法合并统计。
    枚举值必须列在值位置之外：早期把枚举写进 ["…"] 里，模型会把整串枚举原样抄回来当标签。"""
    from ontology import dimensions as _dim_defs
    _D = _dim_defs()

    def _v(dim):
        return "/".join((_D.get(dim) or {}).get("values") or [])

    return (
        "你是一名专业的自动驾驶场景分析专家。请仔细观察这张车载前视摄像头画面，"
        "严格按照下面的 JSON Schema 输出，不要输出任何其他文字、解释或代码块标记。\n"
        "{\n"
        '  "time": ["<从枚举选>"],\n'
        '  "weather": ["<从枚举选>"],\n'
        '  "road": ["<从枚举选>"],\n'
        '  "road_surface": ["<从枚举选>"],\n'
        '  "objects": ["<从枚举选>"],\n'
        '  "events": ["<从枚举选>"],\n'
        '  "risk": ["<从枚举选>"],\n'
        '  "traffic_sign": ["<从枚举选>"],\n'
        '  "scene": ["<从枚举选>"]\n'
        "}\n"
        "可用枚举值：\n"
        "time: " + _v("time") + "\n"
        "weather: " + _v("weather") + "\n"
        "road: " + _v("road") + "\n"
        "road_surface: " + _v("road_surface") + "\n"
        "objects: " + _v("objects") + "\n"
        "events: " + _v("events") + "\n"
        "risk: " + _v("risk") + "\n"
        "traffic_sign: " + _v("traffic_sign") + "\n"
        "scene: " + _v("scene") + "\n"
        "每个维度只输出最匹配的 1 个值（objects/events 可为空数组，events 最多 2 个），必须使用中文，"
        "尖括号 <> 内是占位说明，必须替换为你实际判断出的枚举值，禁止原样照抄枚举清单。\n"
        "⚠️ 必须完整输出全部 9 个字段。注意区分：road 是道路形态(直路/弯道/十字路口…)，"
        "scene 是地点场景(城市道路/高速道路/隧道…)，两者含义不同，都要填。\n"
        + _event_hint_block()
    )
# 自由文本兜底词典：关键词 -> (维度, 标准Tag)
_VLM_KEYWORD_MAP = [
    # time
    ("白天", "time", "白天"),
    ("夜晚", "time", "夜晚"),
    ("夜间", "time", "夜晚"),
    ("黄昏", "time", "黄昏"),
    ("黎明", "time", "黎明"),
    # weather
    ("晴天", "weather", "晴天"),
    ("阴天", "weather", "阴天"),
    ("多云", "weather", "阴天"),
    ("雨天", "weather", "雨天"),
    ("下雨", "weather", "雨天"),
    ("雪天", "weather", "雪天"),
    ("雾天", "weather", "雾天"),
    ("有雾", "weather", "雾天"),
    ("强逆光", "weather", "其他"),
    # road
    # 地点类 -> scene（统一后这些属于地点场景，不再塞进 road）
    ("高速", "scene", "高速道路"),
    ("高架", "scene", "高架"),
    ("城市道路", "scene", "城市道路"),
    ("市区", "scene", "城市道路"),
    ("乡村", "scene", "乡村道路"),
    ("隧道", "scene", "隧道"),
    ("涵洞", "scene", "隧道"),
    ("匝道", "scene", "匝道"),
    ("施工区", "scene", "施工区域"),
    ("停车场", "scene", "停车场"),
    ("加油站", "scene", "加油站"),
    ("服务区", "scene", "服务区"),
    ("收费站", "scene", "收费站"),
    # 道路形态 -> road
    ("直路", "road", "直路"),
    ("弯道", "road", "弯道"),
    ("坡道", "road", "弯道"),
    ("十字路口", "road", "十字路口"),
    ("环岛", "road", "环岛"),
    # objects
    ("行人", "objects", "行人"),
    ("人", "objects", "行人"),
    ("两轮车", "objects", "两轮车"),
    ("三轮车", "objects", "三轮车"),
    ("自行车", "objects", "两轮车"),
    ("摩托车", "objects", "两轮车"),
    ("大货车", "objects", "大车"),
    ("货车", "objects", "大车"),
    ("卡车", "objects", "大车"),
    ("小轿车", "objects", "小车"),
    ("轿车", "objects", "小车"),
    ("小车", "objects", "小车"),
    ("SUV", "objects", "小车"),
    ("公交车", "objects", "大车"),
    ("大巴", "objects", "大车"),
    ("锥桶", "objects", "交通设施"),
    ("水马", "objects", "交通设施"),
    ("护栏", "objects", "交通设施"),
    ("红绿灯", "objects", "交通设施"),
    ("限速牌", "objects", "交通设施"),
    ("路灯", "objects", "交通设施"),
    ("路障", "objects", "交通设施"),
    # events
    ("横穿", "events", "行人横穿"),
    ("加塞", "events", "车辆加塞"),
    ("变道", "events", "无"),
    ("异常停车", "events", "异常停车"),
    ("违停", "events", "异常停车"),
    ("施工", "events", "道路施工"),
    ("拥堵", "events", "车辆拥堵"),
    ("堵车", "events", "车辆拥堵"),
    ("干燥", "road_surface", "干燥"),
    ("湿滑", "road_surface", "湿滑"),
    ("积水", "road_surface", "积水"),
    ("结冰", "road_surface", "结冰"),
    ("破损", "road_surface", "破损"),
    ("路面", "road_surface", "未指定"),
    ("密集", "events", "车辆密集"),
    ("跟车", "events", "无"),
    ("危险", "risk", "高风险"),
    # risk
    ("低风险", "risk", "低风险"),
    ("中风险", "risk", "中风险"),
    ("高风险", "risk", "高风险"),
]
_VLM_VOCAB = None
def _vlm_vocab():
    """统一本体词表（懒加载缓存）：解析 VLM 输出时用来丢弃词表外的值。"""
    global _VLM_VOCAB
    if _VLM_VOCAB is None:
        try:
            from ontology import valid_tags
            _VLM_VOCAB = valid_tags()
        except Exception:
            _VLM_VOCAB = set()
    return _VLM_VOCAB
_TAG_SEP = re.compile(r"[|/、,，;；\s]+")
_NEG_PREFIX = ("无", "没有", "未见", "未发现", "不存在")
def _looks_like_tag(v: str) -> bool:
    """判断一个词表外的取值像不像"一个标签"（而不是一句描述或否定回答）。
    像标签的保留下来当候选（如 公交车切出）；描述句、以及"无车辆切入"这类
    回答"没有发生"的否定词直接丢弃（events 为空数组才是表达"无事件"的方式）。"""
    if v.startswith(_NEG_PREFIX):
        return False
    return 2 <= len(v) <= 10 and not re.search(r"[。！？，,.、；;：:（）()【】\[\]\"'']", v)
def _clean_vlm_tags(dim: str, vals, meta_out: dict = None) -> list:
    """清洗 VLM 某维度的原始取值（字符串或数组）为标签列表。

    模型偶尔会把提示词里的枚举清单整串抄回来（"A|B|C|…|unknown"）——
    单个取值拆开后能命中 3 个以上枚举值即判定为枚举回声，整条丢弃，
    否则一帧会挂上十几个标签、明细表整列被撑开。
    其余值按分隔符拆开，再做三件事：
      1) 命中取值别名 -> 归一到正式值（公交车切出 若已归并到 车辆切出）；
      2) 在本体词表内 -> 保留；
      3) 词表外但像个标签 -> 也保留（它是模型真实看到的东西，会进候选池，
         由「融合语义搜索 → 标签维护」决定升为正式标签还是归并到已有标签）。"""
    if isinstance(vals, str):
        vals = [vals]
    if not isinstance(vals, list):
        return []
    # 事件允许输出成对象（{"type":..., "evidence":[...]}）：取出 type，其余字段不参与打标
    _flat = []
    for _v0 in vals:
        if isinstance(_v0, dict):
            _flat.append(_v0.get("type") or _v0.get("tag") or "")
            # 证据/置信度必须留下：这是"标签凭什么判出来"的唯一依据（提示词本来就要求
            # events 给出 F1→FN 的跨帧证据）。以前只取 type，证据全丢，前端只能看到结论。
            if meta_out is not None:
                _raw = str(_v0.get("type") or _v0.get("tag") or "").strip()
                _rec = {}
                if _v0.get("confidence") is not None:
                    _rec["confidence"] = _v0.get("confidence")
                if _v0.get("evidence"):
                    _rec["evidence"] = _v0.get("evidence")
                if _raw and _rec:
                    try:
                        from ontology import canonical_tag as _ct0
                        _raw = _ct0(dim, _raw)
                    except Exception:
                        pass
                    meta_out.setdefault(dim, {})[_raw] = _rec
        else:
            _flat.append(_v0)
    vals = _flat
    from ontology import canonical_tag, values_of
    vocab = _vlm_vocab()
    # 按【维度】校验，而不是用全量词表：否则"高速道路"(场景维的值)会被塞进"道路"维。
    try:
        _dim_vals = set(values_of(dim)) | {"unknown"}
    except Exception:
        _dim_vals = set(vocab)
    out = []
    for v in vals:
        parts = [p.strip() for p in _TAG_SEP.split(str(v or "").strip()) if p.strip()]
        valid = [p for p in parts if p in vocab]
        if len(valid) >= 3 and len(parts) >= 4:
            continue        # 枚举清单回声：不是对画面的判断，整条丢弃
        for p in parts:
            # 模型常把"无法判断"写成中文的"未知/不确定"，枚举里是 unknown：统一归一，
            # 否则会被当成新词收进标签（实测交通标识维度 91 帧全变成"未知"这种非枚举值）
            if p in ("未知", "不确定", "无法判断", "不详", "无", "N/A", "n/a"):
                p = "unknown"
            p = canonical_tag(dim, p)
            if p in vocab and p not in _dim_vals:
                continue    # 本体里别的维度的合法值，放错维度了 -> 丢弃
            if p not in vocab and not _looks_like_tag(p):
                continue    # 是句子/描述，不是标签
            if p not in out:
                out.append(p)
    cap = 2 if dim == "events" else (4 if dim == "objects" else 3)
    out = out[:cap]
    # 只保留最终留下的标签对应的证据（被清洗掉的标签不留证据，避免脏数据）
    if meta_out is not None and meta_out.get(dim):
        _keep = {k: v for k, v in meta_out[dim].items() if k in out}
        if _keep:
            meta_out[dim] = _keep
        else:
            meta_out.pop(dim, None)
    return out
def _parse_vlm_structured(text: str, meta_out: dict = None) -> dict:
    """解析 VLM 输出为结构化维度标签 {dim: [tag,...]}。

    优先解析 JSON；失败则用关键词词典兜底归类。"""
    text = (text or "").strip()
    # 尝试提取 JSON 对象
    import json as _json
    # 去掉可能的 ```json 围栏
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    # 找到第一个 { 和最后一个 }
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s >= 0 and e > s:
        try:
            obj = _json.loads(cleaned[s : e + 1])
            if isinstance(obj, dict):
                result = {}
                for dim in (
                    "time",
                    "weather",
                    "road",
                    "road_surface",
                    "objects",
                    "events",
                    "risk",
                    "traffic_sign",
                    "scene",
                ):
                    tags = _clean_vlm_tags(dim, obj.get(dim), meta_out)
                    if tags:
                        result[dim] = tags
                if result:
                    return result
        except Exception:
            pass
    # 兜底：关键词归类
    result = {}
    for kw, dim, tag in _VLM_KEYWORD_MAP:
        if kw in text:
            tags = result.setdefault(dim, [])
            if tag not in tags and tag != "无":
                tags.append(tag)
    return result
def _find_db_asset(project: str, image_id: int):
    """直接按 vector_id 从数据库定位 Asset（不依赖 JSON 兼容层）"""
    from db_service import get_project, Asset
    db = get_db_session()
    try:
        proj = get_project(db, project)
        if not proj:
            db.close()
            return None, None, None
        asset = (
            db.query(Asset)
            .filter(
                Asset.project_id == proj.id,
                Asset.vector_id == int(image_id),
            )
            .first()
        )
        if asset is None:
            # 兜底：按 asset_id 前缀匹配（某些场景 image_id 传 asset_id）
            asset = (
                db.query(Asset)
                .filter(
                    Asset.project_id == proj.id,
                    Asset.asset_id == str(image_id),
                )
                .first()
            )
        if asset is None:
            db.close()
            return None, None, None
        return asset, None, db
    except Exception:
        try:
            db.close()
        except Exception:
            pass
        return None, None, None
def _ensure_db_project(db, project_name: str):
    """DB 层项目兜底：同名项目目录存在时，自动补建 DB Project 记录（避免双轨 404）。

    目录不存在的项目名【不】补建：那是已删除（或从未存在）的项目，补建会把它在
    任何带 project 参数的轮询里凭空复活。返回 None 表示项目不存在，调用方按
    "数据库未就绪/项目不存在"处理（各调用点已判空）。"""
    from db_service import get_project, create_project
    proj = get_project(db, project_name)
    if proj is None:
        if not _project_exists_on_disk(project_name):
            _log(f"[DB] 项目 [{project_name}] 目录不存在，不补建记录（已删除项目不复活）")
            return None
        proj = create_project(db, project_name)
        db.commit()  # 持久化, 避免轮询每 3s 重复补建
        _log(f"[DB] 自动补建项目记录: {project_name}")
    return proj
def _asset_to_legacy_item(asset, project: str) -> dict:
    """把 DB Asset 序列化为前端兼容的记录结构"""
    fn = os.path.basename(asset.image_path or "")
    return {
        "id": asset.vector_id,
        "asset_id": asset.asset_id,
        "filename": fn,
        "path": asset.image_path,
        "url": f"/api/image/{project}/{asset.vector_id}",
        "status": asset.status.value if asset.status else "SOURCE",
        "decision_status": (
            asset.decision_status.value if asset.decision_status else None
        ),
        "decision_reason": asset.decision_reason,
        "ai_score": (asset.asset_metadata or {}).get("ai_score"),
        "decision_score": asset.decision_score,
        "final_result": asset.final_result,
        "ai_tags": asset.ai_tags or {},
        "human_tags": asset.human_tags or {},
        "final_tags": asset.final_tags or {},
        "detections": asset.detections or {},
        "frame_index": asset.frame_index,
        "timestamp": asset.timestamp,
        "source_type": asset.source_type.value if asset.source_type else "image",
    }
# ---- VLM 单图预测公共函数（单帧/批量共用；模型互斥按需加载一次）----
def _vlm_vram(tag: str):
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        print(
            f"[VLM][VRAM] {tag}: {free/1e9:.2f} GB free / {total/1e9:.2f} GB",
            flush=True,
        )
    except Exception:
        pass
def _seg_chunks(ordered_ids, seg_size: int) -> list:
    """把某个视频里按时间排好序的帧号切成 seg_size 一段；末尾不足半段的零头并进前一段。

    否则 61 帧会切成 30+30+1 —— 那个 1 帧的"段"会单独出一份标签，
    搜索里也会冒出一条"单帧结果"（用户实测反馈过）。"""
    ids = list(ordered_ids)
    if not ids:
        return []
    chunks = [ids[i:i + seg_size] for i in range(0, len(ids), seg_size)]
    if len(chunks) >= 2 and len(chunks[-1]) < max(1, seg_size // 2):
        _tail = chunks.pop()          # 先弹出，再并到新的最后一段（反过来会索引越界）
        chunks[-1] = chunks[-1] + _tail
    return chunks
def _clip_name_parts(path_or_name: str):
    """Clip 链路抽帧的文件名 → 分段线索：v<N>_<毫秒>_<序号>.jpg。

    返回 (段标识 v<N>, 毫秒, 序号)；认不出来返回 None。"""
    _fn = os.path.basename((path_or_name or "").replace("\\", "/"))
    _m = re.match(r"^(v\d+)_(\d+)_(\d+)\.jpg$", _fn, re.I)
    if not _m:
        return None
    return _m.group(1), int(_m.group(2)), int(_m.group(3))
def _split_clip_runs(rows):
    """rows: [(段标识, 毫秒, 序号, 载荷)] → [((段标识, 第几段), [载荷, ...]), ...]

    同一 v<N> 内按 (毫秒, 序号) 排序，序号一旦回退就切开成新的一段视频 —— v<N> 是每个
    抽帧任务内部各自从 v1 编的，跨任务会重名，混在一起会把两段不同视频拼成一个 Clip。
    片段联合打标与融合语义搜索共用这里，两处的分段口径不会再跑偏。"""
    buckets = {}
    for tag, ms, seq, payload in rows:
        buckets.setdefault(tag, []).append((ms, seq, payload))
    out = []
    for tag in sorted(buckets.keys()):
        run, prev = [], None
        for ms, seq, payload in sorted(buckets[tag]):
            if prev is not None and seq <= prev and run:
                out.append(((tag, len(out)), run))
                run = []
            prev = seq
            run.append(payload)
        if run:
            out.append(((tag, len(out)), run))
    return out
def _group_frames_joint(db, project_id: int, image_ids, seg_size: int = 30, pick: int = 6) -> list:
    """按视频把帧切成 seg_size 帧的"片段"，每片段均匀挑 pick 帧送去 VLM 联合推理。

    返回 [{"sample_ids": 送模型的 pick 帧, "member_ids": 整段 seg_size 帧}, ...]
    —— 一份标签覆盖整个片段，并写回该片段的全部帧（这就是"这一片段发生了什么"）。
    分段/抽样口径与 Clip 链路统一：30 帧为一段（默认），段内 np.linspace 均匀取 pick 帧。
    没有 video_source 的帧（Clip 链路抽出的图）按 v<N>_<毫秒>_<序号>.jpg 恢复成段，
    实在认不出来的才各自成段走单帧判定。"""
    from models import Asset
    amap = {a.vector_id: a for a in db.query(Asset).filter(Asset.project_id == project_id).all()}
    by_video, order = {}, []
    _clip_pending = []  # [(v<N>, 毫秒, 序号, (毫秒, 序号, 帧id))]，没有 video_source 的帧先攒着
    for iid in image_ids:
        a = amap.get(iid)
        vs = ((a.asset_metadata or {}).get("video_source") or {}) if a is not None else {}
        vp = vs.get("video_path") if isinstance(vs, dict) else None
        if vp:
            key = vp
            sort_key = (vs.get("timestamp") or 0, vs.get("frame_index") or 0)
        else:
            # Clip 链路抽的帧没有 video_source。若一帧一段就等于退化成逐帧 VLM
            # （实测 CLIP测试：8455 帧 -> 8455 次单帧调用，整个任务从约 10 分钟涨到两小时）。
            _parts = _clip_name_parts((a.image_path if a is not None else "") or "")
            if _parts:
                _clip_pending.append(
                    (_parts[0], _parts[1], _parts[2], (_parts[1], _parts[2], iid)))
                continue
            key = "__single__%s" % iid
            sort_key = (0, 0)
        if key not in by_video:
            by_video[key] = []
            order.append(key)
        by_video[key].append((sort_key[0], sort_key[1], iid))
    # Clip 帧按文件名恢复成段（与融合语义搜索共用同一套口径，避免两边跑偏）
    for _ck, _run in _split_clip_runs(_clip_pending):
        _key = "__clip__%s_%d" % _ck
        by_video[_key] = list(_run)
        order.append(_key)
    groups = []
    for key in order:
        items = by_video[key]
        items.sort(key=lambda x: (x[0], x[1]))
        ids = [x[2] for x in items]
        if key.startswith("__single__"):
            groups.extend([{"sample_ids": [x], "member_ids": [x]} for x in ids])
            continue
        for seg in _seg_chunks(ids, seg_size):
            s0 = 0
            if len(seg) <= pick:
                picks = list(seg)
            else:
                idx = np.linspace(0, len(seg) - 1, pick).round().astype(int).tolist()
                seen, picks = set(), []
                for k in idx:
                    if k not in seen:
                        seen.add(k)
                        picks.append(seg[k])
            groups.append({"sample_ids": picks, "member_ids": list(seg)})
    return groups

def _event_hint_block() -> str:
    """易混事件判据（真源在本体 scene.json 的 event_hints；帧级与联合提示词共用）。"""
    try:
        from ontology import event_hints_text
        t = event_hints_text()
    except Exception:
        t = ""
    return ("易混事件判据（必须按此区分，禁止凭习惯用词）：\n" + t + "\n") if t else ""


def _vlm_prompt_multi(n: int) -> str:
    """多帧联合提示词：帧逐编号 + 强制跨帧证据，让模型真的按序列推理。

    与帧级单图提示词的区别不只是"多给几张图"：这里明确要求
      · 按时间顺序编号（Frame 1…N），模型据此判断变化；
      · events 必须给出 F1→中间帧→FN 的三帧证据；
      · 6 帧间位置没变化的静止目标，禁止报横穿/切入等动态事件；
    否则模型容易退化成"分别看每帧、再各报一遍"。"""
    from ontology import dimensions as _dim_defs
    dims = _dim_defs()

    def _v(dim):
        return "/".join((dims.get(dim) or {}).get("values") or [])

    seq = " → ".join("Frame %d" % k for k in range(1, n + 1))
    return (
        f"你是自动驾驶场景分析专家。以下 {n} 张图片是同一段车载前视视频按时间顺序抽取的帧，"
        f"依次为 {seq}（注意：不是相邻帧，相邻两帧之间隔了若干原始帧）。\n"
        f"请把这 {n} 帧当成一段连续过程来整体判断，只输出【一份】JSON，"
        "不要分帧输出、不要逐帧重复字段、不要任何解释或代码块标记：\n"
        "{\n"
        '  "time": ["<从枚举选>"],\n'
        '  "weather": ["<从枚举选>"],\n'
        '  "road": ["<从枚举选>"],\n'
        '  "road_surface": ["<从枚举选>"],\n'
        '  "objects": [{"type": "<从枚举选>", "state": ["<从枚举选>"], "confidence": <0.0~1.0真实数值>}],\n'
        '  "events": [{"type": "<从枚举选>", "confidence": <0.0~1.0真实数值>, '
        '"evidence": ["F1: <初始位置>", "F4: <中间变化>", "F6: <最终状态>"]}],\n'
        '  "risk": ["<从枚举选>"],\n'
        '  "traffic_sign": ["<从枚举选>"],\n'
        '  "scene": ["<从枚举选>"]\n'
        "}\n"
        "可用枚举值：\n"
        f"time: {_v('time')}\n"
        f"weather: {_v('weather')}\n"
        f"road: {_v('road')}\n"
        f"road_surface: {_v('road_surface')}\n"
        f"objects.type: {_v('objects')}\n"
        f"events.type: {_v('events')}\n"
        f"risk: {_v('risk')}\n"
        f"traffic_sign: {_v('traffic_sign')}\n"
        f"scene: {_v('scene')}\n"
        "强制规则：\n"
        "1. 判断依据是【帧间的变化】，不是某一帧的孤立画面；scene/road/weather 等全段一致的维度按整体判断；\n"
        "2. 静态目标不要报动态事件：某个行人/车辆虽然出现，但多帧间位置没有明显变化，"
        "不要报行人横穿/车辆切入等动态事件；\n"
        "3. events 的 evidence 必须写出三帧证据（F1 初始位置 → 中间帧变化 → 最后一帧最终状态），"
        "格式如 \"F1: 行人位于画面左侧路缘\"；\n"
        "4. 每个维度只输出最匹配的 1 个值（objects/events 可为空数组，events 最多 2 个）；"
        "确实无法确认的维度填 [\"unknown\"]，禁止臆测；\n"
        "5. 注意区分：road 是道路形态(直路/弯道/十字路口…)，scene 是地点场景(城市道路/高速道路/隧道…)，两者都要填；\n"
        "6. 必须完整输出全部 9 个字段；禁止输出枚举之外的泛化词（如「复杂城市交通」）；"
        "尖括号 <> 内只是占位说明，必须替换为你实际判断出的枚举值，禁止原样照抄枚举清单；\n"
        "8. confidence 必须是 0.0~1.0 的真实数值，反映确信程度，不得统一填 0.0 或照抄示例；\n"
        "9. 同一段里的用词要一致，事件判据见下：\n"
        + _event_hint_block()
    )
def _vlm_predict_text_multi(images) -> str:
    """多帧联合 VLM 判定：N 张图一次送入，综合推理出一份标签。

    与逐帧判定的区别：逐帧各自打标、再在视频层面做并集，会把整段视频打成"什么标签都有"；
    这里一组帧只产出一份总结标签，组内共享。"""
    global vlm_model, vlm_processor
    with _gpu_slot("多帧联合 VLM 判定"):
        _vlm_vram("before_load")
        _free_dino()
        _free_yolo()
        if vlm_model is None:
            try:
                init_vlm_local()
            except Exception as e:
                _log(f"[VLM] 多帧联合: 模型加载失败 {e}")
                return ""
        if vlm_model is None:
            return ""
        try:
            # 必须先缩图：多图一次送进去，原图(1920x1080)的视觉 token 会把 12G 显存撑爆
            # （实测 5 张原图 -> CUDA OOM: Tried to allocate 6.18 GiB）。Clip 链路也是这么做的。
            _thumb = int(os.environ.get("AD_VLM_JOINT_THUMB", "512"))
            _small = []
            for _im in images:
                _im2 = _im.copy()
                _im2.thumbnail((_thumb, _thumb))
                _small.append(_im2)
            content = [{"type": "image"} for _ in _small]
            content.append({"type": "text", "text": _vlm_prompt_multi(len(_small))})
            text_prompt = vlm_processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = vlm_processor(text=[text_prompt], images=_small, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                generated_ids = vlm_model.generate(**inputs, max_new_tokens=220, do_sample=False)
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return vlm_processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0] or ""
        except Exception as e:
            _log(f"[VLM] 多帧联合推理异常: {e}")
            return ""
def _vlm_predict_text(image) -> str:
    """帧级 VLM 单图判定（GPU 串行闸门：忙时排队等待，不与 Clip/YOLO/DINO 并发）"""
    with _gpu_slot("帧级 VLM 判定"):
        return _vlm_predict_text_locked(image)
def _vlm_predict_text_locked(image) -> str:
    """确保 VLM 已加载并返回单图结构化预测文本。返回 "" 表示失败。"""
    global vlm_model, vlm_processor
    _vlm_vram("before_load")
    # DINO/YOLO 不与 VLM 共存：加载 VLM 前先释放，腾出 12G 显存
    _free_dino()
    _free_yolo()
    if vlm_model is None:
        try:
            init_vlm_local()
        except Exception as e:
            _log(f"[VLM] 模型加载失败: {e}")
            return ""
    _vlm_vram("after_load")
    if vlm_model is None:
        return ""
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _vlm_structured_prompt()},
                ],
            }
        ]
        text_prompt = vlm_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm_processor(
            text=[text_prompt], images=[image], return_tensors="pt"
        ).to(DEVICE)
        _vlm_vram("after_input")
        with torch.inference_mode():
            _vlm_vram("before_gen")
            generated_ids = vlm_model.generate(
                **inputs, max_new_tokens=160, do_sample=False
            )
            _vlm_vram("after_gen")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text or ""
    except Exception as e:
        _log(f"[VLM] 推理异常: {e}")
        return ""
# ---- 依据证据组装 Decision Engine 输入（单帧/批量共用）----
def _build_decision_evidence(asset) -> dict:
    """从 asset 现有 AI 标签与检测结果组装决策证据。"""
    ai_tags = asset.ai_tags or {}
    evidence = {
        # VLM(Qwen2-VL)为文本生成, 无真实概率——不给假置信度; 真实置信仅来自 YOLO/DINO 检测
        "vlm_tags_ok": bool(ai_tags),
        "vlm_confidence": None,
        "yolo_labels": [],
        "yolo_max_conf": 0.0,
        "dino_labels": [],
        "dino_max_conf": 0.0,
        "siglip_score": 0.0,
        "final_tags": ai_tags,
    }
    dets = asset.detections or {}
    yolo_dets = dets.get("yolo") or []
    dino_dets = dets.get("dino") or []
    if yolo_dets:
        evidence["yolo_labels"] = [d.get("label") for d in yolo_dets]
        evidence["yolo_max_conf"] = max(
            [float(d.get("confidence", 0)) for d in yolo_dets], default=0.0
        )
    if dino_dets:
        evidence["dino_labels"] = [d.get("label") for d in dino_dets]
        evidence["dino_max_conf"] = max(
            [float(d.get("confidence", 0)) for d in dino_dets], default=0.0
        )
    # 互相校验: VLM 目标类事件 vs 检测无对应目标 -> 疑似编造, 强制人工
    try:
        evidence["vlm_event_conflict"] = _event_detection_conflict(ai_tags, dets)
    except Exception:
        evidence["vlm_event_conflict"] = False
    return evidence
# ---- 将解析出的维度标签合并进 asset.ai_tags（带来源追踪）----
def _merge_dim_tags_into_asset(
    asset, dim_tags: dict, model_name: str, confidence: float = None, tag_meta: dict = None
):
    ai_tags = dict(asset.ai_tags or {})
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for dim, tags in (dim_tags or {}).items():
        # 重跑要点：VLM 判出来的维必须"以本次为准"。原来是纯并集，旧标签只增不减——
        # 于是重跑后每帧仍带着上一轮的旧标签，"段内 30 帧共享一份标签"看着就永远不生效。
        # 人工复核(source=HUMAN)与其它来源的标签保留，只替换 VLM 自己判出来的。
        existing = [t for t in (ai_tags.get(dim) or [])
                    if isinstance(t, dict) and (t.get("source") or "") != "VLM"]
        names = {t.get("tag") for t in existing}
        for tag_name in tags:
            if tag_name not in names:
                _rec = ((tag_meta or {}).get(dim) or {}).get(tag_name) or {}
                _item = {
                    "tag": tag_name,
                    "source": "VLM",
                    "confidence": _rec.get("confidence", confidence),
                    "model": model_name,
                    "version": "v1",
                    "created_at": ts,
                }
                if _rec.get("evidence"):
                    _item["evidence"] = _rec["evidence"]   # 模型给的帧级证据，前端可核对
                existing.append(_item)
                names.add(tag_name)
        ai_tags[dim] = existing
    asset.ai_tags = ai_tags
    return ai_tags
@app.post("/api/asset_analyze_full")
def asset_analyze_full(project: str = Form("default"), image_id: int = Form(...)):
    """完整单帧 AI Pipeline（V2）：

    VLM 结构化判定 -> 解析成维度标签 -> 写入 ai_tags -> Decision Engine 自动决策 -> 落库

    返回 {code, item(结构化 asset), decision}

    """
    _log(f"[V2] asset_analyze_full project={project} image_id={image_id}")
    asset, _ctx, db = _find_db_asset(project, image_id)
    if db is None:
        return {"code": 500, "msg": "数据库未就绪"}
    try:
        if asset is None:
            return {"code": 404, "msg": "图片ID不存在或未入库"}
        img_path = asset.image_path or ""
        if not img_path or not os.path.exists(img_path):
            return {"code": 404, "msg": f"图片文件不存在: {img_path}"}
        # 1) VLM 推理（共享函数：负责加载互斥与释放）
        output_text = _vlm_predict_text(Image.open(img_path).convert("RGB"))
        if not output_text:
            return {"code": 500, "msg": "VLM 推理失败或模型不可用"}
        # 2) 解析结构化输出（顺带收下 evidence/confidence，便于事后核对标签真实性）
        _tag_meta = {}
        dim_tags = _parse_vlm_structured(output_text, _tag_meta)
        # 3) 合并维度标签进 ai_tags（带来源追踪 VLM/model/version）
        model_name = vlm_loaded_name or VLM_MODEL_NAME
        ai_tags = _merge_dim_tags_into_asset(asset, dim_tags, model_name, tag_meta=_tag_meta)
        # 4) Decision Engine 评估
        from db_service import update_asset_decision
        evidence = _build_decision_evidence(asset)
        decision = create_decision_engine(
            _backend_decision_preset if _backend_decision_preset else "balanced"
        ).decide(evidence)
        # 5) 写决策结果 + 更新状态
        from db_service import AssetStatus
        update_asset_decision(
            db,
            asset.asset_id,
            decision.status.value,
            decision.reason,
            (
                round(decision.score, 4)
                if isinstance(decision.score, (int, float))
                else None
            ),
        )
        db.commit()
        db.refresh(asset)
        item = _asset_to_legacy_item(asset, project)
        return {
            "code": 200,
            "vlm_raw_output": output_text,
            "parsed_dim_tags": dim_tags,
            "decision": {
                "status": decision.status.value,
                "reason": decision.reason,
                "score": (
                    round(decision.score, 4)
                    if isinstance(decision.score, (int, float))
                    else None
                ),
            },
            "item": item,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 500, "msg": f"AI Pipeline 异常: {str(e)}"}
    finally:
        try:
            db.close()
        except Exception:
            pass
@app.post("/api/decision/evaluate")
def decision_evaluate(project: str = Form("default"), image_id: int = Form(...)):
    """仅对已推理 asset 重新跑一次决策（不改 AI 标签）"""
    asset, ctx, db = _find_db_asset(project, image_id)
    if db is None:
        return {"code": 500, "msg": "数据库未就绪"}
    try:
        if asset is None:
            return {"code": 404, "msg": "图片ID不存在或未入库"}
        from db_service import update_asset_decision
        ai_tags = asset.ai_tags or {}
        evidence = {
            # VLM(Qwen2-VL)为文本生成, 无真实概率——不给假置信度; 真实置信仅来自 YOLO/DINO 检测
            "vlm_tags_ok": bool(ai_tags),
            "vlm_confidence": None,
            "yolo_labels": [],
            "yolo_max_conf": 0.0,
            "dino_labels": [],
            "dino_max_conf": 0.0,
            "siglip_score": 0.0,
            "final_tags": ai_tags,
        }
        dets = asset.detections or {}
        yolo_dets = dets.get("yolo") or []
        dino_dets = dets.get("dino") or []
        if yolo_dets:
            evidence["yolo_labels"] = [d.get("label") for d in yolo_dets]
            evidence["yolo_max_conf"] = max(
                [float(d.get("confidence", 0)) for d in yolo_dets], default=0.0
            )
        if dino_dets:
            evidence["dino_labels"] = [d.get("label") for d in dino_dets]
            evidence["dino_max_conf"] = max(
                [float(d.get("confidence", 0)) for d in dino_dets], default=0.0
            )
        decision = create_decision_engine("balanced").decide(evidence)
        update_asset_decision(
            db,
            asset.asset_id,
            decision.status.value,
            decision.reason,
            (
                round(decision.score, 4)
                if isinstance(decision.score, (int, float))
                else None
            ),
        )
        db.commit()
        return {
            "code": 200,
            "decision": {
                "status": decision.status.value,
                "reason": decision.reason,
                "score": (
                    round(decision.score, 4)
                    if isinstance(decision.score, (int, float))
                    else None
                ),
            },
        }
    finally:
        db.close()
@app.get("/api/decision/config")
def get_decision_config():
    """查看决策引擎配置（当前用 balanced 预设 + 覆盖项）"""
    from decision_engine import PRESET_CONFIGS
    return {
        "code": 200,
        "preset": "balanced",
        "config": PRESET_CONFIGS.get("balanced", {}),
    }
@app.post("/api/decision/config")
def update_decision_config(preset: str = Form(None), detect_first: int = Form(0)):
    """切换决策引擎预设（strict/lenient/safety_first/balanced）"""
    from decision_engine import PRESET_CONFIGS
    preset = (preset or "balanced") if isinstance(preset, str) else "balanced"
    if preset not in PRESET_CONFIGS:
        return {
            "code": 400,
            "msg": f"未知预设: {preset}，可选 {list(PRESET_CONFIGS.keys())}",
        }
    global _backend_decision_preset
    _backend_decision_preset = preset
    _log(f"[决策] 切换预设: {preset}")
    return {"code": 200, "msg": f"已切换决策预设: {preset}", "preset": preset}
_backend_decision_preset = "balanced"
@app.post("/api/review/submit")
def review_submit(
    project: str = Form("default"),
    image_id: int = Form(...),
    action: str = Form(None),  # confirm / modify / add / delete / reject / skip
    dim: str = Form(None),  # 修改/新增/删除时使用的维度
    tag: str = Form(None),  # 标签名
    comments: str = Form(None),
):
    """人工审核提交（PRD 20-23）：

    confirm: AI Tag -> Final Tag (认可)

    modify/add/delete: 修改 human_tags 后重算 final_tags

    reject: AI 结果 -> FILTERED

    人工结果与 AI 不一致 -> 自动记 Hard Case

    """
    from db_service import (
        update_asset_status,
        update_asset_tags,
        update_asset_review,
        update_asset_final_result,
        create_review_record,
        create_hard_case,
        AssetStatus,
        ReviewAction,
    )
    asset, ctx, db = _find_db_asset(project, image_id)
    if db is None:
        return {"code": 500, "msg": "数据库未就绪"}
    try:
        if asset is None:
            return {"code": 404, "msg": "图片ID不存在或未入库"}
        ai_tags = dict(asset.ai_tags or {})
        human_tags = dict(asset.human_tags or {})
        # 归一化可选表单参数（兼容 FastAPI Form 与直接函数调用）
        _FASTAPI_MARKERS = ("Form", "File", "Query", "Body", "Cookie", "Header")
        def _norm_str(v):
            if v is None:
                return ""
            if not isinstance(v, str):
                # FastAPI 参数标记对象（直接调用时落入默认值）视为未传
                if type(v).__name__ in _FASTAPI_MARKERS:
                    return ""
                return str(v)
            return v
        action = _norm_str(action).lower()
        dim = _norm_str(dim)
        tag = _norm_str(tag)
        comments = _norm_str(comments)
        if action == "confirm":
            # AI -> Final，human_tags 不变（表示认可）
            final_tags = dict(ai_tags)
            asset.final_tags = final_tags
            asset.human_tags = human_tags  # 保持
            update_asset_status(db, asset.asset_id, AssetStatus.APPROVED)
            asset.final_result = {
                "status": "APPROVED",
                "decided_by": "human",
                "action": "confirm",
                "comments": comments,
            }
        elif action in ("modify", "add", "delete"):
            if not dim or not tag:
                return {"code": 400, "msg": f"action={action} 需要 dim 和 tag 参数"}
            # 更新 human_tags
            dim_tags = [t for t in (human_tags.get(dim) or []) if isinstance(t, dict)]
            if action == "delete":
                dim_tags = [t for t in dim_tags if t.get("tag") != tag]
            else:  # modify / add: 人工标签覆盖
                dim_tags = [t for t in dim_tags if t.get("tag") != tag]
                dim_tags.append(
                    {
                        "tag": tag,
                        "source": "Human",
                        "confidence": 1.0,
                        "model": "human",
                        "version": "v1",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )
            if dim_tags:
                human_tags[dim] = dim_tags
            elif dim in human_tags:
                del human_tags[dim]
            # 重算 final：human 优先覆盖同名 AI，其余沿用 AI
            final_tags = {}
            all_dims = set(ai_tags.keys()) | set(human_tags.keys())
            for d in all_dims:
                ht = {
                    t.get("tag")
                    for t in (human_tags.get(d) or [])
                    if isinstance(t, dict)
                }
                merged = [t for t in (human_tags.get(d) or []) if isinstance(t, dict)]
                for t in ai_tags.get(d) or []:
                    if isinstance(t, dict) and t.get("tag") not in ht:
                        merged.append(t)
                if merged:
                    final_tags[d] = merged
            asset.final_tags = final_tags
            asset.human_tags = human_tags
            update_asset_status(db, asset.asset_id, AssetStatus.APPROVED)
            asset.final_result = {
                "status": "APPROVED",
                "decided_by": "human",
                "action": action,
                "comments": comments,
            }
            # AI != Human -> Hard Case
            ai_flat = {
                t.get("tag")
                for dim_list in ai_tags.values()
                for t in dim_list
                if isinstance(t, dict)
            }
            human_flat = {
                t.get("tag")
                for dim_list in human_tags.values()
                for t in dim_list
                if isinstance(t, dict)
            }
            if ai_flat != human_flat:
                create_hard_case(
                    db,
                    asset.project_id,
                    asset.asset_id,
                    ai_tags,
                    human_tags,
                    final_tags,
                    {
                        "ai_only": sorted(ai_flat - human_flat),
                        "human_only": sorted(human_flat - ai_flat),
                    },
                )
        elif action == "reject":
            asset.final_tags = {}
            asset.final_result = {
                "status": "FILTERED",
                "decided_by": "human",
                "action": "reject",
                "comments": comments,
            }
            update_asset_status(db, asset.asset_id, AssetStatus.FILTERED)
            update_asset_final_result(db, asset.asset_id, {"status": "FILTERED"})
        elif action == "skip":
            pass  # 不做任何修改
        else:
            return {
                "code": 400,
                "msg": f"未知 action: {action}，可选 confirm/modify/add/delete/reject/skip",
            }
        # 记录审核流水
        create_review_record(
            db,
            asset.project_id,
            asset.id,
            reviewer="annotator",
            action=getattr(ReviewAction, action.upper(), ReviewAction.SKIP),
            ai_tags_snapshot=ai_tags,
            human_tags=asset.human_tags or {},
            final_tags=asset.final_tags or {},
            comments=comments,
        )
        db.commit()
        db.refresh(asset)
        item = _asset_to_legacy_item(asset, project)
        return {"code": 200, "msg": f"审核完成: {action}", "item": item}
    finally:
        db.close()
@app.get("/api/review/queue")
def review_queue(project: str = Query("default"), page: int = 1, size: int = 20):
    """获取审核队列：decision=REVIEW 或 status=REVIEW 的资产"""
    from db_service import Asset, AssetStatus, DecisionStatus
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪", "items": [], "total": 0}
        q = db.query(Asset).filter(Asset.project_id == proj.id)
        q = q.filter(
            (Asset.status == AssetStatus.REVIEW)
            | (Asset.decision_status == DecisionStatus.REVIEW)
        )
        total = q.count()
        assets = (
            q.order_by(Asset.updated_at.asc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )
        items = [_asset_to_legacy_item(a, project) for a in assets]
        return {"code": 200, "total": total, "page": page, "size": size, "items": items}
    finally:
        db.close()
@app.get("/api/asset_info")
def asset_info(project: str = Query("default"), image_id: int = Query(...)):
    """查看单帧完整结构化信息（含血缘/标签/决策/检测）"""
    asset, ctx, db = _find_db_asset(project, image_id)
    if db is None:
        return {"code": 500, "msg": "数据库未就绪"}
    try:
        if asset is None:
            return {"code": 404, "msg": "图片ID不存在或未入库"}
        # 血缘信息
        lineage = {}
        if asset.source:
            src = asset.source
            lineage = {
                "source_id": src.source_id,
                "source_type": src.source_type.value if src.source_type else None,
                "source_root": src.source_root,
                "relative_path": src.relative_path,
                "directory_chain": src.directory_chain or [],
                "file_name": src.file_name,
                "file_hash": src.file_hash,
                "phash": src.phash,
            }
        item = _asset_to_legacy_item(asset, project)
        item["lineage"] = lineage
        return {"code": 200, "item": item}
    finally:
        db.close()
# ============================================================
# ===== Pipeline 任务系统（PRD 39-43, 53） =====
# 批量 AI Pipeline 走 DB Jobs 表：可持久化 / 断点续跑 / 增量 / 失败重试
# ============================================================
# 后台任务内存信号（cancel/pause；单进程 workers=1 场景）
_ai_pipeline_cancel = set()  # 存 Job.id
_ai_pipeline_pause = set()
_pipeline_lock = threading.Lock()
def _pipeline_cancel_signals():
    with _pipeline_lock:
        return set(_ai_pipeline_cancel), set(_ai_pipeline_pause)
_YOLO_LABEL2OBJ = {
    "person": "行人",
    "bicycle": "两轮车",
    "motorcycle": "两轮车",
    "rickshaw": "三轮车",
    "car": "小车",
    "truck": "大车",
    "bus": "大车",
    "van": "大车",
    "trailer": "大车",
    "traffic light": "交通设施",
    "stop sign": "交通设施",
    "cone": "交通设施",
    "barrier": "交通设施",
}
_EVENT_NEED = {
    "行人横穿": {"person"},
    "行人密集": {"person"},
    "非机动车横穿": {"bicycle", "motorcycle"},
    "车辆加塞": {"car", "truck", "bus"},
    "车辆拥堵": {"car", "truck", "bus"},
    "车辆密集": {"car", "truck", "bus"},
    "异常停车": {"car", "truck", "bus"},
}
# 以上为内置兜底；若本体(scene.json)里定义了 events_need 则以其为准（统一真源）
try:
    from ontology import events_need as _onto_events_need
    _NEED = _onto_events_need()
    if _NEED:
        _EVENT_NEED = {k: set(v) for k, v in _NEED.items()}
except Exception:
    pass
# DINO 标签还原：词表值(英文) -> 中文键；再兜一层标识关键词（模型偶尔输出 ##walk/lane 这类碎词）
_DINO_EN2CN = {}
try:
    _dd = json.load(open(os.path.join(PROJECT_DIR, "dino_dict.json"), encoding="utf-8"))
    for _k, _v in _dd.items():
        _DINO_EN2CN.setdefault(str(_v).strip().lower(), _k)
except Exception:
    _dd = {}
_DINO_SIGN_KW = [
    ("crosswalk", "人行横道"), ("#walk", "人行横道"), ("zebra", "人行横道"),
    ("stop line", "停止线"), ("traffic light", "交通信号灯"), ("signal light", "交通信号灯"),
    ("no entry", "禁止通行"), ("no parking", "禁止停车"), ("stop sign", "停车让行"),
    ("speed limit", "限速"),
]
def _dino_label_to_cn(lb: str):
    """把 DINO 返回的标签还原成中文标准词；还原不出来返回 None（宁可丢，也不把英文串写进标签）。

    DINO 用逗号拼接多类提示词时，会把多个类名连成一串（如 "traffic light traffic light street"），
    所以先整体匹配，再按关键词兜底，最后才考虑单值精确匹配。"""
    t = (lb or "").strip().lower()
    if not t:
        return None
    for kw, cn in _DINO_SIGN_KW:          # 标识类关键词优先（要进 traffic_sign 维度）
        if kw in t:
            return cn
    hit = _DINO_EN2CN.get(t)              # 词表值精确匹配 -> 中文键
    if hit:
        return hit
    if t in _SIGN_TAGS:                   # 直接就是标准词
        return lb.strip()
    # 逐个英文词试：取最长能匹配上的词表项
    toks = [x for x in re.split(r"[^a-z]+", t) if x]
    for n in (3, 2, 1):
        for i in range(0, max(1, len(toks) - n + 1)):
            cand = " ".join(toks[i:i + n])
            if cand in _DINO_EN2CN:
                return _DINO_EN2CN[cand]
    if re.search(r"[a-zA-Z]", t):         # 仍是英文 -> 丢弃
        return None
    return lb.strip()
# 交通标识信号：DINO 词表里属于标线/标志的词（命中后进 traffic_sign 维度，不进 objects）
try:
    from ontology import values_of as _values_of
    _SIGN_TAGS = set(_values_of("traffic_sign")) - {"unknown"}
except Exception:
    _SIGN_TAGS = set()
# DINO 词表里的键名比本体取值更口语（"限速标志" vs 取值"限速"），先归一再判定
_SIGN_ALIAS = {
    "斑马线": "人行横道", "限速标志": "限速", "左转标志": "左转", "右转标志": "右转",
    "掉头标志": "掉头", "直行标志": "直行", "禁止通行标志": "禁止通行",
    "禁止停车标志": "禁止停车", "停车让行标志": "停车让行",
    "注意行人标志": "注意行人", "施工标志": "施工",
}
def _detections_to_objects(detections: dict):
    """把真实检测结果(YOLO/DINO)映射为 ai_tags.objects 标签列表(带真实置信, 不虚报)。"""
    out = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    seen = set()
    cands = []
    for eng in ("yolo", "dino"):
        for d in (detections or {}).get(eng) or []:
            try:
                cf = float(d.get("confidence") or 0)
            except Exception:
                cf = 0.0
            if cf < 0.3:
                continue
            lb = str(d.get("label") or "").strip()
            if not lb:
                continue
            # 交通标识信号：DINO 提示词里的标线/标志词命中后，应进 traffic_sign 维度，
            # 不能混到 objects 里（否则"人行横道"会变成一个目标类）
            # DINO 常返回英文词表值或把多个类名拼成一串，先尽量还原成中文标准词
            _lb2 = lb
            if eng == "dino":
                _lb2 = _dino_label_to_cn(lb)
                if _lb2 is None:
                    continue        # 还原不出来（英文/乱码）就别进标签，避免污染
            _lb_sign = _SIGN_ALIAS.get(_lb2, _lb2)
            if _lb_sign in _SIGN_TAGS:
                cands.append((cf, _lb_sign, "sign"))
                continue
            lb = _lb2
            tag = _YOLO_LABEL2OBJ.get(lb)
            if not tag:
                # DINO 属开放词(提示词通常已是中文，可原样保留)；YOLO 是固定 COCO 类名，
                # 没映射到的一律归为"其他目标"——绝不能把英文类名写进标签
                # (曾出现 suitcase/handbag/train 混入目标标签)
                tag = lb if eng == "dino" else "其他目标"
            cands.append((cf, tag, eng))
    # 标识与目标分开截断：交通标识常是低置信（0.2~0.5），一起排序会被目标挤掉前 6 名之外
    cands.sort(key=lambda x: -x[0])
    _signs = [c for c in cands if c[2] == "sign" and c[0] >= 0.25][:2]
    _objs = [c for c in cands if c[2] != "sign"][:4]
    for cf, tag, eng in _signs:
        if tag in seen:
            continue
        seen.add(tag)
        out.append({"tag": tag, "source": "DINO", "confidence": cf, "dim": "traffic_sign"})
    for cf, tag, eng in _objs:
        if tag in seen:
            continue
        seen.add(tag)
        out.append(
            {
                "tag": tag,
                "source": eng.upper(),
                "confidence": cf,
                "model": eng.upper(),
                "version": "det",
                "created_at": ts,
            }
        )
    return out
AUTO_PASS_SCORE = 70  # AI 综合分(0-100)阈值: >=70 自动 AI_PASS
# 抽样复核比例：自动通过的帧里抽多少转人工复核（0=全自动不抽样）。可用 AD_REVIEW_SAMPLE_RATE 覆盖
REVIEW_SAMPLE_RATE = float(os.environ.get("AD_REVIEW_SAMPLE_RATE", "0.05"))
def _sample_review(asset_id, rate=None) -> bool:
    """按 asset_id 哈希判定是否抽样复核：整体比例≈rate，且同一 id 结果稳定可复现"""
    import hashlib
    r = REVIEW_SAMPLE_RATE if rate is None else rate
    if not r or r <= 0:
        return False
    if r >= 1:
        return True
    try:
        h = int(hashlib.md5(str(asset_id).encode("utf-8")).hexdigest()[:8], 16)
    except Exception:
        return False
    return (h % 10000) < int(r * 10000)
def _ai_score_of(asset, det_best=None):
    """AI 综合分 0-100 = 60%*检测最高置信 + 40%*VLM风险档(低1.0/中0.5/高0)。
    无真实概率不虚报: det_best 取 yolo/dino 检测最高框置信(真实测量); VLM 部分用风险档量化(规则基线, 非模型概率)。"""
    if det_best is None:
        confs = []
        for _eng in ("yolo", "dino"):
            for _d in (asset.detections or {}).get(_eng) or []:
                try:
                    confs.append(float(_d.get("confidence") or 0))
                except Exception:
                    pass
        det_best = max(confs) if confs else 0.0
    risk = 0.6  # 默认中档保守
    try:
        ai = asset.ai_tags or {}
        for t in ai.get("risk") or []:
            r = t.get("tag") if isinstance(t, dict) else t
            if r == "低风险":
                risk = 1.0
            elif r == "高风险":
                risk = 0.0
            else:
                risk = 0.5
    except Exception:
        pass
    return round(60.0 * float(det_best) + 40.0 * risk)
def _event_detection_conflict(ai_tags: dict, detections: dict) -> bool:
    """互相校验: VLM 报了目标类事件, 但 YOLO/DINO 没检到对应目标 -> 疑似编造, 交人工。"""
    labels = set()
    for eng in ("yolo", "dino"):
        for d in (detections or {}).get(eng) or []:
            lb = str(d.get("label") or "").strip()
            if lb:
                labels.add(_YOLO_LABEL2OBJ.get(lb, lb))
    for t in (ai_tags or {}).get("events") or []:
        ev = t.get("tag") if isinstance(t, dict) else t
        need = _EVENT_NEED.get(str(ev))
        if not need:
            continue
        # 关键：labels 是映射后的中文名，need 里写的是英文原始类名，必须同样映射后再比。
        # 否则交集恒为空 -> 只要报了事件就必然判“冲突”，该规则实际从未通过（曾导致
        # 62 帧里 59 帧被强制转人工）。
        need_cn = {_YOLO_LABEL2OBJ.get(x, x) for x in need}
        if not (labels & need_cn):
            return True
    return False
def _joint_infer_group(project: str, project_id, group: dict) -> dict:
    """对一段做一次多帧联合推理（段内取 AD_JOINT_PICK 帧送模型，覆盖该段全部帧）。

    返回 ({帧id: dim_tags}, 证据表 {dim: {tag: {confidence, evidence}}})；
    取不到图/推理失败/无有效标签时返回 ({}, {})，该段的帧由调用方回退为逐帧 VLM。"""
    from models import Asset   # worker 里的同名导入是函数局部的，抽成独立函数必须自己导入
    out = {}
    try:
        _dbp2 = get_db_session()
        try:
            _paths = []
            for _iid in (group.get("sample_ids") or []):
                _a2 = _dbp2.query(Asset).filter(
                    Asset.project_id == project_id, Asset.vector_id == _iid).first()
                _p2 = (_a2.image_path if _a2 else "") or ""
                if _p2 and os.path.exists(_p2):
                    _paths.append((_iid, _p2))
        finally:
            _dbp2.close()
        if not _paths:
            return out, {}
        if len(_paths) == 1:
            _txt = _vlm_predict_text(Image.open(_paths[0][1]).convert("RGB"))
        else:
            _txt = _vlm_predict_text_multi(
                [Image.open(_p).convert("RGB") for _i, _p in _paths])
        _meta2 = {}
        _dt = _parse_vlm_structured(_txt, _meta2) if _txt else {}
        # 模型没按 JSON 输出时会退化成关键词兜底（维度少、可能出现枚举外的词），
        # 把原始输出记下来便于调提示词
        if _txt and ("{" not in _txt or "}" not in _txt):
            _log(f"[Pipeline] 联合打标非JSON输出(已走关键词兜底): {str(_txt)[:220]}")
        if not _dt:
            _log(f"[Pipeline] 片段联合打标无有效标签(组内 {len(_paths)} 帧)，原始输出: "
                 + (str(_txt)[:300] if _txt else "<空>"))
        _members = list(group.get("member_ids") or [])
        for _iid in _members:
            out[_iid] = dict(_dt)
    except Exception as _e:
        _log(f"[Pipeline] 联合打标某组失败(跳过该组): {_e}")
    return out, _meta2
def _ai_batch_worker(
    job_pk: int,
    project: str,
    image_ids: List[int],
    preset: str = "balanced",
    detect_first: bool = False,
    dino_enhance: bool = False,
    joint_mode: bool = True,
    clip_size: int = 0,
):
    """批量 AI Pipeline(检测融合版)：detect_first=1 时先 YOLO 全批检测(真实目标证据)，

    再 VLM 只判 场景/天气/道路/路面/事件/风险；objects 维以检测为准；

    VLM 目标类事件与检测冲突 -> 强制人工 REVIEW。支持 cancel/pause。"""
    from db_service import update_job_status, get_job
    from models import Asset
    db = get_db_session()
    try:
        total = len(image_ids)
        processed_ok = 0
        skipped = []
        failed = []
        dec_engine = create_decision_engine(preset or "balanced")
        from models import JobStatus as _JS
        _job_set(job_pk, status=_JS.RUNNING, progress=0.0, current_stage="任务启动")
        # ============ 阶段 1: YOLO 全批检测(真实目标置信) ============
        yolo_hits = {}  # image_id -> [ {label, confidence, box} ]
        detect_ok = not detect_first   # 未开启检测时不算失败
        if detect_first and image_ids and DEVICE == "cuda":
            try:
                from db_service import get_project as _gp
                # 只对"还没检测过"的帧做检测：中断续跑时不必把已检测过的帧重新检测一遍
                _dbp = get_db_session()
                try:
                    _pj = _gp(_dbp, project)
                    _want = set(image_ids)
                    detect_todo = [
                        a.vector_id
                        for a in _dbp.query(Asset).filter(Asset.project_id == _pj.id).all()
                        if a.vector_id in _want and not (a.detections or {}).get("yolo")
                    ] if _pj is not None else list(image_ids)
                finally:
                    _dbp.close()
                _log(
                    f"[Pipeline] 阶段1 YOLO检测 job={job_pk} 待检测={len(detect_todo)}/{total}（已检测的跳过）"
                )
                # 分批检测 + 每批更新进度：整批一次性检测时进度条会长时间停在 0/N，
                # 看起来像卡死；分批也让"终止/暂停"在检测阶段就能生效（原来只在 VLM 阶段检查信号）。
                _CHUNK = max(50, int(os.environ.get("AD_DETECT_CHUNK", "500")))
                _done = 0
                for _s in range(0, len(detect_todo), _CHUNK):
                    _cs, _ps = _pipeline_cancel_signals()
                    if job_pk in _cs:
                        from models import JobStatus as _JSc
                        _job_set(job_pk, status=_JSc.CANCELLED, progress=round(_done / max(1, total) * 100, 1),
                                 current_stage=f"已中止于检测阶段 {_done}/{total}（剩余 {len(detect_todo) - _done} 帧未检测）")
                        _log(f"[Pipeline] 检测阶段被终止 job={job_pk} 已完成 {_done}/{len(detect_todo)}")
                        return
                    while job_pk in _ps:
                        time.sleep(1)
                        _cs, _ps = _pipeline_cancel_signals()
                        if job_pk in _cs:
                            from models import JobStatus as _JSc2
                            _job_set(job_pk, status=_JSc2.CANCELLED, progress=round(_done / max(1, total) * 100, 1),
                                     current_stage=f"已中止于检测阶段 {_done}/{total}")
                            return
                    _part = detect_todo[_s:_s + _CHUNK]
                    _update_job_progress(job_pk, _done, total,
                                         f"YOLO 检测 {_done}/{len(detect_todo)}（本批 {len(_part)} 帧）")
                    # 复用现有批量检测核心(同进程直接调用端点函数, 返回 {code,results,count})
                    resp = yolo_detect_batch(
                        project=project,
                        image_ids=",".join(str(x) for x in _part),
                        batch_size=16,
                        fp16="1",
                    )
                    # 关键：检测阶段失败必须中止任务。否则 0 命中会被后面的"无检测目标"逻辑
                    # 当成"这批帧都没目标"，把全部帧静默标成 AUTO_PASS 且不产生任何标签
                    # （曾导致 2034 帧空通过、分析中心无数据可统计）。
                    if not isinstance(resp, dict) or resp.get("code") != 200:
                        raise RuntimeError("YOLO 检测失败: " + str((resp or {}).get("msg") or "无返回"))
                    detect_ok = True
                    _done += len(_part)
                    # 每批就把结果落库：中断后已检测过的帧不会白做
                    _dbw = get_db_session()
                    try:
                        for id_str, r in (resp.get("results") or {}).items():
                            try:
                                iid = int(id_str)
                            except Exception:
                                continue
                            dets = []
                            for lb, cf, bx in zip(r.get("labels") or [], r.get("scores") or [], r.get("boxes") or []):
                                dets.append({"label": lb, "confidence": float(cf), "box": bx})
                            if dets:
                                yolo_hits[iid] = dets
                            a = _dbw.query(Asset).filter(Asset.project_id == _pj.id, Asset.vector_id == iid).first()
                            if a is None:
                                continue
                            dd = dict(a.detections or {})
                            dd["yolo"] = dets
                            a.detections = dd
                        _dbw.commit()
                    finally:
                        _dbw.close()
                _update_job_progress(job_pk, total, total, f"YOLO 检测完成 {_done}/{len(detect_todo)}")
                _log(f"[Pipeline] YOLO 完成 命中 {len(yolo_hits)} 帧 / 检测 {_done} 帧")
            except Exception as e:
                # 不再"继续纯VLM"：空 hit_ids 会让所有帧走免VLM直通，等于整轮空跑
                _log(f"[Pipeline] YOLO 阶段失败，中止任务(不标记任何帧): {e}")
                raise
            finally:
                try:
                    _free_yolo()
                except Exception:
                    pass
                # 原缩进错误：empty_cache 被写在 except 里，正常路径从不执行，
                # 导致检测释放的显存不归还驱动，后面的 VLM 因显存不足降级成 2B
                if torch is not None:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        # ============ 阶段 1.5: DINO 开放词增强(可选, 检 YOLO 未覆盖目标) ============
        if dino_enhance and image_ids and DEVICE == "cuda":
            try:
                _log(f"[Pipeline] 阶段1.5 DINO开放词检测 job={job_pk} frames={total}")
                _update_job_progress(job_pk, 0, total, f"DINO 开放词检测")
                from db_service import get_project as _gp2
                _dct = _load_dino_dict()
                dino_prompt = ". ".join(list(_dct.keys())[:60]) + "."
                if not dino_prompt.strip():
                    dino_prompt = "锥桶, 水马, 护栏, 红绿灯, 限速牌, 施工区, 障碍物"
                resp3 = dino_detect_batch(
                    project=project,
                    image_ids=",".join(str(x) for x in image_ids),
                    text_prompt=dino_prompt,
                )
                hits3 = {}
                for id_str, r in (resp3.get("results") or {}).items():
                    try:
                        iid3 = int(id_str)
                    except Exception:
                        continue
                    dd3 = []
                    for lb3, cf3, bx3 in zip(
                        r.get("labels") or [],
                        r.get("scores") or [],
                        r.get("boxes") or [],
                    ):
                        dd3.append({"label": lb3, "confidence": float(cf3), "box": bx3})
                    if dd3:
                        hits3[iid3] = dd3
                if hits3:
                    db3 = get_db_session()
                    try:
                        _proj3 = _gp2(db3, project)
                        if _proj3 is not None:
                            for iid3, dd3 in hits3.items():
                                a3 = (
                                    db3.query(Asset)
                                    .filter(
                                        Asset.project_id == _proj3.id,
                                        Asset.vector_id == iid3,
                                    )
                                    .first()
                                )
                                if a3 is None:
                                    continue
                                d3 = dict(a3.detections or {})
                                d3["dino"] = dd3
                                a3.detections = d3
                        db3.commit()
                    finally:
                        db3.close()
                _log(f"[Pipeline] DINO 完成 命中 {len(hits3)} 帧")
            except Exception as e:
                _log(f"[Pipeline] DINO 阶段失败(跳过): {e}")
            finally:
                try:
                    _free_dino()
                except Exception:
                    pass
                    if torch is not None:
                        torch.cuda.empty_cache()
        # ============ 阶段 2: VLM 场景/事件 + 融合 + 决策 ============
        _log(
            f"[Pipeline] 批量 AI 阶段2(VLM+融合) job={job_pk} project={project} frames={total} detect={detect_first}"
        )
        dino_hits_local = {}
        try:
            dino_hits_local = hits3 or {}
        except Exception:
            dino_hits_local = {}
        # 检测出目标的帧(仅这些帧跑 VLM)。不能只取"本次运行检测到的"：
        # 续跑会跳过已检测的帧(见阶段1)，那时 yolo_hits 几乎为空 -> hit_ids 为空 ->
        # 所有帧都被当成"无检测目标"免VLM直通，跑一轮一帧都不打标（踩过）。
        # 这里以库里的检测记录为准，再并入本次检测结果。
        hit_ids = set(yolo_hits) | set(dino_hits_local)
        if detect_first:
            try:
                from db_service import get_project as _gp_h
                _dbh = get_db_session()
                try:
                    _pjh = _gp_h(_dbh, project)
                    if _pjh is not None:
                        _want = set(image_ids)
                        for _a in _dbh.query(Asset).filter(Asset.project_id == _pjh.id).all():
                            if _a.vector_id in _want:
                                _dd = _a.detections or {}
                                if _dd.get("yolo") or _dd.get("dino"):
                                    hit_ids.add(_a.vector_id)
                finally:
                    _dbh.close()
                _log(f"[Pipeline] 待VLM帧数={len(hit_ids)}/{len(image_ids)}（含历史检测记录）")
            except Exception as _e:
                _log(f"[Pipeline] 读取历史检测记录失败(退化为仅本次检测): {_e}")
        # ============ 阶段 2a: 片段联合打标（按视频相邻 N 帧一组，一次推理出一份标签）============
        # 逐帧打标 + 视频级并集会把整段视频打成"什么标签都有"（事件尤其明显）。
        # 这里改为：同一视频相邻的 N 帧一次送 VLM，综合出一份总结标签，组内帧共享。
        # 落地方式：段内第一帧轮到时就地推理这一段，然后走下面逐帧那同一套写库逻辑。
        # （旧实现是等所有段推理完、标签攒在内存里最后统一落库：推理阶段前端一帧结果都看不到，
        #   中途被终止/重启时几千帧的 VLM 结果全部丢失。现在每段推理完即随帧入库。）
        joint_tags = {}
        joint_meta = {}    # 段下标 -> 证据表（段内帧共享）
        joint_groups = []
        joint_member_of = {}   # 帧 -> 所属段下标
        joint_done = set()     # 已推理过的段下标
        joint_project_id = None
        # 片段级打标要覆盖整段所有帧（含无检测目标的帧），否则那些帧拿不到片段标签
        joint_targets = list(image_ids)
        if joint_mode and joint_targets:
            # 段长优先用前端「Clip帧数」，其次环境变量，默认 30（Clip 统一 30 帧）
            _SEG = max(5, int(clip_size)) if int(clip_size or 0) >= 5 else max(
                5, int(os.environ.get("AD_JOINT_SEG", "30")))
            _PICK = max(2, int(os.environ.get("AD_JOINT_PICK", "6")))
            try:
                from db_service import get_project as _gpj
                _dbg = get_db_session()
                try:
                    _pg = _gpj(_dbg, project)
                    joint_groups = _group_frames_joint(
                        _dbg, _pg.id, joint_targets, _SEG, _PICK) if _pg else []
                    joint_project_id = _pg.id if _pg else None
                finally:
                    _dbg.close()
                for _gi, _g in enumerate(joint_groups):
                    for _iid in (_g.get("member_ids") or []):
                        joint_member_of[_iid] = _gi
                _log(f"[Pipeline] 片段级联合: {len(joint_targets)} 帧 -> {len(joint_groups)} 段"
                     f"（每段≤{_SEG} 帧，段内取 {_PICK} 帧送模型；逐段推理、随帧落库）")
            except Exception as _e:
                _log(f"[Pipeline] 片段联合打标整体失败，退回逐帧模式: {_e}")
                joint_groups = []
                joint_member_of = {}
        for pos, img_id in enumerate(image_ids, 1):
            cancel_set, pause_set = _pipeline_cancel_signals()
            if job_pk in cancel_set:
                from models import JobStatus
                _job_set(
                    job_pk,
                    status=JobStatus.CANCELLED,
                    progress=round((pos - 1) / total * 100, 1),
                    current_stage=f"已中止于 {pos-1}/{total}（剩余 {total - pos + 1} 帧未处理）",
                )
                _log(f"[Pipeline] 任务被取消 job={job_pk}")
                return
            while job_pk in pause_set:
                time.sleep(1)
                cancel_set, pause_set = _pipeline_cancel_signals()
                if job_pk in cancel_set:
                    break
            if job_pk in cancel_set:
                continue
            if joint_groups:
                _gi = joint_member_of.get(img_id)
                if _gi is not None and _gi not in joint_done:
                    joint_done.add(_gi)   # 先登记再推理：同一段只推一次
                    _tags2, _meta2 = _joint_infer_group(
                        project, joint_project_id, joint_groups[_gi])
                    joint_tags.update(_tags2)
                    joint_meta[_gi] = _meta2          # 段级证据，随段共享
                    if not _tags2:
                        _log(f"[Pipeline] 段 {_gi + 1}/{len(joint_groups)} 未产出有效标签，"
                             f"该段帧回退逐帧 VLM")
            asset, _ctx, adb = _find_db_asset(project, img_id)
            if adb is None or asset is None:
                skipped.append(img_id)
                continue
            try:
                img_path = asset.image_path or ""
                if not img_path or not os.path.exists(img_path):
                    skipped.append(img_id)
                    adb.close()
                    continue
                if (detect_first and detect_ok and img_id not in hit_ids
                        and img_id not in joint_tags):
                    # 无检测目标帧: 跳过 VLM(省算力), 直接免人工通过
                    # 注意：若该帧属于某个已做片段级打标的片段，则不跳过（要用片段标签）
                    from db_service import update_asset_decision as _uad0
                    _uad0(
                        adb, asset.asset_id, "AUTO_PASS", "无检测目标(免VLM直通)", None
                    )
                    adb.commit()
                    processed_ok += 1
                    continue
                _tag_meta = {}
                if joint_mode and img_id in joint_tags:
                    # 多帧联合模式：标签来自本组的多帧联合推理结果（组内共享），不再逐帧调用
                    dim_tags = dict(joint_tags.get(img_id) or {})
                    output_text = "joint" if dim_tags else ""
                    # 段级证据（F1→FN）随段共享，直接挂到本帧
                    _tag_meta = joint_meta.get(joint_member_of.get(img_id)) or {}
                else:
                    output_text = _vlm_predict_text(Image.open(img_path).convert("RGB"))
                if not output_text:
                    failed.append(img_id)
                    continue
                else:
                    if not (joint_mode and img_id in joint_tags):
                        dim_tags = _parse_vlm_structured(output_text, _tag_meta)
                    model_name = vlm_loaded_name or VLM_MODEL_NAME
                    # VLM 的 objects 维弃用(以真实检测为准); 若帧无检测则保留 VLM objects 作兜底
                    if img_id in yolo_hits:
                        dim_tags.pop("objects", None)
                    _merge_dim_tags_into_asset(asset, dim_tags, model_name, tag_meta=_tag_meta)
                    # ---- 检测融合: objects 维 = YOLO 真实检测(映射到标准目标类别) ----
                    if img_id in yolo_hits:
                        _d2o = _detections_to_objects(asset.detections or {})
                        _objs = [o for o in _d2o if (o.get("dim") or "objects") == "objects"]
                        _signs = [o for o in _d2o if o.get("dim") == "traffic_sign"]
                        if _objs or _signs:
                            ai = dict(asset.ai_tags or {})
                            if _objs:
                                ai["objects"] = _objs
                            if _signs:
                                # 交通标识信号：与 VLM 判出的标识合并去重
                                _have = {t.get("tag") for t in (ai.get("traffic_sign") or []) if isinstance(t, dict)}
                                _merged = list(ai.get("traffic_sign") or [])
                                for _sg in _signs:
                                    if _sg["tag"] not in _have:
                                        _merged.append(_sg)
                                ai["traffic_sign"] = _merged
                            asset.ai_tags = ai
                    evidence = _build_decision_evidence(asset)
                    decision = dec_engine.decide(evidence)
                    from db_service import update_asset_decision
                # AI 综合分(60%检测置信+40%VLM风险档) -> 阈值判 PASS/REVIEW
                ai_score = _ai_score_of(asset)
                hard_review = decision.status.value == "REVIEW" and (
                    "冲突" in (decision.reason or "")
                )
                if decision.status.value == "FILTER":
                    final_st = "FILTER"
                    final_rs = decision.reason
                elif hard_review:
                    final_st = "REVIEW"
                    final_rs = decision.reason
                elif ai_score >= AUTO_PASS_SCORE:
                    final_st = "AUTO_PASS"
                    final_rs = "AI分%d>=%d 自动通过" % (ai_score, AUTO_PASS_SCORE)
                    # 抽样复核：自动通过的帧按比例抽一部分转人工。
                    # 用 asset_id 哈希取模而非纯随机 —— 同一帧每次抽中结果一致，
                    # 重跑不会换一批，便于复核结果可复现。
                    if _sample_review(asset.asset_id, REVIEW_SAMPLE_RATE):
                        final_st = "REVIEW"
                        final_rs = "抽样复核(%.0f%%)" % (REVIEW_SAMPLE_RATE * 100)
                else:
                    final_st = "REVIEW"
                    final_rs = "AI分%d<%d 需人工" % (ai_score, AUTO_PASS_SCORE)
                from db_service import update_asset_decision as _uad1
                _uad1(
                    adb,
                    asset.asset_id,
                    final_st,
                    final_rs,
                    (round(ai_score / 100.0, 4)),
                )
                try:
                    _md = dict(asset.asset_metadata or {})
                    _md["ai_score"] = ai_score
                    asset.asset_metadata = _md
                except Exception:
                    pass
                adb.commit()
                processed_ok += 1
            except Exception as e:
                failed.append({"id": img_id, "err": str(e)})
            finally:
                try:
                    adb.close()
                except Exception:
                    pass
            _update_job_progress(
                job_pk,
                pos,
                total,
                (f"片段联合打标+融合决策 {pos}/{total}"
                 if joint_groups else f"VLM+融合决策 {pos}/{total}"),
            )
        cancel_set, _ = _pipeline_cancel_signals()
        if job_pk in cancel_set:
            return
        result = {
            "processed": processed_ok,
            "skipped": len(skipped),
            "failed": len(failed),
            "skipped_ids": skipped[:100],
            "failed_detail": failed[:20],
            "detect_first": bool(detect_first),
            "dino_enhance": bool(dino_enhance),
            "detected_frames": len(yolo_hits),
        }
        if processed_ok == 0 and len(failed) > 0:
            _fail_job(
                job_pk,
                f"全部处理失败: {len(failed)} 张异常, 例如 {str(failed[:2])[:220]}",
            )
            _log(
                f"[Pipeline] 批量 AI 失败 job={job_pk} processed=0 skipped={len(skipped)} failed={len(failed)}"
            )
        else:
            if len(failed) > 0:
                result["error"] = f"{len(failed)} 张处理失败: {str(failed[:2])[:180]}"
            _finish_job(job_pk, result)
            _log(
                f"[Pipeline] 批量 AI 完成 job={job_pk} processed={processed_ok} skipped={len(skipped)} failed={len(failed)}"
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        _fail_job(job_pk, str(e))
def _job_set(
    pk,
    status=None,
    progress=None,
    current_stage=None,
    result=None,
    error=None,
    completed_at=None,
):
    """更新 DB Job 字段(供各后台 worker 汇报进度/终态)"""
    try:
        from db_service import get_job
        from models import Job
        _db = get_db_session()
        try:
            if isinstance(pk, int) or str(pk).isdigit():
                _j = _db.query(Job).filter(Job.id == int(pk)).first()
            else:
                _j = get_job(_db, pk)
            if _j is None:
                return
            if status is not None:
                _j.status = status
            if progress is not None:
                _j.progress = progress
            if current_stage is not None:
                _j.current_stage = current_stage
            if result is not None:
                _j.result = result
            if error is not None:
                _j.error = error
            if completed_at is not None:
                _j.completed_at = completed_at
            _db.commit()
        finally:
            _db.close()
    except Exception:
        pass
def _update_job_progress(pk: int, pos: int, total: int, stage: str):
    """更新 DB Job 进度(批量 worker 用)"""
    try:
        _job_set(
            pk,
            progress=round(pos / max(1, total) * 100, 1),
            current_stage="%s (%d/%d)" % (stage, pos, total),
        )
    except Exception:
        pass
def _cleanup_stale_jobs():
    try:
        from db_service import get_db_session
        from models import Job, JobStatus as _JS4
        from datetime import datetime as _dt4
        _d = get_db_session()
        n = 0
        for j in (
            _d.query(Job).filter(Job.status.in_([_JS4.RUNNING, _JS4.PENDING])).all()
        ):
            j.status = _JS4.FAILED
            j.error = "服务重启中断：任务未完成，请重发续跑(已处理帧自动跳过)"
            j.completed_at = _dt4.utcnow()
            n += 1
        _d.commit()
        _d.close()
        if n:
            print(f"[startup] 清理 {n} 条中断任务为 FAILED（可重发续跑）")
    except Exception as e:
        print(f"[startup] 清理中断任务失败: {e}")
def _fail_job(pk: int, err: str):
    from models import JobStatus
    from datetime import datetime as _dt
    _job_set(pk, status=JobStatus.FAILED, error=err, completed_at=_dt.utcnow())
def _finish_job(pk: int, result=None):
    from models import JobStatus
    from datetime import datetime as _dt
    import json as _json
    try:
        res = (
            _json.dumps(result, ensure_ascii=False)
            if isinstance(result, (dict, list))
            else (result or None)
        )
    except Exception:
        res = None
    _job_set(
        pk,
        status=JobStatus.SUCCESS,
        progress=100.0,
        current_stage="完成",
        result=res,
        completed_at=_dt.utcnow(),
    )
@app.post("/api/pipeline/run_ai_batch")
def pipeline_run_ai_batch(
    project: str = Form("default"),
    image_ids: str = Form(None),
    preset: str = Form(None),
    detect_first: int = Form(0),
    dino_enhance: int = Form(0),
    only_unprocessed: int = Form(1),
    joint_frames: int = Form(1),
    clip_size: int = Form(0),
):
    """创建批量 AI Pipeline 任务（VLM 结构化 → 维度标签 → Decision Engine 决策）。

    image_ids 逗号分隔；**留空/传 all 表示整个项目**（由后端挑帧，前端不必再把全量 id 塞过来）。
    only_unprocessed=1（默认）为增量：已有 ai_tags 的帧视为已处理直接跳过，
    这样新入库的帧不会连带把老帧重跑一遍；勾选检测时，已标但没检测记录的帧仍会补跑。

    clip_size 为 Clip 段长（帧），默认 30：前端「Clip帧数」传进来的就是它，
    这样 Clip 链路与片段联合打标用同一个段长，不会一边 50 一边 30。
    """
    from db_service import create_job
    from models import JobType, Asset, AssetStatus
    preset = preset if isinstance(preset, str) and preset else "balanced"
    ids_in = []
    for s in str(image_ids or "").replace(" ", "").split(","):
        if s.isdigit():
            ids_in.append(int(s))
    whole_project = (not ids_in) or str(image_ids or "").strip().lower() == "all"
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        rows = db.query(Asset).filter(Asset.project_id == proj.id).all()
        if not whole_project:
            # 在内存里筛：指定 id 可能上千，SQL 的 IN(...) 参数有上限
            want = set(ids_in)
            rows = [a for a in rows if a.vector_id in want]
        picked, skipped_tagged, redetect = [], 0, 0
        for a in rows:
            tagged = bool(a.ai_tags)
            has_det = bool((a.detections or {}).get("yolo"))
            if only_unprocessed and tagged and (has_det or not detect_first):
                skipped_tagged += 1
                continue
            if only_unprocessed and tagged and detect_first and not has_det:
                redetect += 1   # 已标但缺检测记录：本次补跑检测，仍计入任务
            picked.append(a.vector_id)
        ids = picked
        if not ids:
            return {
                "code": 200,
                "msg": "没有需要处理的帧：%d 帧都已跑过 AI 分析（如需重跑请取消勾选「增量」）" % skipped_tagged,
                "job_id": None,
                "skipped": skipped_tagged,
                "pending": 0,
            }
        _log(
            "[Pipeline] 选帧: 候选=%d 跳过已处理=%d 本次=%d (增量=%s 检测=%s)"
            % (len(rows), skipped_tagged, len(ids), bool(only_unprocessed), bool(detect_first))
        )
        # 创建 Job 记录
        job = create_job(
            db,
            proj.id,
            JobType.TAG,
            {
                "project": project,
                "image_ids": ids,
                "preset": preset,
                "detect_first": bool(detect_first),
                "clip_size": int(clip_size or 0),
                "pipeline": "vlm->tags->decision",
            },
        )
        job_pk = job.id
        db.commit()
    finally:
        db.close()
    # 后台线程执行（避免阻塞；uvicorn workers=1 下 threading 足够）
    t = threading.Thread(
        target=_ai_batch_worker,
        args=(job_pk, project, ids, preset, bool(detect_first), bool(dino_enhance),
              bool(joint_frames), int(clip_size or 0)),
        daemon=True,
    )
    t.start()
    _log(
        f"[Pipeline] 批量 AI 任务已创建 job_pk={job_pk} frames={len(ids)} project={project} preset={preset}"
    )
    extra = "，跳过已处理 %d 帧" % skipped_tagged if skipped_tagged else ""
    if redetect:
        extra += "（其中 %d 帧已标但补跑检测）" % redetect
    return {
        "code": 200,
        "msg": f"批量 AI Pipeline 已启动，本次 {len(ids)} 帧{extra}",
        "job_id": str(job_pk),
        "pending": len(ids),
        "skipped": skipped_tagged,
    }
def _running_tag_jobs(project: str, job_pk: int = None) -> list:
    """当前正在跑的 TAG 任务 id 列表（临时终止/暂停用）"""
    from models import Job, JobType, JobStatus as _JS
    db = get_db_session()
    try:
        if job_pk:
            return [j.id for j in db.query(Job).filter(Job.id == int(job_pk)).all()]
        q = db.query(Job).filter(
            Job.status.in_([_JS.RUNNING, _JS.PENDING]), Job.job_type == JobType.TAG
        )
        proj = _ensure_db_project(db, project)
        if proj is not None:
            q = q.filter(Job.project_id == proj.id)
        return [j.id for j in q.all()]
    finally:
        db.close()
@app.post("/api/pipeline/cancel")
def pipeline_cancel(project: str = Form("default"), job_pk: int = Form(None)):
    """临时终止正在跑的 AI 批任务。

    worker 逐帧检查取消信号，命中即把任务标为 CANCELLED 并保留进度
    （"已中止于 N/M，剩余 K 帧未处理"）；已处理的帧全部落库，
    重发时勾选「增量」即可从未处理的帧续跑，不会重跑已完成的。"""
    ids = _running_tag_jobs(project, job_pk)
    if not ids:
        return {"code": 404, "msg": "当前没有正在运行的 AI 分析任务"}
    with _pipeline_lock:
        _ai_pipeline_pause.difference_update(ids)   # 暂停中的任务也要能被终止
        _ai_pipeline_cancel.update(ids)
    _log(f"[Pipeline] 收到临时终止信号 job={ids}")
    return {
        "code": 200,
        "msg": "已发送终止信号（job %s）：当前帧处理完即停，进度保留，重发勾选「增量」可续跑" % ids,
        "jobs": ids,
    }
@app.post("/api/pipeline/pause")
def pipeline_pause(
    project: str = Form("default"), job_pk: int = Form(None), resume: int = Form(0)
):
    """暂停 / 继续正在跑的 AI 批任务（帧间等待，不丢进度、不重载模型）"""
    ids = _running_tag_jobs(project, job_pk)
    if not ids:
        return {"code": 404, "msg": "当前没有正在运行的 AI 分析任务"}
    with _pipeline_lock:
        if int(resume or 0):
            _ai_pipeline_pause.difference_update(ids)
        else:
            _ai_pipeline_pause.update(ids)
    _log(f"[Pipeline] {'继续' if int(resume or 0) else '暂停'} job={ids}")
    return {
        "code": 200,
        "msg": ("已继续 job %s" if int(resume or 0) else "已暂停 job %s（帧间生效）") % ids,
        "jobs": ids,
        "paused": not int(resume or 0),
    }
@app.get("/api/pipeline/list")
def pipeline_list(
    project: str = Query(None), job_type: str = Query(None), limit: int = 50
):
    """列出任务（可按项目/类型过滤）"""
    db = get_db_session()
    try:
        from models import Job
        q = db.query(Job)
        if project:
            proj = _ensure_db_project(db, project)
            if proj:
                q = q.filter(Job.project_id == proj.id)
            else:
                return {"code": 200, "jobs": []}
        if job_type:
            try:
                from models import JobType as JT
                q = q.filter(Job.job_type == JT(job_type.upper()))
            except Exception:
                pass
        jobs = q.order_by(Job.created_at.desc()).limit(max(1, min(limit, 200))).all()
        out = []
        from db_service import get_project_by_id as _gp_id
        _cancel_set, _pause_set = _pipeline_cancel_signals()
        for j in jobs:
            proj = _gp_id(db, j.project_id)
            out.append(
                {
                    "job_id": str(j.id),
                    "project": proj.name if proj else None,
                    "job_type": j.job_type.value if j.job_type else None,
                    "status": j.status.value if j.status else None,
                    "progress": j.progress,
                    "current_stage": j.current_stage,
                    "error": j.error,
                    "paused": j.id in _pause_set,   # 前端据此显示"已暂停/继续"
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                }
            )
        return {"code": 200, "jobs": out}
    finally:
        db.close()
@app.get("/api/pipeline/{job_pk}")
def pipeline_get(job_pk: int):
    """查询任务状态（PRD 53: GET /api/pipeline/{job_id}）"""
    db = get_db_session()
    try:
        from models import Job
        job = db.query(Job).filter(Job.id == job_pk).first()
        if not job:
            return {"code": 404, "msg": "任务不存在"}
        from db_service import get_project_by_id
        proj = get_project_by_id(db, job.project_id)
        return {
            "code": 200,
            "job_id": str(job.id),
            "project": proj.name if proj else None,
            "job_type": job.job_type.value if job.job_type else None,
            "status": job.status.value if job.status else None,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "error": job.error,
            "result": job.result or {},
            "payload": job.payload or {},
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
    finally:
        db.close()
@app.post("/api/pipeline/{job_pk}/cancel")
def pipeline_cancel(job_pk: int):
    """中止任务（PRD 53: POST /api/pipeline/{job_id}/cancel）"""
    with _pipeline_lock:
        _ai_pipeline_cancel.add(job_pk)
    _log(f"[Pipeline] 收到取消信号 job={job_pk}")
    return {"code": 200, "msg": "取消信号已发送，任务将在当前帧完成后停止"}
@app.post("/api/pipeline/{job_pk}/pause")
def pipeline_pause(job_pk: int):
    """暂停任务（处理完当前帧后暂停；PRD 53: /pause）"""
    with _pipeline_lock:
        _ai_pipeline_pause.add(job_pk)
    _log(f"[Pipeline] 收到暂停信号 job={job_pk}")
    return {"code": 200, "msg": "暂停信号已发送，任务将在当前帧后暂停"}
@app.post("/api/pipeline/{job_pk}/resume")
def pipeline_resume(job_pk: int):
    """恢复暂停任务（PRD 53: /resume）"""
    with _pipeline_lock:
        _ai_pipeline_pause.discard(job_pk)
    _log(f"[Pipeline] 收到恢复信号 job={job_pk}")
    return {"code": 200, "msg": "已恢复任务"}
# ============================================================
# ===== Analytics：数据需求 / 分布分析（PRD 26-33） =====
# ============================================================
@app.get("/api/analytics/overview")
def analytics_overview(project: str = Query("default")):
    """仪表盘概览：总量/已通过/待审核/已过滤/AI覆盖 + 各维度分布（Final Tags）"""
    from db_service import get_project, get_analytics_overview
    from models import Asset, AssetStatus
    from models import DecisionStatus as _DecisionStatus
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        base = get_analytics_overview(db, proj.id)
        filtered = (
            db.query(Asset)
            .filter(
                Asset.project_id == proj.id,
                Asset.status == AssetStatus.FILTERED,
            )
            .count()
        )
        base["filtered_count"] = filtered
        # 无标签资产数（AI 标注覆盖率分母更精确）
        tagged = (
            db.query(Asset)
            .filter(
                Asset.project_id == proj.id,
                Asset.ai_tags.isnot(None),
            )
            .count()
        )
        base["ai_tagged_assets"] = tagged
        # 口径分桶：以前 only 有 approved_count，前端把"人工审过的帧"也算进了「AI自动通过」，
        # 而「已人工通过」只统计 Clip -> 人工审帧后那个数字永远不动，看着像没生效。
        base["pending_count"] = (
            db.query(Asset)
            .filter(
                Asset.project_id == proj.id,
                (Asset.status == AssetStatus.REVIEW)
                | (Asset.decision_status == _DecisionStatus.REVIEW),
            )
            .count()
        )
        base["approved_human_count"] = (
            db.query(Asset)
            .filter(
                Asset.project_id == proj.id,
                Asset.status == AssetStatus.APPROVED,
                Asset.final_result.like("%human%"),
            )
            .count()
        )
        base["approved_auto_count"] = max(
            0, (base.get("approved_count") or 0) - base["approved_human_count"]
        )
        base["code"] = 200
        return base
    finally:
        db.close()
@app.get("/api/analytics/distribution")
def analytics_distribution(
    project: str = Query("default"),
    dim: str = Query("weather"),
    source: str = Query("final"),
):
    """单维度分布（source: final/ai/human）"""
    from db_service import get_project
    from models import Asset, AssetStatus
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        field = {
            "final": Asset.final_tags,
            "ai": Asset.ai_tags,
            "human": Asset.human_tags,
        }.get(source, Asset.final_tags)
        assets = db.query(Asset).filter(Asset.project_id == proj.id).all()
        dist = {}
        for a in assets:
            tags = (
                getattr(
                    a,
                    {"final": "final_tags", "ai": "ai_tags", "human": "human_tags"}[
                        source
                    ],
                )
                or {}
            )
            for t in tags.get(dim, []):
                name = t.get("tag") if isinstance(t, dict) else t
                dist[name] = dist.get(name, 0) + 1
        total = sum(dist.values())
        return {
            "code": 200,
            "dim": dim,
            "source": source,
            "total": total,
            "distribution": dist,
        }
    finally:
        db.close()
@app.post("/api/coverage/analyze")
def coverage_analyze(project: str = Form("default"), requirement: str = Form(None)):
    """需求覆盖度分析（PRD 32-33）：requirement 为 JSON 字符串，如

    {"weather":["雨天"],"road":["城市道路"],"time":["夜晚"],"events":["行人横穿"],"target":10000}"""
    import json as _json
    from db_service import get_project, search_assets_by_tags
    from models import AssetStatus
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        try:
            req = (
                _json.loads(requirement)
                if isinstance(requirement, str) and requirement.strip()
                else {}
            )
        except Exception:
            return {"code": 400, "msg": "requirement 不是合法 JSON"}
        if not req:
            return {"code": 400, "msg": "requirement 不能为空"}
        target = int(req.pop("target", 10000) or 10000)
        # 用 final_tags 优先匹配，回退 ai_tags（search_assets_by_tags 双查）
        _, current = search_assets_by_tags(
            db, proj.id, req, status=AssetStatus.APPROVED, page=1, size=100000
        )
        pct = current / max(1, target) * 100
        return {
            "code": 200,
            "requirement": req,
            "target": target,
            "current": current,
            "coverage": f"{pct:.1f}%",
            "gap": max(0, target - current),
            "msg": f"需求命中 {current} 条，覆盖率 {pct:.1f}%，缺口 {max(0, target - current)}",
        }
    finally:
        db.close()
# ============================================================
# ===== Benchmark 数据集与评测（PRD 24-25） =====
# ============================================================
@app.post("/api/benchmark/samples/add")
def benchmark_sample_add(
    project: str = Form("default"),
    image_id: int = Form(...),
    gt_tags: str = Form(None),
    split: str = Form("val"),
):
    """把某帧加入 Benchmark：gt_tags 为 JSON 字符串，如

    {"weather":["雨天"],"road":["城市道路"],"time":["夜晚"],"objects":["行人"],"events":["行人横穿"],"risk":["中风险"]}
    """
    import json as _json
    from db_service import add_benchmark_sample
    asset, _ctx, db = _find_db_asset(project, image_id)
    if db is None:
        return {"code": 500, "msg": "数据库未就绪"}
    try:
        if asset is None:
            return {"code": 404, "msg": "图片ID不存在或未入库"}
        try:
            gt = (
                _json.loads(gt_tags)
                if isinstance(gt_tags, str) and gt_tags.strip()
                else {}
            )
        except Exception:
            return {"code": 400, "msg": "gt_tags 不是合法 JSON"}
        if not gt:
            return {"code": 400, "msg": "gt_tags 不能为空（需含至少一个维度标注）"}
        sample = add_benchmark_sample(
            db, asset.project_id, asset.asset_id, gt, split or "val"
        )
        db.commit()
        return {
            "code": 200,
            "msg": "已加入 Benchmark",
            "benchmark_id": sample.benchmark_id,
            "asset_id": asset.asset_id,
            "gt_tags": gt,
            "split": split,
        }
    finally:
        db.close()
@app.post("/api/benchmark/samples/import")
def benchmark_samples_import(
    project: str = Form("default"),
    source: str = Form("reviewed"),
    limit: int = Form(200),
    split: str = Form("val"),
):
    """批量加入 Benchmark 样本（真值取自人工审核结果），省掉逐帧手填。

    source:
      reviewed — 人工审核过的帧（final_result.decided_by=human），GT = final_tags
      modified — 人工改过标签的帧（human_tags 非空），GT = final_tags（人工优先覆盖后）
      random   — 随机抽 N 帧，GT 留空待人工核对（评测时自动跳过未填真值的样本）

    ⚠️ 诚实提醒：人工只点了"通过"、没改任何标签的帧，其真值就等于模型自己的输出，
    拿它评测只会得到接近满分、没有参考意义 —— 真值要来自人工的独立判断（改过/重标/外部标注）。
    """
    import random as _random
    from models import Asset
    from db_service import add_benchmark_sample, get_benchmark_samples
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        src = (source or "reviewed").lower()
        if src not in ("reviewed", "modified", "random"):
            return {"code": 400, "msg": f"未知来源: {src}（可选 reviewed/modified/random）"}
        existing = {s.asset_id for s in get_benchmark_samples(db, proj.id)}
        picked = []
        for a in db.query(Asset).filter(Asset.project_id == proj.id).all():
            if a.asset_id in existing:
                continue
            if src == "modified":
                if not (a.human_tags and any((a.human_tags or {}).values())):
                    continue
            elif src == "reviewed":
                if str((a.final_result or {}).get("decided_by") or "") != "human":
                    continue
            picked.append(a)
        if src == "random":
            _random.shuffle(picked)
        picked = picked[: max(1, min(int(limit or 200), 5000))]
        for a in picked:
            gt = {} if src == "random" else dict(a.final_tags or {})
            add_benchmark_sample(db, proj.id, a.asset_id, gt, split or "val")
        db.commit()
        label = {"reviewed": "人工审核过的帧", "modified": "人工改过标签的帧",
                 "random": "随机抽样（待填真值）"}[src]
        return {
            "code": 200,
            "added": len(picked),
            "source": src,
            "split": split,
            "msg": f"已从「{label}」导入 {len(picked)} 个样本"
            + ("（真值留空，请在样本清单里补齐后再评测）" if src == "random" else "（真值取自标签）"),
        }
    finally:
        db.close()
@app.get("/api/benchmark/samples")
def benchmark_samples(project: str = Query("default")):
    """列出 Benchmark 样本（含 asset 缩略信息与 GT）"""
    from db_service import get_benchmark_samples
    from models import Asset
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        samples = get_benchmark_samples(db, proj.id)
        items = []
        for s in samples:
            asset = db.query(Asset).filter(Asset.asset_id == s.asset_id).first()
            items.append(
                {
                    "benchmark_id": s.benchmark_id,
                    "asset_id": s.asset_id,
                    "image_id": asset.vector_id if asset else None,
                    "image_url": (
                        f"/api/image/{project}/{asset.vector_id}" if asset else None
                    ),
                    "split": s.split,
                    "gt_tags": s.gt_tags or {},
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
            )
        return {"code": 200, "total": len(items), "items": items}
    finally:
        db.close()
@app.post("/api/benchmark/samples/remove")
def benchmark_sample_remove(
    project: str = Form("default"), benchmark_id: str = Form(...)
):
    """从 Benchmark 移除样本"""
    from models import Benchmark
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        row = db.query(Benchmark).filter(Benchmark.benchmark_id == benchmark_id).first()
        if not row:
            return {"code": 404, "msg": "样本不存在"}
        db.delete(row)
        db.commit()
        return {"code": 200, "msg": "已移除样本"}
    finally:
        db.close()
@app.post("/api/benchmark/evaluate")
def benchmark_evaluate(project: str = Form("default"), combo: str = Form(None)):
    """

    对 Benchmark 全集评测（离线：基于 Asset 已存的 VLM ai_tags / YOLO·DINO detections / Fusion）。

    combo: 可选 all / vlm / yolo / dino / fusion；默认 all。

    结果存入 benchmark_evaluations，指标 = 每维度 P/R/F1 + macro 均值。

    """
    from benchmark_engine import (
        evaluate_asset_all_sources,
        aggregate_evaluations,
        extract_prediction_sources,
    )
    from db_service import get_benchmark_samples, save_benchmark_evaluation
    from models import Asset
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        samples = get_benchmark_samples(db, proj.id)
        if not samples:
            return {
                "code": 400,
                "msg": "Benchmark 为空，请先添加样本（POST /api/benchmark/samples/add）",
            }
        per_asset = []
        detailed = []
        missing = 0
        no_gt = 0
        for s in samples:
            if not (s.gt_tags or {}):
                no_gt += 1          # 真值留空的样本（随机抽样待标注）不参与评测，避免把 P/R 拉成 0
                continue
            asset = db.query(Asset).filter(Asset.asset_id == s.asset_id).first()
            if asset is None:
                missing += 1
                continue
            src_map = evaluate_asset_all_sources(asset, s.gt_tags or {})
            if src_map:
                per_asset.append((asset.asset_id, src_map))
            detailed.append(
                {
                    "asset_id": asset.asset_id,
                    "benchmark_id": s.benchmark_id,
                    "gt_tags": s.gt_tags,
                    "sources": src_map,
                }
            )
        if not per_asset:
            return {
                "code": 400,
                "msg": "%d 个样本没有可用真值（跳过 %d 个真值留空的，缺失资产 %d 个），无法评测。"
                       "请在样本清单里补齐真值，或点「导入人工审核结果」" % (len(samples), no_gt, missing),
                "skipped_no_gt": no_gt,
            }
        agg = aggregate_evaluations(per_asset)
        # combo 过滤
        combo = (combo or "all").lower()
        if combo != "all":
            agg = {k: v for k, v in agg.items() if k == combo}
            detailed = [d for d in detailed if combo in (d.get("sources") or {})]
        versions = {"mode": "offline", "combo_filter": combo}
        eval_row = save_benchmark_evaluation(
            db, proj.id, combo, versions, {"sources": agg}
        )
        db.commit()
        return {
            "code": 200,
            "eval_id": eval_row.eval_id,
            "combo": combo,
            "sample_count": len(samples),
            "evaluated_count": len(per_asset),
            "missing_asset": missing,
            "skipped_no_gt": no_gt,
            "results": agg,  # {source: {samples, macro, per_dim}}
            "per_sample": detailed,
        }
    finally:
        db.close()
@app.get("/api/benchmark/evaluations")
def benchmark_evaluations(project: str = Query("default"), limit: int = 20):
    """历史评测记录"""
    from models import BenchmarkEvaluation
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        rows = (
            db.query(BenchmarkEvaluation)
            .filter(BenchmarkEvaluation.project_id == proj.id)
            .order_by(BenchmarkEvaluation.created_at.desc())
            .limit(max(1, min(limit, 100)))
            .all()
        )
        out = []
        for r in rows:
            out.append(
                {
                    "eval_id": r.eval_id,
                    "combo": r.model_combo,
                    "metrics": r.metrics or {},
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return {"code": 200, "evaluations": out}
    finally:
        db.close()
# ============================================================
# ===== Hard Case 闭环（PRD 23: AI≠Human -> 分析 -> 优化） =====
# ============================================================
@app.get("/api/hard-cases")
def hard_cases_list(
    project: str = Query("default"),
    status: str = Query(None),
    page: int = 1,
    size: int = 20,
):
    """Hard Case 列表：AI 与人工不一致的样本（含差异与资产信息）"""
    from models import Asset
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        from db_service import list_hard_cases
        rows = list_hard_cases(db, proj.id, status or None)
        total = len(rows)
        start = (page - 1) * size
        page_rows = rows[start : start + size]
        items = []
        for h in page_rows:
            asset = db.query(Asset).filter(Asset.asset_id == h.asset_id).first()
            items.append(
                {
                    "case_id": h.case_id,
                    "asset_id": h.asset_id,
                    "image_id": asset.vector_id if asset else None,
                    "image_url": (
                        f"/api/image/{project}/{asset.vector_id}" if asset else None
                    ),
                    "status": h.status,
                    "diff": h.diff or {},
                    "notes": h.notes,
                    "ai_only": (h.diff or {}).get("ai_only", []),
                    "human_only": (h.diff or {}).get("human_only", []),
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                    "resolved_at": h.resolved_at.isoformat() if h.resolved_at else None,
                }
            )
        return {"code": 200, "total": total, "page": page, "size": size, "items": items}
    finally:
        db.close()
@app.post("/api/hard-cases/update")
def hard_case_update(
    project: str = Form("default"),
    case_id: str = Form(...),
    status: str = Form(None),
    notes: str = Form(None),
):
    """流转 Hard Case：status=open/analyzed/resolved；notes 为分析结论"""
    from models import HardCase
    from datetime import datetime as _dt
    db = get_db_session()
    try:
        row = db.query(HardCase).filter(HardCase.case_id == case_id).first()
        if not row:
            return {"code": 404, "msg": "Hard Case 不存在"}
        if isinstance(status, str) and status in ("open", "analyzed", "resolved"):
            row.status = status
            if status == "resolved":
                row.resolved_at = _dt.utcnow()
        if isinstance(notes, str) and notes:
            row.notes = notes
        db.commit()
        return {
            "code": 200,
            "msg": "已更新",
            "case_id": case_id,
            "status": row.status,
            "notes": row.notes,
        }
    finally:
        db.close()
@app.get("/api/hard-cases/stats")
def hard_cases_stats(project: str = Query("default")):
    """Hard Case 统计：总数 / 未解决 / 分析中 / 已解决 + 高频差异标签"""
    from collections import Counter
    from models import HardCase
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        rows = db.query(HardCase).filter(HardCase.project_id == proj.id).all()
        by_status = Counter((r.status or "open") for r in rows)
        ai_only = Counter()
        human_only = Counter()
        for r in rows:
            d = r.diff or {}
            for t in d.get("ai_only", []):
                ai_only[t] += 1
            for t in d.get("human_only", []):
                human_only[t] += 1
        return {
            "code": 200,
            "total": len(rows),
            "by_status": dict(by_status),
            "top_ai_only": ai_only.most_common(10),
            "top_human_only": human_only.most_common(10),
        }
    finally:
        db.close()
# ============================================================
# ===== 数据导出（PRD 48-50, 63: JSONL/CSV/Parquet/ZIP + manifest） =====
# ============================================================
EXPORT_ROOT = os.path.join(WORKSPACE, "exports")
os.makedirs(EXPORT_ROOT, exist_ok=True)
def _export_worker(
    export_id: str, project: str, fmt: str, approved_only: bool, include_images: bool
):
    """后台导出：查 DB Assets -> 导出 -> ExportRecord 落盘"""
    from db_service import get_project_by_id
    from models import ExportRecord, Asset, AssetStatus
    from exporter import export_project
    from datetime import datetime as _dt
    db = get_db_session()
    try:
        rec = db.query(ExportRecord).filter(ExportRecord.export_id == export_id).first()
        if not rec:
            return
        rec.status = "running"
        db.commit()
        proj = get_project_by_id(db, rec.project_id)
        q = db.query(Asset).filter(Asset.project_id == rec.project_id)
        if approved_only:
            q = q.filter(Asset.status == AssetStatus.APPROVED)
        assets = q.order_by(Asset.vector_id).all()
        if not assets:
            rec.status = "failed"
            rec.error = "没有可导出的资产（approved_only=" + str(approved_only) + "）"
            db.commit()
            return
        # 输出目录按 PRD 47 布局落位：jsonl/parquet -> metadata/；csv/zip -> exports/
        from storage import project_dirs
        pname = proj.name if proj else "default"
        dirs = project_dirs(pname)
        if fmt == "jsonl":
            out_dir = dirs["jsonl"]
        elif fmt == "parquet":
            out_dir = dirs["parquet"]
        else:  # csv / zip 视为交付物
            out_dir = dirs["exports"]
        result = export_project(
            assets, pname, fmt, out_dir, include_images=include_images
        )
        # manifest 副本落 metadata/manifest/<project>/
        import json as _json
        try:
            os.makedirs(dirs["manifest"], exist_ok=True)
            _mname = "manifest_" + time.strftime("%Y%m%d_%H%M%S") + ".json"
            with open(
                os.path.join(dirs["manifest"], _mname), "w", encoding="utf-8"
            ) as _mf:
                _json.dump(result["manifest"], _mf, ensure_ascii=False, indent=2)
        except Exception:
            pass
        rec.output_path = result["path"]
        rec.asset_count = result["count"]
        rec.manifest = result["manifest"]
        rec.status = "completed"
        rec.completed_at = _dt.utcnow()
        db.commit()
        _log(
            f"[导出] 完成 export={export_id} fmt={fmt} count={result['count']} -> {result['path']}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        _log(f"[导出] 失败 export={export_id}: {e}")
        db.rollback()
        rec = db.query(ExportRecord).filter(ExportRecord.export_id == export_id).first()
        if rec:
            rec.status = "failed"
            rec.error = str(e)
            db.commit()
    finally:
        db.close()
@app.post("/api/export/run")
def export_run(
    project: str = Form("default"),
    fmt: str = Form("jsonl"),
    approved_only: int = Form(1),
    include_images: int = Form(1),
):
    """导出资产：fmt=jsonl/csv/parquet/zip；approved_only=1 仅导出已通过；

    include_images=1 zip 含原图。后台执行，进度/下载走 /api/export/list。"""
    from models import ExportRecord
    fmt = (fmt or "jsonl").lower()
    if fmt not in ("json", "jsonl", "csv", "parquet", "zip"):
        return {"code": 400, "msg": "未知格式，可选 json/jsonl/csv/parquet/zip"}
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        export_id = secrets.token_hex(8)
        rec = ExportRecord(
            export_id=export_id,
            project_id=proj.id,
            format=fmt,
            filter_criteria={
                "approved_only": bool(approved_only),
                "include_images": bool(include_images),
            },
            status="pending",
        )
        db.add(rec)
        db.commit()
    finally:
        db.close()
    t = threading.Thread(
        target=_export_worker,
        args=(export_id, project, fmt, bool(approved_only), bool(include_images)),
        daemon=True,
    )
    t.start()
    return {"code": 200, "msg": f"导出任务已启动 ({fmt})", "export_id": export_id}
@app.get("/api/export/list")
def export_list(project: str = Query("default"), limit: int = 20):
    """历史导出记录（含状态/下载路径）"""
    from models import ExportRecord
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        rows = (
            db.query(ExportRecord)
            .filter(ExportRecord.project_id == proj.id)
            .order_by(ExportRecord.created_at.desc())
            .limit(max(1, min(limit, 100)))
            .all()
        )
        out = []
        for r in rows:
            out.append(
                {
                    "export_id": r.export_id,
                    "format": r.format,
                    "status": r.status,
                    "asset_count": r.asset_count,
                    "output_path": r.output_path,
                    "error": r.error,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return {"code": 200, "exports": out}
    finally:
        db.close()
@app.get("/api/export/download/{export_id}")
def export_download(export_id: str):
    """下载导出文件（zip/jsonl/csv/parquet）"""
    from models import ExportRecord
    db = get_db_session()
    try:
        rec = db.query(ExportRecord).filter(ExportRecord.export_id == export_id).first()
        if not rec:
            return JSONResponse(status_code=404, content={"msg": "导出记录不存在"})
        if (
            rec.status != "completed"
            or not rec.output_path
            or not os.path.exists(rec.output_path)
        ):
            return JSONResponse(status_code=400, content={"msg": "文件未就绪或已清理"})
        media = {
            "jsonl": "application/x-ndjson",
            "csv": "text/csv",
            "parquet": "application/octet-stream",
            "zip": "application/zip",
        }.get(rec.format, "application/octet-stream")
        return FileResponse(
            rec.output_path,
            media_type=media,
            filename=os.path.basename(rec.output_path),
        )
    finally:
        db.close()
# ============================================================
# ===== 融合搜索（PRD 35）：自然语言 -> 结构化条件 + Tag + 语义 + 检测 =====
# ============================================================
from typing import Dict, List as _TList
_ONTO_DIM_NAMES = {
    "time": "时间",
    "weather": "天气",
    "road": "道路",
    "objects": "目标",
    "events": "事件",
    "risk": "风险",
}
def _parse_query_tags(query: str) -> Dict[str, List[str]]:
    """用关键词词典从自然语言 query 提取维度标签条件（与 VLM 解析共用词表）"""
    cond: Dict[str, set] = {}
    for kw, dim, tag in _VLM_KEYWORD_MAP:
        if kw in query:
            cond.setdefault(dim, set()).add(tag)
    return {k: sorted(v) for k, v in cond.items()}
@app.get("/api/ontology")
def ontology_list(project: str = Query("default")):
    """融合语义搜索的筛选字段与取值：与明细表同一套字段
    (车型/分辨率/时间/天气/道路/目标/事件/场景)。

    字段取值统一取自本体 ontology/scene.json —— 以前读的是 tag_system.ONTOLOGY 那套旧词表，
    给出的值(道路=城市道路/高速/高架…)与库里实际打的标准标签(道路=直路/弯道/十字路口…)对不上，
    勾了筛不到任何东西。车型/分辨率则来自资产元数据（动态）。"""
    from ontology import dimensions as _dims
    D = _dims()
    order = [
        ("traffic_sign", "交通标识信号"),
        ("vehicle", "车型"),
        ("resolution", "分辨率"),
        ("time", "时间"),
        ("weather", "天气"),
        ("road", "道路"),
        ("objects", "目标"),
        ("events", "事件"),
        ("scene", "场景"),
    ]
    out = {}
    for dim, cn in order:
        vals = [
            str(v)
            for v in ((D.get(dim) or {}).get("values") or [])
            if str(v) and str(v) != "unknown"
        ]
        out[dim] = {"name": cn, "values": vals}
    try:
        veh, res = _asset_facets(project)
        if veh:
            out["vehicle"]["values"] = veh
        if res:
            out["resolution"]["values"] = res
    except Exception as e:
        _log(f"[本体] 车型/分辨率取值读取失败: {e}")
    return {"code": 200, "ontology": out}
def _fresh_vocab():
    """词表被维护后清各处缓存：提示词枚举与解析校验下次调用即用新词表（无需重启）"""
    global _VLM_VOCAB
    _VLM_VOCAB = None
    try:
        import vlm_clip
        from ontology import valid_tags as _vt
        vlm_clip.VALID = _vt()   # Clip 链路在 import 时缓存过一份词表
    except Exception:
        pass
def _project_tag_values(project: str) -> list:
    """项目里实际出现过的标签取值（帧级 DB 标签 + Clip 结果），含出现次数与样例帧号。

    这是候选池的数据面：VLM 真看到、但本体里还没有的词，先在这里露出来，
    由人工在「融合语义搜索 → 标签维护」决定升为正式标签还是归并到已有标签。"""
    from models import Asset
    seen = {}

    def _bump(dim, tag, vid=None):
        if not isinstance(dim, str) or not isinstance(tag, str):
            return
        tag = tag.strip()
        if not tag or "|" in tag:     # 枚举回声这类脏值不进候选池
            return
        it = seen.setdefault((dim, tag), {"dim": dim, "tag": tag, "count": 0, "samples": []})
        it["count"] += 1
        if vid is not None and len(it["samples"]) < 5 and vid not in it["samples"]:
            it["samples"].append(vid)

    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is not None:
            for a in db.query(Asset).filter(Asset.project_id == proj.id).all():
                for col in (a.final_tags, a.ai_tags, a.human_tags):
                    for dim, vals in (col or {}).items():
                        for v in (vals if isinstance(vals, list) else [vals]):
                            _bump(dim, v.get("tag") if isinstance(v, dict) else v, a.vector_id)
    finally:
        db.close()
    # Clip 级 VLM 结果（clips.json）里的标签同样纳入候选
    try:
        from clip_service import load_clips
        ctx = load_project_context(project)
        for c in (load_clips(ctx) or []):
            res = c.get("vlm_result") or {}
            for dim in ("events", "objects", "scene", "road", "road_surface", "weather", "time"):
                if isinstance(res.get(dim), list):
                    for v in res[dim]:
                        _bump(dim, v.get("tag") if isinstance(v, dict) else v)
    except Exception as e:
        _log(f"[本体] Clip 标签收集跳过: {e}")
    return list(seen.values())
@app.get("/api/ontology/candidates")
def ontology_candidates(project: str = Query("default"), dim: str = Query(None)):
    """标签候选池：项目里出现过、但不在本体词表里的取值（事件/场景排最前）。

    常态化维护的入口数据 —— 看到新词就「加入本体」或「归并到已有标签」。"""
    from ontology import (
        valid_tags as _vt, canonical_tag, dimensions as _dims, dim_cn, dismissed_map,
    )
    vocab = _vt()
    dims = _dims()
    dismissed = dismissed_map()      # 已"放弃使用"的词不再提示
    # 排序优先级：事件/场景最高（平台首要关注），其次是明细表那 8 个字段，其余维度最后
    show = ["vehicle", "resolution", "time", "weather", "road", "objects", "events",
            "traffic_sign", "scene"]
    def _prio(dim):
        if dim == "events":
            return 0
        if dim == "scene":
            return 1
        return 2 if dim in show else 3
    out = []
    for it in _project_tag_values(project):
        if dim and it["dim"] != dim:
            continue
        if it["dim"] not in dims:
            continue                     # 旧维度名(road_type/surface 等)不参与维护
        if canonical_tag(it["dim"], it["tag"]) in vocab:
            continue                     # 已是正式值，或已归并到正式值
        if not _looks_like_tag(it["tag"]):
            continue
        if it["tag"] in (dismissed.get(it["dim"]) or []):
            continue                     # 已放弃使用
        out.append(dict(it, cn=dim_cn(it["dim"]), priority=_prio(it["dim"])))
    out.sort(key=lambda x: (x["priority"], -x["count"], x["dim"], x["tag"]))
    gave_up = [
        {"dim": d, "cn": dim_cn(d), "tag": t}
        for d, tags in sorted(dismissed.items())
        for t in tags
    ]
    return {"code": 200, "total": len(out), "candidates": out[:300], "dismissed": gave_up}
@app.post("/api/ontology/dismiss")
def ontology_dismiss(
    project: str = Form("default"),
    dim: str = Form(...),
    tag: str = Form(...),
    mode: str = Form("dismiss"),      # dismiss=放弃使用 / restore=放回候选池
):
    """放弃使用某个候选标签：不进本体、不再出现在候选池（可恢复）。

    与"归并"的区别：归并把它当成某个正式值的别名（写入端会归一、检索端会展开）；
    放弃只是不再提示，数据里已经出现的原词原样保留。"""
    from ontology import dismiss_value, restore_value, dimensions as _dims
    if dim not in _dims():
        return {"code": 400, "msg": f"未知维度: {dim}"}
    tag = (tag or "").strip()
    if not tag:
        return {"code": 400, "msg": "标签为空"}
    try:
        if (mode or "dismiss").lower() == "restore":
            changed, lst = restore_value(dim, tag)
            msg = f"已把「{tag}」放回候选池"
        else:
            changed, lst = dismiss_value(dim, tag)
            msg = f"已放弃使用「{tag}」（不再提示，可恢复）"
    except Exception as e:
        return {"code": 400, "msg": f"操作失败: {e}"}
    _log(f"[本体] {msg}（项目={project}）")
    return {"code": 200, "msg": msg, "changed": changed, "dismissed": lst}
@app.post("/api/ontology/promote")
def ontology_promote(
    project: str = Form("default"),
    dim: str = Form(...),
    tag: str = Form(...),
    mode: str = Form("new"),
    target: str = Form(None),
):
    """把候选标签沉淀进本体（标签维护的归口）。

    new   -> 升为该维度正式取值：提示词枚举、明细表筛选、融合搜索随后都带上它
    alias -> 记为 target 的取值别名：写入端自动归一(公交车切出->车辆切出)，
             检索端自动展开(按 target 筛也能搜到带别名的老数据)
    """
    from ontology import add_value, add_value_alias, dimensions as _dims
    dims = _dims()
    if dim not in dims:
        return {"code": 400, "msg": f"未知维度: {dim}"}
    tag = (tag or "").strip()
    if not tag:
        return {"code": 400, "msg": "标签为空"}
    mode = (mode or "new").lower()
    try:
        if mode == "alias":
            tgt = (target or "").strip()
            if not tgt:
                return {"code": 400, "msg": "归并模式必须指定目标标签"}
            changed, values = add_value_alias(dim, tag, tgt)
            msg = f"已把「{tag}」归并到「{tgt}」"
        elif mode == "new":
            changed, values = add_value(dim, tag)
            msg = f"已把「{tag}」加入「{dims[dim].get('cn') or dim}」正式取值"
        else:
            return {"code": 400, "msg": f"未知模式: {mode}（可选 new/alias）"}
    except Exception as e:
        return {"code": 400, "msg": f"维护失败: {e}"}
    _fresh_vocab()
    _log(f"[本体] {msg}（项目={project} 变更={changed}）")
    return {"code": 200, "msg": msg, "changed": changed, "dim": dim, "values": values}
@app.post("/api/search/fusion")
def search_fusion(
    project: str = Form("default"),
    query: str = Form(""),
    weather: str = Form(None),
    road: str = Form(None),
    time_dim: str = Form(None),
    objects: str = Form(None),
    events: str = Form(None),
    scene: str = Form(None),
    risk: str = Form(None),
    road_surface: str = Form(None),
    vehicle: str = Form(None),
    resolution: str = Form(None),
    top_k: int = Form(200),
    min_score: float = Form(0.0),
):
    """

    三类搜索融合（PRD 35）：

      1) 语义：SigLIP text -> FAISS（模型可用时）

      2) Tag/结构化：query 关键词解析 + 显式维度下拉（weather/road/time_dim/objects/events/risk，逗号分隔多值）

      3) 融合打分排序：semantic*0.7 + tag命中加权，去重返回

    """
    # ---- 显式结构化条件 + query 解析条件合并 ----
    tag_conditions: Dict[str, List[str]] = {}
    explicit = {
        "weather": weather,
        "road": road,
        "time": time_dim,
        "objects": objects,
        "events": events,
        "scene": scene,          # 场景是首要关注维度之一，早期漏了这个参数导致场景筛选被静默忽略
        "risk": risk,
        "road_surface": road_surface,
    }
    for dim, val in explicit.items():
        if isinstance(val, str) and val.strip():
            vals = [v.strip() for v in val.split(",") if v.strip()]
            if vals:
                tag_conditions[dim] = vals
    if isinstance(query, str) and query.strip():
        for dim, vals in _parse_query_tags(query).items():
            tag_conditions.setdefault(dim, []).extend(
                v for v in vals if v not in tag_conditions.get(dim, [])
            )
    # 取值别名展开：按正式值筛时，历史数据里被归并的别名也要能搜到
    try:
        from ontology import value_alias_map as _vam
        for dim, want in list(tag_conditions.items()):
            al = _vam(dim)
            if not al:
                continue
            extra = [a for a, canon in al.items() if canon in want and a not in want]
            if extra:
                tag_conditions[dim] = want + extra
    except Exception as e:
        _log(f"[本体] 别名展开跳过: {e}")
    # ---- 1) 语义检索（SigLIP 可用时；否则该路返回空）----
    sem_scores = {}  # vector_id -> float
    sem_ok = False
    sem_note = ""
    try:
        # SigLIP 可能被 VLM 任务卸载。这里补上按需重载（原来只有文本/以图搜图两个接口调了
        # _ensure_siglip，融合搜索漏了 -> VLM 跑过之后融合搜索静默返回 0 条）。
        # 但"重载"会腾显存把正在跑的 VLM 挤掉，任务会反复重载 -> 有任务在跑时只降级不抢显存。
        if not LITE_MODE and query.strip() and siglip_model is None:
            if _running_tag_jobs(project):
                sem_note = "语义引擎暂不可用：AI 分析任务正在运行（避免抢显存中断任务），本次仅按标签检索"
                _log("[搜索] " + sem_note)
            else:
                _ensure_siglip()
        if not LITE_MODE and siglip_model is not None and query.strip():
            ctx = load_project_context(project)
            idx = ctx["index"]
            if idx is not None and idx.ntotal > 0:
                with _gpu_slot("SigLIP 语义检索"), torch.no_grad():
                    if siglip_model is None or siglip_processor is None:
                        # 排队期间被别的 AI 任务卸了：本轮只按标签检索（由外层 except 收口）
                        raise RuntimeError("SigLIP 已被其它 AI 任务卸载（显存互斥）")
                    inputs = siglip_processor(
                        text=[query],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=64,
                    ).to(DEVICE)
                    if DEVICE == "cuda":
                        with torch.cuda.amp.autocast():
                            tf = siglip_model.get_text_features(**inputs)
                    else:
                        tf = siglip_model.get_text_features(**inputs)
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                k = min(top_k, idx.ntotal)
                scores, indices = idx.search(tf.cpu().numpy().astype(np.float32), k)
                for sc, ix in zip(scores[0], indices[0]):
                    sem_scores[int(ix)] = float(sc)
                sem_ok = True
    except Exception as e:
        _log(f"[搜索] 语义检索降级: {e}")
        if not sem_note:
            sem_note = "语义检索本轮降级（显存被其它 AI 任务占用），结果仅按标签匹配"
    # ---- 2) Tag / 结构化过滤（DB 侧，final/ai 双匹配）----
    tag_hits = set()
    from db_service import search_assets_by_tags, get_project
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is not None and tag_conditions:
            assets, _ = search_assets_by_tags(
                db, proj.id, tag_conditions, page=1, size=100000
            )
            tag_hits = {a.vector_id for a in assets}
    finally:
        db.close()
    # ---- 2.5) 车型/分辨率过滤（来自资产元数据，不是标签）----
    meta_conds = {}
    for _k, _val in (("vehicle", vehicle), ("resolution", resolution)):
        if isinstance(_val, str) and _val.strip():
            _vals = [x.strip() for x in _val.split(",") if x.strip()]
            if _vals:
                meta_conds[_k] = _vals
    meta_hits = set()
    if meta_conds:
        from models import Asset as _AssetM
        dbm = get_db_session()
        try:
            projm = _ensure_db_project(dbm, project)
            if projm is not None:
                for a in dbm.query(_AssetM).filter(_AssetM.project_id == projm.id).all():
                    got = {
                        "vehicle": str((a.asset_metadata or {}).get("vehicle") or "").strip(),
                        "resolution": _asset_resolution(a),
                    }
                    if all(got.get(k) in want for k, want in meta_conds.items()):
                        meta_hits.add(a.vector_id)
        finally:
            dbm.close()
    # ---- 2.8) 事件/场景命中集合（平台里这两个维度的优先级最高，命中加权）----
    prio_hits = set()
    prio_dims = [d for d in ("events", "scene") if tag_conditions.get(d)]
    if prio_dims:
        dbp = get_db_session()
        try:
            projp = _ensure_db_project(dbp, project)
            if projp is not None:
                assets_p, _ = search_assets_by_tags(
                    dbp, projp.id,
                    {d: tag_conditions[d] for d in prio_dims},
                    page=1, size=100000,
                )
                prio_hits = {a.vector_id for a in assets_p}
        except Exception as e:
            _log(f"[搜索] 事件/场景加权集合计算失败: {e}")
        finally:
            dbp.close()
    # ---- 3) 融合打分 ----
    union = set(sem_scores.keys()) | tag_hits
    if meta_conds:
        # 与其它条件取交集；只选了车型/分辨率时则以它为准
        union = (union & meta_hits) if (tag_conditions or sem_scores) else set(meta_hits)
    matched_dims = len(tag_conditions)
    results = []
    for vid in union:
        score = 0.0
        sem = sem_scores.get(vid, 0.0)
        if sem_ok:
            score += sem * 0.7
        if (vid in tag_hits and tag_conditions) or (vid in meta_hits and meta_conds):
            score += 0.3  # 标签/元数据命中基础分（无逐帧命中计数时全命中近似）
        if vid in prio_hits:
            score += 0.1  # 事件/场景命中加权（这两个维度是平台的首要关注）
        if score < min_score:
            continue
        results.append(
            {
                "id": vid,
                "score": round(score, 4),
                "semantic_score": round(sem, 4) if sem_ok else None,
                "tag_hit": vid in tag_hits,
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]
    # 组装与 metadata 对齐的完整 item（含 url/标签摘要）
    ctx = load_project_context(project)
    meta = ctx.get("metadata", [])
    items = []
    for r in results:
        if 0 <= r["id"] < len(meta):
            it = dict(meta[r["id"]])
            it["score"] = r["score"]
            it["semantic_score"] = r["semantic_score"]
            it["tag_hit"] = r["tag_hit"]
            items.append(it)
        else:
            # DB-only asset（未入 JSON metadata）：尝试 DB 取标签摘要
            items.append(
                {
                    "id": r["id"],
                    "asset_id": None,
                    "score": r["score"],
                    "semantic_score": r["semantic_score"],
                    "tag_hit": r["tag_hit"],
                }
            )
    # ---- 按"片段"聚合结果 ----
    # 打标是"每 50 个抽样帧一段、一份标签"，搜索也应以段为单位给出，
    # 否则同一段的 50 帧会重复出现 50 次（标签完全相同，没有信息量）。
    # 分段由后端按各视频的帧顺序现算（同一视频按时间戳排序后每 SEG 帧一段），
    # 与打标时的切段口径一致，不需要重跑打标。
    _SEG = max(5, int(os.environ.get("AD_JOINT_SEG", "30")))
    _seg_of, _frames_of, _by_video = {}, {}, {}
    _clip_rows = []          # 没有源视频的帧（Clip 链路抽出的图）：按文件名恢复成段
    for _m in meta:
        _vs = _m.get("video_source") or {}
        _vp = _vs.get("video_path") if isinstance(_vs, dict) else None
        if _vp:
            _by_video.setdefault(_vp, []).append((_vs.get("timestamp") or 0, _m.get("id")))
            continue
        _parts = _clip_name_parts(_m.get("path") or _m.get("filename") or "")
        if _parts and isinstance(_m.get("id"), int):
            # 载荷就是帧 id（这里只需要段内顺序 + id）
            _clip_rows.append((_parts[0], _parts[1], _parts[2], _m.get("id")))
    for _vp, _lst in _by_video.items():
        _lst.sort(key=lambda x: (x[0], x[1] if x[1] is not None else 0))
        for _si, _seg in enumerate(_seg_chunks([x[1] for x in _lst], _SEG)):
            _k = (_vp, _si)
            for _vid in _seg:
                _seg_of[_vid] = _k
            _frames_of[_k] = list(_seg)
    # Clip 链路的帧同样每 _SEG 帧一段：以前这里直接 continue 丢掉，导致这些帧在
    # "按片段给结果"的搜索里彻底不可见（CLIP测试 实测 40 个命中被丢掉 31 个）。
    for _ck, _run in _split_clip_runs(_clip_rows):
        _tag, _run_i = _ck
        _label = ("Clip " + _tag) if _run_i == 0 else "Clip %s·%d" % (_tag, _run_i + 1)
        # _run 里存的就是帧 id（这一步的行载荷），按文件名恢复出来的顺序已排好
        for _si, _seg in enumerate(_seg_chunks(list(_run), _SEG)):
            _k = (_label, _si)
            for _vid in _seg:
                _seg_of[_vid] = _k
            _frames_of[_k] = list(_seg)
    _grouped, _order = {}, []
    _no_video = 0
    for _it in items:
        _vid = _it.get("id")
        _k = _seg_of.get(_vid)
        if _k is None:
            # 图片直导帧（没有源视频）无法组成 Clip：融合语义搜索按"片段"给结果，
            # 这类帧就不展示了（否则会冒出一条"单帧"结果）。要看单图去图集/明细表。
            _no_video += 1
            continue
        _g = _grouped.get(_k)
        if _g is None:
            _g = _grouped[_k] = dict(_it)
            _g["segment"] = {"video": os.path.basename(_k[0]) if _k[0] else "",
                             "index": _k[1] if isinstance(_k[1], int) else 0,
                             "frame_count": len(_frames_of.get(_k) or [_vid]),
                             "frame_ids": list(_frames_of.get(_k) or [_vid])}
            _g["_best"] = _it.get("score") or 0
            _order.append(_k)
        else:
            _sc = _it.get("score") or 0
            if _sc > _g["_best"]:           # 段的分值取段内最高，代表帧也换成这一帧
                _g["_best"] = _sc
                for _f in ("id", "filename", "image_url", "path", "score", "semantic_score",
                           "tag_hit", "video_path", "timestamp"):
                    if _f in _it:
                        _g[_f] = _it.get(_f)
    if meta and _no_video and not _order:
        items = []          # 结果全是图片直导帧（无源视频）-> 按片段的搜索不展示
    if _order:
        _segs = []
        for _k in _order:
            _g = _grouped[_k]
            _g.pop("_best", None)
            # 代表帧取段内中间那帧（比首帧更能代表整段）
            _fids = _g["segment"]["frame_ids"]
            if _fids:
                _g["id"] = _fids[len(_fids) // 2]
                _g["image_url"] = "/api/image/%s/%s" % (project, _g["id"])
                _mm = next((m for m in meta if m.get("id") == _g["id"]), None)
                if _mm is not None:
                    for _f in ("filename", "path", "video_source", "timestamp", "video_path"):
                        if _mm.get(_f) is not None:
                            _g[_f] = _mm.get(_f)
            _segs.append(_g)
        _segs.sort(key=lambda x: -(x.get("score") or 0))
        items = _segs
    # 每段附上综合标签（代表帧那帧的 ai_tags）：点开 Clip 连播时前端在右下角显示，
    # 用来核对"标签是不是真的对得上画面"。原来只有图片和分值，看不到依据。
    if items:
        for _it in items:
            _it.setdefault("tags", {})
        try:
            from models import Asset as _As
            _ids = [it.get("id") for it in items if isinstance(it.get("id"), int)][:1000]
            if _ids:
                _dbt = get_db_session()
                try:
                    _pt = _ensure_db_project(_dbt, project)
                    _tagmap = {}
                    if _pt is not None:
                        for _a in _dbt.query(_As).filter(
                                _As.project_id == _pt.id, _As.vector_id.in_(_ids)).all():
                            _tagmap[_a.vector_id] = _a.ai_tags or {}
                    for _it in items:
                        _it["tags"] = _tagmap.get(_it.get("id")) or {}
                finally:
                    _dbt.close()
        except Exception as _e:
            _log(f"[搜索] 附加片段标签失败(不影响搜索本身): {_e}")
    return {
        "code": 200,
        "query": query,
        "matched_frames": len(union),
        "segment_count": len(items),
        "hidden_no_video": _no_video,     # 被隐藏的图片直导帧数（无源视频，组不成 Clip）
        "tag_conditions": tag_conditions,
        "meta_conditions": meta_conds,
        "semantic_engine": sem_ok,
        "semantic_note": sem_note,
        "semantic_total": len(sem_scores),
        "tag_hit_total": len(tag_hits),
        "total": len(items),
        "results": items,
    }
# ============================================================
# ===== 存储布局（PRD 47 网盘目录规范） =====
# ============================================================
@app.get("/api/storage/layout")
def storage_layout():
    """返回当前数据目录树（AD_DATA_ROOT 可指向 NAS 挂载）"""
    from storage import layout_info
    return {"code": 200, "layout": layout_info()}
# ============================================================
# ===== 数据根配置（NAS/网盘持久化落库） =====
# ============================================================
@app.get("/api/storage/config")
def storage_config_get():
    """当前生效的数据根配置（来源：env / data_config.json / 默认）"""
    return {"code": 200, "config": _cfg_info()}
@app.post("/api/storage/config")
def storage_config_set(data_root: str = Form(None), db_path: str = Form(None)):
    """持久化配置到项目根 data_config.json。

    data_root: 媒体/项目仓根（NAS/网盘挂载路径，如 Z:/AutodriveData）

    db_path:   SQLite 库路径（建议留空跟随 data_root；或指本地 SSD）

    重启后端后生效（当前会话仍使用旧路径）。

    """
    from settings import save_config
    if not (data_root or db_path):
        return {"code": 500, "msg": "需提供 data_root 或 db_path"}
    if data_root and os.path.isfile(data_root):
        return {"code": 500, "msg": f"路径 {data_root} 是文件，需为目录"}
    cfg = save_config(data_root=data_root, db_path=db_path)
    if data_root:
        try:
            os.makedirs(data_root, exist_ok=True)
        except Exception as e:
            return {"code": 500, "msg": f"目录不可创建: {e}"}
    return {
        "code": 200,
        "msg": "配置已写入 data_config.json，重启后端后生效",
        "config": cfg,
    }
# ============================================================
# ===== API 契约总览（开发对齐用） =====
# ============================================================
@app.get("/api/meta/endpoints")
def api_endpoints():
    """列出全部 API 路由与方法（供前端/文档核对 PRD 53 契约）"""
    from fastapi.routing import APIRoute
    out = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            out.append({"method": sorted(route.methods or []), "path": route.path})
    out.sort(key=lambda r: r["path"])
    return {"code": 200, "total": len(out), "endpoints": out}
# ============================================================
# ===== 数据管理明细表（PRD 59）与任务中心辅助 =====
# ============================================================
def _asset_resolution(a):
    """资产分辨率档位：元数据优先，否则按宽高推断。
    明细表与融合搜索筛选用同一口径，避免两处算出不同的值。"""
    r = (a.asset_metadata or {}).get("resolution")
    if r:
        return r
    if not a.width:
        return ""
    px = (a.width or 0) * (a.height or 0)
    return "8M" if px >= 8e6 else ("2M" if px >= 1.8e6 else "其他")
def _asset_facets(project: str):
    """车型/分辨率的可选取值（来自资产元数据，按出现次数排序）"""
    from models import Asset
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return [], []
        veh, res = {}, {}
        for a in db.query(Asset).filter(Asset.project_id == proj.id).all():
            v = str((a.asset_metadata or {}).get("vehicle") or "").strip()
            if v:
                veh[v] = veh.get(v, 0) + 1
            r = _asset_resolution(a)
            if r:
                res[r] = res.get(r, 0) + 1
        return (
            sorted(veh, key=lambda k: (-veh[k], k)),
            sorted(res, key=lambda k: (-res[k], k)),
        )
    finally:
        db.close()
@app.get("/api/assets/table")
def assets_table(
    project: str = Query("default"),
    page: int = 1,
    size: int = 20,
    status: str = Query(None),
    weather: str = Query(None),
    group: str = Query(None),
):
    """数据管理明细表：DB Asset 行（帧ID/血缘/标签三层状态/决策），支持状态与天气过滤。
    group=video 时按「原始视频路径」聚合：一个视频一行，标签去重合并（抽出的多帧合并体现）。"""
    from models import Asset, AssetStatus
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
        q = db.query(Asset).filter(Asset.project_id == proj.id)
        if isinstance(status, str) and status.strip():
            try:
                q = q.filter(Asset.status == AssetStatus(status.upper()))
            except Exception:
                pass
        grouped = str(group or "").lower() == "video"
        if grouped:
            assets = q.order_by(Asset.vector_id.asc()).all()   # 聚合需要全量，先不分页
            total = len(assets)
        else:
            total = q.count()
            assets = (
                q.order_by(Asset.vector_id.asc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
        def _names(tags, dim):
            vals = (tags or {}).get(dim) or []
            out = []
            for v in vals:
                n = v.get("tag") if isinstance(v, dict) else v
                if isinstance(n, str) and n.strip() and n not in out:
                    out.append(n)
            return out
        rows = []
        for a in assets:
            fn = _names(a.final_tags, "weather")
            if (
                isinstance(weather, str)
                and weather.strip()
                and weather not in fn
                and weather not in _names(a.ai_tags, "weather")
            ):
                continue
            meta = a.asset_metadata or {}
            rows.append(
                {
                    "image_id": a.vector_id,
                    "asset_id": a.asset_id,
                    "filename": os.path.basename(a.image_path or ""),
                    "image_url": f"/api/image/{project}/{a.vector_id}",
                    "frame_index": a.frame_index,
                    "timestamp": a.timestamp,
                    "time": _names(a.final_tags, "time") or _names(a.ai_tags, "time"),
                    "weather": _names(a.final_tags, "weather")
                    or _names(a.ai_tags, "weather"),
                    "road": _names(a.final_tags, "road") or _names(a.ai_tags, "road"),
                    "objects": _names(a.final_tags, "objects")
                    or _names(a.ai_tags, "objects"),
                    "events": _names(a.final_tags, "events")
                    or _names(a.ai_tags, "events"),
                    "scene": _names(a.final_tags, "scene") or _names(a.ai_tags, "scene"),
                    "risk": _names(a.final_tags, "risk") or _names(a.ai_tags, "risk"),
                    "vehicle": meta.get("vehicle", ""),
                    "resolution": _asset_resolution(a),
                    "source_file": a.source.file_name if a.source else "",
                    "source_chain": (
                        (a.source.directory_chain or []) if a.source else []
                    ),
                    "ai_status": ("已标" if a.ai_tags else "未标"),
                    "human_status": (
                        "已审"
                        if (
                            a.human_tags
                            and any(v for v in (a.human_tags or {}).values())
                        )
                        else "未审"
                    ),
                    "status": a.status.value if a.status else "",
                    "decision_status": (
                        a.decision_status.value if a.decision_status else ""
                    ),
                    "decision_score": a.decision_score,
                    "ai_score": (a.asset_metadata or {}).get("ai_score"),
                    "video_path": ((a.asset_metadata or {}).get("video_source") or {}).get("video_path") or "",
                }
            )
        if grouped:
            # 按原始视频路径聚合：一个视频一行；同一视频抽出的多帧标签去重合并
            agg = {}
            for r in rows:
                key = r["video_path"] or ("(非视频来源) " + (r.get("source_file") or ""))
                g = agg.get(key)
                if g is None:
                    g = agg[key] = {
                        "source_file": r.get("source_file") or "",
                        "video_path": r.get("video_path") or "",
                        "vehicle": set(), "resolution": set(),
                        "time": [], "weather": [], "road": [], "objects": [],
                        "scene": [], "events": [], "risk": [],
                        "frames": 0, "tagged": 0, "reviewed": 0, "score_max": None,
                        "status": {}, "decision": {},
                    }
                g["frames"] += 1
                if r.get("ai_status") == "已标":
                    g["tagged"] += 1
                if r.get("human_status") == "已审":
                    g["reviewed"] += 1
                if r.get("ai_score") is not None:
                    g["score_max"] = max(g["score_max"] or 0, r["ai_score"])
                for k in ("vehicle", "resolution"):
                    if r.get(k):
                        g[k].add(r[k])
                for k in ("time", "weather", "road", "objects", "scene", "events", "risk"):
                    for v in (r.get(k) or []):
                        if v not in g[k]:     # 去重合并
                            g[k].append(v)
                if r.get("status"):
                    g["status"][r["status"]] = g["status"].get(r["status"], 0) + 1
                if r.get("decision_status"):
                    g["decision"][r["decision_status"]] = g["decision"].get(r["decision_status"], 0) + 1
            out = []
            for g in agg.values():
                out.append({
                    "source_file": g["source_file"], "video_path": g["video_path"],
                    "vehicle": "/".join(sorted(g["vehicle"])),
                    "resolution": "/".join(sorted(g["resolution"])),
                    "time": g["time"], "weather": g["weather"], "road": g["road"],
                    "objects": g["objects"], "scene": g["scene"], "events": g["events"], "risk": g["risk"],
                    "ai_status": "已标 %d/%d" % (g["tagged"], g["frames"]),
                    "human_status": "已审 %d/%d" % (g["reviewed"], g["frames"]),
                    "ai_score": g["score_max"],
                    "status": "　".join("%s %d" % (k, v) for k, v in sorted(g["status"].items(), key=lambda x: -x[1])),
                    "decision_status": "　".join("%s %d" % (k, v) for k, v in sorted(g["decision"].items(), key=lambda x: -x[1])),
                    "group_frames": g["frames"],
                })
            total = len(out)
            rows = out[(page - 1) * size: page * size]
        return {"code": 200, "total": total, "page": page, "size": size, "rows": rows,
                "grouped": "video" if grouped else ""}
    finally:
        db.close()
# ============================================================
# ===== 导出血缘信息: filename -> source 块(交付 JSON 注入用) =====
# ============================================================
def _orig_source_root(src, vs):
    """导出/血缘用的“原始数据根目录”。
    Source.source_root 是入库时的简化登记(=项目 images 目录)，不含原始数据位置，直接导出没用。
    优先取：入库时记录的原始源目录 src_dir → 视频所在目录 → 兜底回退 source_root。
    注：历史老记录两处都没有时无法还原(当时未记录)。
    """
    sd = (getattr(src, "meta", None) or {}).get("src_dir") if src else None
    if sd:
        return sd
    vp = (vs or {}).get("video_path")
    if vp:
        return os.path.dirname(vp)
    return src.source_root if src else None
@app.post("/api/source_info")
def source_info(project: str = Form("default"), filenames: str = Form("")):
    """按文件名批量返回资产血缘(供前端导出 JSON 注入 source 字段)。
    返回: {map: {filename: {source: {source_root, relative_path, original_filename,
           source_type, video_filename, frame_index, timestamp}}}}
    """
    names = [n.strip() for n in str(filenames or "").split(",") if n.strip()]
    if not names:
        return {"code": 200, "map": {}}
    from db_service import get_db_session
    from models import Asset, Source
    db = get_db_session()
    out = {}
    try:
        proj = _ensure_db_project(db, project)
        if proj is not None:
            assets = db.query(Asset).filter(Asset.project_id == proj.id).all()
            for a in assets:
                fn = os.path.basename(a.image_path or "")
                if fn not in names or fn in out:
                    continue
                s = (
                    db.query(Source).filter(Source.id == a.source_id).first()
                    if a.source_id
                    else None
                )
                meta = dict(a.asset_metadata or {})
                vs = (
                    (meta.get("video_source") or {})
                    if isinstance(meta.get("video_source"), dict)
                    else {}
                )
                src_type = "video" if vs else "image"
                out[fn] = {
                    "source": {
                        "source_root": _orig_source_root(s, vs),
                        "relative_path": (s.relative_path if s else None),
                        "original_filename": (s.file_name if s else fn),
                        "source_type": src_type,
                        "video_filename": vs.get("video_filename"),
                        "video_path": vs.get("video_path"),
                        "frame_index": a.frame_index if src_type == "video" else None,
                        "timestamp": a.timestamp if src_type == "video" else None,
                    }
                }
    finally:
        db.close()
    return {"code": 200, "map": out}
def _write_deliver_meta(project: str, output_dir: str, items: list, img_dir: str):
    """交付包写 deliver_meta.json：每条帧带 source(血缘) / annotations(检测) / final_tags(标签)"""
    import json as _j
    from sqlalchemy.orm import joinedload
    from db_service import get_db_session
    from models import Asset, DetectionCache
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        rows = []
        if proj is not None:
            assets = (
                db.query(Asset)
                .options(joinedload(Asset.source))
                .filter(Asset.project_id == proj.id)
                .all()
            )
            by_fn = {os.path.basename(a.image_path or ""): a for a in assets}
            dets = {}
            for dc in (
                db.query(DetectionCache)
                .filter(DetectionCache.project_id == proj.id)
                .all()
            ):
                dets.setdefault(str(dc.asset_id), []).append(dc)
            for it in items:
                fn = it["img_name"]
                a = by_fn.get(fn)
                if a is None:
                    rows.append({"filename": fn})
                    continue
                src = a.source
                meta = dict(a.asset_metadata or {})
                vs = (
                    meta.get("video_source") or {}
                    if isinstance(meta.get("video_source"), dict)
                    else {}
                )
                src_type = "video" if vs else "image"
                entry = {
                    "filename": fn,
                    "source": {
                        "source_root": _orig_source_root(src, vs),
                        "relative_path": src.relative_path if src else None,
                        "original_filename": src.file_name if src else fn,
                        "source_type": src_type,
                        "video_filename": vs.get("video_filename"),
                        "video_path": vs.get("video_path"),
                        "frame_index": a.frame_index if src_type == "video" else None,
                        "timestamp": a.timestamp if src_type == "video" else None,
                    },
                    "final_tags": {
                        "ai_tags": a.ai_tags or {},
                        "human_tags": a.human_tags or {},
                        "decision": (
                            a.decision_status.value if a.decision_status else None
                        ),
                    },
                }
                rows.append(entry)
        out_path = os.path.join(output_dir, "deliver_meta.json")
        with open(out_path, "w", encoding="utf-8") as f:
            _j.dump(
                {"project": project, "count": len(rows), "items": rows},
                f,
                ensure_ascii=False,
                indent=2,
            )
        _log(f"✔ 交付清单已写: {out_path} ({len(rows)} 条)")
    finally:
        db.close()
# ============================================================
# ===== Clip 级分析（P0 最小闭环）=====
# ============================================================
try:
    from clip_service import (build_clips_from_metadata, load_clips, save_clips,
                              update_clip_result, set_clip_decision)
    from vlm_clip import predict_clip
except Exception:  # 缺 torch 等依赖时进入轻量模式，避免整个应用启动失败
    build_clips_from_metadata = load_clips = save_clips = update_clip_result = set_clip_decision = None
    predict_clip = None
@app.post("/api/clips/scan")
def clips_scan(project: str = Form("default"), clip_size: int = Form(30)):
    """扫描底库，把视频抽出的帧分组成 Clip 并持久化到 clips.json"""
    if build_clips_from_metadata is None:
        return {"code": 500, "msg": "Clip 模块不可用（依赖缺失，服务处于轻量模式）"}
    ctx = load_project_context(project)
    if not os.path.isdir(ctx["img_dir"]):
        return {"code": 404, "msg": f"项目 {project} 尚无底库目录，请先创建项目并完成视频抽帧入库"}
    clips = build_clips_from_metadata(ctx, clip_size)
    save_clips(ctx, clips)
    if not clips:
        return {"code": 200, "msg": "未生成 Clip：底库中没有来自视频抽帧的帧（纯图片直导不计入 Clip）", "count": 0}
    return {"code": 200, "msg": f"已生成 {len(clips)} 个 Clip", "count": len(clips)}
@app.get("/api/clips/list")
def clips_list(project: str = Query("default"), page: int = 1, size: int = 20, light: int = 0):
    ctx = load_project_context(project)
    clips = load_clips(ctx)
    total = len(clips)
    start = (page - 1) * size
    items = clips[start:start + size]
    # 返回时附带代表帧预览（每clip前1帧的url即可，前端列表用）
    for it in items:
        it["frame_count"] = len(it.get("frame_ids") or [])
        it["preview_url"] = f"/api/image/{project}/{it['frame_ids'][0]}" if it.get("frame_ids") else None
        # 采样帧 URL：与 vlm_clip.sample_representative 同一套均匀取点，
        # 供审核界面做"模型实际输入的 5 帧 × 判定结果"对照
        fids = it.get("frame_ids") or []
        if fids:
            _idx = np.linspace(0, len(fids) - 1, min(5, len(fids))).round().astype(int).tolist()
            _seen, _pos = set(), []
            for _k in _idx:
                if _k not in _seen:
                    _seen.add(_k)
                    _pos.append(int(_k))
            it["sample_urls"] = [
                {"pos": k, "url": f"/api/image/{project}/{fids[k]}"} for k in _pos
            ]
        if light:  # 列表/可视化场景不需要逐帧路径，省流量
            it.pop("frame_paths", None)
            it.pop("frame_ids", None)
    return {"code": 200, "total": total, "page": page, "size": size, "items": items}
@app.post("/api/clips/analyze_single")
def clips_analyze_single(project: str = Form("default"), clip_id: str = Form(...)):
    """P0 同步接口：分析单个 Clip（5帧VLM）。用于先验证效果，批量Job在P0通过后再加"""
    ctx = load_project_context(project)
    clips = load_clips(ctx)
    clip = next((c for c in clips if c["clip_id"] == clip_id), None)
    if not clip:
        return {"code": 404, "msg": "Clip不存在，请先 POST /api/clips/scan"}
    # 加载与推理放在同一个 GPU 闸门内：12G 卡上模型不能共存，
    # 也避免"加载完->释放锁->被别人卸掉"这种空窗（原来分两段加锁就有这个缝）
    try:
        with _gpu_slot("Clip VLM 判定"):
            _free_dino(); _free_yolo()
            if vlm_model is None:
                init_vlm_local()
            if vlm_model is None:
                return {"code": 500, "msg": "VLM 加载失败（显存不足或权重缺失）"}
            result, raw = predict_clip(vlm_model, vlm_processor, clip["frame_paths"])
    except Exception as e:
        # 只记错误不落 decision：异常多为环境/依赖问题，保持可重试，别把 Clip 永久标成需人工
        import traceback as _tb
        _log(f"[Clip] VLM推理失败 clip={clip_id}: {e}\n{_tb.format_exc()}")
        # OOM 兜底：清缓存并把 VLM 卸载掉，避免显存被占死导致后续每个 Clip 都失败
        _emsg = str(e).lower()
        if "out of memory" in _emsg or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
            try:
                with _gpu_slot("Clip OOM 清理"):   # 清理也要排队，别在别人推理时抽显存
                    torch.cuda.empty_cache()
                    _free_vlm()
                _vram_log("OOM 后已卸载 VLM")
            except Exception:
                pass
            _log(f"[Clip] VLM 推理 OOM，已卸载 VLM 释放显存；下次调用会重新加载")
        update_clip_result(ctx, clip_id, {"error": str(e)})
        return {"code": 500, "msg": f"VLM推理失败: {e}"}
    if result is None:
        update_clip_result(ctx, clip_id, {"error": "parse_failed"}, decision="REVIEW")
        return {"code": 500, "msg": "VLM输出解析失败", "raw": raw[:500]}
    # 简易决策：有事件且置信>=0.7 且证据齐全 → REVIEW（重点事件必人工）；否则AUTO_PASS
    has_event = any(e["confidence"] >= 0.7 and len(e["evidence"]) >= 2
                    for e in result["events"])
    decision = "REVIEW" if has_event else "AUTO_PASS"
    update_clip_result(ctx, clip_id, result, decision=decision)
    return {"code": 200, "clip_id": clip_id, "decision": decision,
            "result": result, "raw_output": raw}
@app.get("/api/clips/video_summary")
def clips_video_summary(project: str = Query("default")):
    """按“原始视频路径”汇总 Clip 判定结果：同一视频下各片段的场景/事件标签去重合并，
    并给出各标签被多少个片段判出，用于看单个视频里到底有什么场景。"""
    ctx = load_project_context(project)
    clips = load_clips(ctx) or []
    by_video = {}
    for c in clips:
        vid = c.get("video_id") or "(未记录来源)"
        v = by_video.setdefault(vid, {
            "video": vid, "video_name": os.path.basename(str(vid)),
            "clips": 0, "frames": 0, "analyzed": 0,
            "t_start": None, "t_end": None, "decisions": {},
            "scene": {}, "events": {}, "objects": {},
        })
        v["clips"] += 1
        v["frames"] += len(c.get("frame_ids") or [])
        if c.get("vlm_result"):
            v["analyzed"] += 1
        ts, te = c.get("start_timestamp"), c.get("end_timestamp")
        if ts is not None:
            v["t_start"] = ts if v["t_start"] is None else min(v["t_start"], ts)
        if te is not None:
            v["t_end"] = te if v["t_end"] is None else max(v["t_end"], te)
        d = c.get("decision") or "undecided"
        v["decisions"][d] = v["decisions"].get(d, 0) + 1
        r = c.get("vlm_result") or {}
        for dim, vals in (r.get("scene") or {}).items():
            for x in (vals or []):
                if x and x != "unknown":     # 去重：同一片段重复出现只计一次
                    v["scene"].setdefault(dim, {})
                    v["scene"][dim][x] = v["scene"][dim].get(x, 0) + 1
        for e in (r.get("events") or []):
            t = e.get("type")
            if t:
                v["events"][t] = v["events"].get(t, 0) + 1
        for o in (r.get("objects") or []):
            t = o.get("type")
            if t:
                v["objects"][t] = v["objects"].get(t, 0) + 1
    videos = sorted(by_video.values(), key=lambda x: (-x["clips"], x["video_name"]))
    return {"code": 200, "total_videos": len(videos), "total_clips": len(clips), "videos": videos}
# ============================================================
# ===== 单帧上下文回放：回到原始视频，取该帧前后连续画面（不抽稀）=====
# 抽帧是 1:N 稀疏的，搜索命中的帧之间隔着好几帧，连起来看不出运动；
# 这里按帧的原始帧号回原始视频解出前后 span 帧，前端按真实帧率播放。
# ============================================================
_FRAME_CACHE = os.path.join(PROJECT_DIR, "_frame_cache")
def _prune_frame_cache(keep: int = 40):
    """缓存目录只保留最近若干段，避免越积越多（原帧 jpg 每帧约 200KB）"""
    try:
        ds = [os.path.join(_FRAME_CACHE, d) for d in os.listdir(_FRAME_CACHE)]
        ds = [d for d in ds if os.path.isdir(d)]
        if len(ds) <= keep:
            return
        ds.sort(key=lambda d: os.path.getmtime(d))
        for d in ds[:-keep]:
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass
@app.get("/api/video_window")
def video_window(
    project: str = Query("default"),
    image_id: int = Query(...),
    span: int = Query(30),
    id_b: int = Query(None),
    max_side: int = Query(1280),
):
    """取【原始视频】里的连续画面，供前端带完整控制的播放器播放。

    - 只给 image_id：取它前后 span 帧（单帧上下文回放）；
    - 给 image_id + id_b：取这两帧之间（一段 clip 的完整原始画面）；
    返回 {urls, fps, start_frame, center_frame, count}；帧按需解到磁盘缓存。"""
    if cv2 is None:
        return {"code": 400, "msg": "服务器未安装 opencv-python"}
    asset, _c, db = _find_db_asset(project, image_id)
    if db is None:
        return {"code": 500, "msg": "数据库未就绪"}
    try:
        if asset is None:
            return {"code": 404, "msg": "帧不存在"}
        vs = (asset.asset_metadata or {}).get("video_source") or {}
        vpath, fidx = vs.get("video_path"), vs.get("frame_index")
        if not vpath or fidx is None:
            return {"code": 400, "msg": "该帧不是视频抽帧（没有原始视频信息），无法回看原帧"}
        if not os.path.exists(vpath):
            return {"code": 400, "msg": "原始视频不可访问: " + str(vpath)}
        # 段范围模式：id_b 与 id_a 同属一个视频时，取两帧之间的完整画面
        f_end = None
        if id_b is not None:
            try:
                a2, _c2, _db2 = _find_db_asset(project, id_b)
                try:
                    vs2 = ((a2.asset_metadata or {}).get("video_source") or {}) if a2 is not None else {}
                    if vs2.get("video_path") == vpath and vs2.get("frame_index") is not None:
                        f_end = int(vs2["frame_index"])
                finally:
                    if _db2 is not None:
                        _db2.close()
            except Exception:
                f_end = None
        if f_end is not None:
            start = max(0, min(int(fidx), f_end))
            span = max(2, min(abs(int(fidx) - f_end) + 1, 900))
        else:
            span = max(10, min(int(span or 30), 400))
            start = max(0, int(fidx) - span // 2)
        max_side = max(320, min(int(max_side or 1280), 1920))
        base = re.sub(r"[^0-9A-Za-z_\-]", "_", os.path.splitext(os.path.basename(vpath))[0])[-40:]
        key = "%s_%d_%d_%d" % (base, start, span, max_side)
        cdir = os.path.join(_FRAME_CACHE, key)
        os.makedirs(cdir, exist_ok=True)
        fps = 15.0
        stamp = os.path.join(cdir, "done.txt")
        if not os.path.exists(stamp):
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                return {"code": 500, "msg": "原始视频无法打开"}
            try:
                _f = cap.get(cv2.CAP_PROP_FPS)
                if _f and _f > 0:
                    fps = float(_f)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for i in range(span):
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        break
                    try:
                        _h, _w = fr.shape[:2]
                        _mx = max(_h, _w)
                        if _mx > max_side:
                            _sc = max_side / float(_mx)
                            fr = cv2.resize(fr, (int(_w * _sc), int(_h * _sc)),
                                            interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass
                    cv2.imwrite(os.path.join(cdir, "%05d.jpg" % i), fr,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
            finally:
                cap.release()
            try:
                with open(stamp, "w", encoding="utf-8") as f2:
                    f2.write(str(fps))
            except Exception:
                pass
            _prune_frame_cache()
        else:
            try:
                fps = float(open(stamp, encoding="utf-8").read().strip() or fps)
            except Exception:
                pass
        files = sorted(x for x in os.listdir(cdir) if x.endswith(".jpg"))
        return {
            "code": 200,
            "video": os.path.basename(vpath),
            "fps": round(fps, 2),
            "start_frame": start,
            "count": len(files),
            "center_frame": int(fidx),
            "urls": ["/api/frame_cache/%s/%s" % (key, x) for x in files],
        }
    finally:
        db.close()
@app.get("/api/video_clip")
def video_clip(
    project: str = Query("default"),
    id_a: int = Query(...),
    id_b: int = Query(None),
    max_frames: int = Query(600),
):
    """把原始视频里某一段（两帧之间的连续画面）以 MJPEG 流推给前端 —— 就是"播这一段的视频"。

    为什么这么做：抽帧是 1:5 稀疏的，把抽出来的帧一张张切着放会一跳一跳，不像视频；
    这里直接解原始视频的连续帧、按视频真实帧率推流。用 MJPEG 是因为源文件是 AVI
    （浏览器认不了），而服务器上没有 ffmpeg 可转码——MJPEG 浏览器原生支持。
    id_a/id_b 给段内首尾两帧，窗口就正好是这一段；只给 id_a 则默认取前后 50 帧。"""
    from fastapi.responses import StreamingResponse
    if cv2 is None:
        return JSONResponse(status_code=400, content={"msg": "服务器未安装 opencv-python"})
    a1, _c, db = _find_db_asset(project, id_a)
    if db is None:
        return JSONResponse(status_code=500, content={"msg": "数据库未就绪"})
    try:
        if a1 is None:
            return JSONResponse(status_code=404, content={"msg": "帧不存在"})
        vs = (a1.asset_metadata or {}).get("video_source") or {}
        vpath, f1 = vs.get("video_path"), vs.get("frame_index")
        if not vpath or f1 is None:
            return JSONResponse(status_code=400, content={"msg": "该帧不是视频抽帧，无法回原始视频"})
        if not os.path.exists(vpath):
            return JSONResponse(status_code=400, content={"msg": "原始视频不可访问: " + str(vpath)})
        f1 = int(f1)
        f2 = None
        if id_b is not None:
            a2, _c2, _db2 = _find_db_asset(project, id_b)
            try:
                vs2 = ((a2.asset_metadata or {}).get("video_source") or {}) if a2 is not None else {}
                if vs2.get("video_path") == vpath and vs2.get("frame_index") is not None:
                    f2 = int(vs2["frame_index"])
            finally:
                if _db2 is not None:
                    _db2.close()
        if f2 is None:
            f2 = f1 + 50
        start, end = min(f1, f2), max(f1, f2)
        n = max(1, min(int(max_frames), end - start + 1))
        _vp = vpath

        def _gen():
            cap = cv2.VideoCapture(_vp)
            if not cap.isOpened():
                return
            try:
                _fps = cap.get(cv2.CAP_PROP_FPS)
                if not _fps or _fps <= 0:
                    _fps = 15.0
                _delay = 1.0 / float(_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _i in range(n):
                    ok, fr = cap.read()
                    if not ok or fr is None:
                        break
                    ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not ok2:
                        continue
                    data = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    time.sleep(_delay)
            finally:
                cap.release()

        _log(f"[回放] 原视频片段推流 {os.path.basename(_vp)} 帧 {start}~{start + n - 1}")
        return StreamingResponse(
            _gen(), media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )
    finally:
        db.close()
@app.get("/api/frame_cache/{key}/{name}")
def frame_cache_get(key: str, name: str):
    """回看原帧的图片服务（限定在缓存目录内，防目录穿越）"""
    if ("/" in key or ".." in key or "/" in name or ".." in name
            or not name.lower().endswith(".jpg")):
        return JSONResponse(status_code=400, content={"msg": "非法路径"})
    pth = os.path.join(_FRAME_CACHE, key, name)
    if not os.path.isfile(pth):
        return JSONResponse(status_code=404, content={"msg": "不存在"})
    return FileResponse(pth, media_type="image/jpeg")
@app.get("/api/clips/stats")
def clips_stats(project: str = Query("default")):
    """Clip 判定结果汇总：各决策计数 + 场景/事件标签聚合（供审核中心可视化）"""
    ctx = load_project_context(project)
    clips = load_clips(ctx) or []
    dec = {"REVIEW": 0, "AUTO_PASS": 0, "APPROVED": 0, "FILTERED": 0, "undecided": 0}
    scene, events, objects, videos = {}, {}, {}, set()
    for c in clips:
        d = c.get("decision")
        dec[d if d in dec else "undecided"] += 1
        if c.get("video_id"):
            videos.add(c["video_id"])
        r = c.get("vlm_result") or {}
        for dim, vals in (r.get("scene") or {}).items():
            for v in (vals or []):
                if v and v != "unknown":
                    scene.setdefault(dim, {})
                    scene[dim][v] = scene[dim].get(v, 0) + 1
        for e in (r.get("events") or []):
            t = e.get("type")
            if t:
                events[t] = events.get(t, 0) + 1
        for o in (r.get("objects") or []):
            t = o.get("type")
            if t:
                objects[t] = objects.get(t, 0) + 1
    return {"code": 200, "total": len(clips), "video_count": len(videos),
            "decisions": dec, "scene": scene, "events": events, "objects": objects}
@app.post("/api/clips/review")
def clips_review(project: str = Form("default"), clip_id: str = Form(...), action: str = Form(...)):
    """人工审核单个 Clip：approve=通过 / reject=驳回 / reset=退回待审核"""
    act = {"approve": "APPROVED", "reject": "FILTERED", "reset": None}
    if action not in act:
        return {"code": 400, "msg": f"未知动作: {action}"}
    ctx = load_project_context(project)
    if not set_clip_decision(ctx, clip_id, act[action], human={"by": "human", "action": action}):
        return {"code": 404, "msg": "Clip不存在"}
    return {"code": 200, "msg": "已更新", "decision": act[action]}
if __name__ == "__main__":
    import uvicorn
    # 生产部署：可用环境变量覆盖，默认监听 0.0.0.0:8009
    # 注意：本应用有共享的内存态（task_status / 后台任务），必须单进程 workers=1
    host = os.environ.get("AD_HOST", "0.0.0.0")
    port = int(os.environ.get("AD_PORT", "8009"))
    print(f"[i] 启动服务: http://{host}:{port}  (日志目录: {LOG_DIR})")
    _log(f"服务启动 http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)
@app.post("/api/prune_missing")
def prune_missing(project: str = Form("default"), confirm: str = Form(None)):
    """清理失效资产(手动删除图片后残留的黑片记录)：以磁盘文件为唯一真相——
    文件已不存在的资产连带删除(DB 资产+检测缓存+推理缓存+审核+HardCase+孤儿Source+metadata 索引项)。
    红线：本端点只清数据库/索引记录，绝不删除磁盘/网盘任何文件。"""
    _log(f"[清理] prune_missing project={project}")
    ctx = load_project_context(project)
    from db_service import (
        get_db_session,
        get_project,
        delete_assets_by_filenames,
        resync_asset_vector_ids,
    )
    db = get_db_session()
    try:
        proj = get_project(db, project)
        # ---- 1) DB 侧扫描: 文件已不存在 -> 连带删除(不依赖 metadata) ----
        db_removed = 0
        src_removed = 0
        if proj:
            from models import Asset
            import os as _os
            assets = db.query(Asset).filter(Asset.project_id == proj.id).all()
            miss_bn = [
                _os.path.basename(a.image_path or "")
                for a in assets
                if a.image_path and not _os.path.exists(a.image_path)
            ]
            _total_a = len(assets)
            _ratio = (len(miss_bn) / float(_total_a)) if _total_a else 0.0
            # 安全闸：缺失比例过高时先要一次显式确认。
            # 防的是"网盘没挂载 / 路径变了"这种情形——那时所有文件都判为缺失，
            # 一键下去会把整个库的记录（含标签/审核/评测）全删掉。
            if _ratio >= 0.3 and str(confirm or "") != "1":
                _log(f"[清理] 缺失比例 {len(miss_bn)}/{_total_a}，等待确认")
                return {
                    "code": 409,
                    "need_confirm": True,
                    "missing": len(miss_bn),
                    "total": _total_a,
                    "msg": "检测到 %d/%d 帧（%.0f%%）的文件已不存在。\n\n"
                           "继续将【连带删除】这些资产的标签、审核记录、评测记录与索引项"
                           "（只删数据库记录，绝不删磁盘文件）。\n\n"
                           "如果这是你有意删除的图片，可以继续；"
                           "如果是网盘未挂载/路径变动导致的，请先恢复挂载再操作。"
                           % (len(miss_bn), _total_a, _ratio * 100),
                }
            if miss_bn:
                # 清理前自动备份数据库（网络盘/本地盘各留一份，成本几 MB）
                try:
                    # 用 SQLite 官方备份接口：直接 copyfile 一个开着 WAL 的库，
                    # 可能拷到撕裂的中间状态（曾侥幸拷出一份 integrity ok 的）。
                    import sqlite3 as _sq3
                    _bkdir = os.path.join(PROJECT_DIR, "_db_backups")
                    _os.makedirs(_bkdir, exist_ok=True)
                    _bk = os.path.join(
                        _bkdir, "mining_%s.db" % time.strftime("%Y%m%d_%H%M%S"))
                    _src = _sq3.connect(os.path.join(PROJECT_DIR, "workspace", "mining.db"))
                    _dst = _sq3.connect(_bk)
                    try:
                        with _dst:
                            _src.backup(_dst)
                    finally:
                        _src.close()
                        _dst.close()
                    _log(f"[清理] 已备份数据库(SQLite备份接口): {_bk}")
                except Exception as _e:
                    _log(f"[清理] 数据库备份失败(继续清理): {_e}")
                r1 = delete_assets_by_filenames(db, proj.id, miss_bn)
                db_removed = r1["assets"]
                src_removed = r1["sources"]
        # ---- 2) metadata/索引侧: 剔除缺失项, 现存重嵌保序(仅当 metadata 非空) ----
        meta_paths = [m.get("path") for m in ctx["metadata"] if m.get("path")]
        missing = [p for p in meta_paths if not os.path.exists(p)]
        existing = [p for p in meta_paths if os.path.exists(p)]
        rebuilt = False
        vid_updated = 0
        if missing:
            with _project_lock(ctx["name"]):
                ctx["metadata"] = []
                ctx["index"] = (
                    faiss.IndexFlatIP(FEAT_DIM)
                    if (not LITE_MODE and faiss is not None)
                    else _FakeIndex(FEAT_DIM)
                )
                extract_and_index_project(ctx, existing)
                rebuilt = True
            if proj:
                try:
                    r2 = resync_asset_vector_ids(
                        db, project, [m["path"] for m in ctx["metadata"]]
                    )
                    vid_updated = r2["updated"]
                except Exception as e:
                    _log(f"[清理] resync 失败(已删DB残留, 索引已重建): {e}")
        return {
            "code": 200,
            "missing_files": len(missing),
            "existing_files": len(existing),
            "db_assets_removed": db_removed,
            "db_sources_removed": src_removed,
            "index_rebuilt": rebuilt,
            "vector_id_updated": vid_updated,
        }
    finally:
        db.close()
@app.post("/api/pipeline/clear_completed")
def pipeline_clear_completed(project: str = Form("default")):
    """清除 DB 任务中心已完成记录(终态 SUCCESS/FAILED/CANCELLED/PENDING)，运行中保留。
    对应前端 '清除已完成记录' 按钮在 DB 任务中心(#编号任务)的清理。"""
    from models import Job
    from db_service import get_db_session, get_project
    db = get_db_session()
    try:
        proj = get_project(db, project)
        if not proj:
            return {"code": 404, "msg": f"项目不存在: {project}"}
        q = db.query(Job).filter(Job.project_id == proj.id, Job.status != "RUNNING")
        n = q.delete(synchronize_session=False)
        db.commit()
        running = (
            db.query(Job)
            .filter(Job.project_id == proj.id, Job.status == "RUNNING")
            .count()
        )
        _log(
            f"[任务] DB任务中心清除 {n} 条已完成，剩余运行中 {running} 个 (project={project})"
        )
        return {
            "code": 200,
            "cleared": n,
            "running": running,
            "msg": f"已清除 {n} 条已完成记录，剩余运行中 {running} 个",
        }
    finally:
        db.close()
