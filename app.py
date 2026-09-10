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
ADMIN_PASSWORD = os.environ.get("AD_PASSWORD", "admin123")  # 管理员密码（部署前请修改）
ANNOTATOR_PASSWORD = os.environ.get("ANNOTATOR_PASSWORD", "anno123")  # 标注员密码
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
    if password == ADMIN_PASSWORD:
        role = "admin"
    elif password == ANNOTATOR_PASSWORD:
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
    if os.path.exists(index_html):
        return FileResponse(index_html)
    # 兼容旧部署：历史上服务器上的文件名是 前端.html
    legacy = os.path.join(PROJECT_DIR, "前端.html")
    if os.path.exists(legacy):
        return FileResponse(legacy)
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
def extract_and_index_project(ctx, image_paths: List[str], frame_meta: dict = None):
    # SigLIP 可能被 VLM 卸载(显存互斥)，向量化前按需重载；加载失败则跳过本批
    if _ensure_siglip()[0] is None:
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
    if _ensure_siglip()[0] is None:   # VLM 可能占着显存把 SigLIP 卸了，这里按需重载
        return {"code": 500, "msg": "SigLIP 未就绪（显存不足或模型缺失），无法做文本检索"}
    inputs = siglip_processor(
        text=[query], return_tensors="pt", padding="max_length", max_length=64
    ).to(DEVICE)
    with torch.no_grad():
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
    inputs = siglip_processor(images=pil_imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
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
        "scene: " + _v("scene") + "\n"
        "每个维度只输出最匹配的 1 个值（objects/events 可为空数组，events 最多 2 个），必须使用中文，"
        "尖括号 <> 内是占位说明，必须替换为你实际判断出的枚举值，禁止原样照抄枚举清单。\n"
        "⚠️ 必须完整输出全部 8 个字段。注意区分：road 是道路形态(直路/弯道/十字路口…)，"
        "scene 是地点场景(城市道路/高速道路/隧道…)，两者含义不同，都要填。"
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
def _clean_vlm_tags(dim: str, vals) -> list:
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
    from ontology import canonical_tag
    vocab = _vlm_vocab()
    out = []
    for v in vals:
        parts = [p.strip() for p in _TAG_SEP.split(str(v or "").strip()) if p.strip()]
        valid = [p for p in parts if p in vocab]
        if len(valid) >= 3 and len(parts) >= 4:
            continue        # 枚举清单回声：不是对画面的判断，整条丢弃
        for p in parts:
            p = canonical_tag(dim, p)
            if p not in vocab and not _looks_like_tag(p):
                continue    # 是句子/描述，不是标签
            if p not in out:
                out.append(p)
    cap = 2 if dim == "events" else (4 if dim == "objects" else 3)
    return out[:cap]
def _parse_vlm_structured(text: str) -> dict:
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
                    "scene",
                ):
                    tags = _clean_vlm_tags(dim, obj.get(dim))
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
    """DB 层项目兜底：同名项目目录存在/或全新时，自动建 DB Project 记录（避免双轨 404）。

    返回 Project 对象或 None（db 不可用）。"""
    from db_service import get_project, create_project
    proj = get_project(db, project_name)
    if proj is None:
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
def _vlm_predict_text(image) -> str:
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
    asset, dim_tags: dict, model_name: str, confidence: float = None
):
    ai_tags = dict(asset.ai_tags or {})
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for dim, tags in (dim_tags or {}).items():
        existing = [t for t in (ai_tags.get(dim) or []) if isinstance(t, dict)]
        names = {t.get("tag") for t in existing}
        for tag_name in tags:
            if tag_name not in names:
                existing.append(
                    {
                        "tag": tag_name,
                        "source": "VLM",
                        "confidence": confidence,
                        "model": model_name,
                        "version": "v1",
                        "created_at": ts,
                    }
                )
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
        # 2) 解析结构化输出
        dim_tags = _parse_vlm_structured(output_text)
        # 3) 合并维度标签进 ai_tags（带来源追踪 VLM/model/version）
        model_name = vlm_loaded_name or VLM_MODEL_NAME
        ai_tags = _merge_dim_tags_into_asset(asset, dim_tags, model_name)
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
            tag = _YOLO_LABEL2OBJ.get(lb)
            if not tag:
                # DINO 属开放词(提示词通常已是中文，可原样保留)；YOLO 是固定 COCO 类名，
                # 没映射到的一律归为"其他目标"——绝不能把英文类名写进标签
                # (曾出现 suitcase/handbag/train 混入目标标签)
                tag = lb if eng == "dino" else "其他目标"
            cands.append((cf, tag, eng))
    cands.sort(key=lambda x: -x[0])
    for cf, tag, eng in cands[:4]:
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
def _ai_batch_worker(
    job_pk: int,
    project: str,
    image_ids: List[int],
    preset: str = "balanced",
    detect_first: bool = False,
    dino_enhance: bool = False,
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
                _log(f"[Pipeline] 阶段1 YOLO检测 job={job_pk} frames={total}")
                _update_job_progress(job_pk, 0, total, f"YOLO 检测 0/{total}")
                from db_service import get_project as _gp
                # 复用现有批量检测核心(同进程直接调用端点函数, 返回 {code,results,count})
                resp = yolo_detect_batch(
                    project=project,
                    image_ids=",".join(str(x) for x in image_ids),
                    batch_size=16,
                    fp16="1",
                )
                # 关键：检测阶段失败必须中止任务。否则 0 命中会被后面的"无检测目标"逻辑
                # 当成"这批帧都没目标"，把全部帧静默标成 AUTO_PASS 且不产生任何标签
                # （曾导致 2034 帧空通过、分析中心无数据可统计）。
                if not isinstance(resp, dict) or resp.get("code") != 200:
                    raise RuntimeError("YOLO 检测失败: " + str((resp or {}).get("msg") or "无返回"))
                detect_ok = True
                for id_str, r in (resp.get("results") or {}).items():
                    try:
                        iid = int(id_str)
                    except Exception:
                        continue
                    dets = []
                    for lb, cf, bx in zip(
                        r.get("labels") or [],
                        r.get("scores") or [],
                        r.get("boxes") or [],
                    ):
                        dets.append({"label": lb, "confidence": float(cf), "box": bx})
                    if dets:
                        yolo_hits[iid] = dets
                # 落 DB asset.detections["yolo"](供 evidence 与前端)
                db2 = get_db_session()
                try:
                    for iid, dets in yolo_hits.items():
                        a = (
                            db2.query(Asset)
                            .filter(
                                Asset.project_id == _gp(db2, project).id,
                                Asset.vector_id == iid,
                            )
                            .first()
                            if _gp(db2, project)
                            else None
                        )
                        if a is None:
                            continue
                        dd = dict(a.detections or {})
                        dd["yolo"] = dets
                        a.detections = dd
                    db2.commit()
                finally:
                    db2.close()
                _log(f"[Pipeline] YOLO 完成 命中 {len(yolo_hits)} 帧")
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
                dino_prompt = ", ".join(list(_dct.keys())[:60])
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
        hit_ids = set(yolo_hits) | set(
            dino_hits_local
        )  # 检测出目标的帧(仅这些帧跑 VLM)
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
                if detect_first and detect_ok and img_id not in hit_ids:
                    # 无检测目标帧: 跳过 VLM(省算力), 直接免人工通过
                    from db_service import update_asset_decision as _uad0
                    _uad0(
                        adb, asset.asset_id, "AUTO_PASS", "无检测目标(免VLM直通)", None
                    )
                    adb.commit()
                    processed_ok += 1
                    continue
                output_text = _vlm_predict_text(Image.open(img_path).convert("RGB"))
                if not output_text:
                    failed.append(img_id)
                    continue
                else:
                    dim_tags = _parse_vlm_structured(output_text)
                    model_name = vlm_loaded_name or VLM_MODEL_NAME
                    # VLM 的 objects 维弃用(以真实检测为准); 若帧无检测则保留 VLM objects 作兜底
                    if img_id in yolo_hits:
                        dim_tags.pop("objects", None)
                    _merge_dim_tags_into_asset(asset, dim_tags, model_name)
                    # ---- 检测融合: objects 维 = YOLO 真实检测(映射到标准目标类别) ----
                    if img_id in yolo_hits:
                        objs = _detections_to_objects(asset.detections or {})
                        if objs:
                            ai = dict(asset.ai_tags or {})
                            ai["objects"] = objs
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
            _update_job_progress(job_pk, pos, total, f"VLM+融合决策 {pos}/{total}")
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
    image_ids: str = Form(...),
    preset: str = Form(None),
    detect_first: int = Form(0),
    dino_enhance: int = Form(0),
):
    """创建批量 AI Pipeline 任务（VLM 结构化 → 维度标签 → Decision Engine 决策）。

    image_ids 为逗号分隔的数字。返回 {job_id, status}，进度走 GET /api/pipeline/{job_id}。"""
    from db_service import create_job
    from models import JobType
    # 解析 image_ids
    ids = []
    for s in str(image_ids or "").replace(" ", "").split(","):
        if s.isdigit():
            ids.append(int(s))
    if not ids:
        return {"code": 400, "msg": "未解析到有效的 image_ids"}
    preset = preset if isinstance(preset, str) and preset else "balanced"
    db = get_db_session()
    try:
        proj = _ensure_db_project(db, project)
        if proj is None:
            return {"code": 500, "msg": "数据库未就绪"}
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
        args=(job_pk, project, ids, preset, bool(detect_first), bool(dino_enhance)),
        daemon=True,
    )
    t.start()
    _log(
        f"[Pipeline] 批量 AI 任务已创建 job_pk={job_pk} frames={len(ids)} project={project} preset={preset}"
    )
    return {
        "code": 200,
        "msg": f"批量 AI Pipeline 已启动，共 {len(ids)} 帧",
        "job_id": str(job_pk),
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
        for s in samples:
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
    show = ["vehicle", "resolution", "time", "weather", "road", "objects", "events", "scene"]
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
    try:
        if not LITE_MODE and siglip_model is not None and query.strip():
            ctx = load_project_context(project)
            idx = ctx["index"]
            if idx is not None and idx.ntotal > 0:
                inputs = siglip_processor(
                    text=[query],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                ).to(DEVICE)
                with torch.no_grad():
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
    return {
        "code": 200,
        "query": query,
        "tag_conditions": tag_conditions,
        "meta_conditions": meta_conds,
        "semantic_engine": sem_ok,
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
def clips_scan(project: str = Form("default"), clip_size: int = Form(50)):
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
    # 复用现有VLM互斥加载逻辑
    _free_dino(); _free_yolo()
    if vlm_model is None:
        try:
            with _GPU_MODEL_LOCK:
                init_vlm_local()
        except Exception as e:
            return {"code": 500, "msg": f"VLM加载失败: {e}"}
    try:
        with _GPU_MODEL_LOCK:   # 推理期间不允许别的请求把模型卸掉
            result, raw = predict_clip(vlm_model, vlm_processor, clip["frame_paths"])
    except Exception as e:
        # 只记错误不落 decision：异常多为环境/依赖问题，保持可重试，别把 Clip 永久标成需人工
        import traceback as _tb
        _log(f"[Clip] VLM推理失败 clip={clip_id}: {e}\n{_tb.format_exc()}")
        # OOM 兜底：清缓存并把 VLM 卸载掉，避免显存被占死导致后续每个 Clip 都失败
        _emsg = str(e).lower()
        if "out of memory" in _emsg or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _free_vlm()
            _vram_log("OOM 后已卸载 VLM")
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
def prune_missing(project: str = Form("default")):
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
            if miss_bn:
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
