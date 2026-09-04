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
    cv2 = None   # 未安装 opencv-python 时，视频接口会友好提示

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request, BackgroundTasks
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
    from transformers import AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection
    from ultralytics import YOLO
except Exception as _dep_err:
    torch = Dataset = DataLoader = transforms = faiss = None
    AutoProcessor = AutoModel = AutoModelForZeroShotObjectDetection = YOLO = None
    LITE_MODE = True
    print(f"[i] 轻量模式：模型依赖缺失，已跳过模型功能 ({_dep_err})。")

# ===== DINO 中英词典（离线方案）：支持直接输入中文 =====
# 服务器无法访问外网，故不使用在线翻译。
# 优先读取外部 JSON 文件 dino_dict.json（可随时编辑增词，无需改代码），
# 缺失或未覆盖时回退到内置默认词典。
_DEFAULT_DINO_DICT = {
    "卡车": "truck", "货车": "truck", "大货车": "heavy truck",
    "汽车": "car", "小车": "car", "越野车": "SUV",
    "行人": "pedestrian", "人": "person", "小孩": "child",
    "锥桶": "traffic cone", "反光锥": "traffic cone", "水马": "water-filled barrier",
    "路标": "traffic sign", "指示牌": "traffic sign", "限速牌": "speed limit sign",
    "红绿灯": "traffic light", "交通灯": "traffic light",
    "车道线": "lane marker", "斑马线": "crosswalk", "停止线": "stop line",
    "路灯": "streetlight", "树木": "tree", "道路": "road",
    "自行车": "bicycle", "摩托车": "motorcycle", "三轮车": "tricycle"
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
WORKSPACE = os.path.join(PROJECT_DIR, "workspace")
PROJECTS_ROOT = os.path.join(WORKSPACE, "projects")
os.makedirs(PROJECTS_ROOT, exist_ok=True)

LOG_DIR = os.environ.get("AD_LOG_DIR", os.path.join(PROJECT_DIR, "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_RETENTION_DAYS = 30

# 交付打包时，伴生文件(.bin/.xml/_raw.jpg)的来源根目录（可多个，逗号/分号分隔）。
# 图片入库时若记录了 src_dir 会优先精确查找；历史已入库数据用这里配置的根目录递归按同名匹配。
SOURCE_ROOTS = [p.strip() for p in re.split(r"[,;，；]", os.environ.get("AD_SOURCE_ROOTS", "")) if p.strip()]


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
                    _log(f"已压缩过期日志: {os.path.basename(path)} -> {os.path.basename(gz)}")
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
ADMIN_PASSWORD = os.environ.get("AD_PASSWORD", "admin123")        # 管理员密码（部署前请修改）
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
        json.dump({"admin": ADMIN_PASSWORD, "annotator": ANNOTATOR_PASSWORD}, f, ensure_ascii=False)

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
# 需服务器 transformers>=4.57 且 pip install qwen-vl-utils；加载失败会自动回退 Qwen2-VL-2B
VLM_MODEL_NAME = os.environ.get("AD_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
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
def change_password(request: Request, old_password: str = Form(...), new_password: str = Form(...)):
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
    if YOLO is None or not LOAD_MODELS:
        return False
    try:
        yolo_model = YOLO(YOLO_MODEL_NAME)   # 本地有则用，没有则联网下载
        print("[i] YOLO 已按需加载。")
        return True
    except Exception as e:
        print(f"[!] YOLO 按需加载失败: {e}")
        yolo_model = None
        return False


def _free_yolo():
    """释放 YOLO 占用的显存，让给 DINO/SigLIP。"""
    global yolo_model
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
    if AutoModelForZeroShotObjectDetection is None or AutoProcessor is None or not LOAD_MODELS:
        return False
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    dino_processor = dino_model = None
    try:
        dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME, local_files_only=True)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_MODEL_NAME, torch_dtype=dtype, local_files_only=True).to(DEVICE)
    except Exception:
        try:
            print("[i] 本地无 DINO 权重，尝试联网下载...")
            dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_MODEL_NAME, torch_dtype=dtype).to(DEVICE)
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


def _free_vlm():
    """释放 VLM 占用的显存（Qwen2-VL 与 DINO/YOLO 不共存）。"""
    global vlm_model, vlm_processor, vlm_loaded_name
    if 'vlm_model' in globals() and vlm_model is not None:
        try:
            del vlm_model
        except Exception:
            pass
        vlm_model = None
    if 'vlm_processor' in globals():
        vlm_processor = None
    vlm_loaded_name = ""
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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
class SigLIPImageDataset(Dataset if (not LITE_MODE and Dataset is not None) else object):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

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
    clean = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', (name or "").strip())
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
        return (_np.zeros((n, k), dtype=_np.float32), _np.zeros((n, k), dtype=_np.int64))
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
        "name": p_name, "dir": p_dir, "img_dir": img_dir,
        "idx_path": idx_path, "meta_path": meta_path,
        "index": index, "metadata": metadata
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
    global siglip_model, siglip_processor, dino_model, dino_processor, yolo_model
    print(f"[*] 启动 Maxieye 旗舰 AI 挖掘平台，运行设备: {DEVICE}")
    _log(f"[*] 服务启动，日志目录: {LOG_DIR}")
    _compress_old_logs()  # 启动时先压缩一次过期日志
    _start_compression()  # 后台线程每小时压缩

    _load_passwords()  # 读取持久化密码（若有）

    # 显式初始化默认项目目录（仅启动/新建时创建目录，读路径不建）
    _dp = get_project_paths("default")
    os.makedirs(_dp[2], exist_ok=True)  # images 目录
    load_project_context("default")

    if not LOAD_MODELS:
        print("[i] 测试模式：已跳过模型加载 (AD_LOAD_MODELS=0)。部署时请将 AD_LOAD_MODELS 置为 1 再启用。")
        return

    # ===== 模型加载：生产优先（本地有就加载，没有则联网下载），失败自动跳过 =====
    print(f"[*] [1/3] 装载 Google SigLIP: {SIGLIP_MODEL_NAME} ...")
    siglip_processor = siglip_model = None
    try:
        siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, local_files_only=True)
        siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, local_files_only=True).to(DEVICE)
    except Exception:
        try:
            print("[i] 本地无 SigLIP 权重，尝试联网下载...")
            siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
        except Exception:
            siglip_processor = siglip_model = None
            print("[!] SigLIP 加载/下载失败，已跳过。")
    if siglip_model is not None:
        siglip_model.eval()
        print("[i] SigLIP 已就绪。")

    print("[*] Grounding DINO / Qwen2-VL 已改为【按需懒加载 + FP16】：不再常驻，"
          "仅在首次调用目标检测/VLM 时加载；DINO 与 VLM 互斥，切换时会自动互相释放，避免 12G 显存 OOM。")
    print("🚀 启动完成（生产模式：本地有则加载，没有则下载，失败自动跳过）")

# ----------------- API 路由 -----------------
@app.get("/")
def read_index():
    # 优先服务生产环境正在编辑的前端文件；没有则回退到旧版文件名
    index_html = os.path.join(PROJECT_DIR, "生产.html")
    if not os.path.exists(index_html):
        index_html = os.path.join(PROJECT_DIR, "前端.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {"msg": "前端.html / 生产.html 均不存在"}

@app.get("/api/projects")
def list_projects():
    if not os.path.exists(PROJECTS_ROOT):
        os.makedirs(PROJECTS_ROOT, exist_ok=True)
    items = [d for d in os.listdir(PROJECTS_ROOT) if os.path.isdir(os.path.join(PROJECTS_ROOT, d))]
    # 永远保留 default 项目，避免下拉列表里消失
    if "default" not in items:
        items.insert(0, "default")
    return {"projects": sorted(items)}

@app.post("/api/projects/create")
def create_project(project_name: str = Form(...)):
    clean_name = sanitize_project_name(project_name)
    _log(f"[项目] 新建项目请求: {project_name} -> {clean_name}")
    if not clean_name:
        raise HTTPException(status_code=400, detail="项目名不合法")
    _cp = get_project_paths(clean_name)
    os.makedirs(_cp[2], exist_ok=True)  # 新建时创建 images 目录
    load_project_context(clean_name)
    return {"code": 200, "msg": f"项目 [{clean_name}] 创建成功", "project": clean_name}

@app.post("/api/projects/rename")
@app.post("/api/rename_project")
def rename_project(old_name: str = Form(...), new_name: str = Form(...)):
    """重命名项目。若该项目仍有后台任务在跑则拦截，避免文件夹被占用/数据错乱。
    同时兼容旧路径 /api/projects/rename 与新路径 /api/rename_project。"""
    _log(f"[项目] 重命名请求: {old_name} -> {new_name}")
    clean_old = sanitize_project_name(old_name)
    clean_new = sanitize_project_name(new_name)
    if not clean_old or not clean_new:
        return {"code": 400, "msg": "项目名无效"}

    # ==========================================
    # 🛑 核心保护锁：拦截该项目运行中的后台任务
    # ==========================================
    for _folder_path, task in extract_task_pool.items():
        # 抽帧并发池：属于该项目且正在运行
        if task.get("project") == clean_old and task.get("is_running"):
            return {
                "code": 403,
                "msg": f"拦截重命名！当前项目 [{clean_old}] 有抽帧任务正在后台运行，请等待任务完成或点击【⏹️ 强制中止】后再修改项目名。"
            }
    # 单任务(直导)运行中且属于该项目 → 拦截重命名
    if task_status.get("is_running") and task_status.get("task_type") == "import" and task_status.get("project") == clean_old:
        return {
            "code": 403,
            "msg": f"拦截重命名！当前项目 [{clean_old}] 有直导任务正在后台运行，请等待完成或先中止后再改名。"
        }
    # 单任务(打包)运行中且属于该项目 → 拦截重命名
    if delivery_status.get("is_running") and delivery_status.get("task_type") == "delivery" and delivery_status.get("project") == clean_old:
        return {
            "code": 403,
            "msg": f"拦截重命名！当前项目 [{clean_old}] 有打包任务正在后台运行，请等待完成或先中止后再改名。"
        }

    old_dir = os.path.join(PROJECTS_ROOT, clean_old)
    new_dir = os.path.join(PROJECTS_ROOT, clean_new)

    if not os.path.exists(old_dir):
        return {"code": 404, "msg": "原项目不存在"}
    if os.path.exists(new_dir):
        return {"code": 400, "msg": "新项目名称已存在，请换一个"}

    if clean_old in project_cache:
        del project_cache[clean_old]

    try:
        # 1. 移动磁盘物理目录（shutil.move 跨盘更稳）
        shutil.move(old_dir, new_dir)

        # 2. 同步修复新目录 metadata.json 里的 path / url，避免改名后旧路径失效
        new_meta_path = os.path.join(new_dir, "metadata.json")
        if os.path.exists(new_meta_path):
            try:
                with open(new_meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                new_img_dir = os.path.join(new_dir, "images")
                for item in metadata:
                    fn = item.get("filename") or os.path.basename(item.get("path") or "")
                    if fn:
                        # 在移动后的 images 目录里递归定位实际文件，兼容平铺/子目录结构
                        found = _find_file_by_name(new_img_dir, fn)
                        item["path"] = found or os.path.join(new_img_dir, fn)
                    item["url"] = f"/api/image/{clean_new}/{item.get('id')}"
                if metadata:
                    with open(new_meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False)
            except Exception as me:
                _log(f"[项目] 重命名后修复 metadata 异常: {me}")

        # 3. 加载新项目上下文（含修复后的 metadata）
        load_project_context(clean_new)
        return {"code": 200, "msg": f"项目已成功重命名为 [{clean_new}]", "project": clean_new}
    except Exception as e:
        return {"code": 500, "msg": f"重命名失败: {str(e)}"}

@app.post("/api/projects/delete")
def delete_project(project_name: str = Form(...)):
    clean_name = sanitize_project_name(project_name)
    _log(f"[项目] 删除项目请求: {clean_name}")
    if clean_name == "default":
        return {"code": 400, "msg": "default 是系统保留的默认项目，无法删除，您可以通过清理数据来清空它。"}
        
    p_dir = os.path.join(PROJECTS_ROOT, clean_name)
    if os.path.exists(p_dir):
        try:
            shutil.rmtree(p_dir)
        except Exception as e:
            return {"code": 500, "msg": f"磁盘删除失败: {str(e)}"}
            
    if clean_name in project_cache:
        del project_cache[clean_name]
        
    return {"code": 200, "msg": f"项目 [{clean_name}] 及其所有底层数据已彻底粉碎"}

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
    raw_images = [f for f in os.listdir(ctx["img_dir"]) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    raw_count = len(raw_images)
    processed_count = ctx["index"].ntotal if ctx["index"] is not None else 0
    return {
        "project": ctx["name"],
        "raw_count": raw_count,
        "processed_count": processed_count,
        "pending_count": max(0, raw_count - processed_count)
    }

def extract_and_index_project(ctx, image_paths: List[str]):
    """带项目级写锁的向量化入库：同一项目并发写底库时串行化，避免 metadata/faiss 竞争；跨项目并行。"""
    with _project_lock(ctx["name"]):
        return _extract_and_index_unlocked(ctx, image_paths)


def _extract_and_index_unlocked(ctx, image_paths: List[str]):
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
            mask = (valids == 1)
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
                extracted_records.append({
                    "id": current_id,
                    "filename": os.path.basename(p),
                    "path": p,
                    "url": f"/api/image/{ctx['name']}/{current_id}"
                })

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
        existing_filenames = set([os.path.basename(m.get("path", "")) for m in ctx["metadata"] if "path" in m])
        new_saved_paths = []

        # 兼容处理单路径或多个逗号分隔的路径
        paths = [p.strip() for p in source_path.split(',') if p.strip()]
        src_map = {}   # 文件名 -> 原始源目录(伴生文件所在)

        cancelled = False
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
                    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp'] or ext.replace('.', '').isdigit():
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
                            src_map[f] = root   # 伴生文件与源 jpg 同目录
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
            task_status["msg"] = f"⚠️ 直导已手动中止！已登记 {len(new_saved_paths)} 张未向量化图片。"
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
                            rec["src_dir"] = src_map[fn]
                    save_project_context(ctx)

        task_status["processed_count"] = processed_count
        task_status["msg"] = f"全自动导入与特征向量化已全部完成！本次向量化 {processed_count} 张新图，库内共 {ctx['index'].ntotal} 张"
    except Exception as e:
        task_status["msg"] = f"后台处理异常: {str(e)}"
        _log("✖ 直导异常: " + str(e))
    finally:
        task_status["is_running"] = False
        _log("✔ 直导结束: " + task_status["msg"])


# 🌟 极速直导接口：秒回响应，绝不卡死前端
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
    return {"code": 200, "msg": f"已清除 {len(finished)} 条已完成任务记录，剩余运行中 {running} 个"}


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
        files.append({
            "name": os.path.basename(p),
            "size": os.path.getsize(p),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))),
        })
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
        entry = extract_task_pool.setdefault(path, {
            "project": project,
            "task_type": "extract",
            "is_running": False,
            "is_cancelled": False,   # 刹车信号
            "msg": "",
            "current_path": path,
            "processed_count": 0,
            "total_count": 0,
        })
        if not entry.get("project"):
            entry["project"] = project
        entry.setdefault("is_cancelled", False)   # 兼容历史记录
        return entry


def _video_entry(video_path, project):
    """取/建某视频对应的处理任务记录（并入抽帧并发池，便于大盘展示/急停/并发）。"""
    with _extract_lock:
        entry = extract_task_pool.setdefault(video_path, {
            "project": project,
            "task_type": "video",
            "is_running": False,
            "is_cancelled": False,
            "msg": "",
            "current_path": video_path,
            "processed_count": 0,
            "total_count": 0,
        })
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
        part = jobs[c0:c0 + chunk]
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


def _run_extract_one(folder_path: str, frame_step: int, project: str, vectorize: bool, unit: str):
    """执行单条路径的稀疏采样抽帧，返回结果字符串。进度写入该路径对应的并发池记录。"""
    folder_path = _to_nas_local(folder_path)  # 自动把 UNC 路径转成服务器本地挂载路径
    entry = _extract_entry(folder_path, project)
    entry["current_path"] = folder_path
    entry["processed_count"] = 0
    entry["total_count"] = 0
    entry["is_cancelled"] = False   # 新任务开始前清除急停信号
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
            # 按帧数间隔：从第 frame_step 个起，每隔 frame_step 抽 1 张
            selected = [jpg_files[i] for i in range(frame_step, len(jpg_files), frame_step + 1)]
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

        # 4) 并发拷贝一份到 sampled_frames（预览目录，始终保留，供人工查看抽出的帧）
        target_dir = os.path.join(folder_path, "sampled_frames")
        os.makedirs(target_dir, exist_ok=True)
        entry["total_count"] = len(selected)
        preview_jobs = [(p, os.path.join(target_dir, os.path.basename(p))) for p in selected]
        _parallel_copy_files(preview_jobs, entry)

        # 5) 自动入底库向量化（再并发拷一份进项目库；sampled_frames=预览，images=入库/交付）
        if vectorize:
            if entry.get("is_cancelled"):
                entry["msg"] = "⚠️ 任务已手动中止！sampled_frames 预览已保留已抽帧。"
                return "⚠️ 任务已手动中止！sampled_frames 预览已保留已抽帧。"
            ctx = load_project_context(project)
            existing_filenames = set([os.path.basename(m.get("path", "")) for m in ctx["metadata"] if "path" in m])
            jobs = []
            src_map = {}
            for p in selected:
                fn = os.path.basename(p)
                if fn in existing_filenames:
                    continue
                dst = os.path.join(ctx["img_dir"], fn)
                jobs.append((p, dst))
                src_map[fn] = os.path.dirname(p)   # 伴生文件与源 jpg 同目录
                existing_filenames.add(fn)
            new_saved_paths = _parallel_copy_files(jobs, entry)
            if entry.get("is_cancelled"):
                entry["msg"] = f"⚠️ 任务已手动中止！保留已入 {len(new_saved_paths)} 张图片。"
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
            return f"✅ [{project}] 已抽 {len(selected)} 帧至 sampled_frames/ 预览，并入库向量化 {vec} 张（库内共 {ctx['index'].ntotal} 张）"

        # 非 vectorize：只出 sampled_frames 预览
        if entry.get("is_cancelled"):
            entry["msg"] = "⚠️ 任务已手动中止！sampled_frames 预览已保留已抽帧。"
            return "⚠️ 任务已手动中止！sampled_frames 预览已保留已抽帧。"
        if unit == "count":
            return f"✅ [{project}] 共 {len(jpg_files)} 帧，按每 {frame_step} 帧抽 1 帧，共抽 {len(selected)} 帧至 sampled_frames/"
        return f"✅ [{project}] 共 {len(jpg_files)} 帧，按 {frame_step}s 时间戳间隔抽取 {len(selected)} 帧至 sampled_frames/"
    except Exception as e:
        _log("✖ 抽帧异常: " + str(e))
        return f"❌ 抽帧异常: {str(e)}"


def background_extract_manager(jobs, frame_step: int, vectorize: bool, unit: str):
    """后台抽帧总调度：各条路径写入独立并发池记录，不同目录可并行（同一目录由接口层拦截重复）。
    进度以 extract_task_pool 各条目为准（前端大盘读取），不再占用单槽 task_status，
    从而可与打包/直导等任务并发执行，避免误判"后台正忙"。"""
    _log(f"▶ 抽帧任务开始: 共 {len(jobs)} 条路径 (间隔:{frame_step},{unit})")
    results = []
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
        results.append(res)
        entry["msg"] = res
        entry["is_running"] = False
    _log("✔ 抽帧结束: " + " | ".join(results))


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
            entry = extract_task_pool.get(raw)   # 原始键容错
    if entry and entry.get("is_running"):
        entry["is_cancelled"] = True
        entry["msg"] = "正在中止任务，等待当前帧落盘..."
        _log(f"[任务] 已发送抽帧中止信号: {key}")
        return {"code": 200, "msg": "中止信号已发送"}

    # 2) 直导(import)：按类型前缀/路径/项目匹配即视为命中
    if task_status.get("is_running") and task_status.get("task_type") == "import":
        cur = task_status.get("current_path") or task_status.get("project") or ""
        if (raw.startswith("import:")) or raw == cur or raw == task_status.get("project", ""):
            task_status["is_cancelled"] = True
            task_status["msg"] = "正在中止任务，等待当前步骤落盘..."
            _log(f"[任务] 已发送直导中止信号: {raw}")
            return {"code": 200, "msg": "中止信号已发送"}
    # 3) 打包(delivery)：独立单槽单独判定
    if delivery_status.get("is_running") and delivery_status.get("task_type") == "delivery":
        cur = delivery_status.get("current_path") or delivery_status.get("project") or ""
        if (raw.startswith("delivery:")) or raw == cur or raw == delivery_status.get("project", ""):
            delivery_status["is_cancelled"] = True
            delivery_status["msg"] = "正在中止任务，等待当前步骤落盘..."
            _log(f"[任务] 已发送打包中止信号: {raw}")
            return {"code": 200, "msg": "中止信号已发送"}

    return {"code": 400, "msg": "任务不存在或已结束"}


# ================= 视频抽帧（后台任务，并入并发池） =================
# 支持的视频格式（与图片类似，可按目录递归扫描）
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v",
              ".mpg", ".mpeg", ".3gp", ".ts", ".m2ts", ".rmvb", ".rm", ".vob")


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


def _extract_video_frames(ctx, video_path: str, step: int, unit: str, entry, existing_filenames, new_saved_paths, name_prefix: str):
    """对单个视频按「帧/秒」间隔稀疏抽帧，写入项目底库目录，返回本视频实际保存的帧数。
    unit == 'count' → 每隔 step 帧抽 1 帧；unit == 'time' → 每隔 step 秒抽 1 帧。
    急停由 entry['is_cancelled'] 触发；命名带 name_prefix+时间戳 避免跨视频/与底库冲突。"""
    if step <= 0:
        step = 1
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("无法打开视频文件（可能不是有效视频或编码不受支持）: " + os.path.basename(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    if unit == "time":
        # 每隔 step 秒抽 1 帧 → 换算成帧间隔
        frame_interval = max(1, int(round(step * video_fps)))
    else:
        # 每隔 step 帧抽 1 帧
        frame_interval = max(1, step)

    saved = 0
    count = 0
    while cap.isOpened():
        # 🛑 刹车点：可被大盘急停
        if entry.get("is_cancelled"):
            break
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            # 用高精度时间戳+序号命名，避免与底库已有文件/跨视频冲突
            fn = f"{name_prefix}_{int(time.time() * 1000)}_{saved:06d}.jpg"
            if fn not in existing_filenames:
                dst = os.path.join(ctx["img_dir"], fn)
                try:
                    cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if os.path.exists(dst):
                        new_saved_paths.append(dst)
                        existing_filenames.add(fn)
                        saved += 1
                except Exception:
                    pass
        count += 1
    cap.release()
    return saved


def _video_interval_desc(step: int, unit: str) -> str:
    """把 步长+单位 转成可读描述，如 每隔 1 秒抽1帧 / 每隔 5 帧抽1帧。"""
    if unit == "time":
        return f"每隔 {step} 秒抽 1 帧"
    return f"每隔 {step} 帧抽 1 帧"


def _run_video_list(entry, project: str, videos, step: int, unit: str, kind: str):
    """对一组视频（单文件或目录递归收集）逐个按「帧/秒」间隔抽帧，统一写入项目底库并自动向量化。
    entry 需已由调用方置为 is_running。"""
    if not videos:
        entry["msg"] = "未找到任何视频文件"
        return
    ctx = load_project_context(project)
    existing_filenames = set(os.path.basename(m.get("path", "")) for m in ctx["metadata"] if m.get("path"))
    new_saved_paths = []
    entry["total_count"] = len(videos)

    total_saved = 0
    cancelled = False
    for idx, vp in enumerate(videos, 1):
        if entry.get("is_cancelled"):
            cancelled = True
            break
        entry["current_path"] = vp
        entry["msg"] = f"[{idx}/{len(videos)}] 正在抽帧: {os.path.basename(vp)}"
        try:
            saved = _extract_video_frames(ctx, vp, step, unit, entry, existing_filenames, new_saved_paths, f"v{idx}")
            total_saved += saved
        except Exception as e:
            _log(f"✖ 单个视频抽帧失败 {vp}: {str(e)}")
            entry["msg"] = f"[{idx}/{len(videos)}] {os.path.basename(vp)} 抽帧失败: {str(e)}"
        entry["processed_count"] = idx

    vec = 0
    if new_saved_paths:
        entry["msg"] = f"共从 {len(videos)} 个视频中抽帧 {total_saved} 张，开始执行 GPU 向量化..."
        vec = extract_and_index_project(ctx, new_saved_paths)
        save_project_context(ctx)

    if cancelled:
        entry["msg"] = f"⚠️ 任务已手动中止！已处理 {entry['processed_count']}/{len(videos)} 个视频，抽帧 {total_saved} 张，向量化入库 {vec} 张。"
    else:
        entry["msg"] = f"✅ 视频抽帧与特征入库完成：{kind}共扫描 {len(videos)} 个视频，抽帧 {total_saved} 张，向量化 {vec} 张，库内共 {ctx['index'].ntotal} 张"


def _run_video_job(entry, project: str, local: str, step: int, unit: str):
    """对一条提交路径处理：目录→递归收集全部视频；单个视频文件→仅该文件。"""
    if cv2 is None:
        entry["msg"] = "视频处理失败：服务器未安装 opencv-python (cv2)"
        return
    if os.path.isdir(local):
        entry["msg"] = f"正在递归扫描目录内视频文件 ({_video_interval_desc(step, unit)})..."
        _run_video_list(entry, project, _collect_videos(local), step, unit, "目录")
    elif os.path.isfile(local) and _is_video_file(local):
        entry["msg"] = f"正在解析视频并稀疏抽帧 ({_video_interval_desc(step, unit)})..."
        _run_video_list(entry, project, [local], step, unit, "单视频")
    else:
        entry["msg"] = f"❌ 路径无效（不是目录也不是视频文件）: {local}"


def background_video_manager(jobs, step: int, unit: str):
    """后台视频抽帧总调度：像图片抽帧一样支持多路径（可含 项目名::/路径 归入不同项目），
    各条路径写入独立并发池记录；每条路径可被大盘【⏹️ 强制中止】。"""
    for idx, (path, proj) in enumerate(jobs, 1):
        local = _to_nas_local(path)
        entry = _video_entry(local, proj)
        entry["is_running"] = True
        entry["is_cancelled"] = False
        entry["processed_count"] = 0
        entry["total_count"] = 0
        entry["current_path"] = local
        entry["msg"] = f"[{idx}/{len(jobs)}] 正在处理 第{idx}条 [{proj}] {path}..."
        _log(f"[视频] 批次 {idx}/{len(jobs)} 项目={proj} 路径={path} step={step} unit={unit}")
        try:
            _run_video_job(entry, proj, local, step, unit)
        except Exception as e:
            entry["msg"] = f"视频处理出错: {str(e)}"
            _log("✖ 视频处理异常: " + str(e))
        finally:
            entry["is_running"] = False


@app.post("/api/process_video")
async def process_video(
    project: str = Form(default="default"),
    video_path: str = Form(...),
    frame_step: int = Form(default=1),
    unit: str = Form(default="time"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    if cv2 is None:
        return {"code": 400, "msg": "服务器未安装 opencv-python (cv2)，无法处理视频"}
    # 与图片抽帧一致：video_path 可含多条路径（换行/逗号/分号分隔，可用 项目名::/路径 指定归属项目）
    jobs = _parse_extract_jobs(video_path, project)
    if not jobs:
        return {"code": 400, "msg": "未解析到有效的视频路径"}

    # 并发控制：同一路径正在处理时拦截；不同路径可并发
    for path, proj in jobs:
        entry = _video_entry(_to_nas_local(path), proj)
        if entry.get("is_running"):
            return {"code": 400, "msg": f"该路径的视频任务正在运行中，请勿重复提交: {path}"}

    for path, proj in jobs:
        _video_entry(_to_nas_local(path), proj)["is_running"] = True
    background_tasks.add_task(background_video_manager, jobs, frame_step, unit)
    _log(f"[视频] 收到视频抽帧请求 路径数={len(jobs)} step={frame_step} unit={unit} 默认项目={project}")
    return {"code": 200, "msg": f"视频抽帧任务已启动：共 {len(jobs)} 条路径，{_video_interval_desc(frame_step, unit)}，帧将抽入对应项目"}


# ================= 三件套交付打包（后台任务） =================
class ExportDeliveryRequest(BaseModel):
    project: str
    selected_image_names: List[str]
    extensions: List[str] = [".bin", ".xml"]
    export_dir_name: str = "delivery_package"
    output_dir: Optional[str] = None   # 用户指定的输出目录（可填服务器/NAS路径，支持 UNC）
    source_roots: Optional[List[str]] = None   # 手动指定伴生文件源根目录(历史数据兜底)


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
        needed = {}   # 伴生文件名 -> 找到的绝对路径(未找到为 None)
        for img_name in req.selected_image_names:
            base = os.path.splitext(img_name)[0]
            rec = meta_by_fn.get(img_name) or {}
            src_dir = rec.get("src_dir")
            # 若文件名带 _raw，先去 _raw 再找关联文件（与挑图工具一致）
            match_base = base.replace('_raw', '') if '_raw' in os.path.basename(base).lower() else base
            ext_names = [f"{match_base}{ext}" for ext in req.extensions]
            items.append({"img_name": img_name, "src_dir": src_dir, "ext_names": ext_names})
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
            delivery_status["msg"] = f"⚠️ 打包已手动中止！已拷贝 {copied_count} 个文件，缺失 {missing_count} 个。"
            _log("✔ 打包被手动中止: " + delivery_status["msg"])
        else:
            delivery_status["msg"] = f"✅ 交付包打包完成！共拷贝 {copied_count} 个文件，缺失 {missing_count} 个，输出: {output_dir}"
            _log("✔ 打包结束: " + delivery_status["msg"])
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
            return {"code": 400, "msg": f"该目录的抽帧任务正在运行中，请勿重复提交: {path}"}

    # 先同步置为运行中（立即拦截重复提交），再丢入后台线程池
    for path, proj in jobs:
        _extract_entry(_to_nas_local(path), proj)["is_running"] = True
    background_tasks.add_task(background_extract_manager, jobs, frame_step, bool(vectorize), unit)
    _log(f"[抽帧] 收到抽帧请求 路径数={len(jobs)} step={frame_step} unit={unit} vectorize={vectorize} 默认项目={project}")
    return {"code": 200, "msg": f"稀疏抽帧任务已启动：共 {len(jobs)} 条路径，间隔单位: {'时间戳(秒)' if unit == 'time' else '帧数'}"}


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
    _log(f"[打包] 收到打包请求 project={req.project} frames={len(req.selected_image_names)} exts={req.extensions}")
    return {"code": 200, "msg": f"正在后台自动打包 {len(req.selected_image_names)} 帧数据及其伴生文件！"}


@app.post("/api/upload_batch")
async def upload_batch(project: str = Form("default"), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="未接收到文件")

    ctx = load_project_context(project)
    existing_filenames = set([os.path.basename(m.get("path", "")) for m in ctx["metadata"] if "path" in m])
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
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) and not member.startswith('__MACOSX'):
                            fn = os.path.basename(member)
                            if fn and fn not in existing_filenames:
                                ext_path = os.path.join(ctx["img_dir"], fn)
                                with zip_ref.open(member) as s, open(ext_path, "wb") as t:
                                    shutil.copyfileobj(s, t)
                                new_saved_paths.append(ext_path)
                                existing_filenames.add(fn)
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        elif safe_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            if safe_name not in existing_filenames:
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                new_saved_paths.append(file_path)
                existing_filenames.add(safe_name)

    processed_count = 0
    if new_saved_paths:
        processed_count = extract_and_index_project(ctx, new_saved_paths)
    _log(f"[上传] 上传批次完成，新增并向量化 {processed_count} 张，库内 {ctx['index'].ntotal} 张")

    return {
        "code": 200,
        "msg": f"本次新增并向量化 {processed_count} 张图片",
        "processed_count": ctx["index"].ntotal
    }

@app.post("/api/build_index_online")
async def build_index_online(project: str = Query("default")):
    _log(f"[向量化] 触发在线向量化 project={project}")
    ctx = load_project_context(project)
    all_imgs = [os.path.join(ctx["img_dir"], f) for f in os.listdir(ctx["img_dir"]) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    indexed_paths = set([m['path'] for m in ctx["metadata"] if 'path' in m])
    unprocessed_paths = [p for p in all_imgs if p not in indexed_paths]

    if not unprocessed_paths:
        return {"code": 200, "msg": "该项目所有图片均已完成向量化"}

    processed_count = extract_and_index_project(ctx, unprocessed_paths)
    return {"code": 200, "msg": f"成功向量化 {processed_count} 张图片！"}

# ----------------- 检索路由 -----------------
@app.get("/api/search")
def search_text(project: str = Query("default"), query: str = Query(..., min_length=1), top_k: int = 500):
    _log(f"[检索] 文本检索 project={project} query={query} top_k={top_k}")
    ctx = load_project_context(project)
    if ctx["index"] is None or ctx["index"].ntotal == 0:
        return {"code": 200, "results": [], "total": 0}

    inputs = siglip_processor(text=[query], return_tensors="pt", padding="max_length", max_length=64).to(DEVICE)
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
    results = [m for m in ctx["metadata"] if filename_query.lower() in m.get("filename", "").lower()]
    return {"code": 200, "results": results}

@app.post("/api/search_by_external_image")
async def search_by_external_image(project: str = Form("default"), files: List[UploadFile] = File(...), top_k: int = 500):
    _log(f"[检索] 以图搜图 project={project} 参考图={len(files)} 张")
    ctx = load_project_context(project)
    if ctx["index"] is None or ctx["index"].ntotal == 0 or not files:
        return {"code": 200, "results": []}

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

    sorted_indices = sorted(max_scores.keys(), key=lambda k: max_scores[k], reverse=True)
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
        "code": 200, "total": total, "total_pages": total_pages, "current_page": page,
        "items": items
    }

# ----------------- 数据清洗与标注推理 -----------------
@app.post("/api/dedup_stats")
def dedup_stats(project: str = Form("default"), sim_threshold: float = Form(0.95)):
    _log(f"[降重] 特征矩阵扫描 project={project} threshold={sim_threshold}")
    ctx = load_project_context(project)
    total = len(ctx["metadata"])
    if total < 2 or ctx["index"] is None or LITE_MODE:
        return {"total_images": total, "unique_images": total, "duplicate_count": 0, "dedup_rate": "0.0%", "clusters": []}

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
        "total_images": total, "unique_images": unique_images,
        "duplicate_count": duplicate_count, "dedup_rate": rate,
        "clusters": clusters
    }

@app.post("/api/delete_and_sync")
def delete_and_sync(project: str = Form("default"), image_ids: str = Form(...)):
    _log(f"[降重] 永久粉碎冗余 project={project} ids={image_ids}")
    ctx = load_project_context(project)
    del_ids = set([int(x) for x in image_ids.split(",") if x.strip().isdigit()])
    if not del_ids:
        return {"deleted_count": 0}

    remaining_paths = []
    deleted_count = 0

    for m in ctx["metadata"]:
        if m["id"] in del_ids:
            if os.path.exists(m["path"]):
                try:
                    os.remove(m["path"])
                except Exception:
                    pass
            deleted_count += 1
        else:
            remaining_paths.append(m["path"])

    with _project_lock(ctx["name"]):
        ctx["metadata"] = []
        ctx["index"] = faiss.IndexFlatIP(FEAT_DIM) if (not LITE_MODE and faiss is not None) else _FakeIndex(FEAT_DIM)
        extract_and_index_project(ctx, remaining_paths)

    return {"code": 200, "deleted_count": deleted_count}

@app.post("/api/ground_detect")
def ground_detect(project: str = Form("default"), image_id: int = Form(...), text_prompt: str = Form(...)):
    """Grounding DINO 真实目标级开放词汇推理（支持中英文自动转译）"""
    _log(f"[检测] DINO project={project} image_id={image_id} prompt={text_prompt}")
    global dino_model, dino_processor
    ctx = load_project_context(project)
    # DINO 与 VLM/YOLO 不共存：先腾显存，再按需加载(FP16)
    _free_vlm(); _free_yolo()
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
        if not prompt.endswith('.'):
            prompt += '.'

        print(f"[*] DINO 原始提示词: [{text_prompt}] ---> 处理后: [{prompt}]")

        inputs = dino_processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = dino_model(**inputs)

        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(h, w)]
        )[0]

        # 将英文标签翻译回中文（词典只维护后端这一份）
        dino_reverse = {en: zh for zh, en in dino_dict.items()}
        labels = [dino_reverse.get(lbl, lbl) for lbl in results["labels"]]

        res = {
            "image_id": image_id,
            "engine": "dino",
            "prompt": (text_prompt or "").strip(),   # 记录所用提示词，供前端判断缓存是否需失效
            "scores": results["scores"].cpu().numpy().tolist(),
            "labels": labels,
            "boxes": results["boxes"].cpu().numpy().tolist(),
            "width": w, "height": h
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
    _free_dino(); _free_vlm()
    if not _ensure_yolo():   # 按需懒加载
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
            "width": w, "height": h
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
                arr = np.ascontiguousarray(arr[:, :, ::-1])   # RGB -> BGR
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
def yolo_detect_batch(project: str = Form("default"), image_ids: str = Form(...),
                      batch_size: int = Form(16), fp16: str = Form("1")):
    """YOLO 批量推理：一次提交一批图片，由 ultralytics 自动分批切到 GPU，榨干显存/算力。
    单张单发只有 ~10% 占用，瓶颈在『喂太慢』；批量后占用率与吞吐都会大幅提升。
    返回 {str(image_id): res}，并一次性落盘持久化。"""
    _log(f"[检测] YOLO-BATCH project={project} ids={image_ids}")
    global yolo_model
    ctx = load_project_context(project)
    # YOLO 不与 DINO/VLM 共存：先释放它们，腾出 12G 显存再按需加载
    _free_dino(); _free_vlm()
    if not _ensure_yolo():   # 按需懒加载
        return {"code": 500, "msg": "YOLO 模型加载失败/未启用 (AD_LOAD_MODELS=0?)"}

    ids = []
    for s in image_ids.replace(" ", "").split(","):
        if s == "":
            continue
        try:
            ids.append(int(s))
        except Exception:
            pass
    # 只保留底库里真实存在的帧，避免越界
    valid = [(i, ctx["metadata"][i]["path"]) for i in ids if 0 <= i < len(ctx["metadata"])]
    if not valid:
        return {"code": 200, "results": {}, "count": 0}

    paths = [p for _, p in valid]
    idxs = [i for i, _ in valid]

    # 并行解码，避免 CPU 串行读图成为 GPU 瓶颈；失败帧过滤掉
    items = _preload_bgr(paths, workers=8)          # [(bgr,w,h) | None, ...]
    loaded_idx = [idxs[k] for k in range(len(idxs)) if items[k] is not None]
    loaded_arr = [items[k][0] for k in range(len(idxs)) if items[k] is not None]
    if not loaded_arr:
        return {"code": 200, "results": {}, "count": 0}
    sizes = {idxs[k]: (items[k][1], items[k][2]) for k in range(len(idxs)) if items[k] is not None}

    bs = max(1, min(int(batch_size) or 16, len(loaded_arr)))

    # 固定 imgsz 下让 cuDNN 自动挑选最优卷积算法（同分辨率批量时提速明显）
    if DEVICE == "cuda" and torch is not None:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    try:
        results = yolo_model(
            loaded_arr,         # 传预解码 BGR 数组，直接分批上 GPU（不再让 ultralytics 串行读盘）
            batch=bs,           # 一批多少张
            imgsz=640,          # 固定分辨率 → 触发 cudnn.benchmark 最优内核
            device=DEVICE,
            verbose=False,
            half=((str(fp16) == "1") and DEVICE == "cuda"),  # FP16：4070 Ti Tensor Core 约 2x（仅 CUDA）
        )
    except Exception as e:
        print(f"[!] YOLO 批量推理异常: {e}")
        return {"code": 500, "msg": f"YOLO 批量推理异常: {e}"}

    out = {}
    with _detections_lock:
        for i, idx in enumerate(loaded_idx):
            try:
                r = results[i]
                if r is None or r.boxes is None:
                    continue
                w, h = sizes.get(idx, (1920, 1080))
                cls_indices = r.boxes.cls.cpu().numpy().astype(int).tolist()
                res = {
                    "image_id": idx,
                    "engine": "yolo",
                    "scores": r.boxes.conf.cpu().numpy().tolist(),
                    "labels": [r.names[c] for c in cls_indices],
                    "boxes": r.boxes.xyxy.cpu().numpy().tolist(),
                    "width": w, "height": h
                }
                sub = detections_cache.setdefault(project or "default", {})
                sub[str(idx)] = res
                out[str(idx)] = res
            except Exception:
                continue
        _persist_detections()  # 批量一次落盘，避免逐张反复写文件
    return {"code": 200, "results": out, "count": len(out)}


@app.post("/api/dino_detect_batch")
def dino_detect_batch(project: str = Form("default"), image_ids: str = Form(...),
                      text_prompt: str = Form(...)):
    """Grounding DINO 满载批处理：一次喂多张图 + 单条提示词，FP16 加速，中英自动互译并回译。
    单张单发极慢，批量后可显著提升 GPU 占用与吞吐。按项目隔离并逐帧落盘。"""
    _log(f"[检测] DINO-BATCH project={project} ids={image_ids} prompt={text_prompt}")
    global dino_model, dino_processor
    ctx = load_project_context(project)
    # DINO 与 VLM/YOLO 不共存：先腾显存，再按需加载(FP16)
    _free_vlm(); _free_yolo()
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
    dino_prompt_key = prompt   # 记录原始提示词，供前端判断缓存是否需失效
    sorted_keys = sorted(dino_dict.keys(), key=len, reverse=True)
    for zh in sorted_keys:
        prompt = prompt.replace(zh, dino_dict[zh])
    if not prompt.endswith('.'):
        prompt += '.'
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
    items = _preload_rgb([p for _, p in candidates], workers=8)   # [(rgb,w,h) | None, ...]
    rgb_imgs = []
    target_sizes = []      # 用「原图」尺寸，把模型输出框映射回原分辨率
    valid = []             # (image_id, orig_w, orig_h)
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
            "prompt": dino_prompt_key,   # 记录所用提示词，供前端判断缓存是否需失效
            "scores": pr["scores"].cpu().numpy().tolist(),
            "labels": labels,
            "boxes": pr["boxes"].cpu().numpy().tolist(),
            "width": w, "height": h
        }
        save_detection_record(project, img_id, res)
        return res

    def _run_batch(batch_idx):
        """跑一个子批，返回写入的 out；OOM 返回 None（调用方回退单张）。"""
        chunk_rgb = [feed_imgs[i] for i in batch_idx]
        chunk_ts = [target_sizes[i] for i in batch_idx]
        inputs = dino_processor(images=chunk_rgb, text=[prompt] * len(chunk_rgb), return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = dino_model(**inputs)
            else:
                outputs = dino_model(**inputs)
        prs = dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.20, text_threshold=0.20, target_sizes=chunk_ts)
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
                try: torch.cuda.empty_cache()
                except Exception: pass
            return False

    out = {}
    DINO_SUB_BATCH = 4     # 12G 卡谨慎：先试 4，OOM 自动逐张退
    try:
        for c0 in range(0, len(feed_imgs), DINO_SUB_BATCH):
            batch_idx = list(range(c0, min(c0 + DINO_SUB_BATCH, len(feed_imgs))))
            try:
                _run_batch(batch_idx)
            except torch.cuda.OutOfMemoryError:
                if DEVICE == "cuda":
                    try: torch.cuda.empty_cache()
                    except Exception: pass
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
vlm_loaded_name = ""   # 实际加载成功的 VLM 模型名（用于确认是否 3B / 回退 2B）

def _try_load_vlm(name: str, dtype):
    """按服务器 transformers 版本兼容加载 Qwen2.5-VL；失败自动回退 Qwen2-VL-2B。"""
    from transformers import AutoProcessor
    # 1) 优先 Qwen2.5-VL（需要 transformers>=4.57）
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
        model = _ModelCls.from_pretrained(name, torch_dtype=dtype, device_map="auto")
        try:
            proc = AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        except Exception:
            proc = AutoProcessor.from_pretrained(name)
        return model, proc, name
    except Exception as _e:
        print(f"[!] Qwen2.5-VL 加载失败({_e})，自动回退 Qwen2-VL-2B...")
        from transformers import Qwen2VLForConditionalGeneration
        fallback = "Qwen/Qwen2-VL-2B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(fallback, torch_dtype=dtype, device_map="auto")
        proc = AutoProcessor.from_pretrained(fallback)
        return model, proc, fallback


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
        print(f"[✓] VLM 场景判定引擎就绪 ({used})")

@app.post("/api/vlm_analyze")
def vlm_analyze_scene(project: str = Form("default"), image_id: int = Form(...)):
    """
    第五层 VLM 综合场景判定：对指定图片进行多模态理解，并自动生成驾驶场景标签
    """
    global vlm_model, vlm_processor, vlm_loaded_name
    ctx = load_project_context(project)
    _log(f"[VLM] 场景判定 model={vlm_loaded_name or '(未加载)'} project={project} image_id={image_id}")
    if not (0 <= image_id < len(ctx["metadata"])):
        return {"code": 400, "msg": "图片ID不存在"}

    # DINO/YOLO 不与 VLM 共存：加载 VLM 前先把它们释放，腾出 12G 显存
    _free_dino(); _free_yolo()
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
                    {"type": "text", "text":
                        "你是一名专业的自动驾驶场景分析专家。请仔细观察这张车载前视摄像头画面，只输出场景标签词，不要任何解释、编号或多余标点。\n"
                        "请尽量从以下维度判断：\n"
                        "1) 天气与光照：白天/夜晚/黄昏/雨天/雪天/雾天/阴天/强逆光\n"
                        "2) 道路与环境：高速公路/城市道路/乡村/隧道/高架桥/匝道/施工区\n"
                        "3) 路面状况：干燥/湿滑/积水/结冰/破损\n"
                        "4) 关键目标：前方大货车/小轿车/行人/锥桶/水马/护栏/红绿灯/限速牌/路灯\n"
                        "5) 风险与驾驶状态：近距离跟车/变道/拥堵/危险/畅通\n"
                        "综合输出 3~6 个词，用中文逗号分隔。例如：夜晚,高速公路,干燥,前方大货车,近距离跟车"}
                ],
            }
        ]

        text_prompt = vlm_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = vlm_processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=80, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 解析 VLM 返回的标签字符串并自动写入该图片的标签库
        tags = [t.strip() for t in output_text.replace("，", ",").split(",") if t.strip()]

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
            "tags": item["userTags"]
        }
    except Exception as e:
        return {"code": 500, "msg": f"VLM 推理异常: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # 生产部署：可用环境变量覆盖，默认监听 0.0.0.0:8009
    # 注意：本应用有共享的内存态（task_status / 后台任务），必须单进程 workers=1
    host = os.environ.get("AD_HOST", "0.0.0.0")
    port = int(os.environ.get("AD_PORT", "8009"))
    print(f"[i] 启动服务: http://{host}:{port}  (日志目录: {LOG_DIR})")
    _log(f"服务启动 http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)