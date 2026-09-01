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
from typing import List, Optional
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

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
# 默认策略（最稳健，部署零配置）：
#   - 服务器（依赖齐全，非轻量模式）→ 自动加载模型，无需任何环境变量
#   - 本地轻量模式（缺 torch 等依赖）→ 自动跳过，仅界面测试
# 也可用环境变量 AD_LOAD_MODELS 强制指定：1=加载模型 0=不加载（快速测试用）
LOAD_MODELS = (os.environ.get("AD_LOAD_MODELS", "1" if not LITE_MODE else "0")) == "1"

# 旗舰高配模型设置
SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
DINO_MODEL_NAME = "IDEA-Research/grounding-dino-base"
YOLO_MODEL_NAME = os.path.join(PROJECT_DIR, "yolov8x.pt")
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
    os.makedirs(img_dir, exist_ok=True)
    return p_name, p_dir, img_dir, idx_path, meta_path

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

    _load_passwords()  # 读取持久化密码（若有）

    get_project_paths("default")
    load_project_context("default")

    if not LOAD_MODELS:
        print("[i] 测试模式：已跳过模型加载 (AD_LOAD_MODELS=0)。部署时请将 AD_LOAD_MODELS 置为 1 再启用。")
        return

    print(f"[*] [1/3] 正在装载 Google SigLIP 旗舰底模: {SIGLIP_MODEL_NAME} ...")
    try:
        siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, local_files_only=True)
        siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, local_files_only=True).to(DEVICE)
    except Exception:
        siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
    siglip_model.eval()

    print(f"[*] [2/3] 正在装载 Grounding DINO-Base: {DINO_MODEL_NAME} ...")
    try:
        dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME, local_files_only=True)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_NAME, local_files_only=True).to(DEVICE)
    except Exception:
        dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_NAME).to(DEVICE)
    dino_model.eval()

    print(f"[*] [3/3] 正在装载 YOLOv8x 顶配模型: {YOLO_MODEL_NAME} ...")
    try:
        yolo_model = YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        pass

    print("🚀 RTX 4070 Ti 旗舰配置 已就绪！")

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
        return {"projects": ["default"]}
    items = [d for d in os.listdir(PROJECTS_ROOT) if os.path.isdir(os.path.join(PROJECTS_ROOT, d))]
    return {"projects": sorted(items) if items else ["default"]}

@app.post("/api/projects/create")
def create_project(project_name: str = Form(...)):
    clean_name = sanitize_project_name(project_name)
    if not clean_name:
        raise HTTPException(status_code=400, detail="项目名不合法")
    get_project_paths(clean_name)
    load_project_context(clean_name)
    return {"code": 200, "msg": f"项目 [{clean_name}] 创建成功", "project": clean_name}

@app.post("/api/projects/rename")
def rename_project(old_name: str = Form(...), new_name: str = Form(...)):
    clean_old = sanitize_project_name(old_name)
    clean_new = sanitize_project_name(new_name)
    if not clean_old or not clean_new:
        return {"code": 400, "msg": "项目名无效"}
        
    old_dir = os.path.join(PROJECTS_ROOT, clean_old)
    new_dir = os.path.join(PROJECTS_ROOT, clean_new)
    
    if not os.path.exists(old_dir):
        return {"code": 404, "msg": "原项目不存在"}
    if os.path.exists(new_dir):
        return {"code": 400, "msg": "新项目名称已存在，请换一个"}
        
    if clean_old in project_cache:
        del project_cache[clean_old]
        
    try:
        os.rename(old_dir, new_dir)
        load_project_context(clean_new)
        return {"code": 200, "msg": f"项目已成功重命名为 [{clean_new}]", "project": clean_new}
    except Exception as e:
        return {"code": 500, "msg": f"重命名失败: {str(e)}"}

@app.post("/api/projects/delete")
def delete_project(project_name: str = Form(...)):
    clean_name = sanitize_project_name(project_name)
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
        path = meta[image_id].get("path")
        if path and os.path.exists(path):
            return FileResponse(path)
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
    if LITE_MODE or torch is None or siglip_model is None or not image_paths:
        return 0

    dataset = SigLIPImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

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
@app.post("/api/import_from_path")
async def import_from_path(project: str = Form("default"), source_path: str = Form(...)):
    if not os.path.exists(source_path):
        return {"code": 404, "msg": f"服务器端未找到路径: {source_path}"}
    
    ctx = load_project_context(project)
    existing_filenames = set([os.path.basename(m.get("path", "")) for m in ctx["metadata"] if "path" in m])
    new_saved_paths = []
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) and not file.startswith('__MACOSX'):
                if file not in existing_filenames:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(ctx["img_dir"], file)
                    try:
                        shutil.copy2(src_file, dst_file)
                        new_saved_paths.append(dst_file)
                        existing_filenames.add(file)
                    except Exception as e:
                        pass
                        
    processed_count = 0
    if new_saved_paths:
        processed_count = extract_and_index_project(ctx, new_saved_paths)
        
    return {
        "code": 200,
        "msg": f"成功从挂载路径读取并向量化了 {processed_count} 张新图",
        "processed_count": ctx["index"].ntotal
    }

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

    return {
        "code": 200,
        "msg": f"本次新增并向量化 {processed_count} 张图片",
        "processed_count": ctx["index"].ntotal
    }

@app.post("/api/build_index_online")
async def build_index_online(project: str = Query("default")):
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
    ctx = load_project_context(project)
    results = [m for m in ctx["metadata"] if filename_query.lower() in m.get("filename", "").lower()]
    return {"code": 200, "results": results}

@app.post("/api/search_by_external_image")
async def search_by_external_image(project: str = Form("default"), files: List[UploadFile] = File(...), top_k: int = 500):
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

    ctx["metadata"] = []
    ctx["index"] = faiss.IndexFlatIP(FEAT_DIM) if (not LITE_MODE and faiss is not None) else _FakeIndex(FEAT_DIM)
    extract_and_index_project(ctx, remaining_paths)

    return {"code": 200, "deleted_count": deleted_count}

@app.post("/api/ground_detect")
def ground_detect(project: str = Form("default"), image_id: int = Form(...), text_prompt: str = Form(...)):
    """Grounding DINO 真实目标级开放词汇推理（支持中英文自动转译）"""
    global dino_model, dino_processor
    ctx = load_project_context(project)
    if not (0 <= image_id < len(ctx["metadata"])) or dino_model is None:
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

        return {
            "scores": results["scores"].cpu().numpy().tolist(),
            "labels": labels,
            "boxes": results["boxes"].cpu().numpy().tolist(),
            "width": w, "height": h
        }
    except Exception as e:
        print(f"[!] DINO 检测异常: {e}")
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}

@app.post("/api/yolo_detect")
def yolo_detect(project: str = Form("default"), image_id: int = Form(...)):
    global yolo_model
    ctx = load_project_context(project)
    if not (0 <= image_id < len(ctx["metadata"])) or yolo_model is None:
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}

    img_path = ctx["metadata"][image_id]["path"]
    try:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        results = yolo_model(image, verbose=False, device=DEVICE)[0]
        cls_indices = results.boxes.cls.cpu().numpy().astype(int).tolist()

        return {
            "scores": results.boxes.conf.cpu().numpy().tolist(),
            "labels": [results.names[idx] for idx in cls_indices],
            "boxes": results.boxes.xyxy.cpu().numpy().tolist(),
            "width": w, "height": h
        }
    except Exception as e:
        return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}

# ----------------- [可选扩展] Qwen2-VL 综合场景判定 -----------------
vlm_model = None
vlm_processor = None

def init_vlm_local():
    global vlm_model, vlm_processor
    if not LOAD_MODELS:
        raise RuntimeError("VLM 未启用：请将 AD_LOAD_MODELS 置为 1 后再调用")
    if vlm_model is None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # 使用 2B 版本：体积小、能完整放进 12GB 显存，不卸载到 CPU，更快更稳不 OOM
        print("[*] 正在加载 VLM 综合场景判定模型 (Qwen2-VL-2B)...")
        vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        vlm_model.eval()
        print("[✓] VLM 场景判定引擎就绪 (Qwen2-VL-2B)")

@app.post("/api/vlm_analyze")
def vlm_analyze_scene(project: str = Form("default"), image_id: int = Form(...)):
    """
    第五层 VLM 综合场景判定：对指定图片进行多模态理解，并自动生成驾驶场景标签
    """
    global vlm_model, vlm_processor
    ctx = load_project_context(project)
    if not (0 <= image_id < len(ctx["metadata"])):
        return {"code": 400, "msg": "图片ID不存在"}

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
                    {"type": "text", "text": "请分析这张自动驾驶前视摄像头图片。提取 2 到 3 个核心场景特征标签（例如：城市夜晚、无路灯高速、雨天湿滑、大货车前车等），只输出简短的标签词，用逗号分隔。"}
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
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=64)
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
            "vlm_raw_output": output_text,
            "tags": item["userTags"]
        }
    except Exception as e:
        return {"code": 500, "msg": f"VLM 推理异常: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009, access_log=False)