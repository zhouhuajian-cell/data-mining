# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import sys
import json
import shutil
import zipfile
import re
from typing import List, Optional
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import faiss
from transformers import AutoProcessor, CLIPModel

# ----------------- 路径与轻量配置 -----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(PROJECT_DIR, "workspace")
PROJECTS_ROOT = os.path.join(WORKSPACE, "projects")
os.makedirs(PROJECTS_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 极速轻量模型 (~300MB，CPU 秒跑)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEAT_DIM = 512  # Base-Patch32 特征维度为 512

app = FastAPI(title="Maxieye Lightweight Scene Mining")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(PROJECT_DIR):
    app.mount("/static_root", StaticFiles(directory=PROJECT_DIR), name="static_root")

# 全局状态
clip_model = None
clip_processor = None
project_cache = {}

# ----------------- 224x224 轻量 DataLoader -----------------
class FastImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
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
            return torch.zeros((3, 224, 224)), path, 0

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

def load_project_context(project_name: str):
    p_name, p_dir, img_dir, idx_path, meta_path = get_project_paths(project_name)
    if p_name in project_cache:
        return project_cache[p_name]

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        try:
            index = faiss.read_index(idx_path)
            if index.d != FEAT_DIM:
                print(f"[!] 项目 {p_name} 索引维度 ({index.d}) 与当前轻量模型 ({FEAT_DIM}) 不匹配，初始化新索引")
                index = faiss.IndexFlatIP(FEAT_DIM)
                metadata = []
            else:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
        except Exception as e:
            print(f"[!] 读取项目 {p_name} 索引异常: {e}，初始化新索引")
            index = faiss.IndexFlatIP(FEAT_DIM)
            metadata = []
    else:
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
    faiss.write_index(ctx["index"], ctx["idx_path"])
    with open(ctx["meta_path"], "w", encoding="utf-8") as f:
        json.dump(ctx["metadata"], f, ensure_ascii=False)

# ----------------- 系统启动 -----------------
@app.on_event("startup")
def startup_event():
    global clip_model, clip_processor
    print(f"[*] 启动轻量级测试环境，运行设备: {DEVICE}")

    get_project_paths("default")
    load_project_context("default")

    print(f"[*] 正在载入轻量 CLIP 模型: {CLIP_MODEL_NAME} ...")
    try:
        clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(DEVICE)
    except Exception:
        print("[!] 本地无缓存，尝试通过国内镜像同步下载...")
        clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    
    clip_model.eval()
    print("[✓] 轻量 CLIP 模型已就绪")

# ----------------- API 路由 -----------------
@app.get("/")
def read_index():
    index_html = os.path.join(PROJECT_DIR, "前端.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {"msg": "前端.html 不存在"}

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
        "pending_count": max(0, raw_count - processed_count),
        "total_raw": raw_count,
        "indexed": processed_count,
        "unprocessed": max(0, raw_count - processed_count)
    }

def extract_and_index_project(ctx, image_paths: List[str]):
    if not image_paths:
        return 0

    dataset = FastImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    extracted_feats = []
    extracted_records = []

    with torch.no_grad():
        for tensors, paths, valids in dataloader:
            mask = (valids == 1)
            if not mask.any():
                continue
            valid_tensors = tensors[mask].to(DEVICE)
            feats = clip_model.get_image_features(pixel_values=valid_tensors)
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

@app.post("/api/upload_batch")
async def upload_batch(project: str = Form("default"), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="未接收到文件")

    ctx = load_project_context(project)
    new_saved_paths = []

    for file in files:
        file_path = os.path.join(ctx["img_dir"], file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file.filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) and not member.startswith('__MACOSX'):
                            fn = os.path.basename(member)
                            if fn:
                                ext_path = os.path.join(ctx["img_dir"], fn)
                                with zip_ref.open(member) as s, open(ext_path, "wb") as t:
                                    shutil.copyfileobj(s, t)
                                new_saved_paths.append(ext_path)
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            new_saved_paths.append(file_path)

    processed_count = extract_and_index_project(ctx, new_saved_paths)
    return {
        "code": 200,
        "msg": f"成功入库并向量化 {processed_count} 张图片",
        "processed_count": ctx["index"].ntotal,
        "total_indexed": ctx["index"].ntotal
    }

@app.post("/api/build_index_online")
async def build_index_online(project: str = Query("default")):
    ctx = load_project_context(project)
    all_imgs = [os.path.join(ctx["img_dir"], f) for f in os.listdir(ctx["img_dir"]) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    indexed_paths = set([m['path'] for m in ctx["metadata"] if 'path' in m])
    unprocessed_paths = [p for p in all_imgs if p not in indexed_paths]

    if not unprocessed_paths:
        return {"code": 200, "msg": "该项目所有图片均已完成向量化", "processed": 0}

    processed_count = extract_and_index_project(ctx, unprocessed_paths)
    return {
        "code": 200,
        "msg": f"成功完成 {processed_count} 张图片的向量化！",
        "processed_count": ctx["index"].ntotal
    }

@app.get("/api/search")
def search_text(project: str = Query("default"), query: str = Query(..., min_length=1), top_k: int = 500):
    ctx = load_project_context(project)
    if ctx["index"] is None or ctx["index"].ntotal == 0:
        return {"code": 200, "results": [], "total": 0}

    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**inputs)
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

    return {"code": 200, "results": results, "total": len(results)}

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

    inputs = clip_processor(images=pil_imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_feats = clip_model.get_image_features(**inputs)
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
    total_pages = int(np.ceil(total / size)) if total > 0 else 1
    return {
        "code": 200, "total": total, "total_raw": total,
        "current_page": page, "total_pages": total_pages,
        "items": items, "results": items
    }

@app.post("/api/dedup_stats")
def dedup_stats(project: str = Form("default"), sim_threshold: float = Form(0.95)):
    ctx = load_project_context(project)
    total = len(ctx["metadata"])
    if total < 2 or ctx["index"] is None:
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
    ctx["index"] = faiss.IndexFlatIP(FEAT_DIM)
    extract_and_index_project(ctx, remaining_paths)

    return {"code": 200, "deleted_count": deleted_count}

# 本地测试占位
@app.post("/api/ground_detect")
def ground_detect(project: str = Form("default"), image_id: int = Form(...), text_prompt: str = Form(...)):
    return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}

@app.post("/api/yolo_detect")
def yolo_detect(project: str = Form("default"), image_id: int = Form(...)):
    return {"scores": [], "labels": [], "boxes": [], "width": 1920, "height": 1080}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8009, access_log=False)