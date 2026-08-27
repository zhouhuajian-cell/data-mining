import io
import os
import json

# 👇 新增：强制让 Hugging Face 使用国内镜像加速下载 👇
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 👆 新增结束 👆

import zipfile
import tempfile
import math
import asyncio
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import FormData
from starlette.background import BackgroundTask
from PIL import Image
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO

app = FastAPI(title="AD Scene Mining Backend")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomFormParserMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        async def custom_form() -> FormData:
            if not hasattr(request, "_form"):
                form = await request._form(max_files=100000, max_fields=100000)
                request._form = form
            return request._form
        request.form = custom_form
        return await call_next(request)

app.add_middleware(CustomFormParserMiddleware)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)
weight_dtype = torch.float16 if device == "cuda" else torch.float32

# 1. 加载 CLIP 
clip_model_name = "openai/clip-vit-large-patch14"
print(f"Loading CLIP model {clip_model_name} on {device} (FP16)...")
clip_model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=weight_dtype).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# 2. 加载 Grounding DINO
dino_model_name = "IDEA-Research/grounding-dino-tiny"
print(f"Loading Grounding DINO model on {device} (FP16)...")
dino_processor = AutoProcessor.from_pretrained(dino_model_name)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name, torch_dtype=weight_dtype).to(device)

# 3. YOLOv8x (按需加载)
yolo_model = None
print("YOLOv8x is set to lazy-load.")

WORKSPACE_DIR = "./workspace"
IMG_DIR = os.path.join(WORKSPACE_DIR, "images")
INDEX_PATH = os.path.join(WORKSPACE_DIR, "index.faiss")
META_PATH = os.path.join(WORKSPACE_DIR, "metadata.json")
FEAT_PATH = os.path.join(WORKSPACE_DIR, "features.npy")

os.makedirs(IMG_DIR, exist_ok=True)
embedding_dim = 768

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and os.path.exists(FEAT_PATH):
    print("检测到历史数据，正在恢复...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        image_database = json.load(f)
    image_features_list = np.load(FEAT_PATH).tolist()
    print(f"✅ 成功加载 {len(image_database)} 帧历史特征！")
else:
    print("初始化全新空库。")
    index = faiss.IndexFlatIP(embedding_dim)
    image_database = []
    image_features_list = []

def save_persistence():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(image_database, f, ensure_ascii=False)
    np.save(FEAT_PATH, np.array(image_features_list, dtype=np.float32))

def extract_clip_image_feature(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(device, dtype=weight_dtype)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        if hasattr(feats, "pooler_output"): feats = feats.pooler_output
        elif hasattr(feats, "image_embeds"): feats = feats.image_embeds
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)

def extract_clip_text_feature(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"): feats = feats.pooler_output
        elif hasattr(feats, "text_embeds"): feats = feats.text_embeds
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)

@app.get("/api/db_stats")
def get_db_stats():
    if not os.path.exists(IMG_DIR):
        return {"total_raw": 0, "indexed": len(image_database), "unprocessed": 0}
    raw_count = sum(1 for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    indexed_count = len(image_database)
    unprocessed = max(0, raw_count - indexed_count)
    return {"total_raw": raw_count, "indexed": indexed_count, "unprocessed": unprocessed}

@app.post("/api/upload_batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    global image_database, index, image_features_list
    if index.ntotal + len(files) > 30000:
        return JSONResponse(status_code=400, content={"error": "容量限制 30000 张。"})
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            img_filename = file.filename
            img_path = os.path.join(IMG_DIR, img_filename)
            image.save(img_path, format="JPEG", quality=90)
            feat = extract_clip_image_feature(image)
            index.add(feat)
            image_database.append({"filename": img_filename, "path": img_path})
            image_features_list.append(feat[0])
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"入库出错: {e}")
    save_persistence()
    return {"status": "success", "total_indexed": index.ntotal}

@app.get("/api/list_all")
async def list_all_images(page: int = 1, size: int = 60):
    total = len(image_database)
    if total == 0: return {"results": [], "total": 0, "total_pages": 0, "current_page": 1}
    start_idx = (page - 1) * size
    end_idx = min(start_idx + size, total)
    results = [{"id": idx, "filename": image_database[idx]["filename"], "score": 1.0} for idx in range(start_idx, end_idx)]
    return {"results": results, "total": total, "total_pages": math.ceil(total / size), "current_page": page}

@app.post("/api/search")
async def search_scenes(query: str = Form(...), top_k: int = Form(24)):
    if index.ntotal == 0: return {"results": []}
    text_feat = extract_clip_text_feature(query)
    scores, indices = index.search(text_feat, min(top_k, index.ntotal))
    results = [{"id": int(idx), "filename": image_database[idx]["filename"], "score": float(score)} 
               for score, idx in zip(scores[0], indices[0]) if idx != -1]
    return {"results": results}

@app.post("/api/search_by_filename")
async def search_by_filename(filename_query: str = Form(...)):
    if index.ntotal == 0 or not filename_query: return {"results": []}
    query = filename_query.lower()
    results = [{"id": idx, "filename": item["filename"], "score": 1.0} 
               for idx, item in enumerate(image_database) if query in item["filename"].lower()]
    return {"results": results}

@app.post("/api/search_by_external_image")
async def search_by_external_image(files: List[UploadFile] = File(...), top_k: int = Form(24)):
    """多图联合检索：合并结果并保留最高分数"""
    if index.ntotal == 0: return {"results": []}
    all_results_dict = {}
    
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_feat = extract_clip_image_feature(image)
        scores, indices = index.search(image_feat, min(top_k, index.ntotal))
        
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                idx = int(idx)
                score = float(score)
                # 去重并保留针对同一张底库图片检索出的最高分
                if idx not in all_results_dict or score > all_results_dict[idx]["score"]:
                    all_results_dict[idx] = {
                        "id": idx, 
                        "filename": image_database[idx]["filename"], 
                        "score": score
                    }
                    
    merged_results = sorted(list(all_results_dict.values()), key=lambda x: x["score"], reverse=True)
    return {"results": merged_results}

@app.post("/api/search_by_image")
async def search_by_image(image_id: int = Form(...), top_k: int = Form(24)):
    if index.ntotal == 0 or image_id < 0 or image_id >= len(image_database): return {"results": []}
    img_path = image_database[image_id]["path"]
    query_image = Image.open(img_path).convert("RGB")
    image_feat = extract_clip_image_feature(query_image)
    scores, indices = index.search(image_feat, min(top_k, index.ntotal))
    results = [{"id": int(idx), "filename": image_database[idx]["filename"], "score": float(score)} 
               for score, idx in zip(scores[0], indices[0]) if idx != -1]
    return {"results": results}

@app.get("/api/image/{image_id}")
async def get_image(image_id: int):
    if image_id < 0 or image_id >= len(image_database): 
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(image_database[image_id]["path"], media_type="image/jpeg")

@app.post("/api/ground_detect")
def ground_detect(image_id: int = Form(...), text_prompt: str = Form(...)):
    img_path = image_database[image_id]["path"]
    image = Image.open(img_path).convert("RGB")
    prompt = text_prompt.strip()
    if not prompt.endswith("."): prompt += "."
    inputs = dino_processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=weight_dtype)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=[image.size[::-1]]
    )[0]
    return {
        "image_id": image_id, "width": image.width, "height": image.height,
        "boxes": results["boxes"].cpu().tolist(),
        "labels": results["labels"],
        "scores": results["scores"].cpu().tolist()
    }

@app.post("/api/yolo_detect")
def api_yolo_detect(image_id: int = Form(...)):
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("yolov8x.pt")
    img_path = image_database[image_id]["path"]
    image = Image.open(img_path).convert("RGB")
    results = yolo_model(image, verbose=False)[0]
    return {
        "image_id": image_id, "width": image.width, "height": image.height,
        "boxes": results.boxes.xyxy.cpu().tolist(),
        "labels": [results.names[int(c)] for c in results.boxes.cls.cpu().tolist()],
        "scores": results.boxes.conf.cpu().tolist()
    }

@app.post("/api/dedup_stats")
async def dedup_stats(sim_threshold: float = Form(0.95)):
    n = len(image_database)
    if n == 0 or len(image_features_list) == 0:
        return {"total_images": 0, "unique_images": 0, "duplicate_count": 0, "dedup_rate": "0.00%", "clusters": []}
    feats = np.array(image_features_list)
    sim_matrix = np.dot(feats, feats.T)
    parent = list(range(n))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j: parent[root_j] = root_i

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= sim_threshold: union(i, j)
    clusters_dict = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict: clusters_dict[root] = []
        clusters_dict[root].append(i)

    duplicate_clusters = []
    redundant_frames_count = 0
    for root, members in clusters_dict.items():
        if len(members) > 1:
            redundant_frames_count += (len(members) - 1)
            cluster_items = []
            for m in members:
                score = float(sim_matrix[members[0], m]) if m != members[0] else 1.0
                cluster_items.append({"id": m, "filename": image_database[m]["filename"], "score": score})
            duplicate_clusters.append({"group_size": len(members), "items": cluster_items})

    return {
        "total_images": n, "unique_images": n - redundant_frames_count,
        "duplicate_count": redundant_frames_count, 
        "dedup_rate": f"{(redundant_frames_count / n) * 100:.2f}%" if n > 0 else "0.00%",
        "clusters": duplicate_clusters
    }

@app.post("/api/delete_and_sync")
async def delete_and_sync(image_ids: str = Form(...)):
    global image_database, index, image_features_list
    if not image_ids: return {"deleted_count": 0, "total_indexed": index.ntotal}
    ids_to_delete = set(int(x) for x in image_ids.split(",") if x.strip().isdigit())
    deleted_count = 0
    new_db, new_feats = [], []
    for i, item in enumerate(image_database):
        if i in ids_to_delete:
            if os.path.exists(item["path"]):
                try:
                    os.remove(item["path"])
                    deleted_count += 1
                except: pass
        else:
            new_db.append(item)
            new_feats.append(image_features_list[i])
    image_database = new_db
    image_features_list = new_feats
    index = faiss.IndexFlatIP(embedding_dim)
    if len(new_feats) > 0: index.add(np.array(new_feats, dtype=np.float32))
    save_persistence()
    return {"deleted_count": deleted_count, "total_indexed": index.ntotal}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return Response(content=b"", media_type="image/x-icon")

# 托管前端静态图片资源（引导页背景、顶栏 Logo 等）
@app.get("/landing-bg.jpg", include_in_schema=False)
async def landing_bg(): return FileResponse("landing-bg.jpg", media_type="image/jpeg")

@app.get("/TT.png", include_in_schema=False)
async def logo_png(): return FileResponse("TT.png", media_type="image/png")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "前端.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f: return f.read()
    return f"""<h3>前端.html 文件未找到！</h3>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)