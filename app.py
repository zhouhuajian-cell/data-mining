import io
import os
import json
import zipfile
import tempfile
import random
import math
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

clip_model_name = "openai/clip-vit-large-patch14"
print(f"Loading CLIP model {clip_model_name} on {device}...")
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

dino_model_name = "IDEA-Research/grounding-dino-tiny"
print(f"Loading Grounding DINO model on {device}...")
dino_processor = AutoProcessor.from_pretrained(dino_model_name)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name).to(device)

print(f"Loading YOLO model on {device}...")
yolo_model = YOLO("yolov8x.pt") 

WORKSPACE_DIR = "./workspace"
IMG_DIR = os.path.join(WORKSPACE_DIR, "images")
INDEX_PATH = os.path.join(WORKSPACE_DIR, "index.faiss")
META_PATH = os.path.join(WORKSPACE_DIR, "metadata.json")
FEAT_PATH = os.path.join(WORKSPACE_DIR, "features.npy")

os.makedirs(IMG_DIR, exist_ok=True)
embedding_dim = 768

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and os.path.exists(FEAT_PATH):
    print("检测到本地历史数据，正在从硬盘恢复...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        image_database = json.load(f)
    image_features_list = np.load(FEAT_PATH).tolist()
    print(f"✅ 成功加载 {len(image_database)} 帧历史特征！")
else:
    print("未检测到历史数据，初始化全新空库。")
    index = faiss.IndexFlatIP(embedding_dim)
    image_database = []
    image_features_list = []

def save_persistence():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(image_database, f, ensure_ascii=False)
    np.save(FEAT_PATH, np.array(image_features_list, dtype=np.float32))

def extract_clip_image_feature(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
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

@app.post("/api/upload_batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    global image_database, index, image_features_list
    if index.ntotal + len(files) > 30000:
        return JSONResponse(status_code=400, content={"error": "系统容量限制为 30000 张。"})

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            img_filename = f"{len(image_database)}_{file.filename}"
            img_path = os.path.join(IMG_DIR, img_filename)
            image.save(img_path, format="JPEG", quality=90)
            feat = extract_clip_image_feature(image)
            index.add(feat)
            image_database.append({"filename": file.filename, "path": img_path})
            image_features_list.append(feat[0])
        except Exception as e:
            print(f"处理出错: {e}")
            
    save_persistence()
    return {"status": "success", "total_indexed": index.ntotal}

@app.get("/api/list_all")
async def list_all_images(page: int = 1, size: int = 60):
    total = len(image_database)
    if total == 0:
        return {"results": [], "total": 0, "total_pages": 0, "current_page": 1}
    
    start_idx = (page - 1) * size
    end_idx = min(start_idx + size, total)
    
    results = []
    for idx in range(start_idx, end_idx):
        results.append({
            "id": idx,
            "filename": image_database[idx]["filename"],
            "score": 1.0 
        })
        
    return {
        "results": results,
        "total": total,
        "total_pages": math.ceil(total / size),
        "current_page": page
    }

@app.post("/api/search")
async def search_scenes(query: str = Form(...), top_k: int = Form(24)):
    if index.ntotal == 0: return {"results": []}
    text_feat = extract_clip_text_feature(query)
    scores, indices = index.search(text_feat, min(top_k, index.ntotal))
    results = [{"id": int(idx), "filename": image_database[idx]["filename"], "score": float(score)} 
               for score, idx in zip(scores[0], indices[0]) if idx != -1]
    return {"results": results}

# 👇 新增：按文件名检索接口 👇
@app.post("/api/search_by_filename")
async def search_by_filename(filename_query: str = Form(...)):
    if index.ntotal == 0 or not filename_query:
        return {"results": []}
    
    query = filename_query.lower()
    results = []
    for idx, item in enumerate(image_database):
        # 支持模糊匹配（只要包含该字符串即返回）
        if query in item["filename"].lower():
            results.append({
                "id": idx,
                "filename": item["filename"],
                "score": 1.0 # 按名字搜，分数固定给 1.0
            })
            
    return {"results": results}

@app.post("/api/search_by_external_image")
async def search_by_external_image(file: UploadFile = File(...), top_k: int = Form(24)):
    if index.ntotal == 0: return {"results": []}
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_feat = extract_clip_image_feature(image)
    scores, indices = index.search(image_feat, min(top_k, index.ntotal))
    results = [{"id": int(idx), "filename": image_database[idx]["filename"], "score": float(score)} 
               for score, idx in zip(scores[0], indices[0]) if idx != -1]
    return {"results": results}

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
    if image_id < 0 or image_id >= len(image_database): return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(image_database[image_id]["path"], media_type="image/jpeg")

@app.post("/api/ground_detect")
async def ground_detect(image_id: int = Form(...), text_prompt: str = Form(...)):
    img_path = image_database[image_id]["path"]
    image = Image.open(img_path).convert("RGB")
    prompt = text_prompt.strip()
    if not prompt.endswith("."): prompt += "."
    inputs = dino_processor(images=image, text=prompt, return_tensors="pt").to(device)
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
async def api_yolo_detect(image_id: int = Form(...)):
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

@app.post("/api/export_images")
async def export_images(image_ids: str = Form(...)):
    if not image_ids:
        return JSONResponse({"error": "没有提供图片ID"}, status_code=400)
    
    ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
    
    fd, temp_zip_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    
    with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_STORED) as zip_file:
        for img_id in ids:
            if 0 <= img_id < len(image_database):
                img_path = image_database[img_id]["path"]
                filename = image_database[img_id]["filename"]
                arcname = f"ID{img_id}_{filename}" 
                if os.path.exists(img_path):
                    zip_file.write(img_path, arcname)
    
    return FileResponse(
        temp_zip_path,
        media_type="application/zip",
        filename=f"AD_Selected_Images.zip",
        background=BackgroundTask(os.remove, temp_zip_path)
    )

@app.post("/api/delete_and_sync")
async def delete_and_sync(image_ids: str = Form(...)):
    global image_database, index, image_features_list
    if not image_ids:
        return {"deleted_count": 0, "total_indexed": index.ntotal}
    
    ids_to_delete = set(int(x) for x in image_ids.split(",") if x.strip().isdigit())
    deleted_count = 0
    
    new_db = []
    new_feats = []
    
    for i, item in enumerate(image_database):
        if i in ids_to_delete:
            file_path = item["path"]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除物理文件失败 {file_path}: {e}")
        else:
            new_db.append(item)
            new_feats.append(image_features_list[i])
            
    image_database = new_db
    image_features_list = new_feats
    
    index = faiss.IndexFlatIP(embedding_dim)
    if len(new_feats) > 0:
        index.add(np.array(new_feats, dtype=np.float32))
        
    save_persistence()
    
    return {"deleted_count": deleted_count, "total_indexed": index.ntotal}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "前端.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"""<h3 style="color:red;">文件未找到！</h3>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)