import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm 

# 路径配置
WORKSPACE_DIR = "./workspace"
IMG_DIR = os.path.join(WORKSPACE_DIR, "images")
INDEX_PATH = os.path.join(WORKSPACE_DIR, "index.faiss")
META_PATH = os.path.join(WORKSPACE_DIR, "metadata.json")
FEAT_PATH = os.path.join(WORKSPACE_DIR, "features.npy")

# 👇 专为纯 CPU 护肝模式调整的参数 👇
BATCH_SIZE = 8   # 每次处理 8 张，防内存爆满
NUM_WORKERS = 2  # 适中的多线程读图，不卡死系统

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    def __len__(self): 
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        return Image.open(path).convert("RGB"), os.path.basename(path), path

def collate_fn(batch):
    images, filenames, paths = zip(*batch)
    return list(images), filenames, paths

def build_database():
    print(f"📁 正在扫描目录: {IMG_DIR}")
    dataset = ImageDataset(IMG_DIR)
    if len(dataset) == 0:
        return print("❌ 目录下没有图片！")
    
    print(f"🚀 共找到 {len(dataset)} 张图片，开始纯 CPU 模式特征提取...")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    device = "cpu"
    weight_dtype = torch.float32 
    
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    db_meta = []
    feat_list = []
    
    for images, filenames, paths in tqdm(loader, desc="提取特征"):
        inputs = processor(images=images, return_tensors="pt").to(device, dtype=weight_dtype)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            
        feat_list.append(feats.cpu().numpy().astype(np.float32))
        for fn, p in zip(filenames, paths):
            db_meta.append({"filename": fn, "path": p.replace("\\", "/")})

    print("\n💾 正在将数据落盘 (FAISS & Metadata)...")
    all_feats = np.vstack(feat_list)
    index = faiss.IndexFlatIP(all_feats.shape[1])
    index.add(all_feats)

    faiss.write_index(index, INDEX_PATH)
    np.save(FEAT_PATH, all_feats)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(db_meta, f, ensure_ascii=False)
        
    print("🎉 3万+ 图片底层数据库构建完毕！现在可以启动你的 Web 界面了。")

if __name__ == "__main__":
    build_database()