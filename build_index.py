import os
import json
import argparse
import torch
import faiss
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

import config as cfgmod

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        return Image.open(path).convert("RGB"), os.path.basename(path), path

def collate_fn(batch):
    images, filenames, paths = zip(*batch)
    return list(images), filenames, paths

def build_database(args):
    workspace_dir = args.workspace
    img_dir = os.path.join(workspace_dir, "images")
    index_path = os.path.join(workspace_dir, "index.faiss")
    meta_path = os.path.join(workspace_dir, "metadata.json")
    feat_path = os.path.join(workspace_dir, "features.npy")

    os.makedirs(workspace_dir, exist_ok=True)

    print(f"Scanning: {img_dir}")
    dataset = ImageDataset(img_dir)
    if len(dataset) == 0:
        print("No images found!")
        return

    print(f"Found {len(dataset)} images")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device} ({weight_dtype})")

    torch.set_num_threads(args.num_workers * 2)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    model = CLIPModel.from_pretrained(cfgmod.clip_model(),
                                      torch_dtype=weight_dtype,
                                      local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(cfgmod.clip_model(),
                                              local_files_only=True)

    db_meta = []
    feat_list = []

    for images, filenames, paths in tqdm(loader, desc="Extracting"):
        inputs = processor(images=images, return_tensors="pt").to(device, dtype=weight_dtype)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            # 兼容不同 transformers 版本：可能返回原始张量或 BaseModelOutputWithPooling 对象
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

        feat_list.append(feats.cpu().numpy().astype(np.float32))
        for fn, p in zip(filenames, paths):
            db_meta.append({"filename": fn, "path": p.replace("\\", "/")})

    print("\nSaving to disk...")
    all_feats = np.vstack(feat_list)
    index = faiss.IndexFlatIP(all_feats.shape[1])
    index.add(all_feats)

    # faiss 的 C++ fopen 无法处理含中文的 Windows 路径，改用 Python 写字节
    with open(index_path, "wb") as f:
        f.write(faiss.serialize_index(index))
    np.save(feat_path, all_feats)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(db_meta, f, ensure_ascii=False)

    print(f"Done! {len(dataset)} images indexed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLIP feature index")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--batch-size", type=int, default=cfgmod.batch_size())
    parser.add_argument("--num-workers", type=int, default=cfgmod.num_workers())
    parser.add_argument("--device", default=cfgmod.device(), choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    build_database(args)
