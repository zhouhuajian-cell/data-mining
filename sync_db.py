import os
import json
import numpy as np
import faiss
import argparse

def sync_db(args):
    workspace_dir = args.workspace
    index_path = os.path.join(workspace_dir, "index.faiss")
    meta_path = os.path.join(workspace_dir, "metadata.json")
    feat_path = os.path.join(workspace_dir, "features.npy")

    print("Reading feature database...")
    with open(meta_path, "r", encoding="utf-8") as f:
        image_database = json.load(f)
    features = np.load(feat_path)

    new_db = []
    new_feats = []

    print("Checking files on disk...")
    for i, item in enumerate(image_database):
        if os.path.exists(item["path"]):
            new_db.append(item)
            new_feats.append(features[i])

    print("Rebuilding index...")
    embedding_dim = 768
    index = faiss.IndexFlatIP(embedding_dim)

    if len(new_feats) > 0:
        feats_array = np.array(new_feats, dtype=np.float32)
        index.add(feats_array)
    else:
        feats_array = np.array([], dtype=np.float32)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(new_db, f, ensure_ascii=False)
    np.save(feat_path, feats_array)

    print("="*50)
    print(f"Before: {len(image_database)} images")
    print(f"After:  {len(new_db)} images")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync database with disk files")
    parser.add_argument("--workspace", default="./workspace")
    args = parser.parse_args()
    sync_db(args)
