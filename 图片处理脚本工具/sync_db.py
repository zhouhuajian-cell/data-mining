import os
import json
import numpy as np
import faiss

WORKSPACE_DIR = "./workspace"
INDEX_PATH = os.path.join(WORKSPACE_DIR, "index.faiss")
META_PATH = os.path.join(WORKSPACE_DIR, "metadata.json")
FEAT_PATH = os.path.join(WORKSPACE_DIR, "features.npy")

print("正在读取历史特征库...")
with open(META_PATH, "r", encoding="utf-8") as f:
    image_database = json.load(f)
features = np.load(FEAT_PATH)

new_db = []
new_feats = []

print("正在核对硬盘实体文件...")
for i, item in enumerate(image_database):
    # 核心：只保留硬盘上真实存在的图片特征
    if os.path.exists(item["path"]):
        new_db.append(item)
        new_feats.append(features[i])

print(f"核对完毕！准备重建特征库...")
embedding_dim = 768
index = faiss.IndexFlatIP(embedding_dim)

if len(new_feats) > 0:
    feats_array = np.array(new_feats, dtype=np.float32)
    index.add(feats_array)
else:
    feats_array = np.array([], dtype=np.float32)

# 覆盖保存旧的数据库文件
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(new_db, f, ensure_ascii=False)
np.save(FEAT_PATH, feats_array)

print("="*50)
print(f"✅ 底库同步成功！")
print(f"清理前记录: {len(image_database)} 张")
print(f"清理后剩余: {len(new_db)} 张")
print("="*50)