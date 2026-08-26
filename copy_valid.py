# -*- coding: utf-8 -*-
"""从 AD_Selected_Data json 提取 filename，在 workspace/images 匹配对应图片，复制到"有效"文件夹。"""
import json
import os
import shutil

json_path = r"C:\Users\zhouhuajian\Downloads\AD_Selected_Data_With_Tags_1787550785216.json"
project_root = r"C:\Users\zhouhuajian\Desktop\数据挖掘"
image_dir = os.path.join(project_root, "workspace", "images")
valid_dir = os.path.join(project_root, "有效", "0.2 0.22")  # 新建的有效文件夹

with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

filenames = [d.get("filename") for d in data if d.get("filename")]
print(f"JSON 条目数: {len(data)}")
print(f"filename 数量: {len(filenames)}")
print(f"filename 唯一数: {len(set(filenames))}")

os.makedirs(valid_dir, exist_ok=True)

# 建立图片目录内文件名索引
all_files = set(os.listdir(image_dir))
name_to_path = {fn: os.path.join(image_dir, fn) for fn in all_files}

moved = 0
missing = 0
matched_names = []

for d in data:
    fn = d.get("filename")
    if not fn:
        continue
    fp = d.get("file_path", "").lstrip("./").replace("/", "\\")
    src = None
    # 1) 优先用 file_path 精确定位
    cand_path = os.path.join(project_root, fp)
    if fp and os.path.isfile(cand_path):
        src = cand_path
        matched_names.append(os.path.basename(cand_path))
    else:
        # 2) 其次按 filename 在图片目录中匹配（带 id_ 前缀或同名）
        candidates = [fn] + [f for f in all_files if f.endswith("_" + fn)]
        for c in candidates:
            p = os.path.join(image_dir, c)
            if os.path.isfile(p):
                src = p
                matched_names.append(c)
                break
    if src is None:
        print(f"未匹配: {fn}")
        missing += 1
        continue

    dst = os.path.join(valid_dir, os.path.basename(src))
    shutil.copy2(src, dst)  # 复制，保留源文件
    moved += 1
    print(f"已复制: {os.path.basename(src)}")

valid_count = len([f for f in os.listdir(valid_dir) if os.path.isfile(os.path.join(valid_dir, f))])
print("\n===== 完成 =====")
print(f"匹配并复制: {moved} 张")
print(f"未匹配: {missing} 张")
print(f"有效文件夹: {valid_dir}")
print(f"有效文件夹内图片数: {valid_count}")
