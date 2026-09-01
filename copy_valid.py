# -*- coding: utf-8 -*-
"""从 JSON 导出文件中提取 filename，匹配 workspace/images 中的图片，复制到交付文件夹。"""
import json
import os
import shutil
import argparse
from tqdm import tqdm

def copy_valid(args):
    json_path = args.json
    project_root = args.root
    image_dir = args.image_dir
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    filenames = [d.get("filename") for d in data if d.get("filename")]
    print(f"JSON entries: {len(data)}")
    print(f"filename count: {len(filenames)}")
    print(f"filename unique: {len(set(filenames))}")

    all_files = set(os.listdir(image_dir))

    copied = 0
    missing = 0

    for d in tqdm(data, desc="导出图片", unit="张"):
        fn = d.get("filename")
        if not fn:
            continue

        src = None

        if fn in all_files:
            src = os.path.join(image_dir, fn)

        if src is None:
            for f in all_files:
                if f.endswith("_" + fn):
                    src = os.path.join(image_dir, f)
                    break

        if src is None:
            fp = d.get("file_path", "").lstrip("./")
            if fp:
                fp = fp.replace("\\", "/")
                candidate = os.path.join(project_root, fp)
                if os.path.isfile(candidate):
                    src = candidate

        if src is None:
            missing += 1
            continue

        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        copied += 1

    valid_count = len([f for f in os.listdir(output_dir)
                       if os.path.isfile(os.path.join(output_dir, f))])
    print(f"\n===== Done =====")
    print(f"Copied: {copied}")
    print(f"Missing: {missing}")
    print(f"Output: {output_dir}")
    print(f"Output count: {valid_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy valid images from JSON export")
    parser.add_argument("--json", required=True, help="Path to exported JSON file")
    parser.add_argument("--root", default="/opt/ad_mining", help="Project root")
    parser.add_argument("--image-dir", default=None, help="Image directory")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    if args.image_dir is None:
        args.image_dir = os.path.join(args.root, "workspace", "images")
    if args.output is None:
        args.output = os.path.join(args.root, "有效", "Clip初筛")

    copy_valid(args)
