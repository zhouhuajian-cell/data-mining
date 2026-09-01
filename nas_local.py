#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAS → 本地 增量拷贝图片脚本（交互式 Python 版）
功能：交互输入多源路径、递归遍历所有子目录、跳过已存在文件、
      tqdm 进度条、完成提示
用法：运行后按提示输入源路径（可多个），回车开始拷贝
"""
import os
import sys
import shutil

# 允许的图片扩展名（白名单）
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 默认目标目录（可直接回车使用默认值）
DEFAULT_DEST = os.path.join(os.path.expanduser("~"), "Desktop", "图片集", "夜晚")


def input_sources():
    """交互式收集源路径，支持多条，输入空行结束"""
    sources = []
    print("请输入源路径（NAS 或本地文件夹），支持多条。")
    print("提示：一条一行，全部输完后直接回车（空行）即可开始。")
    print("-" * 60)
    while True:
        line = input("源路径（空行结束）: ").strip().strip('"').strip("'")
        if not line:
            if sources:
                break
            print("[提示] 尚未输入任何路径，至少需要一条。")
            continue
        sources.append(line)
    return sources


def collect_images(sources):
    """递归收集所有源路径下的图片，返回绝对路径列表"""
    all_files = []
    for src in sources:
        if not os.path.isdir(src):
            print(f"[WARN] 源路径不可访问，已跳过: {src}")
            continue
        found = []
        for dirpath, _, filenames in os.walk(src):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in EXTS:
                    found.append(os.path.join(dirpath, fn))
        print(f"源 [{src}]")
        print(f"     发现 {len(found)} 张图片")
        all_files.extend(found)
    return all_files


def main():
    # ---- 1. 目标目录 ----
    dest = input(f"目标文件夹（回车用默认: {DEFAULT_DEST}）: ").strip().strip('"').strip("'")
    if not dest:
        dest = DEFAULT_DEST
    os.makedirs(dest, exist_ok=True)
    print(f"目标文件夹: {dest}")
    print()

    # ---- 2. 交互输入源路径 ----
    sources = input_sources()
    if not sources:
        print("未提供任何有效源路径，退出。")
        input("按回车键退出...")
        return

    # ---- 3. 递归收集 ----
    print()
    all_files = collect_images(sources)
    if not all_files:
        print("\n所有源路径均未找到符合条件的图片。")
        input("按回车键退出...")
        return

    # ---- 4. 增量过滤 ----
    existing = {f for f in os.listdir(dest)
                if os.path.splitext(f)[1].lower() in EXTS} if os.path.isdir(dest) else set()
    to_copy = [f for f in all_files if os.path.basename(f) not in existing]

    print(f"\n扫描到图片总数: {len(all_files)}   目标已存在: {len(existing)}   本次待拷贝: {len(to_copy)}")

    # ---- 4.5 拷贝前确认 ----（再次核对目标文件夹路径和源数据路径）
    print("\n----------------- 请确认以下信息 -----------------")
    print(f"需要拷贝到的本地文件夹路径: {dest}")
    print("源数据路径:")
    for src in sources:
        print(f"  - {src}")
    print("--------------------------------------------------")
    confirm = input("是否确认拷贝？(y/Y 确认，其他任意键取消): ").strip().lower()
    if confirm != 'y':
        print("已取消拷贝，未复制任何文件。")
        input("按回车键退出...")
        return

    if not to_copy:
        print("没有需要拷贝的新图片，全部已存在。")
        print(f"目标文件夹现有图片: {len(existing)} 张")
        input("按回车键退出...")
        return

    # ---- 5. 拷贝 + 进度条 ----
    try:
        from tqdm import tqdm
        iterator = tqdm(to_copy, desc="拷贝图片", unit="张")
    except ImportError:
        print("未安装 tqdm，使用普通循环。可执行: pip install tqdm")
        iterator = to_copy

    copied = 0
    failed = 0
    for src_path in iterator:
        name = os.path.basename(src_path)
        dst_path = os.path.join(dest, name)
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name} -> {e}")

    total = len([f for f in os.listdir(dest) if os.path.splitext(f)[1].lower() in EXTS])
    print("\n================= 完成 =================")
    print(f"本次成功拷贝: {copied} 张")
    print(f"本次失败:     {failed} 张")
    print(f"目标文件夹现有图片总数: {total} 张")
    print("========================================")
    input("按回车键退出...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消。")
        sys.exit(130)
