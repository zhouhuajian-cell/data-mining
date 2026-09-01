# -*- coding: utf-8 -*-
"""从服务器按导出的 JSON 直接下载图片到本地。

最简用法（会交互式选择下载目录）:
    python download_from_server.py --json "你的导出文件.json"

指定输出目录（跳过交互）:
    python download_from_server.py \
        --json "C:\\Users\\zhouhuajian\\Downloads\\AD_测试数据_AD_Selected_Total_499_1788167467933.json" \
        --output "C:\\Users\\zhouhuajian\\Desktop\\data_mining\\有效\\Clip初筛"

可选参数:
    --host        服务器 IP（默认 10.2.248.34）
    --user        SSH 用户名（默认 root）
    --remote-dir  服务器图片目录（默认 .../测试数据/images）
"""
import json
import os
import sys
import argparse
import paramiko
from tqdm import tqdm

DEFAULT_OUT = r"C:\Users\zhouhuajian\Desktop\data_mining\有效\Clip初筛"


def interactive_choose_json():
    """交互式选择导出 JSON 文件（默认扫描下载文件夹），返回路径。"""
    search_dirs = [
        r"C:\Users\zhouhuajian\Downloads",
        r"C:\Users\zhouhuajian\Desktop",
    ]
    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".json") and ("AD_" in f or "Selected" in f):
                    p = os.path.join(d, f)
                    candidates.append(p)
    # 按修改时间倒序，最新的在前
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    if not candidates:
        print("\n未在下载/桌面找到导出的 JSON 文件。")
        while True:
            p = input("请直接输入 JSON 文件完整路径: ").strip().strip('"')
            if p and os.path.isfile(p):
                return p
            print("路径无效或文件不存在，请重试")

    print("\n===== 选择导出 JSON 文件 ====")
    for i, p in enumerate(candidates[:15], 1):
        print(f"  [{i}] {os.path.basename(p)}")
    print("  或直接输入一个自定义 JSON 路径")

    while True:
        choice = input("请输入编号（默认选最新的 1）: ").strip()
        if not choice:
            return candidates[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= min(15, len(candidates)):
                return candidates[idx - 1]
            print(f"编号超出范围 (1-{min(15, len(candidates))})，请重试")
            continue
        p = os.path.abspath(os.path.expanduser(choice.strip('"')))
        if os.path.isfile(p):
            return p
        print("路径无效或文件不存在，请重试")


def find_suggestion_folders():
    """扫描常见位置，收集已有的"有效"目录作为建议。"""
    candidates = []
    roots = [
        r"C:\Users\zhouhuajian\Desktop\data_mining\有效",
        r"C:\Users\zhouhuajian\Desktop\有效",
    ]
    for r in roots:
        if os.path.isdir(r):
            for d in sorted(os.listdir(r)):
                p = os.path.join(r, d)
                if os.path.isdir(p):
                    candidates.append(p)
    return candidates


def interactive_choose_output():
    """交互式选择下载目录，返回所选路径。"""
    suggestions = find_suggestion_folders()
    if DEFAULT_OUT not in suggestions:
        suggestions.insert(0, DEFAULT_OUT)
    # 去重保持顺序
    seen, uniq = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    suggestions = uniq

    print("\n===== 选择下载目录 =====")
    print("可用选项：")
    for i, s in enumerate(suggestions, 1):
        print(f"  [{i}] {s}")
    print("  或直接输入一个自定义文件夹路径")

    while True:
        choice = input("请输入编号或路径（直接回车用默认）: ").strip().strip('"')
        if not choice:
            return DEFAULT_OUT
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(suggestions):
                return suggestions[idx - 1]
            print(f"编号超出范围 (1-{len(suggestions)})，请重试")
            continue
        return os.path.abspath(os.path.expanduser(choice))


def build_lookup(sftp, remote_dir):
    """列出服务器图片目录，返回 {文件名: 远端完整路径} 与纯文件名集合。"""
    names = [n for n in sftp.listdir(remote_dir) if not n.startswith(".")]
    base = remote_dir.rstrip("/")
    full = {n: base + "/" + n for n in names}
    return full, set(names)


def main(args):
    if not args.json:
        args.json = interactive_choose_json()
        print(f"\n已选择 JSON: {args.json}")
    if not args.output:
        args.output = interactive_choose_output()
        print(f"\n已选择下载目录: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    filenames = [d.get("filename") for d in data if d.get("filename")]
    print(f"JSON entries: {len(data)}")
    print(f"filename count: {len(filenames)}")
    print(f"filename unique: {len(set(filenames))}")

    key_path = os.path.expanduser(args.key)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        key_filename=key_path,
        timeout=15,
    )
    sftp = ssh.open_sftp()
    print(f"已连接 {args.user}@{args.host}:{args.port}")

    # 兜底：列出 --remote-dir 里的文件用于按文件名匹配
    full, all_files = build_lookup(sftp, args.remote_dir)
    print(f"服务器图片目录: {args.remote_dir}")
    print(f"该目录图片文件数: {len(all_files)}")

    copied = 0
    missing = 0

    for d in tqdm(data, desc="下载图片", unit="张"):
        fn = d.get("filename")
        if not fn:
            continue

        src_remote = None

        # 1) 优先用 file_path 精确定位服务器上的真实路径
        fp = d.get("file_path", "").lstrip("./").replace("\\", "/")
        if fp:
            candidate = args.root + "/" + fp
            try:
                sftp.stat(candidate)
                src_remote = candidate
            except IOError:
                src_remote = None

        # 2) 兜底：按文件名在 --remote-dir 里匹配
        if src_remote is None:
            if fn in all_files:
                src_remote = full[fn]
            else:
                for f in all_files:
                    if f.endswith("_" + fn):
                        src_remote = full[f]
                        break

        if src_remote is None:
            missing += 1
            continue

        dst = os.path.join(args.output, os.path.basename(src_remote))
        if os.path.exists(dst):
            # 已存在则跳过（可断点续传）
            copied += 1
            continue
        sftp.get(src_remote, dst)
        copied += 1

    sftp.close()
    ssh.close()

    valid_count = len([f for f in os.listdir(args.output)
                       if os.path.isfile(os.path.join(args.output, f))])
    print(f"\n===== Done =====")
    print(f"Downloaded: {copied}")
    print(f"Missing: {missing}")
    print(f"Output: {args.output}")
    print(f"Output count: {valid_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从服务器按 JSON 下载图片到本地")
    parser.add_argument("--json", default=None, help="导出的 JSON 文件路径（不填则交互式选择）")
    parser.add_argument("--host", default="10.2.248.34", help="服务器 IP")
    parser.add_argument("--port", type=int, default=22, help="SSH 端口")
    parser.add_argument("--user", default="root", help="SSH 用户名")
    parser.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH 私钥路径")
    parser.add_argument("--root", default="/opt/ad_mining",
                        help="服务器项目根目录（用于拼 file_path 定位图片）")
    parser.add_argument("--remote-dir", default="/opt/ad_mining/workspace/projects/测试数据/images",
                        help="服务器上图片所在目录（仅作按文件名匹配的兜底）")
    parser.add_argument("--output", default=None,
                        help="本地输出目录（不填则交互式选择）")
    args = parser.parse_args()

    main(args)
