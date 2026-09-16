# -*- coding: utf-8 -*-
"""
VLM(AWQ) 服务器环境自检脚本 —— 阶段0
作用：在目标 GPU 服务器上跑一遍，确认能否加载 Qwen2.5-VL AWQ 模型，
      避免环境不支持 AWQ / 版本太旧 / 缺依赖时直接上业务代码白忙。
用法（在服务器上）：
    python check_vlm_env.py                 # 只检查依赖/版本/连通性
    python check_vlm_env.py --load          # 额外真正加载 7B-AWQ 并报告显存占用
可选参数：
    --model "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"   # 覆盖要检查的模型
    --no-download                                # 跳过联网下载连通性测试
"""
import os
import sys
import argparse

MODEL_DEFAULT = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

def banner(t):
    print("\n" + "=" * 60)
    print("  " + t)
    print("=" * 60)

def section(t):
    print("\n--- " + t + " ---")

def fmt_ok(s):  return "  [OK]   " + str(s)
def fmt_warn(s):return "  [WARN] " + str(s)
def fmt_fail(s):return "  [FAIL] " + str(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("AD_VLM_MODEL", MODEL_DEFAULT))
    ap.add_argument("--load", action="store_true", help="真正加载模型测显存")
    ap.add_argument("--no-download", action="store_true", help="跳过联网下载测试")
    args = ap.parse_args()

    # ---------- 1. 基础 ----------
    banner("1) 基础 / Python")
    print(fmt_ok(f"Python: {sys.version.split()[0]}  (exe: {sys.executable})"))

    # ---------- 2. PyTorch + CUDA + 显存 ----------
    banner("2) PyTorch / CUDA / GPU 显存")
    try:
        import torch
        print(fmt_ok(f"torch: {torch.__version__}"))
        print(fmt_ok(f"CUDA available: {torch.cuda.is_available()}"))
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(fmt_ok(f"GPU count: {n}"))
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                vram_gb = p.total_memory / (1024**3)
                print(fmt_ok(f"GPU[{i}] {p.name}  total VRAM={vram_gb:.1f} GB"))
        else:
            print(fmt_fail("CUDA 不可用 —— AWQ/VLM 必须跑在 CUDA 上"))
    except Exception as e:
        print(fmt_fail(f"torch import 失败: {e}"))

    # ---------- 3. transformers 及 Qwen2.5-VL 支持 ----------
    banner("3) transformers + Qwen2.5-VL 支持")
    try:
        import transformers
        print(fmt_ok(f"transformers: {transformers.__version__}"))
        # Qwen2.5-VL 需要较新的 transformers（一般 >=4.57 才有 Qwen2_5_VL 类）
        has_q25 = False
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # noqa
            has_q25 = True
        except Exception as e:
            print(fmt_fail(f"无 Qwen2_5_VLForConditionalGeneration（transformers 可能过旧）: {e}"))
        if not has_q25:
            try:
                from transformers import AutoModelForImageTextToText  # noqa
                print(fmt_warn("无 Qwen2_5_VL 专用类，但有 AutoModelForImageTextToText（可试）"))
            except Exception:
                print(fmt_fail("连 AutoModelForImageTextToText 也没有"))
        # AWQ 支持
        try:
            from transformers import AwqConfig  # noqa
            print(fmt_ok("AwqConfig 可用"))
        except Exception:
            print(fmt_warn("无 AwqConfig（AWQ 量化加载可能走 autoawq 旧接口）"))
    except Exception as e:
        print(fmt_fail(f"transformers import 失败: {e}"))

    # ---------- 4. AWQ / bitsandbytes / qwen-vl-utils ----------
    banner("4) AWQ 相关依赖")
    for mod, label in [("awq", "autoawq(awq)"), ("bitsandbytes", "bitsandbytes"),
                       ("qwen_vl_utils", "qwen-vl-utils"),
                       ("torchvision", "torchvision")]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            print(fmt_ok(f"{label}: installed (v{v})"))
        except Exception as e:
            if mod in ("awq", "bitsandbytes"):
                # 这两个是 AWQ 的常用后端，缺了就大概率跑不了 AWQ 量化
                print(fmt_fail(f"{label}: 未安装 —— AWQ 量化加载通常需要它 -> {e}"))
            else:
                print(fmt_warn(f"{label}: 未安装 -> {e}"))

    # ---------- 5. 模型权重是否已在本地缓存 ----------
    banner("5) 模型本地缓存检查")
    target = args.model
    print(fmt_ok(f"目标模型: {target}"))
    try:
        from huggingface_hub import try_to_load_from_cache
        cache_found = False
        for fname in ["config.json", "model.safetensors", "model-00001-of-0000X.safetensors"]:
            p = try_to_load_from_cache(target, fname)
            if p:
                print(fmt_ok(f"  命中缓存: {p}"))
                cache_found = True
                break
        if not cache_found:
            print(fmt_warn("本地 HF 缓存中未找到该模型 -> 首次需要联网下载（若离线需先手动放入缓存）"))
    except Exception as e:
        print(fmt_warn(f"无法检查 HF 缓存: {e}"))

    # ---------- 6. 连通性 / 下载(小文件)测试 ----------
    if not args.no_download:
        section("6) 联网下载连通性（仅拉 processor，小文件）")
        try:
            from transformers import AutoProcessor
            print(f"    尝试 AutoProcessor.from_pretrained('{target}') ...")
            AutoProcessor.from_pretrained(target)
            print(fmt_ok("Processor 下载/加载成功（网络与镜像可达）"))
        except Exception as e:
            print(fmt_fail(f"Processor 下载失败（网络/镜像/HF_ENDPOINT 问题）: {e}"))
            print(fmt_warn("若为离线服务器：请把模型手动放进 HF 缓存后加 --no-download 重跑"))
    else:
        print(fmt_warn("已跳过联网下载测试 (--no-download)"))

    # ---------- 7. (可选) 真正加载 AWQ 模型 ----------
    if args.load:
        banner("7) 真正加载 AWQ 模型测显存（可能较慢/较大）")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            print(f"    正在加载 {target} ...")
            model = AutoModelForImageTextToText.from_pretrained(
                target, device_map="auto"
            )
            proc = AutoProcessor.from_pretrained(target)
            model.eval()
            print(fmt_ok("AWQ 模型加载成功！"))
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(fmt_ok(f"  当前显存占用: {alloc:.2f} / {total:.1f} GB"))
            # 释放
            del model, proc
            torch.cuda.empty_cache()
            print(fmt_ok("已释放模型，empty_cache 完成"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(fmt_fail(f"AWQ 加载失败: {e}"))
            print(fmt_warn("根据上面的报错判断是：版本太旧 / 缺 autoawq / 权重不在 / 显存不足 哪一种"))

    print("\n" + "=" * 60)
    print("自检完成。判断标准：")
    print("  1) CUDA 可用且显存充足(>=11G)")
    print("  2) transformers 支持 Qwen2.5-VL（尽量新版本）")
    print("  3) autoawq / bitsandbytes 至少一个可用")
    print("  4) --load 能真正把 AWQ 模型加载上 GPU 且不 OOM")
    print("全部满足 -> 再回来铺 vlm_tag / vlm_scene_check 业务代码。")
    print("=" * 60)

if __name__ == "__main__":
    main()
