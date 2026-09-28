#!/venv/bin/python
# -*- coding: utf-8 -*-
"""
一次性脚本：训练闭集交通标志（语义大类）检测器。

干什么：用 /opt/datasets/tsr_yolo 训练 YOLOv8，产出能区分"真交通牌 vs 广告牌"的闭集模型。
怎么跑：
  # 必须先停服务（12G 卡装不下 训练+VLM；历史上硬抢显存把机器打爆过）
  systemctl stop ad_mining
  AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python \\
      /opt/ad_mining/tools/_train_tsr.py --weights /opt/datasets/tsr_weights/yolov8s.pt --epochs 60
  训练完必须把服务起回来： systemctl start ad_mining
怎么判定成功：
  末尾打印 best.pt 路径 + val 的 mAP50/mAP50-95；
  再用 tools/_eval_tsr_ownframes.py 在**我们自己的帧**上量广告牌误检率（val mAP 高 ≠ 这个好）。
"""
import argparse, os, shutil, subprocess, sys

def gpu_free_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[0]
        return float(out) / 1024.0
    except Exception:
        return -1.0

def service_running():
    try:
        r = subprocess.run(["systemctl", "is-active", "ad_mining"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return r.stdout.decode().strip() == "active"
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/opt/datasets/tsr_yolo/data.yaml")
    ap.add_argument("--weights", default="/opt/datasets/tsr_weights/yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="tsr_v1")
    ap.add_argument("--lr0", type=float, default=0.01, help="暖启动微调时调小（如 0.005）")
    ap.add_argument("--save-period", type=int, default=-1,
                    help=">0 时每 N 轮存一份 weights/epochN.pt —— 用来后置跑『逐类×每轮』P/R 曲线")
    a = ap.parse_args()

    # 闸门：服务在跑就拒绝（除非显式 AD_TRAIN_FORCE=1）。别在 12G 卡上跟 VLM 抢显存。
    if service_running() and os.environ.get("AD_TRAIN_FORCE") != "1":
        print("✗ ad_mining 正在运行 —— 先 systemctl stop ad_mining（12G 装不下 训练+VLM）。")
        print("  确认要硬跑就加 AD_TRAIN_FORCE=1（不推荐，会 OOM 并可能连 VS Code 一起被杀）。")
        sys.exit(2)
    free = gpu_free_gb()
    print("[GPU] 空闲显存 %.1f GB（yolov8s@%d batch%d 约需 6~8G）" % (free, a.imgsz, a.batch))
    if 0 <= free < 6.0:
        print("✗ 显存不足，别跑。停掉占用进程后重试。"); sys.exit(3)
    if not os.path.exists(a.data) or not os.path.exists(a.weights):
        print("✗ 缺 %s 或 %s" % (a.data, a.weights)); sys.exit(4)
    n = sum(len(os.listdir(os.path.join(os.path.dirname(a.data), "images", s)))
            for s in ("train", "val") if os.path.isdir(os.path.join(os.path.dirname(a.data), "images", s)))
    print("[数据] %d 张图；weights=%s" % (n, a.weights))

    from ultralytics import YOLO
    m = YOLO(a.weights)
    m.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers,
            project="/opt/datasets/tsr_runs", name=a.name, device=0, seed=0, patience=15,
            cos_lr=True, close_mosaic=10, val=True, plots=True, lr0=a.lr0, save_period=a.save_period,
            # 小目标为主 → 不做大尺度增强裁切，保留原尺度分布
            scale=0.3, mosaic=1.0, mixup=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4)
    print("\n=== 训练结束 ===")
    print("best.pt → /opt/datasets/tsr_runs/%s/weights/best.pt" % a.name)
    try:
        r = m.val(data=a.data, imgsz=a.imgsz, split="val")
        print("val mAP50=%.4f  mAP50-95=%.4f" % (r.box.map50, r.box.map))
        for j, ci in enumerate(list(r.box.ap_class_index)):
            nm = r.names[int(ci)] if isinstance(r.names, dict) else r.names[int(ci)]
            print("   %-16s P=%.3f R=%.3f mAP50=%.3f" % (nm, float(r.box.p[j]), float(r.box.r[j]),
                                                         float(r.box.all_ap[j][0])))
    except Exception as e:
        print("val 失败（不影响权重）:", str(e)[:200])
    # ⚠️ 别把 best.pt 直接拷成 /opt/ad_mining/tsr_sign.pt —— val mAP 高 ≠ 自有帧好
    #    （2026-09-24 教训：二轮统一模型 val mAP50 0.674、自有帧段级只有 8%，而现役模型 98%）。
    print("→ 下一步：先跑 tools/_eval_checkpoints.py 在**自有帧**上对比现役模型，通过后才部署（并先备份旧权重）")


if __name__ == "__main__":
    main()
