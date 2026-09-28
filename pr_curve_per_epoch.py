# 干什么：把训练过程中 save_period=1 存下的每轮权重（weights/*.pt）逐个跑一次验证，
#         产出【逐类 × 每轮】的 P/R/mAP 表 —— Ultralytics 每轮只把总体 P/R 写进 results.csv，
#         逐类 P/R 必须用每轮权重后置跑才能拿到（2026-09-24 用户明确要过，上一轮因没存权重拿不到）。
# 怎么跑：AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python -u tools/_pr_curve_per_epoch.py
# 怎么判定成功：打印 "轮次 × 类" 矩阵（P/R/mAP50），并写 reports/pr_curve_<run>.html。
# ⚠️ 训练刚结束时用：GPU 空着才跑得动；imgsz 640、batch 4，别在现场抢显存。
import glob, io, os, re, sys

RUN = os.environ.get("AD_PR_RUN", "/opt/datasets/tsr_runs/tsr7_v2")
DATA = os.environ.get("AD_PR_DATA", "/opt/datasets/tsr7_v2/data.yaml")
MINI = "/opt/datasets/tsr7_v2_mini"          # 小验证子集，20 个权重才跑得完
N_MINI = 300
CLASSES = ["speed_limit", "prohibition", "warning", "mandatory", "guide", "signal", "crosswalk"]


def build_mini():
    import random
    src_l = os.path.join(os.path.dirname(DATA), "labels", "val")
    files = [f for f in os.listdir(src_l) if os.path.getsize(os.path.join(src_l, f)) > 0]
    random.Random(0).shuffle(files)
    files = files[:N_MINI]
    for s in ("images/val", "labels/val"):
        os.makedirs(os.path.join(MINI, s), exist_ok=True)
    root = os.path.dirname(DATA)
    n = 0
    for f in files:
        stem = os.path.splitext(f)[0]
        ip = os.path.join(root, "images", "val", stem + ".jpg")
        if not os.path.exists(ip):
            continue
        di = os.path.join(MINI, "images", "val", stem + ".jpg")
        dl = os.path.join(MINI, "labels", "val", stem + ".txt")
        for a, b in ((ip, di), (os.path.join(src_l, f), dl)):
            if not os.path.exists(b):
                try:
                    os.symlink(a, b)
                except Exception:
                    pass
        n += 1
    with open(os.path.join(MINI, "data.yaml"), "w") as fh:
        fh.write("path: %s\ntrain: images/val\nval: images/val\nnames:\n" % MINI)
        for i, c in enumerate(CLASSES):
            fh.write("  %d: %s\n" % (i, c))
    print("[mini val] %d 张（有框的验证图）" % n)
    return n


def main():
    if not os.path.isdir(MINI):
        if build_mini() == 0:
            print("✗ mini val 建不出来"); sys.exit(2)
    ws = []
    for p in glob.glob(os.path.join(RUN, "weights", "*.pt")):
        m = re.search(r"(\d+)", os.path.basename(p))
        ep = int(m.group(1)) if m else None
        ws.append((ep if ep is not None else 9999, p))
    ws.sort()
    if not ws:
        print("✗ %s/weights 下没有权重" % RUN); sys.exit(3)
    print("[权重] %d 个：%s" % (len(ws), "、".join(os.path.basename(p) for _e, p in ws)))

    from ultralytics import YOLO
    rows = {}
    for ep, p in ws:
        try:
            m = YOLO(p)
            r = m.val(data=os.path.join(MINI, "data.yaml"), imgsz=640, batch=4, device=0,
                      split="val", plots=False, verbose=False, project="/tmp/pr_curve",
                      name="ep%s" % ep, exist_ok=True)
        except Exception as e:
            print("  ep%s 失败: %s" % (ep, str(e)[:120])); continue
        tab = {}
        for j, ci in enumerate(list(r.box.ap_class_index)):
            tab[CLASSES[int(ci)]] = (float(r.box.p[j]), float(r.box.r[j]), float(r.box.all_ap[j][0]))
        rows[ep] = tab
        print("  ep%-3s 总体 mAP50=%.3f（牌子类均值 %.3f）" % (
            ep, float(r.box.map50),
            sum(tab[c][2] for c in tab) / max(1, len(tab))), flush=True)
        del m
        try:
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
    if not rows:
        print("✗ 一个权重都没跑成功"); sys.exit(4)

    print("\n=== 逐类 × 每轮（mAP50）===")
    eps = sorted(rows)
    print("%-16s %s" % ("类", " ".join("ep%-4s" % e for e in eps)))
    for c in CLASSES:
        print("%-16s %s" % (c, " ".join(("%-6.3f" % rows[e][c][2]) if c in rows[e] else "  --  " for e in eps)))
    print("\n=== 逐类 × 每轮（R 召回）===")
    print("%-16s %s" % ("类", " ".join("ep%-4s" % e for e in eps)))
    for c in CLASSES:
        print("%-16s %s" % (c, " ".join(("%-6.3f" % rows[e][c][1]) if c in rows[e] else "  --  " for e in eps)))
    print("\n=== 逐类 × 每轮（P 精确率）===")
    print("%-16s %s" % ("类", " ".join("ep%-4s" % e for e in eps)))
    for c in CLASSES:
        print("%-16s %s" % (c, " ".join(("%-6.3f" % rows[e][c][0]) if c in rows[e] else "  --  " for e in eps)))
    print("\n[口径] mini val = %d 张有框验证图；imgsz 640；P/R 取 IoU=0.5 最优点" % N_MINI)


if __name__ == "__main__":
    main()
