# 干什么：用现役 tsr_sign.pt（红绿灯再叠 yolov8x 共识）在【自有帧】上挖伪标签，供今晚那轮训练用。
#         动机（2026-09-24 实测）：公开数据堆到 9.5 万张，自有帧 A 组段级命中反而从 88% 掉到 8%
#         —— 公开数据再多也动不了真实帧；真实帧的域匹配只能靠自有帧自己的标注。
#   正样本：判为「有标识」的段，sign 模型 conf≥0.35 的框；**只保留有框的帧**（无框帧不当负样本）。
#   红绿灯：sign 模型 conf≥0.35 且与 yolov8x(COCO traffic light) 的框 IoU≥0.25 —— 两个独立模型都认才算，
#          规避「路灯被当成红绿灯」这种自伤（YOLO 单模型在自有帧上就有这个毛病）。
#   负样本：判为「无标识」的段，整帧空标注（哪怕 sign 模型给了框也一律丢掉 —— 这些段已确认没有标识）。
# 输出：/opt/datasets/ownframes/{images,labels}/{train,val} + data.yaml（7 类，与生产同序）
#       ⚠️ 帧图一律**软链**回 NAS 原始路径（不复制！复制 1 万张要几十 GB，且 NAS 是"能建不能删"）；
#          YOLO 靠把路径里的 /images/ 换成 /labels/ 找标签，所以软链名必须与标签同名。
#       正样本帧再建 K_COPY 个软链（不同名指向同一图）—— 在 YOLO 里等价于给自有帧加权。
# 怎么跑：AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python -u tools/_mine_ownframe_labels.py
# 怎么判定成功：末尾打印 正样本帧/框数、负样本帧数、逐类框数，以及 ownframes/data.yaml；磁盘几乎不涨。
import hashlib, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _eval_checkpoints as E                                                 # noqa: E402

STORE = E.STORE
OUT = "/opt/datasets/ownframes_v2"   # 第二轮：阈值降到 0.25、另存一份，不覆盖第一轮的
SIGN_W = "/opt/ad_mining/tsr_sign.pt"
YOLO_W = "/opt/ad_mining/yolov8x.pt"
CLASSES = ["speed_limit", "prohibition", "warning", "mandatory", "guide", "signal", "crosswalk"]
CONF_SIGN = float(os.environ.get("AD_MINE_CONF", "0.25"))   # 第二轮降到 0.25：多挖正样本，针对 A 组禁令/警告回退
CONF_LIGHT_COCO = 0.25    # COCO 侧放低一点，只用来做共识
IOU_LIGHT = 0.25
K_COPY = 2                # 正样本帧复制份数（软链，零磁盘）
MAX_POS_SEG = 200
PER_SEG = 30              # 一段最多取多少帧
MAX_NEG_SEG = 80
NEG_PER_SEG = 3
MAX_FRAMES = 12000        # 正样本总帧数上限（控制构建时长）


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def split_of(key):
    return "val" if int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"


def link(src, dst):
    """软链（同机可用）；失败再退回复制。"""
    if os.path.exists(dst):
        return True
    try:
        os.symlink(src, dst)
        return True
    except Exception:
        try:
            shutil.copy(src, dst)
            return True
        except Exception:
            return False


def main():
    projects = [p for p in sorted(os.listdir(STORE))
                if os.path.exists(os.path.join(STORE, p, "clips.json")) and p != "thumbs"]
    print("[项目] %s" % "、".join(projects))
    pos, neg = [], []
    for p in projects:
        has, non = E.pick_segments(p, MAX_POS_SEG, 20260923)
        meta = E.load_json(os.path.join(STORE, p, "metadata.json"))
        for _cid, fids in has:
            for iid in fids[:PER_SEG]:
                if 0 <= int(iid) < len(meta):
                    fp = meta[int(iid)].get("path")
                    if fp and os.path.exists(fp):
                        pos.append(fp)
        for _cid, fids in non[:MAX_NEG_SEG]:
            for iid in fids[:NEG_PER_SEG]:
                if 0 <= int(iid) < len(meta):
                    fp = meta[int(iid)].get("path")
                    if fp and os.path.exists(fp):
                        neg.append(fp)
        if len(pos) >= MAX_FRAMES:
            print("  → 已达正样本上限 %d，后续项目不再取" % MAX_FRAMES)
            break
    pos = pos[:MAX_FRAMES]
    print("[候选] 正样本帧 %d ｜ 负样本帧 %d" % (len(pos), len(neg)))
    if not pos:
        print("✗ 没有正样本帧，停止（先确认项目的 traffic_sign 判定结果还在）"); sys.exit(2)

    from ultralytics import YOLO
    ms = YOLO(SIGN_W)
    names = ms.names
    print("[sign 模型] %s（%d 类）阈值 %.2f" % (os.path.basename(SIGN_W), len(names), CONF_SIGN))

    keep = []            # (path, [(cls_i, xyxy)], orig_shape)
    sig_frames = []      # 有 signal 候选的帧
    sig_boxes = {}       # path -> [xyxy, ...]
    print("[第一遍] %d 帧 …" % len(pos))
    for k in range(0, len(pos), 16):
        batch = pos[k:k + 16]
        for fp, r in zip(batch, ms.predict(batch, imgsz=1280, conf=CONF_SIGN, verbose=False)):
            box, sig = [], []
            for b in r.boxes:
                nm = str(names[int(b.cls[0])])
                bb = [float(v) for v in b.xyxy[0]]
                if nm == "signal":
                    sig.append(bb)
                elif nm in CLASSES:
                    box.append((CLASSES.index(nm), bb))
            if sig:
                sig_frames.append(fp)
                sig_boxes[fp] = sig
            if box:
                keep.append((fp, box, r.orig_shape))
    print("[正样本] 有高置信框的帧 %d / %d ｜ 有红绿灯候选的帧 %d" % (len(keep), len(pos), len(sig_frames)))

    n_light = 0
    if sig_frames:
        my = YOLO(YOLO_W)
        yn = my.names
        for k in range(0, len(sig_frames), 8):
            batch = sig_frames[k:k + 8]
            for fp, r in zip(batch, my.predict(batch, imgsz=1280, conf=CONF_LIGHT_COCO, verbose=False)):
                cb = [[float(v) for v in b.xyxy[0]] for b in r.boxes
                      if str(yn[int(b.cls[0])]).lower().strip() == "traffic light"]
                if not cb:
                    continue
                ok = [bb for bb in sig_boxes.get(fp, []) if any(iou(bb, c) >= IOU_LIGHT for c in cb)]
                if ok:
                    for fp2, box, sh in keep:
                        if fp2 == fp:
                            for bb in ok:
                                box.append((CLASSES.index("signal"), bb))
                            break
                    n_light += len(ok)
        del my
        print("[红绿灯] 双模型共识通过 %d 框" % n_light)
    del ms
    try:
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass

    if os.path.isdir(OUT):
        for s in ("images", "labels"):
            shutil.rmtree(os.path.join(OUT, s), ignore_errors=True)
    for s in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)

    cnt = {c: 0 for c in CLASSES}
    npos = nlink = 0
    for fp, box, sh in keep:
        H, W = float(sh[0]), float(sh[1])
        lines = []
        for ci, bb in box:
            cx, cy = (bb[0] + bb[2]) / 2 / W, (bb[1] + bb[3]) / 2 / H
            w, h = (bb[2] - bb[0]) / W, (bb[3] - bb[1]) / H
            if w <= 0 or h <= 0 or w > 1 or h > 1:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (ci, cx, cy, w, h))
            cnt[CLASSES[ci]] += 1
        if not lines:
            continue
        sp = split_of(os.path.basename(fp))
        for j in range(K_COPY):
            nm = "own%d_%s" % (j, os.path.basename(fp))
            if link(fp, os.path.join(OUT, "images", sp, nm)):
                nlink += 1
                with open(os.path.join(OUT, "labels", sp, os.path.splitext(nm)[0] + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
        npos += 1
    nneg = 0
    for fp in neg:
        sp = split_of("neg" + os.path.basename(fp))
        nm = "ofneg_" + os.path.basename(fp)
        if link(fp, os.path.join(OUT, "images", sp, nm)):
            open(os.path.join(OUT, "labels", sp, os.path.splitext(nm)[0] + ".txt"), "w").close()
            nneg += 1
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % OUT)
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))
    print("\n[完成] %s" % OUT)
    print("  正样本帧 %d（软链 %d 个）｜ 负样本帧 %d" % (npos, nlink, nneg))
    for c in CLASSES:
        print("    %-16s %6d 框（按原帧计）" % (c, cnt[c]))
    for sp in ("train", "val"):
        print("  %s: 图 %d / 标 %d" % (sp, len(os.listdir(os.path.join(OUT, "images", sp))),
                                       len(os.listdir(os.path.join(OUT, "labels", sp)))))


if __name__ == "__main__":
    main()
