# 干什么：重建【纯 TSR + 红绿灯 + 自有帧】的 7 类数据集（用户 2026-09-24 定："今晚把 tsr 和红绿灯数据
#         重新训练一轮，其他不要加了"，目标是"提升 TSR 和红绿灯在真实帧的 P 和 R"）。
#         类序与生产 tsr_yolo 完全一致 → 可从生产验证过的 tsr_s16/best.pt 暖启动。
# 包含：tsr_yolo(4 源标志整图) + 官方 MTSD 全量【整图】 + StarLine + BDD 的 traffic light
#       + 自有帧伪标签（A 组高置信框 / B 组空标注）+ 历史背景负样本。
# 不含：人/两轮车/小车/大车（BDD/KITTI）、任何地面标线（RLMD/CDSet/zebra）—— 二轮就是被这些把牌子摊薄的。
# ★ 两处关键修正（都有实测依据）：
#   ① MTSD 用**整图**而非裁剪块。二轮把牌子裁成"占画面 1/4"的块（half=max(bw,bh)*2）→
#      教出"牌子都很大"的先验，而自有帧牌子很小；这解释了二轮"训得越久自有帧越差"（epoch10 52% > epoch20 38%）。
#      生产验证过的 tsr_s16 吃的正是 MTSD 整图。整图直接写 zip 里的原始字节，不重编码，快。
#   ② 加**自有帧伪标签**。公开数据堆到 9.5 万张也动不了自有帧（88%→8%），真实帧只能靠自有帧域数据。
# 怎么跑：AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python -u tools/_build_tsr7_v2.py
# 怎么判定成功：末尾 train/val 图数一致、逐类框数打印，且 tsr7_v2/data.yaml 的 names 是 7 类。
# ⚠️ 前置：先跑 tools/_mine_ownframe_labels.py 生成 /opt/datasets/ownframes（本脚本只做软链，不复制）。
import io, json, os, shutil, sys, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _build_unified_dataset as U                                            # noqa: E402
from _build_tsr_dataset import mtsd_group                                     # noqa: E402

OUT = "/opt/datasets/tsr7_v2"
OWNF = "/opt/datasets/ownframes"
CLASSES = ["speed_limit", "prohibition", "warning", "mandatory", "guide", "signal", "crosswalk"]
U.OUT = OUT
U.CID = {"speed_limit": 0, "prohibition": 1, "warning": 2, "mandatory": 3,
         "guide": 4, "traffic_light": 5, "crosswalk_sign": 6}


def stage_bdd_lights(rep):
    """BDD100K 里**只取红绿灯**框（其余人车类一律不要）。"""
    root = os.path.join(U.D, "stage2", "dgural_bdd100k")
    sp = os.path.join(root, "samples.json")
    if not os.path.exists(sp):
        print("  bdd_lights: 未下载，跳过"); return
    d = json.load(io.open(sp, encoding="utf-8"))
    n = 0
    for s in d.get("samples") or []:
        fp = os.path.join(root, s.get("filepath") or "")
        if not os.path.exists(fp):
            continue
        lines = []
        for det in ((s.get("detections") or {}).get("detections") or []):
            if str(det.get("label", "")).lower().strip() != "traffic light":
                continue
            bb = det.get("bounding_box") or []
            if len(bb) != 4:
                continue
            x, y, w, h = [float(v) for v in bb]
            if w <= 0 or h <= 0 or w * h < 1e-5:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (U.CID["traffic_light"], x + w / 2, y + h / 2, w, h))
            rep["boxes"]["traffic_light"] += 1
        if lines:
            key = "bddlight_" + os.path.basename(fp)
            if U._copy(fp, key, lines, U.split_of(key)):
                n += 1
    print("  bdd_lights: %d 张（只有红绿灯框）" % n)


def stage_mtsd_full_images(rep, max_imgs=50000):
    """官方 MTSD 全量【整图】：直接写 zip 里的原始 JPEG 字节（不裁剪、不重编码）。

    ⚠️ 只排除物理上不在画面里的框（exterior / out-of-frame）；**不能用 properties.included 过滤**
      —— 官方这份里 included=False 占 94%，其中绝大多数是完全干净的框（2026-09-24 实测）。
    """
    root = os.path.join(U.D, "tsr_more", "crimedetector_roadsign")
    az = os.path.join(root, "mtsd_fully_annotated_annotation.zip")
    if not os.path.exists(az):
        print("  mtsd_full: 标注包不在，跳过"); return
    ann = {}
    zf = zipfile.ZipFile(az)
    for nm in zf.namelist():
        if not nm.endswith(".json"):
            continue
        try:
            d = json.load(io.BytesIO(zf.read(nm)))
        except Exception:
            continue
        iid = os.path.splitext(os.path.basename(nm))[0]
        W, H = float(d.get("width") or 0), float(d.get("height") or 0)
        if not W or not H:
            continue
        objs = []
        for o in d.get("objects") or []:
            p = o.get("properties") or {}
            if p.get("exterior") or p.get("out-of-frame"):
                continue
            # ⚠️ mtsd_group 返回**元组** (组名, 官方类名)，只接一个值会让 `g in CID` 永远为假
            #    → 整批 MTSD 静默变成 0 张（2026-09-24 实测踩到）
            _g, _k = mtsd_group(o.get("label") or "")
            g = U._uni(_g) if _g else None
            bb = o.get("bbox") or {}
            if g and g in U.CID and all(k in bb for k in ("xmin", "ymin", "xmax", "ymax")):
                objs.append((g, bb))
        if objs:
            ann[iid] = (W, H, objs)
    print("  mtsd_full: 有可用标注的图 %d 张（整图口径）" % len(ann))
    if not ann:
        return
    picked = set(sorted(ann)[:max_imgs])
    n = 0
    for part in ("test", "train.0", "train.1", "train.2", "val"):
        z = os.path.join(root, "mtsd_fully_annotated_images.%s.zip" % part)
        if not os.path.exists(z):
            continue
        iz = zipfile.ZipFile(z)
        for nm in iz.namelist():
            if not nm.lower().endswith(".jpg"):
                continue
            iid = os.path.splitext(os.path.basename(nm))[0]
            if iid not in picked:
                continue
            W, H, objs = ann[iid]
            lines = []
            for g, bb in objs:
                w, h = (bb["xmax"] - bb["xmin"]) / W, (bb["ymax"] - bb["ymin"]) / H
                if w <= 0 or h <= 0 or w > 1 or h > 1:
                    continue
                lines.append("%d %.6f %.6f %.6f %.6f" % (
                    U.CID[g], (bb["xmin"] + bb["xmax"]) / 2 / W, (bb["ymin"] + bb["ymax"]) / 2 / H, w, h))
                rep["boxes"][g] += 1
            if not lines:
                continue
            key = "mtsdfull_%s" % iid
            sp = U.split_of(key)
            ip = os.path.join(OUT, "images", sp, key + ".jpg")
            lp = os.path.join(OUT, "labels", sp, key + ".txt")
            if os.path.exists(ip):
                continue
            with open(ip, "wb") as f:                 # 原始字节直写，不重编码
                f.write(iz.read(nm))
            with open(lp, "w") as f:
                f.write("\n".join(lines) + "\n")
            n += 1
            picked.discard(iid)
    print("  mtsd_full: 落盘 %d 张整图" % n)


NN = {"0": "speed_limit", "1": "prohibition", "2": "warning", "3": "mandatory",
      "4": "guide", "5": "traffic_light", "6": "crosswalk_sign"}


def stage_ownframes(rep):
    """自有帧伪标签（软链回 ownframes）。空标签的就是负样本，原样带过来。"""
    if not os.path.isdir(os.path.join(OWNF, "labels")):
        print("  ownframes: 未生成，跳过（先跑 _mine_ownframe_labels.py）"); return
    n = nbox = 0
    for sp in ("train", "val"):
        ld = os.path.join(OWNF, "labels", sp)
        if not os.path.isdir(ld):
            continue
        for fn in os.listdir(ld):
            stem = os.path.splitext(fn)[0]
            src = os.path.join(OWNF, "images", sp, stem + ".jpg")
            if not os.path.exists(src):
                continue
            lines = [ln.strip() for ln in io.open(os.path.join(ld, fn), encoding="utf-8") if ln.strip()]
            key = "ownf_" + stem
            ip = os.path.join(OUT, "images", sp, key + ".jpg")
            lp = os.path.join(OUT, "labels", sp, key + ".txt")
            try:
                os.symlink(src, ip)
            except Exception:
                try:
                    os.link(src, ip)
                except Exception:
                    shutil.copy(src, ip)
            with open(lp, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n += 1
            for ln in lines:
                # 自有帧标签是 7 类 id，计数要查回**统一侧类名**再进 rep["boxes"]
                # （写成 U.CLASSES[名字] 是拿字符串索引列表 → TypeError，已踩）
                rep["boxes"][NN[str(ln.split()[0])]] += 1
                nbox += 1
    print("  ownframes: %d 帧（框 %d）" % (n, nbox))


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for s in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    # stage_starline 会往 rep["unmapped"] 里记未映射的类名 —— 少这个键直接 KeyError（已踩）
    rep = {"boxes": {c: 0 for c in U.CLASSES}, "unmapped": []}
    print("[7 类重建] 类序: %s" % " ".join(CLASSES))
    U.stage_signs(rep)
    stage_bdd_lights(rep)
    U.stage_lisa(rep)
    stage_mtsd_full_images(rep, max_imgs=50000)
    U.stage_starline(rep, per_img=1)
    U.stage_negs(rep)
    stage_ownframes(rep)

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % OUT)
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))

    tot = {}
    for sp in ("train", "val"):
        ni = len(os.listdir(os.path.join(OUT, "images", sp)))
        nl = len(os.listdir(os.path.join(OUT, "labels", sp)))
        nbox = 0
        for fn in os.listdir(os.path.join(OUT, "labels", sp)):
            for ln in io.open(os.path.join(OUT, "labels", sp, fn), encoding="utf-8"):
                if ln.strip():
                    nbox += 1
                    tot[CLASSES[int(ln.split()[0])]] = tot.get(CLASSES[int(ln.split()[0])], 0) + 1
        print("  %s: 图 %d / 标 %d / 框 %d%s" % (sp, ni, nl, nbox, "" if ni == nl else "  ✗ 图数不符"))
    print("  合计框 %d" % sum(tot.values()))
    for c in CLASSES:
        print("    %-16s %6d" % (c, tot.get(c, 0)))
    print("[完成] %s/data.yaml" % OUT)


if __name__ == "__main__":
    main()
