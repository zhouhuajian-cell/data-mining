#!/venv/bin/python
# -*- coding: utf-8 -*-
"""构建**统一闭集检测器**训练集：交通标识 + 红绿灯 + 目标物 + 地面标线（用户 2026-09-23 定的主线）。

为什么要统一：现在牌子由闭集模型给、目标物靠 YOLO COCO、标线**没有任何模型**；
合成一个模型后 —— ① 少一次读图/推理；② **路灯与红绿灯能成为并列类**（治"路灯被认成红绿灯"，阈值救不了）；
③ 类别直接对齐本体（`objects` / `traffic_sign` / `road_marking` 三维一次给全）。

数据来源与门槛（**用户 2026-09-23 授权："你自己评估，过你那关就直接训练"**）：
 ✔ Mapillary/越南/德国/TT100K（已在 `/opt/datasets/tsr_yolo`，7 类标志）→ 直接复用并重映射
 ✔ BDD100K（1 万图 1280×720 **原始分辨率**；FiftyOne json，label + 归一化 xywh）→ 行人/两轮/小车/大车/红绿灯
 ✔ KITTI（7,481 图 1224×370，parquet；绝对 xywh + 类别 int）→ 小车/大车/行人/两轮
 ✔ RLMD（2,137 图 **25 类掩码 PNG**，图与掩码同名配对）→ 人行横道/停止线/导流线/网状线/待行区/车道线（掩码→框）
 ✔ CDSet-3434（**自带 YOLO 格式**）→ 人行横道
 ✗ 印尼停止线：train 仅 17 个标注文件（标注不全）→ **不整集入训**（只用其有标注的 test）
 ✗ 三轮车/异形车/锥桶/围挡/路灯：公开集没有 → 自标（后续补，本脚本先留类位）
 ✔ 自有帧负样本：`无标识`段整帧（空标注）

怎么跑：/venv/bin/python /opt/ad_mining/tools/_build_unified_dataset.py --stages signs,bdd,kitti,rlmd,cdset,negs
怎么判定成功：末尾打印 每类框数 与 **未归类命中（必须为空）**；images/labels 数量一致；
  data.yaml 的 names 与本体映射表一致；再抽查几张图看框对不对（用 `_tsr_mapping_check.py` 类似做法）。
"""
import argparse, glob, hashlib, io, json, os, shutil, sys, zipfile
from collections import Counter, defaultdict

OUT = "/opt/datasets/unified"
D = "/opt/datasets"


# 复用 _build_tsr_dataset 里已验证过的类名归并规则（官方 MTSD 的 label 就是这套官方类名）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build_tsr_dataset import mtsd_group, tt100k_group, to_group      # noqa: E402


def hashlib_s(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ── 统一类别表（顺序即 YOLO 类别 id；名称与本体对应关系写在 CN 里）──
CLASSES = [
    # 交通标识（沿用闭集模型 7 类）
    "speed_limit", "prohibition", "warning", "mandatory", "guide", "traffic_light", "crosswalk_sign",
    # 目标物（本体 objects）
    "pedestrian", "two_wheeler", "car", "big_vehicle",
    # 地面标线（本体 road_marking）
    "marking_crosswalk", "marking_stopline", "marking_channelizing", "marking_box_junction",
    "marking_waiting", "lane_line",
]
CID = {c: i for i, c in enumerate(CLASSES)}
CN = {"speed_limit": "限速", "prohibition": "禁令标志", "warning": "警告标志", "mandatory": "指示标志",
      "guide": "指路标志", "traffic_light": "红绿灯", "crosswalk_sign": "人行横道标志",
      "pedestrian": "行人", "two_wheeler": "两轮车", "car": "小车", "big_vehicle": "大车",
      "marking_crosswalk": "人行横道", "marking_stopline": "停止线", "marking_channelizing": "导流线",
      "marking_box_junction": "网状线", "marking_waiting": "待行区", "lane_line": "车道线"}

# BDD100K → 统一类
BDD_MAP = {"pedestrian": "pedestrian", "bicycle": "two_wheeler", "motorcycle": "two_wheeler",
           "rider": "two_wheeler", "car": "car", "bus": "big_vehicle", "truck": "big_vehicle",
           "traffic light": "traffic_light"}
# KITTI 类别 id 顺序（官方 detection benchmark：0 Car,1 Van,2 Truck,3 Pedestrian,
# 4 Person_sitting,5 Cyclist,6 Tram,7 Misc）
KITTI_MAP = {0: "car", 1: "car", 2: "big_vehicle", 3: "pedestrian", 4: "pedestrian",
             5: "two_wheeler", 6: "big_vehicle"}
# RLMD 掩码值 → 统一类（值为 rlmd.csv 的 id）
RLMD_MAP = {1: "marking_box_junction", 2: "marking_crosswalk", 3: "marking_stopline",
            16: "marking_channelizing", 20: "marking_waiting", 21: "marking_waiting",
            4: "lane_line", 5: "lane_line", 6: "lane_line", 7: "lane_line", 8: "lane_line",
            9: "lane_line", 10: "lane_line"}


def _emit(split, data, name, lines, tag):
    ip = os.path.join(OUT, "images", split, name)
    lp = os.path.join(OUT, "labels", split, os.path.splitext(name)[0] + ".txt")
    if os.path.exists(ip):
        return False
    with open(ip, "wb") as f:
        f.write(data)
    with open(lp, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return True


def _copy(path, name, lines, split="train"):
    """优先硬链接（省磁盘），失败再复制。"""
    ip = os.path.join(OUT, "images", split, name)
    lp = os.path.join(OUT, "labels", split, os.path.splitext(name)[0] + ".txt")
    if os.path.exists(ip):
        return False
    try:
        os.link(path, ip)
    except Exception:
        shutil.copy(path, ip)
    with open(lp, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return True


def split_of(key):
    """确定性地按名字哈希分 train/val（10% 进 val）。"""
    return "val" if hashlib_s(key) % 10 == 0 else "train"


# ───────────────────────── stages ─────────────────────────
def stage_signs(rep):
    """复用已建好的 7 类标志集（tsr_yolo），把类别 id 重映射到统一表。"""
    src = os.path.join(D, "tsr_yolo")
    old = ["speed_limit", "prohibition", "warning", "mandatory", "guide", "signal", "crosswalk"]
    remap = {i: CID["traffic_light" if c == "signal" else ("crosswalk_sign" if c == "crosswalk" else c)]
             for i, c in enumerate(old)}
    n = 0
    for sp in ("train", "val"):
        ld = os.path.join(src, "labels", sp)
        if not os.path.isdir(ld):
            continue
        for fn in os.listdir(ld):
            stem = os.path.splitext(fn)[0]
            ip = os.path.join(src, "images", sp, stem + ".jpg")
            if not os.path.exists(ip):
                continue
            lines = []
            for ln in io.open(os.path.join(ld, fn), encoding="utf-8"):
                p = ln.split()
                if len(p) >= 5:
                    k = int(p[0])
                    if k in remap:
                        lines.append("%d %s" % (remap[k], " ".join(p[1:5])))
                        rep["boxes"][old[k] if old[k] not in ("signal", "crosswalk")
                                     else ("traffic_light" if old[k] == "signal" else "crosswalk_sign")] += 1
            if lines:
                _copy(ip, "sign_%s_%s.jpg" % (sp, stem), lines, split_of("sign" + stem))
                n += 1
    print("  signs: %d 张（复用 tsr_yolo + 类别重映射）" % n)


def stage_bdd(rep, per_img=0):
    """BDD100K（FiftyOne json）：label + 归一化 [x,y,w,h] → YOLO [cx,cy,w,h]。"""
    root = os.path.join(D, "stage2", "dgural_bdd100k")
    sp = os.path.join(root, "samples.json")
    if not os.path.exists(sp):
        print("  bdd: 未下载，跳过"); return
    d = json.load(io.open(sp, encoding="utf-8"))
    n = 0
    for s in d.get("samples") or []:
        fp = os.path.join(root, s.get("filepath") or "")
        if not os.path.exists(fp):
            continue
        lines = []
        for det in ((s.get("detections") or {}).get("detections") or []):
            tgt = BDD_MAP.get(str(det.get("label", "")).lower().strip())
            bb = det.get("bounding_box") or []
            if not tgt or len(bb) != 4:
                continue
            x, y, w, h = [float(v) for v in bb]
            if w <= 0 or h <= 0 or w * h < 1e-5:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (CID[tgt], x + w / 2, y + h / 2, w, h))
            rep["boxes"][tgt] += 1
        if lines:
            key = "bdd" + os.path.basename(fp)
            _copy(fp, key, lines, split_of(key))
            n += 1
    print("  bdd: %d 张" % n)


def stage_kitti(rep):
    """KITTI：parquet 内嵌 PNG + 绝对 [x,y,w,h]（category 是 int，需按官方类序查表）。"""
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(os.path.join(D, "stage2",
                                       "KingRam_Kitti-Object-Detection-Evaluation-2012", "data", "*.parquet")))
    if not fs:
        print("  kitti: 未下载，跳过"); return
    n = 0
    for f in fs:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=24):
            for r in batch.to_pylist():
                W, H = float(r["width"]), float(r["height"])
                lines = []
                o = r.get("objects") or {}
                for cat, bb in zip(o.get("category") or [], o.get("bbox") or []):
                    tgt = KITTI_MAP.get(int(cat))
                    if int(cat) == 8:
                        continue          # KITTI 的 DontCare = 官方"忽略区域"，本就不是目标（11,517 个，正常）
                    if not tgt or len(bb) != 4:
                        rep["unmapped"].append("KITTI:%s" % cat); continue
                    x, y, w, h = [float(v) for v in bb]
                    if w <= 0 or h <= 0:
                        continue
                    lines.append("%d %.6f %.6f %.6f %.6f" % (CID[tgt], (x + w / 2) / W, (y + h / 2) / H,
                                                            w / W, h / H))
                    rep["boxes"][tgt] += 1
                if lines:
                    key = "kitti_" + str(r["image_id"])
                    if _emit(split_of(key), r["image"]["bytes"], key + ".png", lines, "kitti"):
                        n += 1
    print("  kitti: %d 张" % n)


def stage_rlmd(rep, min_area=150):
    """RLMD：掩码 PNG（**RGB 配色图**，不是索引图）→ 连通域外接框。

    ⚠️ 2026-09-23 实测踩到：第一版按"像素值 == 类别 id"解，结果 2,137 对只出 210 张、
      `导流线/网状线/待行区` 全 0 —— 因为掩码是 **RGB 着色图**（否则 `rlmd.csv` 里不会带 r,g,b 三列）。
      改成**按颜色查表**（csv: id,name,r,g,b）再解。
    太小的连通域（<min_area 像素）丢掉（噪点/极细虚线碎段），避免把噪声喂进去。
    """
    import cv2
    import numpy as np
    from PIL import Image
    root = os.path.join(D, "stage2", "veetinator_Road_Line_Marking_Dataset")
    z = os.path.join(root, "RLMD_1080p-20250613T030116Z-1-001.zip")
    if not os.path.exists(z):
        print("  rlmd: 未下载，跳过"); return
    # ⚠️ 两份 csv 列数不同：仓库根 `classes.csv` 是 id,name,**abbr**,r,g,b（六列），
    #    zip 里 `rlmd.csv` 是 id,name,r,g,b（五列）→ **取最后三列当 r,g,b** 两种都兼容。
    csvs = [c for c in (os.path.join(root, "classes.csv"),) if os.path.exists(c)]
    color2cls = {}
    for csv in csvs:
        for ln in io.open(csv, encoding="utf-8"):
            p_ = [x.strip() for x in ln.split(",")]
            if len(p_) >= 5 and p_[0].isdigit():
                tgt = RLMD_MAP.get(int(p_[0]))
                try:
                    r_, g_, b_ = [int(x) for x in p_[-3:]]
                except Exception:
                    continue
                if tgt:
                    color2cls[(r_, g_, b_)] = tgt
    print("  rlmd: 颜色表 %d 项（按 rlmd.csv 的 r,g,b 解码）" % len(color2cls))
    zf = zipfile.ZipFile(z)
    names = zf.namelist()
    imgs = {os.path.splitext(os.path.basename(x))[0]: x for x in names if "/images/" in x}
    labs = {os.path.splitext(os.path.basename(x))[0]: x for x in names if "/labels/" in x}
    pairs = [(st_, imgs[st_], labs[st_]) for st_ in sorted(set(imgs) & set(labs))]
    n = 0
    for stem, ip, lp in pairs:
        try:
            im = Image.open(io.BytesIO(zf.read(ip))).convert("RGB")
            mk = np.array(Image.open(io.BytesIO(zf.read(lp))).convert("RGB")).astype(np.int32)
        except Exception as e:
            print("    读取失败 %s: %s" % (stem, str(e)[:50])); continue
        W, H = im.size
        packed = (mk[:, :, 0] << 16) | (mk[:, :, 1] << 8) | mk[:, :, 2]
        lines = []
        for (r, g, b), tgt in color2cls.items():
            m = (packed == ((r << 16) | (g << 8) | b)).astype(np.uint8)
            if int(m.sum()) < min_area:
                continue
            nlab, _lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i2 in range(1, nlab):
                x, y, w, h, area = stats[i2]
                if area < min_area:
                    continue
                lines.append("%d %.6f %.6f %.6f %.6f" % (CID[tgt], (x + w / 2) / W, (y + h / 2) / H,
                                                        w / W, h / H))
                rep["boxes"][tgt] += 1
        if not lines:
            continue
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
        key = "rlmd_" + stem
        if _emit(split_of(key), buf.getvalue(), key + ".jpg", lines, "rlmd"):
            n += 1
    print("  rlmd: %d / %d 张（颜色查表→连通域→框）" % (n, len(pairs)))


def stage_cdset(rep):
    """CDSet-3434：自带 YOLO 格式（labels/{train,test}/*.txt）。"""
    z = os.path.join(D, "stage2", "zzd0225_crosswalk-detection-dataset", "CDSet.zip")
    if not os.path.exists(z):
        print("  cdset: 未下载，跳过"); return
    zf = zipfile.ZipFile(z)
    names = [x for x in zf.namelist() if x.endswith((".jpg", ".png"))]
    nl = {}
    for x in zf.namelist():
        if x.endswith(".txt") and "/labels/" in x and not x.endswith("labels.txt"):
            nl[os.path.splitext(os.path.basename(x))[0]] = x
    n = 0
    for ip in names:
        stem = os.path.splitext(os.path.basename(ip))[0]
        if stem not in nl:
            continue
        lines = []
        for ln in zf.read(nl[stem]).decode("utf-8", "replace").splitlines():
            p = ln.split()
            if len(p) >= 5:
                lines.append("%d %s" % (CID["marking_crosswalk"], " ".join(p[1:5])))
                rep["boxes"]["marking_crosswalk"] += 1
        if lines:
            key = "cdset_" + stem
            if _emit(split_of(key), zf.read(ip), key + ".jpg", lines, "cdset"):
                n += 1
    print("  cdset: %d 张（YOLO 格式，全归 人行横道）" % n)


def stage_lisa(rep):
    """LISA 红绿灯（YOLO 格式，全部归 `traffic_light`）—— 用户强调"红绿灯准确性比较重要"。

    为什么值得单独收：BDD100K 的红绿灯是 1280×720 行车图里的，而 LISA 是美国专门的红绿灯数据集
    （86k 文件、白天/夜间序列），补上它能明显加厚红绿灯这一类（当前 27,784 框里几乎全来自 BDD）。
    """
    base = os.path.join(D, "tsr_more", "dronefreak_LISA-Traffic-Lights")
    if not os.path.isdir(base):
        print("  lisa: 未下载，跳过"); return
    n = 0
    for r, _d, fs in os.walk(base):
        if os.path.basename(r) != "images":
            continue
        ld = os.path.join(os.path.dirname(r), "labels")
        for f in fs:
            if not f.lower().endswith((".jpg", ".png")):
                continue
            lp = os.path.join(ld, os.path.splitext(f)[0] + ".txt")
            if not os.path.exists(lp):
                continue
            lines = []
            for ln in io.open(lp, encoding="utf-8"):
                p_ = ln.split()
                if len(p_) >= 5:
                    lines.append("%d %s" % (CID["traffic_light"], " ".join(p_[1:5])))
                    rep["boxes"]["traffic_light"] += 1
            if lines:
                key = "lisa_" + f
                if _copy(os.path.join(r, f), key, lines, split_of(key)):
                    n += 1
    print("  lisa: %d 张（全归 红绿灯）" % n)


def stage_zebra(rep):
    """斑马线 OBB（**旋转框** 8 点格式）→ 取外接矩形转普通检测框。

    为什么这么转：我们的统一模型是普通检测器（轴对齐框），OBB 的 8 点无法直接喂；
    斑马线本身是横贯路面的长条，外接矩形与它的真实覆盖基本一致，损失很小。
    """
    base = os.path.join(D, "stage2", "SightLinks_YOLO-OBB-Zebra-Crossings-Dataset")
    if not os.path.isdir(base):
        print("  zebra: 未下载，跳过"); return
    n = 0
    for r, _d, fs in os.walk(base):
        if os.path.basename(r) != "images":
            continue
        ld = os.path.join(os.path.dirname(r), "labels")
        for f in fs:
            if not f.lower().endswith((".jpg", ".png")):
                continue
            lp = os.path.join(ld, os.path.splitext(f)[0] + ".txt")
            if not os.path.exists(lp):
                continue
            lines = []
            for ln in io.open(lp, encoding="utf-8"):
                p_ = ln.split()
                if len(p_) >= 9:                       # OBB：cls + 8 个归一化坐标
                    xs = [float(v) for v in p_[1:9:2]]
                    ys = [float(v) for v in p_[2:9:2]]
                    cx = (min(xs) + max(xs)) / 2; cy = (min(ys) + max(ys)) / 2
                    w = max(xs) - min(xs); h = max(ys) - min(ys)
                    if w > 0 and h > 0:
                        lines.append("%d %.6f %.6f %.6f %.6f" % (CID["marking_crosswalk"], cx, cy, w, h))
                        rep["boxes"]["marking_crosswalk"] += 1
                elif len(p_) == 5:                     # 有的子集是普通框
                    lines.append("%d %s" % (CID["marking_crosswalk"], " ".join(p_[1:5])))
                    rep["boxes"]["marking_crosswalk"] += 1
            if lines:
                key = "zebra_" + f
                if _copy(os.path.join(r, f), key, lines, split_of(key)):
                    n += 1
    print("  zebra: %d 张（OBB→外接矩形，全归 人行横道）" % n)

# ── 官方 MTSD 全量（42G，41,919 个标注 JSON）与俄罗斯 StarLine 的类名映射 ──
# 官方 MTSD 的 label 用的是同一套官方类名（regulatory--no-entry--g1 这种）→ 直接复用 mtsd_group()。
# StarLine 的 label 是 GOST 编号（如 8_22_3）→ 按首位数字映射：
#   1.x 警告 / 2.x 优先(归指示) / 3.x 禁令 / 4.x 指示 / 5.x 指路 / 6.x 服务(归指路) / 7.x、8.x 兜底(交通标识牌)
GOST_MAP = {"1": "warning", "2": "mandatory", "3": "prohibition", "4": "mandatory",
            "5": "guide", "6": "guide", "7": "crosswalk_sign", "8": "crosswalk_sign"}


def _uni(g):
    """tsr 侧的类名 → 统一侧类名（`_build_tsr_dataset` 里叫 crosswalk/signal，统一表里叫
    crosswalk_sign/traffic_light）—— 不做这层转换就会 CID['crosswalk'] KeyError（2026-09-24 实测踩到）。"""
    return {"crosswalk": "crosswalk_sign", "signal": "traffic_light"}.get(g, g)


def stage_mtsd_full(rep, max_imgs=10000, per_img=2):
    """官方 MTSD 全量（用户下的 42G那份）：标注 zip 里每个 JSON 一个图，含 width/height/objects。

    ⚠️ 关键细节：objects 里有 `properties.included` —— 官方把"不该计入"的框标成 included=false
      （外框/歧义/店招之类），**必须排除**，否则等于把噪声当正样本喂进去。
    体积控制：41,919 个标注图全用会训太久，默认只取 max_imgs 张、每图最多 per_img 个裁剪块。
    """
    import zipfile
    from PIL import Image
    root = os.path.join(D, "tsr_more", "crimedetector_roadsign")
    az = os.path.join(root, "mtsd_fully_annotated_annotation.zip")
    if not os.path.exists(az):
        print("  mtsd_full: 未下载，跳过"); return
    zf = zipfile.ZipFile(az)
    ann = {}
    for n in zf.namelist():
        if not n.endswith(".json"):
            continue
        try:
            d = json.load(io.BytesIO(zf.read(n)))
        except Exception:
            continue
        iid = os.path.splitext(os.path.basename(n))[0]
        objs = []
        for o in d.get("objects") or []:
            _p = o.get("properties") or {}
            # ⚠️ 2026-09-24 实测纠正：**不能用 `included` 过滤**！官方这份里 included=False 占 94%，
            #    而其中 414/442 是完全干净的框（不在画面外、不遮挡、不歧义）→ `included` 不是质量标志
            #    （看语义更像"是否属于官方 benchmark 子集"），照它过滤会白扔九成标志数据。
            #    正确口径：只排除**物理上不在画面内**的（exterior / out-of-frame）。
            if _p.get("exterior") or _p.get("out-of-frame"):
                continue
            g, _k = mtsd_group(o.get("label") or "")
            bb = o.get("bbox") or {}
            g = _uni(g) if g else None
            if g and g in CID and all(k in bb for k in ("xmin", "ymin", "xmax", "ymax")):
                objs.append((g, [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]]))
        if objs:
            ann[iid] = (float(d.get("width") or 0), float(d.get("height") or 0), objs)
    print("  mtsd_full: 有可用标注的图 %d 张（已按 included 过滤）" % len(ann))
    if not ann:
        return
    picked = sorted(ann)[:max_imgs]
    want = set(picked)
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
            if iid not in want:
                continue
            W, H, objs = ann[iid]
            if not W or not H:
                continue
            try:
                im = Image.open(io.BytesIO(iz.read(nm))).convert("RGB")
            except Exception:
                continue
            for g, cb in objs[:per_img]:
                bw, bh = cb[2] - cb[0], cb[3] - cb[1]
                if bw < 4 or bh < 4:
                    continue
                cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
                half = max(bw, bh) * 2.0
                x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
                x1, y1 = min(int(W), int(cx + half)), min(int(H), int(cy + half))
                if x1 - x0 < 64 or y1 - y0 < 64:
                    continue
                lines = []
                for g2, b2 in objs:
                    if b2[0] >= x0 and b2[1] >= y0 and b2[2] <= x1 and b2[3] <= y1 and g2 in CID:
                        lines.append("%d %.6f %.6f %.6f %.6f" % (
                            CID[g2], ((b2[0] + b2[2]) / 2 - x0) / (x1 - x0),
                            ((b2[1] + b2[3]) / 2 - y0) / (y1 - y0),
                            (b2[2] - b2[0]) / (x1 - x0), (b2[3] - b2[1]) / (y1 - y0)))
                        rep["boxes"][g2] += 1
                if not lines:
                    continue
                buf = io.BytesIO(); im.crop((x0, y0, x1, y1)).save(buf, "JPEG", quality=88)
                key = "mtsdfull_%s_%d" % (part, n)
                if _emit(split_of(key), buf.getvalue(), key + ".jpg", lines, "mtsdfull"):
                    n += 1
            want.discard(iid)
            if not want:
                break
        if not want:
            break
    print("  mtsd_full: 出 %d 个裁剪块（取 %d 张图，每图最多 %d 块）" % (n, min(len(picked), max_imgs), per_img))


def stage_starline(rep, per_img=2):
    """俄罗斯 StarLine：annotations.parquet（label 是 GOST 编号，bbox 是绝对 [x,y,w,h]）。"""
    import zipfile
    from PIL import Image
    z = os.path.join(D, "tsr_more", "StarLineResearch_Russian_Road_Signs_Dataset", "road_signs_dataset.zip")
    if not os.path.exists(z):
        print("  starline: 未下载，跳过"); return
    import pyarrow.parquet as pq
    zf = zipfile.ZipFile(z)
    rows = pq.read_table(io.BytesIO(zf.read("road_signs_dataset/annotations.parquet"))).to_pylist()
    by_img = {}
    for r in rows:
        lb = str(r.get("label") or "")
        g = GOST_MAP.get(lb.split("_")[0])
        if not g:
            rep["unmapped"].append("StarLine:%s" % lb); continue
        bb = r.get("bbox_coords") or []
        if len(bb) == 4:
            by_img.setdefault(r["image_path"], []).append((g, [float(v) for v in bb]))
    print("  starline: 有标注的图 %d 张（GOST 编号 → 语义类）" % len(by_img))
    n = 0
    for rel, objs in by_img.items():
        try:
            im = Image.open(io.BytesIO(zf.read("road_signs_dataset/" + rel))).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        lines = []
        for g, (x, y, w, h) in objs[:per_img]:
            if w <= 0 or h <= 0:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (CID[g], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
            rep["boxes"][g] += 1
        if not lines:
            continue
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
        key = "starline_" + os.path.basename(rel)
        if _emit(split_of(key), buf.getvalue(), key, lines, "starline"):
            n += 1
    print("  starline: 出 %d 张" % n)

def stage_negs(rep):
    """自有帧负样本（`无标识`段整帧，空标注）—— 教"这种场景什么都没有"。"""
    src = os.path.join(D, "negatives", "frames")
    if not os.path.isdir(src):
        print("  negs: 无负样本目录，跳过"); return
    n = 0
    for r, _d, fs in os.walk(src):
        for f in fs:
            if not f.lower().endswith((".jpg", ".png")):
                continue
            key = "neg_" + f
            if _copy(os.path.join(r, f), key, [], split_of(key)):
                n += 1
    print("  negs: %d 张（空标注背景图）" % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="signs,bdd,kitti,rlmd,cdset,lisa,zebra,negs")
    ap.add_argument("--mtsd-full-imgs", type=int, default=10000, help="官方 MTSD 全量取多少张图（每图 2 块）")
    a = ap.parse_args()
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)                      # 幂等：全量重建
    for sp in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", sp), exist_ok=True)
    rep = {"boxes": Counter(), "unmapped": []}
    for st in a.stages.split(","):
        fn = {"signs": stage_signs, "bdd": stage_bdd, "kitti": stage_kitti,
              "rlmd": stage_rlmd, "cdset": stage_cdset, "lisa": stage_lisa,
              "zebra": stage_zebra, "negs": stage_negs, "mtsd_full": stage_mtsd_full,
              "starline": stage_starline}.get(st)
        if not fn:
            print("  未知 stage: %s" % st); continue
        print("… %s" % st)
        if st == "mtsd_full":
            fn(rep, max_imgs=a.mtsd_full_imgs)
        else:
            fn(rep)
        sys.stdout.flush()

    print("\n=== 统计 ===")
    for sp in ("train", "val"):
        ni = len(os.listdir(os.path.join(OUT, "images", sp)))
        nl_ = len(os.listdir(os.path.join(OUT, "labels", sp)))
        print("  %s: 图 %d / 标 %d %s" % (sp, ni, nl_, "✓" if ni == nl_ else "✗ 不一致!"))
    print("  每类框数（本体名）:")
    tot = 0
    for c in CLASSES:
        v = rep["boxes"].get(c, 0); tot += v
        print("    %-20s %-8s %7d %s" % (c, CN[c], v, "" if v else "← 本类暂无数据（后续自标补）"))
    print("  合计框 %d" % tot)
    print("  未归类命中（应为空）:", rep["unmapped"][:10], "共", len(rep["unmapped"]))
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % OUT)
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))
    with open(os.path.join(OUT, "classes_cn.json"), "w", encoding="utf-8") as f:
        json.dump({c: CN[c] for c in CLASSES}, f, ensure_ascii=False, indent=1)
    print("  已写 %s/data.yaml + classes_cn.json" % OUT)


if __name__ == "__main__":
    main()
