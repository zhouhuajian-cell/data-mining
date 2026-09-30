#!/venv/bin/python
# -*- coding: utf-8 -*-
"""
一次性脚本：把国外交通标志数据集并成**语义大类**的 YOLO 训练集（闭集 TSR 检测器用）。

干什么：
  越南（YOLO 格式） + 德国（Roboflow COCO） + Mapillary MTSD（parquet）→ /opt/datasets/tsr_yolo/{images,labels}/{train,val}
  类别是**语义大类**（不是细类），这样检测输出能直接填本体的 traffic_sign，且每类样本量够。

怎么跑（不碰 GPU，不影响 ad_mining）：
  /venv/bin/python /opt/ad_mining/tools/_build_tsr_dataset.py --stage all
怎么判定成功：
  末尾打印每类样本数 + 未被归类命中的原始类名（必须为空或人工确认过）；
  /opt/datasets/tsr_yolo/data.yaml 存在，且 images/train 与 labels/train 数量一致。
"""
import argparse, json, os, re, shutil, sys, zipfile

OUT = "/opt/datasets/tsr_yolo"
RAW = "/opt/datasets"
CLASSES = ["speed_limit", "prohibition", "warning", "mandatory", "guide", "signal", "crosswalk"]
CID = {c: i for i, c in enumerate(CLASSES)}
# 语义大类 → 本体 traffic_sign 的值（接入时要一致）
CN = {"speed_limit": "限速标志", "prohibition": "禁令标志", "warning": "警告标志",
      "mandatory": "指示标志", "guide": "指路标志", "signal": "交通信号灯", "crosswalk": "人行横道"}

# 关键字规则：**顺序敏感**（先特殊后一般）。命中第一个就停。
RULES = [
    ("signal", ["traffic light", "traffic signal", "signal light", "green light", "red light",
                "yellow light", "đèn xanh", "đèn đỏ", "signal-ahead", "traffic-light",
                "signal ahead", "light ahead"]),
    ("crosswalk", ["crosswalk", "crossing", "pedestrian crossing", "pedestrian lane", "zebra",
                   "người đi bộ", "人行横道"]),
    ("speed_limit", ["speed limit", "speed-limit", "speedlimit", "giới hạn tốc độ", "限速",
                     "end of", "km/h", "mph"]),
    ("prohibition", ["no entry", "no parking", "no stopping", "no turn", "no left", "no right",
                     "no u-turn", "no uturn", "no straight", "no traffic", "no trucks", "no bus",
                     "no cars", "no moto", "no overtak", "no horn", "prohibit", "stop", "yield",
                     "give way", "height limit", "low clearance", "weight limit", "width limit",
                     "no two", "no three", "clearance", "restriction", "cấm", "禁令", "禁止"]),
    ("warning", ["warning", "danger", "children", "curve", "bend", "sharp", "slippery", "road work",
                 "roadwork", "construction", "speed bump", "uneven", "narrow", "steep", "animal",
                 "snow", "level crossing", "railway", "obstacle", "accident", "intersection",
                 "slown down", "slow down", "hill", "bridge", "bump", "cảnh báo", "警告"]),
    ("guide", ["guide", "direction", "destination", "expressway", "highway", "route", "priority",
               "place name", "information", "hospital", "bus stop", "parking", "one way",
               "dual carriageway", "residential", "populated", "camera", "no through", "dead end",
               "指路"]),
    ("mandatory", ["mandatory", "keep left", "keep right", "roundabout", "turn left", "turn right",
                   "go left", "go right", "go straight", "ahead only", "lane allocation", "lane",
                   "u-turn area", "minimum speed", "chỉ dẫn", "指示"]),
]


def to_group(name):
    n = str(name).lower().strip()
    for g, kws in RULES:
        for k in kws:
            if k in n:
                return g, k
    return None, None


# ---------- 越南：已经是 YOLO txt ----------
def build_vietnam(rep, neg):
    z = os.path.join(RAW, "VietnamSign", "traffic-sign-detection-vietnam.zip")
    if not os.path.exists(z):
        print("  越南: 缺 zip，跳过"); return
    cid = {}
    with open(os.path.join(RAW, "VietnamSign", "classid.csv"), encoding="utf-8-sig") as f:
        for ln in f.read().splitlines()[1:]:
            p = ln.split(",")
            if len(p) >= 2:
                cid[int(p[0])] = p[1]
    gmap = {}
    for i, nm in cid.items():
        g, k = to_group(nm)
        if g:
            gmap[i] = g
        else:
            rep["unmapped"].append("越南:%s" % nm)
    print("  越南: %d 类 → 归并 %d 类" % (len(cid), len(gmap)))
    ex = os.path.join(RAW, "VietnamSign", "unzipped")
    if not os.path.isdir(ex):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ex)
    for split, out_split in (("train", "train"), ("test", "val")):
        img_d = os.path.join(ex, "dataset", split, "images")
        lb_d = os.path.join(ex, "dataset", split, "labels")
        if not os.path.isdir(img_d):
            continue
        n = 0
        for fn in os.listdir(img_d):
            if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            lp = os.path.join(lb_d, os.path.splitext(fn)[0] + ".txt")
            lines = []
            if os.path.exists(lp):
                for ln in open(lp):
                    p = ln.split()
                    if len(p) >= 5 and int(p[0]) in gmap:
                        lines.append("%d %s" % (CID[gmap[int(p[0])]], " ".join(p[1:5])))
            if lines:
                _emit(out_split, open(os.path.join(img_d, fn), "rb").read(), fn, lines, neg)
                n += 1
        print("  越南 %s: %d 张（有可用标注）" % (split, n))


# ---------- 德国：Roboflow COCO → YOLO ----------
def build_german(rep, neg):
    import json as _j
    base = os.path.join(RAW, "GermanSign", "data")
    for split, out_split in (("train", "train"), ("valid", "val")):
        z = os.path.join(base, "%s.zip" % split)
        if not os.path.exists(z):
            continue
        ex = z[:-4]
        if not os.path.isdir(ex):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ex)
        js = [os.path.join(ex, f) for f in os.listdir(ex) if f.endswith(".json")]
        if not js:
            print("  德国 %s: 无 coco json，跳过" % split); continue
        d = _j.load(open(js[0], encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        gmap = {}
        for i, nm in cats.items():
            g, k = to_group(nm)
            if g:
                gmap[i] = g
            else:
                rep["unmapped"].append("德国:%s" % nm)
        by_img = {}
        for a in d["annotations"]:
            if a["category_id"] not in gmap:
                continue
            x, y, w, h = a["bbox"]
            W, H = a.get("image_width"), a.get("image_height")
            by_img.setdefault(a["image_id"], []).append((gmap[a["category_id"]], x, y, w, h))
        imgs = {im["id"]: im for im in d["images"]}
        n = 0
        for iid, anns in by_img.items():
            im = imgs[iid]
            W = im["width"]; H = im["height"]
            lines = []
            for g, x, y, w, h in anns:
                lines.append("%d %.6f %.6f %.6f %.6f" % (
                    CID[g], (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
            p = os.path.join(ex, im["file_name"])
            if os.path.exists(p):
                _emit(out_split, open(p, "rb").read(), im["file_name"], lines, neg)
                n += 1
        print("  德国 %s: %d 张" % (split, n))


# ---------- Mapillary MTSD：parquet（objects.category 是 0..399 的官方类名 ID）----------
# 为什么按"裁剪块"而不是整图：MTSD 是 2592x1936 的街景全景，牌子在整图里只有几十像素，
# 整图缩到 imgsz=960 会把牌子压到学不动的尺寸。以框为中心取 4 倍上下文，
# 既保住牌子像素，又保留周边（对"广告牌 vs 交通牌"的判别恰恰要靠周边上下文）。
MTSD_RULES = [
    # 先处理会跨组的特例
    ("crosswalk", ["pedestrians-crossing", "crossing"]),
    ("speed_limit", ["speed-limit", "minimum-speed", "radar-enforced", "speed-zone"]),
    ("prohibition", ["no-entry", "no-parking", "no-stopping", "no-turn", "no-left", "no-right",
                     "no-u-turn", "no-turns", "no-mopeds", "wrong-way", "no-straight",
                     "no-overtaking", "no-bicycles", "no-buses", "no-motor", "no-motorcycles",
                     "no-pedestrians", "no-hawkers", "no-vehicles", "no-heavy", "do-not-block",
                     "do-not-stop", "give-way", "stop", "height-limit", "width-limit",
                     "weight-limit", "road-closed", "end-of-prohibition", "parking-restrictions",
                     "tow-away-zone", "yield"]),
    # ⚠️ complementary-- 细分语义（2026-09-28）：不是所有附加板都该进 guide ——
    #    chevron（急弯导向标）/obstacle-delineator（障碍物标）本质是**警告**类（放在弯道/障碍前），
    #    其余 complementary（go-left/keep-left/distance 等）维持 guide（指路）。
    #    规则按顺序匹配：这两条必须排在 mandatory/guide 之前（warning 表里有 chevron 已覆盖一部分）。
    ("warning", ["warning--", "chevron", "obstacle-delineator", "accident-area", "road-bump",
                 "children", "horizontal-alignment", "curve", "bend", "slippery", "narrow",
                 "steep", "railway", "school", "risk", "danger"]),
    ("mandatory", ["keep-left", "keep-right", "go-straight", "turn-left", "turn-right",
                   "roundabout", "one-way", "one-direction", "dual-lanes", "dual-path",
                   "shared-path", "pedestrians-only", "bicycles-only", "buses-only",
                   "mopeds-and-bicycles-only", "pass-on-either-side", "pass-right",
                   "priority-road", "priority-over", "detour", "lane-control",
                   "reversible-lanes", "passing-lane-ahead", "central-lane", "end-of-priority",
                   "end-of-bicycles", "end-of-buses", "end-of-maximum", "end-of-no-parking",
                   "end-of-speed-limit", "end-of-living", "end-of-motorway", "end-of-built-up",
                   "end-of-limited", "end-of-pedestrians", "left-turn-yield", "no-turn-on-red",
                   "except-bicycles", "go-left", "go-right", "u-turn"]),
    ("guide", ["information--", "service--", "complementary--", "dead-end", "bus-stop",
               "tram-bus-stop", "hospital", "gas-station", "motorway", "highway-exit",
               "interstate", "bike-route", "airport", "parking", "distance", "both-directions",
               "lodging", "food", "camp", "telephone", "stairs", "disabled", "emergency",
               "safety-area", "extent-of-prohibition", "buses", "trucks", "text-"]),
]


def mtsd_group(name):
    n = name.lower()
    for g, kws in MTSD_RULES:
        for k in kws:
            if k in n:
                return g, k
    return None, None


def build_mtsd(rep, shards=15, per_img=2, val_shards=1):
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    root = os.path.join(RAW, "MTSD2")
    lbl = os.path.join(root, "id2labels.txt")
    if not os.path.exists(lbl):
        print("  MTSD: 缺 id2labels.txt，跳过"); return
    txt = open(lbl, encoding="utf-8-sig").read()
    id2 = {int(i): n for i, n in re.findall(r"(\d+):\s*'([^']+)'", txt)}
    gmap = {}
    for i, nm in id2.items():
        g, k = mtsd_group(nm)
        if g:
            gmap[i] = g
        else:
            rep["unmapped"].append("MTSD:%s" % nm)
    print("  MTSD: %d 类 → 归并 %d 类" % (len(id2), len(gmap)))
    for sub, out_split, lim in (("train_mtsd", "train", shards), ("val_mtsd", "val", val_shards)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            print("  MTSD %s: 目录缺，跳过" % sub); continue
        fs = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))[:lim]
        n_img = n_box = 0
        for fn in fs:
            pf = pq.ParquetFile(os.path.join(d, fn))
            # 按批流式读：一个分片 577 张 3264x2448 的 JPEG，整片 to_pylist 要 ~1G 内存，
            # 而 31G 机器上 ad_mining（VLM/检测）还在跑，不能赌内存。
            for batch in pf.iter_batches(batch_size=24):
                for r in batch.to_pylist():
                    W, H = r["width"], r["height"]
                    objs = r["objects"] or {}
                    cats = objs.get("category") or []
                    bxs = objs.get("bbox") or []
                    keep = []
                    for c, b in zip(cats, bxs):
                        if c in gmap and len(b) == 4:
                            keep.append((gmap[c], [float(v) for v in b]))
                    if not keep:
                        continue
                    try:
                        im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                    except Exception:
                        continue
                    # 裁剪块：以每个框为中心取 4 倍上下文，块内保留全部可用框
                    cuts = keep if per_img <= 0 else keep[:per_img]
                    for cg, cb in cuts:
                        bw, bh = cb[2] - cb[0], cb[3] - cb[1]
                        cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
                        half = max(bw, bh) * 2.0
                        x0 = max(0, int(cx - half)); y0 = max(0, int(cy - half))
                        x1 = min(W, int(cx + half)); y1 = min(H, int(cy + half))
                        if x1 - x0 < 64 or y1 - y0 < 64:
                            continue
                        lines = []
                        for g2, b2 in keep:
                            if b2[0] >= x0 and b2[1] >= y0 and b2[2] <= x1 and b2[3] <= y1:
                                lines.append("%d %.6f %.6f %.6f %.6f" % (
                                    CID[g2], ((b2[0] + b2[2]) / 2 - x0) / (x1 - x0),
                                    ((b2[1] + b2[3]) / 2 - y0) / (y1 - y0),
                                    (b2[2] - b2[0]) / (x1 - x0), (b2[3] - b2[1]) / (y1 - y0)))
                        if not lines:
                            continue
                        buf = io.BytesIO()
                        im.crop((x0, y0, x1, y1)).save(buf, "JPEG", quality=88)
                        _emit(out_split, buf.getvalue(),
                              "mtsd_%s_%d" % (fn.split("-of")[0].replace("train_", "").replace("val_", ""), n_box),
                              lines, {"tag": "mtsd"})
                        n_box += 1
                    n_img += 1
                    if n_img % 2000 == 0:
                        print("    %s: %d 图 / %d 块" % (sub, n_img, n_box)); sys.stdout.flush()
        print("  MTSD %s: %d 图 → %d 训练块" % (sub, n_img, n_box))


# ---------- TT100K（中国）：parquet，objects 是 [{category:'pl40', bbox:[x,y,w,h]}] ----------
# 类别是代码，按前缀归并（TT100K 自己的 taxonomy：p=禁令 / w=警告 / i=指示，pl/pr/il=限速族）。
# 2026-09-22 用户："国内也加上吧，一起训练吧" —— 国内是唯一来源（我们自己没有国内项目）。
TT100K_RULES = [
    ("crosswalk", ("ip",)),
    ("speed_limit", ("pl", "pr", "il")),
    ("warning", ("w",)),
    ("mandatory", ("i", "io")),
    ("prohibition", ("pm", "ph", "pn", "pne", "ps", "pg", "pb", "pc", "pa", "pi", "po", "p")),
]


def tt100k_group(code):
    c = str(code).lower()
    for g, pfx in TT100K_RULES:
        for p in pfx:
            if c.startswith(p):
                return g, p
    return None, None


def build_tt100k(rep, shards=30, per_img=2):
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    d = os.path.join(RAW, "TT100K", "data")
    if not os.path.isdir(d):
        print("  TT100K: 目录缺，跳过"); return
    fs = sorted(f for f in os.listdir(d)
                if f.startswith("train-") and f.endswith(".parquet"))[:shards]
    print("  TT100K: 用 %d 个完整分片" % len(fs))
    n_img = n_box = 0
    for fn in fs:
        pf = pq.ParquetFile(os.path.join(d, fn))
        for batch in pf.iter_batches(batch_size=8):     # 每张 2048x2048 PNG，别一次解太多
            for r in batch.to_pylist():
                W, H = r["width"], r["height"]
                keep = []
                for o in (r["objects"] or []):
                    g, k = tt100k_group(o.get("category"))
                    if not g:
                        rep["unmapped"].append("TT100K:%s" % o.get("category"))
                        continue
                    b = o.get("bbox") or []
                    if len(b) == 4:
                        x, y, w, h = [float(v) for v in b]
                        keep.append((g, [x, y, x + w, y + h]))
                if not keep:
                    continue
                try:
                    im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                except Exception:
                    continue
                for cg, cb in (keep if per_img <= 0 else keep[:per_img]):
                    bw, bh = cb[2] - cb[0], cb[3] - cb[1]
                    if bw < 4 or bh < 4:
                        continue
                    cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
                    half = max(bw, bh) * 2.0
                    x0 = max(0, int(cx - half)); y0 = max(0, int(cy - half))
                    x1 = min(W, int(cx + half)); y1 = min(H, int(cy + half))
                    if x1 - x0 < 64 or y1 - y0 < 64:
                        continue
                    lines = []
                    for g2, b2 in keep:
                        if b2[0] >= x0 and b2[1] >= y0 and b2[2] <= x1 and b2[3] <= y1:
                            lines.append("%d %.6f %.6f %.6f %.6f" % (
                                CID[g2], ((b2[0] + b2[2]) / 2 - x0) / (x1 - x0),
                                ((b2[1] + b2[3]) / 2 - y0) / (y1 - y0),
                                (b2[2] - b2[0]) / (x1 - x0), (b2[3] - b2[1]) / (y1 - y0)))
                    if not lines:
                        continue
                    buf = io.BytesIO()
                    im.crop((x0, y0, x1, y1)).save(buf, "JPEG", quality=88)
                    _emit("train", buf.getvalue(), "tt100k_%s_%d" % (fn[6:11], n_box),
                          lines, {"tag": "tt"})
                    n_box += 1
                n_img += 1
                if n_img % 500 == 0:
                    print("    TT100K: %d 图 / %d 块" % (n_img, n_box)); sys.stdout.flush()
    print("  TT100K: %d 图 → %d 训练块" % (n_img, n_box))


def _emit(split, data, fn, lines, neg):
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(fn)[0])[:80]
    key = "%s_%s" % (neg["tag"], stem)
    ip = os.path.join(OUT, "images", split, key + ".jpg")
    lp = os.path.join(OUT, "labels", split, key + ".txt")
    k = 0
    while os.path.exists(ip):
        k += 1
        ip = os.path.join(OUT, "images", split, "%s_%d.jpg" % (key, k))
        lp = os.path.join(OUT, "labels", split, "%s_%d.txt" % (key, k))
    with open(ip, "wb") as f:
        f.write(data)
    with open(lp, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="vietnam,german")
    ap.add_argument("--mtsd-shards", type=int, default=15)
    ap.add_argument("--mtsd-val-shards", type=int, default=1)
    ap.add_argument("--mtsd-per-img", type=int, default=2, help="每图最多取几个裁剪块，0=全部")
    ap.add_argument("--tt100k-shards", type=int, default=30)
    ap.add_argument("--tt100k-per-img", type=int, default=2)
    a = ap.parse_args()
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)          # 幂等：每次全量重建，避免旧样本残留
    for s in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    rep = {"unmapped": [], "neg": 0}
    st = a.stage.split(",")
    if "vietnam" in st:
        print("越南…"); build_vietnam(rep, {"tag": "vn"})
    if "german" in st:
        print("德国…"); build_german(rep, {"tag": "de"})
    if "mtsd" in st:
        print("Mapillary MTSD…（%d 个 train 分片）" % a.mtsd_shards)
        build_mtsd(rep, a.mtsd_shards, a.mtsd_per_img, a.mtsd_val_shards)
    if "tt100k" in st:
        print("TT100K（中国）…（%d 个分片）" % a.tt100k_shards)
        build_tt100k(rep, a.tt100k_shards, a.tt100k_per_img)
    print("\n--- 统计 ---")
    for sp in ("train", "val"):
        ni = len(os.listdir(os.path.join(OUT, "images", sp)))
        nl = len(os.listdir(os.path.join(OUT, "labels", sp)))
        print("  %s: 图 %d / 标 %d %s" % (sp, ni, nl, "✓" if ni == nl else "✗ 不一致!"))
    cnt = {c: 0 for c in CLASSES}
    for sp in ("train", "val"):
        for fn in os.listdir(os.path.join(OUT, "labels", sp)):
            for ln in open(os.path.join(OUT, "labels", sp, fn)):
                if ln.strip():
                    cnt[CLASSES[int(ln.split()[0])]] += 1
    print("  每类框数:", json.dumps(cnt, ensure_ascii=False))
    print("  未归类命中（必须为空）:", rep["unmapped"][:20], "共", len(rep["unmapped"]))
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % OUT)
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))
    print("  已写 %s/data.yaml" % OUT)


if __name__ == "__main__":
    main()
