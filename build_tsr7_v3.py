# 干什么：第四轮数据集 tsr7_v3 —— 在 v2 基础上加【俄国数据】，自有帧换成阈值 0.25 重挖的 v2 版。
#   用户 2026-09-25 定："要不要把现在下载的数据再来一轮训练" → 我给的配方（有实测依据）：
#     ① 加：俄标 YOLO 集（只取**图与标签都齐**的配对）+ RTSD detection（d1/d2/d3 + full-gt.csv）
#     ② 排除：车道线/锥桶/立柱/裁切分类集（r1/r3、rtsd_cleaned）——**不属于 7 类，只稀释**（二轮就崩在这）
#     ③ 自有帧用 v2（阈值 0.25，比 v1 多约 29% 正样本）——针对二轮 A 组"禁令/警告回退"的正解
# 怎么跑：AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python -u tools/_build_tsr7_v3.py
# 怎么判定成功：末尾 train/val 图数一致、逐类框数都 > 0，且日志里 ru_yolo / rtsd 两行有合理数量。
# ⚠️ 图片用**软链**指向 NAS（省本地盘；训练实测读得到，自有帧那批已这么用好几天了）。
# ⚠️ 类别 id 一律以 classes.txt 的**行号（0 基）**为准 —— 抽样验证过（标签里出现 42，对应第 43 行）。
import csv
import io
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _build_tsr7_v2 as V                                                  # noqa: E402

OUT = "/opt/datasets/tsr7_v3"
RU_YOLO = "/mnt/Data_Platform/Russia_test/russian-road-signs"
RTSD = "/mnt/Data_Platform/Russia_test/rtsd_official"
CLASSES = V.CLASSES
CN = {"speed_limit": "限速", "prohibition": "禁令", "warning": "警告", "mandatory": "指示",
      "guide": "指路", "signal": "红绿灯", "crosswalk": "人行横道"}


def gost_to_uni(code):
    """GOST 标志编号 → 我们的 7 类（统一侧名）。认不出返回 None（宁缺毋滥，别硬塞）。
    ⭐ 与 v2 用的 starline 映射的关键差别：**3.24 是限速**，不能并进禁令（v2 把 3.x 全当禁令，太粗）。"""
    c = str(code).strip()
    low = c.lower()
    if "trafficlight" in low or "traffic_light" in low or "светофор" in low:
        return "traffic_light"
    c = c.replace("_", ".")
    m = re.match(r"(\d+)", c)
    if not m:
        return None
    g = m.group(1)
    if g == "1":
        return "warning"
    if g == "3":
        return "speed_limit" if c.startswith("3.24") else "prohibition"
    if g in ("2", "4"):
        return "mandatory"
    if g in ("5", "6"):
        return "guide"
    # ⚠️ 7.x/8.x 是"附加信息板"（随主牌一起出现的小矩形板），**不是人行横道** ——
    #    v2 把它们并进 crosswalk，等于教"小矩形板=人行横道"，会污染这一类 → 这里丢弃（宁缺毋滥）。
    return None


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) or os.path.islink(dst):
        return False
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy(src, dst)
    return True


def write_lab(path, lines):
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def stage_ru_yolo(rep):
    """俄标路牌 YOLO 集（Dognellaf/russian-road-signs）：只取**图片与标签都齐**的配对。"""
    ct = os.path.join(RU_YOLO, "classes.txt")
    if not os.path.exists(ct):
        print("  ru_yolo: 缺 classes.txt，跳过"); return
    names = [l.strip() for l in io.open(ct, encoding="utf-8", errors="ignore") if l.strip()]
    id2uni = {}
    for i, nm in enumerate(names):
        u = gost_to_uni(nm)
        id2uni[i] = u
    lab_root = os.path.join(RU_YOLO, "labels")
    img_root = os.path.join(RU_YOLO, "images")
    n = 0
    unmapped = {}
    for r, _d, fs in os.walk(img_root):
        for f in fs:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(f)[0]
            lp = os.path.join(lab_root, os.path.relpath(r, img_root), stem + ".txt")
            if not os.path.exists(lp):
                continue                      # 标签没下齐的先不要（宁缺毋滥）
            lines = []
            for ln in io.open(lp, encoding="utf-8", errors="ignore"):
                p = ln.split()
                if len(p) < 5:
                    continue
                u = id2uni.get(int(p[0]))
                if not u:
                    unmapped[names[int(p[0])] if int(p[0]) < len(names) else int(p[0])] = \
                        unmapped.get(names[int(p[0])] if int(p[0]) < len(names) else int(p[0]), 0) + 1
                    continue
                cls = {"traffic_light": "signal", "crosswalk_sign": "crosswalk"}.get(u, u)
                lines.append("%d %s" % (CLASSES.index(cls), " ".join(p[1:5])))
            if not lines:
                continue
            key = "ru_%s" % stem
            if link(os.path.join(r, f), os.path.join(OUT, "images", V.U.split_of(key), key + ".jpg")):
                write_lab(os.path.join(OUT, "labels", V.U.split_of(key), key + ".txt"), lines)
                n += 1
            for ln in lines:
                rep["boxes"][u if u != "signal" else "traffic_light"] = \
                    rep["boxes"].get(u if u != "signal" else "traffic_light", 0) + 1
    print("  ru_yolo: %d 张带标注图（配对）｜未映射类 %d 种（前几个：%s）" % (
        n, len(unmapped), list(unmapped.items())[:4]))


def stage_rtsd_detection(rep):
    """RTSD 官方 detection（d1/d2/d3 帧 + full-gt.csv）：绝对 xywh → YOLO 归一化。"""
    gt = os.path.join(RTSD, "full-gt.csv")
    if not os.path.exists(gt):
        print("  rtsd: 缺 full-gt.csv，跳过"); return
    # 先把三个分卷的帧建索引：basename -> 路径
    idx = {}
    for part in ("d1_frames", "d2_frames", "d3_frames"):
        # ⚠️ 解包时落到了 detection/ 底下（dec 的输出目录是 detection/dX_frames），别漏这一层
        root = os.path.join(RTSD, "detection", part)
        if not os.path.isdir(root):
            continue
        for r, _d, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx.setdefault(f, os.path.join(r, f))
    if not idx:
        print("  rtsd: 没找到解出来的帧，跳过"); return
    from PIL import Image
    by_img = {}
    n_row = n_skip = 0
    with io.open(gt, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            if fn not in idx:
                n_skip += 1
                continue
            u = gost_to_uni(row.get("sign_class") or "")
            if not u:
                n_skip += 1
                continue
            try:
                x, y, w, h = (float(row["x_from"]), float(row["y_from"]),
                              float(row["width"]), float(row["height"]))
            except Exception:
                n_skip += 1
                continue
            by_img.setdefault(fn, []).append((u, x, y, w, h))
            n_row += 1
    n = 0
    for fn, boxes in by_img.items():
        src = idx[fn]
        try:
            W, H = Image.open(src).size
        except Exception:
            continue
        if not W or not H:
            continue
        lines = []
        for u, x, y, w, h in boxes:
            cls = {"traffic_light": "signal", "crosswalk_sign": "crosswalk"}.get(u, u)
            if cls not in CLASSES:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append("%d %.6f %.6f %.6f %.6f" % (CLASSES.index(cls), cx, cy, w / W, h / H))
            rep["boxes"][u if u != "signal" else "traffic_light"] = \
                rep["boxes"].get(u if u != "signal" else "traffic_light", 0) + 1
        if not lines:
            continue
        key = "rtsd_%s" % os.path.splitext(fn)[0]
        if link(src, os.path.join(OUT, "images", V.U.split_of(key), key + ".jpg")):
            write_lab(os.path.join(OUT, "labels", V.U.split_of(key), key + ".txt"), lines)
            n += 1
    print("  rtsd: %d 张图（用到 %d 个框；gt 里跳过 %d 行：图没下到或类认不出）" % (n, n_row, n_skip))


def main():
    # 复用 v2 的构建流程，但换输出目录、换自有帧目录（v2 = 0.25 重挖的那批）
    V.OUT = OUT
    V.U.OUT = OUT
    V.OWNF = "/opt/datasets/ownframes_v2"
    # 同上：StarLine 阶段是按 GOST **首位数**查表的，7/8 也一并改成丢弃（保持与上面一致）
    V.U.GOST_MAP = {"1": "warning", "2": "mandatory", "3": "prohibition", "4": "mandatory",
                    "5": "guide", "6": "guide", "7": None, "8": None}
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for s in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    rep = {"boxes": {c: 0 for c in V.U.CLASSES}, "unmapped": []}
    print("[v3 重建] 类序: %s" % " ".join(CLASSES))
    print("[v3] 复用 v2 的公开数据源（整图 MTSD / StarLine / BDD 灯 / 背景负样本）")
    V.U.stage_signs(rep)
    V.stage_bdd_lights(rep)
    V.stage_mtsd_full_images(rep, max_imgs=50000)
    V.U.stage_starline(rep, per_img=1)
    V.U.stage_negs(rep)
    print("[v3] 自有帧（阈值 0.25 重挖的 v2）")
    V.stage_ownframes(rep)
    print("[v3] 俄国数据（只加俄标 YOLO 集 + RTSD detection；车道线/锥桶/立柱/裁切集一律不加）")
    stage_ru_yolo(rep)
    stage_rtsd_detection(rep)

    with io.open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
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
                    k = CLASSES[int(ln.split()[0])]
                    tot[k] = tot.get(k, 0) + 1
        print("  %s: 图 %d / 标 %d / 框 %d%s" % (sp, ni, nl, nbox, "" if ni == nl else "  ✗ 图数不符"))
    print("  合计框 %d" % sum(tot.values()))
    for c in CLASSES:
        print("    %-16s %6d" % (c, tot.get(c, 0)))
    ru = sum(1 for f in os.listdir(os.path.join(OUT, "images", "train")) if f.startswith("ru_"))
    rt = sum(1 for f in os.listdir(os.path.join(OUT, "images", "train")) if f.startswith("rtsd_"))
    print("  [俄国数据入 train] ru_yolo %d 张 ／ rtsd %d 张" % (ru, rt))
    print("[完成] %s/data.yaml" % OUT)


if __name__ == "__main__":
    main()
