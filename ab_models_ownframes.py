#!/venv/bin/python
# -*- coding: utf-8 -*-
"""**训练前后对比**：新旧两个闭集模型在同一批自有帧上并排比（用户 2026-09-23："对比训练前后的结果 给html"）。

为什么不用现成的 `_eval_tsr_ownframes.py`：它只支持"单模型 vs 开放词表 YOLO-World"，
回答不了"新模型比旧模型好还是差" —— 而全量重训之后必须回答这个。

比什么：
 ① **A 组（有标识段）召回** + **B 组（无标识段）误检**：新旧各自的段级/帧级命中率；
 ② **逐帧四象限**：旧命中新也命中 / 旧命中新漏 / 旧漏新命中 / 都漏 —— 一眼看出新模型把哪些搞好了、哪些搞坏了；
 ③ **段级语义值对比**：同一段，旧模型能给的语义值 vs 新模型能给的（含现存判定值）。

口径纪律（沿用已修过的教训）：
 - **未判定的段一律排除**（曾把"未判定"当"无标识"，整组结论作废）；
 - B 组 = 用户确认过"确实没标识"的段，命中即误检；
 - 两侧同 imgsz / 同 conf / 同阈值规则，避免"超参对比冒充模型对比"；
 - 页面结构用 `.card/.imgbox/.bx`（**放大补丁只认 .imgbox**，裸 img 挂不上）+ 结尾调放大补丁，
   并在末尾**自检 `.imgbox` 数量 > 0**（2026-09-23 踩过"补丁打印成功但一张没挂上"）。

怎么跑（要 GPU；训练窗口内服务是停的，不能用 app API）：
  AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python \\
    /opt/ad_mining/tools/_ab_models_ownframes.py \\
    --old /opt/ad_mining/tsr_sign.pt --new /opt/datasets/tsr_runs/tsr_v2/weights/best.pt \\
    --project Oversea_欧洲 --segments 40
怎么判定成功：控制台打印 A/B 两组"旧 vs 新"的命中率与四象限计数；页面可开、可点击放大、
   `.imgbox` 计数 > 0；结论能回答"误检是否更低、召回是否不降"。
"""
import argparse, io, json, os, random, subprocess, sys, urllib.parse

STORE = os.environ.get("AD_INDEX_STORE", "/opt/ad_mining/index_store")
NO_SIGN = "无标识"
SKIP_VALS = {"unknown", ""}
CLASSES = ["speed_limit", "prohibition", "warning", "mandatory", "guide", "signal", "crosswalk"]
# ⚠️ 只比**两个模型共有的牌子类**：旧模型是 7 类、新模型是 18 类（多了行人/车/标线），
# 若把任意框都算"命中"，新模型会凭"多出来的类"虚高 —— 那就不是模型对比、是类数对比。
SIGN_KEYS = ("speed_limit", "prohibition", "warning", "mandatory", "guide", "signal",
             "crosswalk", "crosswalk_sign", "traffic_light", "限速", "禁令标志", "警告标志",
             "指示标志", "指路标志", "红绿灯", "交通标识牌", "人行横道标志")
# 类别中文名（按类明细表要显示）—— 之前加"按类明细"时漏了这个表，跑到那儿就 NameError（2026-09-24 实测）
CN = {"speed_limit": "限速", "prohibition": "禁令标志", "warning": "警告标志", "mandatory": "指示标志",
      "guide": "指路标志", "signal": "红绿灯", "crosswalk": "人行横道标志", "crosswalk_sign": "人行横道标志",
      "traffic_light": "红绿灯", "pedestrian": "行人", "two_wheeler": "两轮车", "car": "小车",
      "big_vehicle": "大车", "marking_crosswalk": "人行横道", "marking_stopline": "停止线",
      "marking_channelizing": "导流线", "marking_box_junction": "网状线", "marking_waiting": "待行区",
      "lane_line": "车道线"}
TO_ONT = {"speed_limit": "限速", "prohibition": "禁令标志", "warning": "警告标志",
          "mandatory": "指示标志", "guide": "指路标志", "signal": "红绿灯",
          "crosswalk": "交通标识牌"}


def out_path(old_w, new_w, project):
    def tag(p):
        return os.path.basename(os.path.dirname(os.path.dirname(p)))
    pj = "".join(ch for ch in project if ch.isalnum() or ch in "_-")
    return "/opt/ad_mining/reports/ab_train_%s_vs_%s_%s.html" % (tag(old_w), tag(new_w), pj)


def load_json(p):
    sz = os.path.getsize(p) / 1e6
    if sz > 400:
        print("  ⚠️ %s 有 %.0f MB，整份读很占内存（AGENTS.md：大文件先分流）" % (p, sz))
    return json.load(io.open(p, encoding="utf-8"))


def pick_segments(project, n, seed):
    clips = load_json(os.path.join(STORE, project, "clips.json"))
    items = clips.get("clips") if isinstance(clips, dict) else clips
    if isinstance(items, dict):
        items = list(items.values())
    has, non, unj = [], [], 0
    for c in items:
        if not isinstance(c, dict):
            continue
        ts = ((c.get("vlm_result") or {}).get("traffic_sign") or c.get("traffic_sign") or [])
        if isinstance(ts, str):
            ts = [ts]
        ts = [str(x).strip() for x in ts if str(x).strip() and str(x).strip() not in SKIP_VALS]
        fid = list(c.get("frame_ids") or [])
        if not fid:
            continue
        if not ts:
            unj += 1
            continue                                  # **未判定：排除**
        (non if all(t == NO_SIGN for t in ts) else has).append((c.get("clip_id"), fid))
    print("  %s：有标识 %d 段 / 无标识 %d 段 / 未判定 %d 段（已排除）" % (project, len(has), len(non), unj))
    rnd = random.Random(seed)
    rnd.shuffle(has); rnd.shuffle(non)
    return has[:n], non[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="旧模型（现役）权重")
    ap.add_argument("--new", required=True, help="新模型（重训后）权重")
    ap.add_argument("--project", default="Oversea_欧洲")
    ap.add_argument("--segments", type=int, default=40)
    ap.add_argument("--per-seg", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-low", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    for p in (a.old, a.new):
        if not os.path.exists(p):
            print("✗ 权重不存在:", p); sys.exit(2)

    has, non = pick_segments(a.project, a.segments, a.seed)
    meta = load_json(os.path.join(STORE, a.project, "metadata.json"))
    rnd = random.Random(a.seed)

    def sample(groups):
        out = []
        for cid, fids in groups:
            f = sorted(rnd.sample(fids, min(a.per_seg, len(fids))))
            fr = []
            for iid in f:
                p = meta[int(iid)].get("path") if 0 <= int(iid) < len(meta) else None
                if p and os.path.exists(p):
                    fr.append((int(iid), p))
            if fr:
                out.append((cid, fr))
        return out
    g_has, g_non = sample(has), sample(non)
    allf, seen = [], set()
    for _c, fr in g_has + g_non:
        for iid, p in fr:
            if iid not in seen:
                seen.add(iid); allf.append((iid, p))
    print("  抽样：A 组 %d 段 / B 组 %d 段 → 去重 %d 帧" % (len(g_has), len(g_non), len(allf)))
    if not allf:
        print("✗ 没抽到帧"); sys.exit(4)
    paths = [p for _i, p in allf]

    from ultralytics import YOLO
    res = {}
    for tag, w in (("old", a.old), ("new", a.new)):
        print("  跑 %s 模型 … %s" % (tag, os.path.basename(w)))
        m = YOLO(w)
        names = m.names
        d = {}
        for k in range(0, len(allf), 16):
            for (iid, _p), r in zip(allf[k:k + 16],
                                    m.predict(paths[k:k + 16], imgsz=a.imgsz, conf=a.conf_low,
                                              verbose=False)):
                hi, lo, W, H = [], [], float(r.orig_shape[1]), float(r.orig_shape[0])
                for b in r.boxes:
                    cf = float(b.conf[0]); c = names[int(b.cls[0])]
                    if str(c) not in SIGN_KEYS and str(c) not in CLASSES:
                        continue                      # 非牌子类（行人/车/标线…）不参与牌子对比
                    (hi if cf >= a.conf else lo).append((c, cf, [float(v) for v in b.xyxy[0]]))
                d[iid] = (hi, lo, W, H)
        res[tag] = d
        del m
        try:
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass

    def hit(tag, iid, low=False):
        hi, lo, _W, _H = res[tag].get(iid, ([], [], 0, 0))
        return bool(hi or (low and lo))

    def stats(groups, name):
        nseg = max(1, len(groups)); nfr = sum(len(f) for _c, f in groups) or 1
        r = {}
        for tag in ("old", "new"):
            r[tag + "_seg"] = sum(1 for _c, f in groups if any(hit(tag, i) for i, _p in f))
            r[tag + "_seg_low"] = sum(1 for _c, f in groups if any(hit(tag, i, True) for i, _p in f))
            r[tag + "_fr"] = sum(1 for _c, f in groups for i, _p in f if hit(tag, i))
        quad = {"both": 0, "old_only": 0, "new_only": 0, "none": 0}
        for _c, f in groups:
            for i, _p in f:
                o, n = hit("old", i), hit("new", i)
                quad["both" if (o and n) else "old_only" if o else "new_only" if n else "none"] += 1
        print("  [%s] %d 段 / %d 帧" % (name, len(groups), sum(len(f) for _c, f in groups)))
        print("      段级命中：旧 %.0f%%（@%.2f %.0f%%）→ 新 %.0f%%（@%.2f %.0f%%）"
              % (100.0 * r["old_seg"] / nseg, a.conf_low, 100.0 * r["old_seg_low"] / nseg,
                 100.0 * r["new_seg"] / nseg, a.conf_low, 100.0 * r["new_seg_low"] / nseg))
        print("      帧级四象限：都命中 %d ｜ 旧命中新漏 **%d** ｜ 旧漏新命中 **%d** ｜ 都漏 %d"
              % (quad["both"], quad["old_only"], quad["new_only"], quad["none"]))
        return r, quad

    print("\n=== 训练前后对比（A=有标识段比召回，B=真无标识段比误检）===")
    RA, QA = stats(g_has, "A 有标识段")
    RB, QB = stats(g_non, "B 无标识段（命中即误检）")
    won = QB["new_only"] - QB["old_only"]
    print("\n  结论要点：")
    print("   · B 组误检帧：旧 %d → 新 %d（新多报 %+d 帧，负值更好）"
          % (QB["both"] + QB["old_only"], QB["both"] + QB["new_only"], won))
    print("   · A 组召回帧：旧 %d → 新 %d（新漏 %d 帧、新找回 %d 帧）"
          % (RA["old_fr"], RA["new_fr"], QA["old_only"], QA["new_only"]))

    # ── 按类详细对比（用户 2026-09-23："前后对比效果要详细列出来"）──
    # 每个类分别数：旧/新各自的命中帧数、四象限（都命中/旧命中新漏/旧漏新命中/都漏）、段级命中率。
    def per_class(frames, name):
        rows_c = []
        for cn in CLASSES:
            f_old = f_new = 0
            q = {"both": 0, "old_only": 0, "new_only": 0, "none": 0}
            seg_o = seg_n = 0
            for _c, fr in frames:
                o_any = n_any = False
                for iid, _p in fr:
                    ho = any(str(x[0]) == cn for x in res["old"].get(iid, ([], [], 0, 0))[0])
                    hn = any(str(x[0]) == cn for x in res["new"].get(iid, ([], [], 0, 0))[0])
                    f_old += 1 if ho else 0
                    f_new += 1 if hn else 0
                    o_any = o_any or ho
                    n_any = n_any or hn
                    q["both" if (ho and hn) else "old_only" if ho else "new_only" if hn else "none"] += 1
                seg_o += 1 if o_any else 0
                seg_n += 1 if n_any else 0
            rows_c.append((cn, CN.get(cn, cn), f_old, f_new, q, seg_o, seg_n, max(1, len(frames))))
        print("  [%s] 按类明细（帧级命中 旧→新 ｜ 四象限 都命中/旧only/新only/都漏 ｜ 段级命中 旧→新）" % name)
        for cn, ccn, fo, fn, q, so, sn, ns in rows_c:
            if fo == 0 and fn == 0:
                continue                       # 两边都没检出的类不占版面
            print("     %-16s %-8s 帧 %4d→%-4d ｜ %4d/%4d/%4d/%4d ｜ 段 %3d/%3d（%.0f%%→%.0f%%）"
                  % (cn, ccn, fo, fn, q["both"], q["old_only"], q["new_only"], q["none"],
                     so, sn, 100.0 * so / ns, 100.0 * sn / ns))
        return rows_c
    print("")
    print("=== 按类明细（A 组：有标识段）===")
    pc_has = per_class(g_has, "A 有标识段")
    print("")
    print("=== 按类明细（B 组：真无标识段，命中即误检）===")
    pc_non = per_class(g_non, "B 真无标识段")

    # 段级语义值对比
    seg_rows = []
    for grp, tagn in ((g_has, "有标识"), (g_non, "无标识")):
        for cid, fr in grp:
            vo, vn = set(), set()
            for iid, _p in fr:
                for c, _cf, _b in res["old"].get(iid, ([], [], 0, 0))[0]:
                    vo.add(TO_ONT.get(c, c))
                for c, _cf, _b in res["new"].get(iid, ([], [], 0, 0))[0]:
                    vn.add(TO_ONT.get(c, c))
            seg_rows.append((tagn, cid, "、".join(sorted(vo)) or NO_SIGN, "、".join(sorted(vn)) or NO_SIGN))

    def bx(items, W, H, color, dashed=False):
        out = []
        for c, cf, (x1, y1, x2, y2) in items:
            st = ("left:%.2f%%;top:%.2f%%;width:%.2f%%;height:%.2f%%"
                  % (x1 / W * 100, y1 / H * 100, (x2 - x1) / W * 100, (y2 - y1) / H * 100))
            out.append('<div class="bx%s" style="%s;border-color:%s"><span style="background:%s">%s %.2f'
                       '</span></div>' % (" dash" if dashed else "", st, color, color,
                                          TO_ONT.get(c, c), cf))
        return "".join(out)

    def cards(groups):
        out = []
        for cid, fr in groups:
            for iid, _p in fr:
                ho, lo, Wo, Ho = res["old"].get(iid, ([], [], 0, 0))
                hn, ln, Wn, Hn = res["new"].get(iid, ([], [], 0, 0))
                out.append(
                    '<div class="card"><div class="cap">帧 %d ｜ <b style="color:#f59e0b">旧 %d 框</b>'
                    '(+%d 低) → <b style="color:#22c55e">新 %d 框</b>(+%d 低)</div><div class="row">'
                    '<div class="col"><div class="imgbox"><img loading="lazy" src="/api/image/%s/%d?w=800">'
                    '%s%s</div></div>'
                    '<div class="col"><div class="imgbox"><img loading="lazy" src="/api/image/%s/%d?w=800">'
                    '%s%s</div></div></div></div>'
                    % (iid, len(ho), len(lo), len(hn), len(ln),
                       urllib.parse.quote(a.project), iid, bx(ho, Wo, Ho, "#f59e0b"),
                       bx(lo, Wo, Ho, "#fbbf24", True),
                       urllib.parse.quote(a.project), iid, bx(hn, Wn, Hn, "#22c55e"),
                       bx(ln, Wn, Hn, "#a3e635", True)))
        return "".join(out)

    rows = "".join('<tr><td>%s</td><td class="mono">%s</td><td>%s</td><td><b>%s</b></td></tr>' % r
                   for r in seg_rows)

    def cls_table(pc, title):
        tr = []
        for cn, ccn, fo, fn, q, so, sn, ns in pc:
            if fo == 0 and fn == 0:
                continue
            tr.append('<tr><td>%s</td><td>%s</td><td>%d → <b>%d</b></td>'
                      '<td>%d / <span style="color:#f87171">%d</span> / '
                      '<span style="color:#22c55e">%d</span> / %d</td>'
                      '<td>%d → <b>%d</b>（%.0f%% → %.0f%%）</td></tr>'
                      % (cn, ccn, fo, fn, q["both"], q["old_only"], q["new_only"], q["none"],
                         so, sn, 100.0 * so / ns, 100.0 * sn / ns))
        return ('<h2>%s · 按类明细</h2><table><tr><th>类别</th><th>本体取值</th><th>帧级命中 旧→新</th>'
                '<th>四象限 都命中/旧命中新漏/旧漏新命中/都漏</th><th>段级命中 旧→新</th></tr>%s</table>'
                % (title, "".join(tr)))
    cls_html = cls_table(pc_has, "A 组（有标识段）") + cls_table(pc_non, "B 组（真无标识段，越低越好）")
    nA, nB = max(1, len(g_has)), max(1, len(g_non))
    html = ("""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>训练前后对比</title><style>
body{background:#0b1220;color:#e2e8f0;font:14px/1.6 "Microsoft YaHei",system-ui,sans-serif;margin:0;padding:16px}
.sum{background:rgba(148,163,184,.10);border:1px solid rgba(148,163,184,.25);border-radius:10px;padding:10px 14px;margin:8px 0 14px}
h2{margin:18px 0 8px;font-size:16px}
table{border-collapse:collapse;width:100%%;font-size:12px;margin-bottom:14px}
th,td{border:1px solid rgba(148,163,184,.25);padding:3px 6px;text-align:left}
th{background:rgba(148,163,184,.12)}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:11px}
.card{display:inline-block;width:calc(50%% - 6px);vertical-align:top;background:rgba(148,163,184,.07);
 border:1px solid rgba(148,163,184,.2);border-radius:10px;padding:8px;margin:0 6px 10px 0}
.cap{font-size:12px;margin-bottom:4px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.col{min-width:0}
.imgbox{position:relative;line-height:0}
.imgbox img{width:100%%;height:auto;display:block;border-radius:4px;background:#000;cursor:zoom-in}
.bx{position:absolute;border:2px solid;border-radius:2px;line-height:1}
.bx.dash{border-style:dashed;opacity:.75}
.bx span{position:absolute;left:-2px;top:-12px;font-size:9px;padding:0 2px;border-radius:3px;
 color:#0b1220;white-space:nowrap;font-weight:700}
</style></head><body>
<h1>训练前后对比 · __PROJ__</h1>
<div class="sum">
 <div>旧模型 <b style="color:#f59e0b">__OLD__</b> ｜ 新模型 <b style="color:#22c55e">__NEW__</b>
  （同批帧/同 imgsz=__IMGSZ__/同 conf≥__CONF__，虚线=__CLOW__~__CONF__ 低置信框）</div>
 <div><b>A 组 有标识段</b>（__NSA__ 段）：段级命中 旧 <b>__AO__%%</b> → 新 <b>__AN__%%</b>
  （@__CLOW__ __AOL__%% → __ANL__%%）</div>
 <div><b>B 组 真无标识段</b>（__NSB__ 段，命中即误检）：段级命中 旧 <b>__BO__%%</b> → 新 <b>__BN__%%</b>
  （@__CLOW__ __BOL__%% → __BNL__%%）</div>
 <div>帧级四象限（B 组）：都命中 __QB__ ｜ <span style="color:#f87171">旧命中新漏 __QO__</span> ｜
  <span style="color:#22c55e">旧漏新命中 __QN__</span> ｜ 都漏 __QNO__</div>
 <div style="color:#94a3b8">判读：<b>B 组越低越好</b>（误检）；<b>A 组越高越好</b>（召回）但不许明显掉。
  两侧同口径、未判定段已排除；页面可点击放大看框。</div>
</div>
__CLS__
<h2>段级语义值对比（同一段：旧 vs 新）</h2>
<table><tr><th>组</th><th>clip_id</th><th>旧模型给出</th><th>新模型给出</th></tr>__ROWS__</table>
<h2>A · 有标识段（左=旧模型，右=新模型）</h2>
__CARDS_A__
<h2>B · 真无标识段（左=旧，右=新；新模型应该更少框）</h2>
__CARDS_B__
</body></html>
""").replace("__PROJ__", a.project).replace("__OLD__", os.path.basename(a.old)) \
        .replace("__NEW__", os.path.basename(a.new)).replace("__IMGSZ__", str(a.imgsz)) \
        .replace("__CONF__", str(a.conf)).replace("__CLOW__", str(a.conf_low)) \
        .replace("__NSA__", str(len(g_has))).replace("__NSB__", str(len(g_non))) \
        .replace("__AO__", "%.0f" % (100.0 * RA["old_seg"] / nA)) \
        .replace("__AN__", "%.0f" % (100.0 * RA["new_seg"] / nA)) \
        .replace("__AOL__", "%.0f" % (100.0 * RA["old_seg_low"] / nA)) \
        .replace("__ANL__", "%.0f" % (100.0 * RA["new_seg_low"] / nA)) \
        .replace("__BO__", "%.0f" % (100.0 * RB["old_seg"] / nB)) \
        .replace("__BN__", "%.0f" % (100.0 * RB["new_seg"] / nB)) \
        .replace("__BOL__", "%.0f" % (100.0 * RB["old_seg_low"] / nB)) \
        .replace("__BNL__", "%.0f" % (100.0 * RB["new_seg_low"] / nB)) \
        .replace("__QB__", str(QB["both"])).replace("__QO__", str(QB["old_only"])) \
        .replace("__QN__", str(QB["new_only"])).replace("__QNO__", str(QB["none"])) \
        .replace("__CLS__", cls_html).replace("__ROWS__", rows).replace("__CARDS_A__", cards(g_has)).replace("__CARDS_B__", cards(g_non))
    out = out_path(a.old, a.new, a.project)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(html)
    try:
        subprocess.run(["/venv/bin/python", "/opt/ad_mining/tools/_sign_probe_zoom.py", out], check=False)
    except Exception as e:
        print("  放大补丁失败（不影响页面）:", str(e)[:80])
    # 自检：放大补丁只认 .imgbox，裸 img 会"打印成功但一张没挂上"（2026-09-23 踩过）
    txt = io.open(out, encoding="utf-8").read()
    nbox = txt.count('class="imgbox"')
    print("\nHTML → %s\n      http://10.2.248.34:8009/static_root/reports/%s"
          % (out, os.path.basename(out)))
    print("  自检：.imgbox=%d %s｜放大补丁标记=%d %s"
          % (nbox, "✓" if nbox > 0 else "✗ 放大挂不上", txt.count("__zoom_patch__"),
             "✓" if txt.count("__zoom_patch__") else "✗"))


if __name__ == "__main__":
    main()
