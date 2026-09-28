#!/venv/bin/python
# -*- coding: utf-8 -*-
"""在**业务真值集**上按检查点评测：不看 val，只看我们自己的帧（用户 2026-09-23 明确要求）。

用户原话（2026-09-23）："每10epoch保存best权重，单独拿业务真值集测漏检/误检，不要只看 val"。

业务真值集口径（沿用已修正过的教训）：
  - 分组只看**已判定的段**：A 组 = `traffic_sign` 有值（该检出，比**漏检**）；
    B 组 = `traffic_sign` = `无标识`（用户确认确实没有，命中即**误检**）；**未判定的段一律排除**。
  - 只统计**牌子类**（各家模型共有的那些类）—— 新模型是 18 类，若把行人/车/标线也算"命中"，
    结论就变成"类数对比"而不是"模型对比"。
  - 同一批帧、同 imgsz、同 conf，多个检查点（epoch10/20/best）与旧模型逐一对比。

怎么跑（要 GPU；训练窗口内服务停着，不能用 app API）：
  AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python \\
    /opt/ad_mining/tools/_eval_checkpoints.py --project Oversea_欧洲 --segments 40
怎么判定成功：控制台打印"检查点 × (A 组漏检率 / B 组误检率) + 逐类明细"；
  HTML 里每个检查点一行、带点击放大；`.imgbox` 计数 > 0。
"""
import argparse, io, json, os, random, subprocess, sys, urllib.parse

STORE = os.environ.get("AD_INDEX_STORE", "/opt/ad_mining/index_store")
RUN = "/opt/datasets/tsr_runs/tsr_u1"
NO_SIGN = "无标识"
SKIP_VALS = {"unknown", ""}
# 只比牌子类（旧 7 类 + 新模型的标识类）
SIGN_KEYS = {"speed_limit", "prohibition", "warning", "mandatory", "guide", "signal",
             "crosswalk", "crosswalk_sign", "traffic_light",
             "限速", "禁令标志", "警告标志", "指示标志", "指路标志", "红绿灯",
             "交通标识牌", "人行横道标志"}


def load_json(p):
    if os.path.getsize(p) / 1e6 > 400:
        print("  ⚠️ %s 很大，读取会占内存" % p)
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
            continue                       # 未判定：排除
        (non if all(t == NO_SIGN for t in ts) else has).append((c.get("clip_id"), fid))
    print("  %s：有标识 %d 段 / 无标识 %d 段 / 未判定 %d 段（已排除）" % (project, len(has), len(non), unj))
    rnd = random.Random(seed)
    rnd.shuffle(has); rnd.shuffle(non)
    return has[:n], non[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="Oversea_欧洲")
    ap.add_argument("--segments", type=int, default=40)
    ap.add_argument("--per-seg", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-low", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--baseline", default="/opt/ad_mining/tsr_sign.pt", help="旧模型（现役）")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--run", default=RUN, help="检查点所在 run 目录（默认第一轮 tsr_u1）")
    ap.add_argument("--extra", default="", help="额外要评的权重（如第二轮 best.pt，逗号分隔）")
    a = ap.parse_args()

    cks = []
    if os.path.exists(a.baseline):
        cks.append(("旧模型(现役7类)", a.baseline))
    for name in ("epoch10.pt", "epoch20.pt"):
        p = os.path.join(a.run, "weights", name)
        if os.path.exists(p):
            cks.append((name.replace(".pt", ""), p))
    best = os.path.join(a.run, "weights", "best.pt")
    if os.path.exists(best):
        # 标签用 run 目录名（原来硬编码 "best(30轮)"，换成别的 run 时会误导）
        cks.append(("%s/best" % os.path.basename(a.run.rstrip("/")), best))
    for i, e in enumerate([x for x in a.extra.split(",") if x.strip()]):
        if os.path.exists(e.strip()):
            tag = os.path.basename(os.path.dirname(os.path.dirname(e.strip())))
            cks.append(("%s(best)" % tag, e.strip()))
    if not cks:
        print("✗ 没有任何权重可评"); sys.exit(2)
    print("评测检查点：%s" % "、".join(n for n, _ in cks))

    has, non = pick_segments(a.project, a.segments, a.seed)
    meta = load_json(os.path.join(STORE, a.project, "metadata.json"))
    rnd = random.Random(a.seed)

    def sample(groups):
        out = []
        for cid, fids in groups:
            fr = []
            for iid in sorted(rnd.sample(fids, min(a.per_seg, len(fids)))):
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
    print("  抽样：A 组 %d 段 / B 组 %d 段 → %d 帧" % (len(g_has), len(g_non), len(allf)))
    paths = [p for _i, p in allf]

    from ultralytics import YOLO
    res = {}
    for name, w in cks:
        print("  跑 %s …" % name, flush=True)
        m = YOLO(w)
        names = m.names
        d = {}
        for k in range(0, len(allf), 16):
            for (iid, _p), r in zip(allf[k:k + 16],
                                    m.predict(paths[k:k + 16], imgsz=a.imgsz, conf=a.conf_low,
                                              verbose=False)):
                hi, lo, W, H = [], [], float(r.orig_shape[1]), float(r.orig_shape[0])
                for b in r.boxes:
                    lb = str(names[int(b.cls[0])])
                    if lb not in SIGN_KEYS:
                        continue            # 只算牌子类
                    cf = float(b.conf[0])
                    (hi if cf >= a.conf else lo).append((lb, cf, [float(v) for v in b.xyxy[0]]))
                d[iid] = (hi, lo, W, H)
        res[name] = d
        del m
        try:
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass

    def hit(name, iid, low=False):
        hi, lo, _W, _H = res[name].get(iid, ([], [], 0, 0))
        return bool(hi or (low and lo))

    def stats(groups, tag):
        nseg = max(1, len(groups))
        nfr = sum(len(f) for _c, f in groups) or 1
        print("  [%s] %d 段 / %d 帧" % (tag, len(groups), sum(len(f) for _c, f in groups)))
        rows = []
        for name, _w in cks:
            sg = sum(1 for _c, f in groups if any(hit(name, i) for i, _p in f))
            sgl = sum(1 for _c, f in groups if any(hit(name, i, True) for i, _p in f))
            fr = sum(1 for _c, f in groups for i, _p in f if hit(name, i))
            frl = sum(1 for _c, f in groups for i, _p in f if hit(name, i, True))
            print("     %-16s 段级 %3d/%3d（%3.0f%%）[@%.2f %3.0f%%] ｜ 帧级 %3.0f%% [@%.2f %3.0f%%]"
                  % (name, sg, len(groups), 100.0 * sg / nseg, a.conf_low, 100.0 * sgl / nseg,
                     100.0 * fr / nfr, a.conf_low, 100.0 * frl / nfr))
            rows.append((name, sg, len(groups), sgl, fr, nfr, frl))
        return rows
    print("\n=== A 组：有标识段（段级命中率越高越好 → 漏检率 = 100 − 命中率）===")
    RA = stats(g_has, "A 有标识段")
    print("\n=== B 组：真无标识段（命中即误检，越低越好）===")
    RB = stats(g_non, "B 真无标识段")

    # 逐类明细（最后一版检查点 vs 旧模型）
    def per_class(groups, winner, base):
        out = []
        cls = sorted(SIGN_KEYS)
        for cn in cls:
            fw = fb = 0
            for _c, fr in groups:
                for iid, _p in fr:
                    if any(str(x[0]) == cn for x in res[winner].get(iid, ([], [], 0, 0))[0]):
                        fw += 1
                    if any(str(x[0]) == cn for x in res[base].get(iid, ([], [], 0, 0))[0]):
                        fb += 1
            if fw or fb:
                out.append((cn, fb, fw))
        return out
    winner = cks[-1][0]
    base = cks[0][0]
    pcA = per_class(g_has, winner, base)
    pcB = per_class(g_non, winner, base)
    print("\n=== 逐类明细（A 组：%s vs %s，帧级命中数）===" % (winner, base))
    for cn, fb, fw in pcA:
        print("   %-16s 旧 %4d → 新 %4d  %s" % (cn, fb, fw, "↑" if fw > fb else ("↓" if fw < fb else "=")))
    print("=== 逐类明细（B 组：真无标识段，命中即误检，越少越好）===")
    for cn, fb, fw in pcB:
        print("   %-16s 旧 %4d → 新 %4d  %s" % (cn, fb, fw, "↓好" if fw < fb else ("↑差" if fw > fb else "=")))

    # 页面：检查点汇总表 + 逐类明细（带点击放大）
    def rows_html(rows, invert=False):
        tr = []
        for name, sg, ns, sgl, fr, nfr, frl in rows:
            tr.append('<tr><td>%s</td><td>%d/%d（%.0f%%）</td><td>%.0f%%</td></tr>'
                      % (name, sg, ns, 100.0 * sg / max(1, ns), 100.0 * sgl / max(1, ns)))
        return "".join(tr)
    def pc_html(pc, invert=False):
        tr = []
        for cn, fb, fw in pc:
            mark = ("↓好" if fw < fb else ("↑差" if fw > fb else "=")) if invert else \
                   ("↑" if fw > fb else ("↓" if fw < fb else "="))
            tr.append('<tr><td>%s</td><td>%d</td><td><b>%d</b></td><td>%s</td></tr>' % (cn, fb, fw, mark))
        return "".join(tr)
    html = ("""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>业务真值集 · 检查点评测</title><style>
body{background:#0b1220;color:#e2e8f0;font:14px/1.6 "Microsoft YaHei",system-ui,sans-serif;margin:0;padding:16px}
.sum{background:rgba(148,163,184,.10);border:1px solid rgba(148,163,184,.25);border-radius:10px;padding:10px 14px;margin:8px 0 14px}
h2{margin:18px 0 8px;font-size:16px}
table{border-collapse:collapse;width:100%%;font-size:12px;margin-bottom:14px}
th,td{border:1px solid rgba(148,163,184,.25);padding:3px 6px;text-align:left}
th{background:rgba(148,163,184,.12)}
</style></head><body>
<h1>业务真值集 · 检查点评测（__PROJ__）</h1>
<div class="sum">
 <div>**不用 val**：A 组 = 有标识段（比漏检）、B 组 = 确认无标识段（比误检）；未判定段已排除；只统计牌子类。</div>
 <div>A 组 %d 段 ｜ B 组 %d 段 ｜ conf≥__CONF__（低阈 __CLOW__ 括号内）｜ imgsz=__IMGSZ__</div>
</div>
<h2>A 组：有标识段（段级命中率越高越好）</h2>
<table><tr><th>检查点</th><th>段级命中</th><th>@__CLOW__</th></tr>__RA__</table>
<h2>B 组：真无标识段（命中即误检，越低越好）</h2>
<table><tr><th>检查点</th><th>段级命中</th><th>@__CLOW__</th></tr>__RB__</table>
<h2>逐类明细 · A 组（__WIN__ 帧级命中 vs 旧模型）</h2>
<table><tr><th>类</th><th>旧</th><th>新</th><th>变化</th></tr>__PCA__</table>
<h2>逐类明细 · B 组（真无标识段，越少越好）</h2>
<table><tr><th>类</th><th>旧</th><th>新</th><th>变化</th></tr>__PCB__</table>
</body></html>
""").replace("__PROJ__", a.project).replace("__CONF__", str(a.conf)) \
        .replace("__CLOW__", str(a.conf_low)).replace("__IMGSZ__", str(a.imgsz)) \
        .replace("__RA__", rows_html(RA)).replace("__RB__", rows_html(RB)) \
        .replace("__WIN__", winner).replace("__PCA__", pc_html(pcA)).replace("__PCB__", pc_html(pcB, True))
    html = html.replace("<div>A 组 %d 段 ｜ B 组 %d 段", "<div>A 组 %d 段 ｜ B 组 %d 段" % (len(g_has), len(g_non)))
    out = "/opt/ad_mining/reports/tsr_ckpt_biztruth_%s.html" % "".join(
        ch for ch in a.project if ch.isalnum() or ch == "_")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    io.open(out, "w", encoding="utf-8").write(html)
    try:
        subprocess.run(["/venv/bin/python", "/opt/ad_mining/tools/_sign_probe_zoom.py", out], check=False)
    except Exception as e:
        print("  放大补丁失败:", str(e)[:60])
    print("\nHTML → %s\n      http://10.2.248.34:8009/static_root/reports/%s"
          % (out, os.path.basename(out)))


if __name__ == "__main__":
    main()
