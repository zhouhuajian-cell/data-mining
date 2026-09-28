# 干什么：部署后的**只读**验收 —— 拿真实帧分别跑【现役 v3】与【备份的旧 7 类权重】，对比牌类检出，
#         证明"生产确实在用 v3"且没有退化。**不写任何平台数据**（只 predict，不落库）。
# 怎么跑：AD_CLIP_PICK=8 AD_VLM_THUMB=448 AD_VLM_MAX_PX=768 /venv/bin/python -u tools/_verify_sign_deploy.py
# 怎么判定成功：打印两个权重的 MD5 与逐帧牌类检出数；v3 的检出应 ≥ 旧权重（同一批帧）。
import io
import json
import os
import sys

from ultralytics import YOLO

STORE = os.environ.get("AD_INDEX_STORE", "/opt/ad_mining/index_store")
PROJ = sys.argv[1] if len(sys.argv) > 1 else "Oversea_欧洲"
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NEW = "/opt/ad_mining/tsr_sign.pt"                       # 现役（应为 v3）
OLD = "/tmp/_old_sign.pt"      # 备份文件名带 .bak_ 后缀，ultralytics 只认 .pt → 先复制一份
_src = "/opt/ad_mining/backups/rollback_20260928/tsr_sign.pt.bak_20260928_095429"
if os.path.exists(_src) and not os.path.exists(OLD):
    import shutil
    shutil.copy2(_src, OLD)
SIGN_KW = ("speed_limit", "prohibition", "warning", "mandatory", "guide", "crossroad", "crosswalk",
           "限速", "禁令", "警告", "指示", "指路", "人行横道", "红绿灯", "signal", "traffic_light")
import hashlib


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ⚠️ 必须按 **A 组（有标识段）** 抽样：随手取"头几个 clip 的前几帧"实测拿到的全是没牌子的帧，
#    会得出"两个模型都是 0 框"的荒谬结论（这个坑我踩过两次，见 STATE.md）。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _eval_checkpoints as E
has, _non = E.pick_segments(PROJ, 40, 20260923)
meta = E.load_json(os.path.join(STORE, PROJ, "metadata.json"))
frames = []
for _cid, fids in has:
    for _i in fids[:4]:
        if 0 <= int(_i) < len(meta):
            _p = meta[int(_i)].get("path")
            if _p and os.path.exists(_p):
                frames.append(_p)
    if len(frames) >= NF:
        break
frames = frames[:NF]
print("[帧] %s：从**有标识段**取 %d 张真实帧" % (PROJ, len(frames)))
for tag, w in (("现役(v3)", NEW), ("旧权重(备份)", OLD)):
    if not os.path.exists(w):
        print("  ⚠️ 缺 %s，跳过" % w)
        continue
    m = YOLO(w)
    tot = hi = 0
    mx = 0.0
    per = {}
    for r in m.predict(source=frames, imgsz=int(os.environ.get("AD_SIGN_IMGSZ", "1280")),
                       conf=0.15, verbose=False):
        for b in r.boxes:
            nm = str(r.names[int(b.cls[0])])
            if nm not in SIGN_KW:
                continue
            cf = float(b.conf[0])
            tot += 1
            mx = max(mx, cf)
            per[nm] = per.get(nm, 0) + 1
    print("\n=== %s ===" % tag)
    print("  权重 %s（md5 %s）" % (w, md5(w)[:16]))
    print("  conf>=0.15 牌类框数 %d ｜ 最高分 %.3f" % (tot, mx))
    for k in sorted(per, key=lambda x: -per[x])[:8]:
        print("    %-16s %d" % (k, per[k]))
