#!/venv/bin/python
# -*- coding: utf-8 -*-
"""Oversea_东南亚 **整批按最新链路重跑**（2026-09-23，用户："东南亚重新跑一下，按最新的"）。

为什么要"重检测 + 重判定"而不是只判：
  闭集交通标志模型（`src="sign"`）是 2026-09-23 才接进**检测阶段**的 —— 已有记录的牌子框
  来自 YOLO-World（开放词表，实测段级误检 98%）。只重判不重检，牌子那一路还是旧证据，
  结果会是"新旧混口径"。所以整批重检测。

覆盖口径：**全部段覆盖**（用户 09-22 定"真值现在还不错直接覆盖"，且还没上人判）。
中途可停：`touch workspace/CLIP_JUDGE_STOP` → 当前段判完即停（检测阶段随时可停，重跑会重来）。

怎么跑（服务器侧 nohup；要 3.5~4.5 小时）：
  nohup /venv/bin/python -u /opt/ad_mining/tools/_rerun_sea_all.py > /opt/ad_mining/logs/rerun_sea_all.log 2>&1 &
怎么判定成功：日志末尾"完成：检测 X 帧 / 判定成功 N 段"；`clip_judge_status.json` 的 total=段数、done 逐步上升；
  抽查若干段的 `vlm_result.traffic_sign` 里出现 限速/禁令标志/警告标志/指示标志/指路标志 之一（证明新模型生效）。
"""
import json, os, sys, time, urllib.parse, urllib.request

sys.path.insert(0, "/opt/ad_mining")
from clip_service import load_clips, flush_pending          # noqa: E402

PROJ = os.environ.get("AD_RERUN_PROJ", "Oversea_东南亚")   # 欧洲 100 段：AD_RERUN_PROJ=Oversea_欧洲
LIMIT = int(os.environ.get("AD_TASK_LIMIT", "100"))   # 测试任务默认只跑 100 段（用户 2026-09-23 明确）
B = "http://127.0.0.1:8009"
CTX = {"name": PROJ, "dir": "/opt/ad_mining/index_store/" + PROJ}
# ⚠️ 不能用 clip_judge_status.json：那个文件在前端被硬编码成「段级判定(VLM)」，
# 而重跑的检测阶段单位是**帧**，复用会把"检测 14000 帧"显示成"已判 14000 段"
# （用户 2026-09-23 当场发现）。改用独立文件 + task_type=sign_rerun。
STATUS = "/opt/ad_mining/workspace/tsr_rerun_status.json"
STOP = "/opt/ad_mining/workspace/CLIP_JUDGE_STOP"
SNAP = "/opt/ad_mining/backups/clips_snapshots/rerun_sea_all_%s.json" % time.strftime("%Y%m%d_%H%M%S")
LOG = os.environ.get("AD_RERUN_LOG", "/opt/ad_mining/logs/rerun_sea_all.log")
DET_CHUNK = int(os.environ.get("AD_RERUN_CHUNK", "200"))


def L(m):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(s, flush=True)


def _get(path, params=None, timeout=120):
    u = B + path + ("?" + urllib.parse.urlencode(params) if params else "")
    with urllib.request.urlopen(u, timeout=timeout) as r:
        return json.load(r)


def _post(path, data, timeout=1800):
    req = urllib.request.Request(B + path, data=urllib.parse.urlencode(data).encode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _status(done_, total_, running_=True, phase="", unit="", task_type="sign_detect"):
    """写重跑状态。三个字段各有分工（2026-09-23 用户定名）：
      task_type：`sign_detect`→总览显示「YOLO批量检测」；`sign_judge`→「VLM场景综合判断」
      phase    ：msg 前缀（"重检测"/"重判定"）
      unit     ：单位（" 帧"/" 段"）—— 之前把帧数当段数显示，就是这里没区分单位。"""
    try:
        json.dump({"project": PROJ, "running": running_, "done": int(done_), "total": int(total_),
                   "phase": phase, "unit": unit, "task_type": task_type, "ts": time.time()},
                  open(STATUS, "w", encoding="utf-8"))
    except Exception:
        pass


def _vlm_ok():
    """硬规则：非 7B 不许判。"""
    try:
        s = _get("/api/vlm_status", {"warm": 1}, timeout=600)
        return bool(s.get("ok") and s.get("is_7b")), "%s is_7b=%s hidden=%s" % (
            s.get("model"), s.get("is_7b"), s.get("hidden"))
    except Exception as e:
        return False, "vlm_status 失败: %s" % str(e)[:80]


def main():
    clips = load_clips(CTX)
    segs = [c for c in clips if c.get("clip_id") and c.get("frame_ids")]
    segs.sort(key=lambda c: str(c.get("clip_id")))
    if LIMIT and len(segs) > LIMIT:      # 测试任务口径：默认 100 段，别动辄全量
        segs = segs[:LIMIT]
    total = len(segs)
    L("=== %s 整批重跑：%d 段（重检测 + 重判定） ===" % (PROJ, total))
    try:
        os.makedirs(os.path.dirname(SNAP), exist_ok=True)
        json.dump({"project": PROJ, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "n": total,
                   "clip_ids": [c["clip_id"] for c in segs]},
                  open(SNAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        L("段清单快照 → %s" % SNAP)
    except Exception as _e:
        L("快照写入失败(不影响): %s" % _e)

    ok7, desc = _vlm_ok()
    L("VLM 闸门: %s（%s）" % ("✓ 7B" if ok7 else "✗ 非 7B", desc))
    if not ok7:
        L("!! 非 7B，按硬规则中止（不许静默降级）")
        _status(0, total, running_=False, phase="非7B中止")
        sys.exit(2)

    # ---- 1. 重检测（全部帧；检测阶段内部会跑 YOLO + YOLO-World + 闭集标志）----
    all_ids = []
    for c in segs:
        all_ids += [int(x) for x in (c.get("frame_ids") or [])]
    seen, uniq = set(), []
    for i in all_ids:
        if i not in seen:
            seen.add(i); uniq.append(i)
    L("待检测帧 %d（去重后 %d）" % (len(all_ids), len(uniq)))
    _status(0, len(uniq), phase="检测", unit=" 帧", task_type="sign_detect")
    t0 = time.time(); nfail = 0
    for k in range(0, len(uniq), DET_CHUNK):
        part = uniq[k:k + DET_CHUNK]
        try:
            r = _post("/api/yolo_detect_batch",
                      {"project": PROJ, "image_ids": ",".join(str(x) for x in part)})
            if (r or {}).get("code") != 200:
                nfail += 1
                L("  检测块失败 code=%s msg=%s" % ((r or {}).get("code"), str((r or {}).get("msg"))[:80]))
        except Exception as e:
            nfail += 1
            L("  检测块异常: %s" % str(e)[:100])
        done = min(k + DET_CHUNK, len(uniq))
        _status(done, len(uniq), phase="检测", unit=" 帧", task_type="sign_detect")
        el = time.time() - t0
        L("  检测 %d/%d（失败块 %d）已用 %.1f 分，均 %.2f 秒/帧，预计剩余 %.0f 分"
          % (done, len(uniq), nfail, el / 60, el / max(1, done), (len(uniq) - done) * el / max(1, done) / 60))
    L("检测阶段结束：%d 帧，用时 %.1f 分" % (len(uniq), (time.time() - t0) / 60))

    # ---- 2. 重判定（整批覆盖）----
    _status(0, total, phase="判定", unit=" 段", task_type="sign_judge")
    ok = bad = 0; failed = []
    t1 = time.time()
    for i, c in enumerate(segs, 1):
        if os.path.exists(STOP):
            L("检测到 CLIP_JUDGE_STOP，停止（已判 %d/%d，进度保留）" % (ok, total))
            break
        cid = c["clip_id"]
        r, last = None, None
        for attempt in range(3):
            try:
                r = _post("/api/clips/analyze_single", {"project": PROJ, "clip_id": cid})
                break
            except Exception as e:
                last = e; time.sleep(3 * (attempt + 1))
        code = (r or {}).get("code")
        if code == 409:
            L("!! 非 7B 被拒(409)，中止 %s" % str((r or {}).get("msg"))[:120])
            _status(ok, total, running_=False, phase="非7B中止")
            sys.exit(2)
        if code == 200:
            ok += 1
        else:
            bad += 1
            failed.append((cid, str((r or {}).get("msg") or last)[:100]))
            L("  失败 %s: %s" % (cid[-26:], failed[-1][1]))
        if i % 5 == 0 or i == total:
            el = time.time() - t1
            _status(ok, total, phase="判定", unit=" 段", task_type="sign_judge")
            L("  判定 %d/%d（成功 %d 失败 %d）已用 %.1f 分，预计剩余 %.0f 分"
              % (i, total, ok, bad, el / 60, (total - i) * el / i / 60))
    # ⚠️ 收尾必须**调应用的接口**落盘：判定结果先攒在应用内存里、定期刷盘，
    #    脚本跑完立刻去读 clips.json 会缺最后几段（2026-09-28 实测缺 5 段，95/100）。
    #    原来这里写的是 flush_pending() —— 那是**应用里**的函数，脚本里不存在，
    #    NameError 被下面的 except 静默吞掉，等于根本没刷。别改回函数调用，就用接口。
    try:
        _post("/api/clips/flush", {"project": PROJ})
        L("收尾落盘：已调 /api/clips/flush")
    except Exception as e:
        L("收尾落盘失败（结果仍会在应用下次刷盘时落盘）: %s" % str(e)[:80])
    _status(ok, total, running_=False, phase="完成")
    L("=== 完成：检测 %d 帧 / 判定成功 %d 段 / 失败 %d，总用时 %.1f 分 ==="
      % (len(uniq), ok, bad, (time.time() - t0) / 60))
    for cid, m in failed[:20]:
        L("   失败 %s  %s" % (cid[-26:], m))


if __name__ == "__main__":
    main()
