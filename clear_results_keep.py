#!/venv/bin/python
# -*- coding: utf-8 -*-
"""清理某项目的段级结果，但**保留一份清单里的段**（用户 2026-09-23：欧洲清历史结果，只在数据审核看新的 100 段）。

为什么不能"现在就全清"：训练链第④步的**训练前后对比依赖现有段级结果做 A/B 分组**（有标识段 / 无标识段），
清早了对比页就没数据了。所以顺序必须是：**对比页跑完 → 100 段判完 → 再清掉其余**。

清什么：`vlm_result` / `decision` / `_diag` / `final_tags`（模型判定产物）。
**不动 human_tags / human_rejected**（那是人工标注，删了不可逆 —— 用户说"还没上人判"，但也不该由脚本来删）。
备份：清之前 `cp -p clips.json clips.json.bak_clearres_<时间戳>`（回退就是拷回来）。

怎么跑：
  # 自动取最新一份"欧洲"的运行快照作为保留清单
  /venv/bin/python tools/_clear_results_keep.py --project Oversea_欧洲 --keep-from-latest-snapshot
  # 或显式给保留清单
  /venv/bin/python tools/_clear_results_keep.py --project Oversea_欧洲 --keep-json /path/xxx.json
怎么判定成功：日志打印"清掉 N 段 / 保留 M 段"；复查 `已有结果 == M`；备份文件存在。
"""
import argparse, glob, io, json, os, shutil, sys, time

STORE = os.environ.get("AD_INDEX_STORE", "/opt/ad_mining/index_store")
SNAP_DIR = "/opt/ad_mining/backups/clips_snapshots"
MODEL_FIELDS = ("vlm_result", "decision", "_diag", "final_tags")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--keep-json", default="", help="含 clip_ids 的 JSON（运行快照）")
    ap.add_argument("--keep-from-latest-snapshot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    keep = set()
    if a.keep_from_latest_snapshot:
        cands = sorted(glob.glob(os.path.join(SNAP_DIR, "rerun_sea_all_*.json")), reverse=True)
        for f in cands:
            try:
                d = json.load(io.open(f, encoding="utf-8"))
            except Exception:
                continue
            if str(d.get("project") or "") == a.project:
                keep = set(str(x) for x in (d.get("clip_ids") or []))
                print("保留清单来自快照: %s（%d 段，%s）" % (os.path.basename(f), len(keep), d.get("ts")))
                break
    elif a.keep_json:
        d = json.load(io.open(a.keep_json, encoding="utf-8"))
        # ⚠️ 别写成 (d.get("clip_ids") or d if isinstance(d, list) else []) —— 三元优先级会让
        #    字典输入时整个表达式变成 []，保留清单变空（实测踩到，幸好 dry-run 挡住了）
        ids = d if isinstance(d, list) else (d.get("clip_ids") or [])
        keep = set(str(x) for x in ids)
        print("保留清单来自: %s（%d 段）" % (a.keep_json, len(keep)))
    if not keep:
        print("✗ 没有保留清单，拒绝执行（防误清）"); sys.exit(2)

    p = os.path.join(STORE, a.project, "clips.json")
    d = json.load(io.open(p, encoding="utf-8"))
    items = d.get("clips") if isinstance(d, dict) else d
    is_dict = isinstance(items, dict)
    seq = list(items.values()) if is_dict else items
    n_had = sum(1 for c in seq if isinstance(c, dict) and any(c.get(k) for k in MODEL_FIELDS))
    cleared = 0
    for c in seq:
        if not isinstance(c, dict):
            continue
        if str(c.get("clip_id")) in keep:
            continue
        if any(c.get(k) for k in MODEL_FIELDS):
            for k in MODEL_FIELDS:
                c.pop(k, None)
            cleared += 1
    kept = sum(1 for c in seq if isinstance(c, dict) and any(c.get(k) for k in MODEL_FIELDS))
    print("清除前有结果 %d 段 → 清掉 %d 段、保留 %d 段" % (n_had, cleared, kept))
    if a.dry_run:
        print("（--dry-run：未写盘）"); return
    bak = p + ".bak_clearres_%s" % time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(p, bak)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print("已写盘；备份 %s（%.1f MB）" % (bak, os.path.getsize(bak) / 1e6))
    print("建议复查：/api/benchmark/clip_label/stats?project=%s 的 total 应等于 %d" % (a.project, kept))


if __name__ == "__main__":
    main()
