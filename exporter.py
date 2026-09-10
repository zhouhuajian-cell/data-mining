# -*- coding: utf-8 -*-
"""
数据资产导出器 - PRD 48-50, 63
JSONL（一行一 Asset） / CSV / Parquet / ZIP（原图+元数据）+ manifest.json
Asset 序列化遵循 PRD 48 字段结构。
"""

import os
import csv
import json
import shutil
import zipfile
import time
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

TAG_DIMS = ["time", "weather", "road", "objects", "events", "risk"]


def _tags_to_names(tags_obj: Optional[Dict], dim: str) -> List[str]:
    """{dim: [{tag,...}|str]} -> 标签名数组（前端/JSONL 友好）"""
    vals = (tags_obj or {}).get(dim) or []
    out = []
    for v in vals:
        n = v.get("tag") if isinstance(v, dict) else v
        if isinstance(n, str) and n.strip() and n not in out:
            out.append(n.strip())
    return out


def serialize_asset(asset, source_chain: Optional[List[str]] = None,
                    source_root: Optional[str] = None) -> dict:
    """
    把 DB Asset 序列化为 PRD 48 的 JSONL 行结构。
    """
    src = asset.source
    ai = {}
    for d in TAG_DIMS:
        names = _tags_to_names(asset.ai_tags, d)
        if names:
            ai[d] = names
    final = {}
    for d in TAG_DIMS:
        names = _tags_to_names(asset.final_tags, d)
        if names:
            final[d] = names
    if not final:
        # 未人工确认则 final 结构仍输出空对象便于分析
        final = {d: [] for d in TAG_DIMS}

    business = dict(asset.asset_metadata or {})
    if not business.get("resolution") and asset.width:
        # 简单按宽度归 2M/8M（3200w 以上算 8M；工程近似）
        px = asset.width * asset.height
        business.setdefault("resolution", "8M" if px >= 8_000_000 else ("2M" if px >= 1_800_000 else "其他"))

    row = {
        "asset_id": asset.asset_id,
        "source_id": src.source_id if src else None,
        "image_path": asset.image_path,
        "source": {
            "relative_path": src.relative_path if src else (asset.image_path or ""),
            "directory_chain": source_chain if source_chain is not None else ((src.directory_chain or []) if src else []),
            "source_root": source_root if source_root is not None else (src.source_root if src else ""),
        },
        "frame": {
            "frame_index": asset.frame_index or 0,
            "timestamp": asset.timestamp or 0.0,
        },
        "business": business,
        "ai_tags": ai,
        "human_tags": {d: _tags_to_names(asset.human_tags, d) for d in TAG_DIMS if _tags_to_names(asset.human_tags, d)},
        "final_tags": final,
        "final": {
            "status": asset.status.value if asset.status else "SOURCE",
            "decision_status": asset.decision_status.value if asset.decision_status else None,
            "decision_score": asset.decision_score,
        },
        "detections": {
            "yolo_count": len((asset.detections or {}).get("yolo") or []),
            "dino_count": len((asset.detections or {}).get("dino") or []),
        },
    }
    return row


def _flatten_for_csv(row: dict) -> dict:
    """JSONL 行 -> CSV 平铺行"""
    flat = {
        "asset_id": row["asset_id"],
        "source_id": row.get("source_id") or "",
        "image_path": row.get("image_path") or "",
        "relative_path": row["source"].get("relative_path") or "",
        "directory_chain": "/".join(row["source"].get("directory_chain") or []),
        "frame_index": row["frame"].get("frame_index"),
        "timestamp": row["frame"].get("timestamp"),
        "vehicle": (row.get("business") or {}).get("vehicle", ""),
        "resolution": (row.get("business") or {}).get("resolution", ""),
        "status": row["final"].get("status", ""),
        "decision_status": row["final"].get("decision_status", "") or "",
    }
    for d in TAG_DIMS:
        flat[f"ai_{d}"] = "|".join(row.get("ai_tags", {}).get(d, []))
        flat[f"final_{d}"] = "|".join(row.get("final_tags", {}).get(d, []))
    return flat


# ==================== 各格式导出 ====================

def export_jsonl(rows: List[dict], out_path: str) -> int:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def export_csv(rows: List[dict], out_path: str) -> int:
    if not rows:
        open(out_path, "w", encoding="utf-8-sig").close()
        return 0
    flat = [_flatten_for_csv(r) for r in rows]
    keys = list(flat[0].keys())
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    return len(rows)


def export_parquet(rows: List[dict], out_path: str) -> int:
    """需要 pandas + pyarrow；缺失则抛出明确提示"""
    import pandas as pd  # noqa
    flat = [_flatten_for_csv(r) for r in rows]
    df = pd.DataFrame(flat)
    df.to_parquet(out_path, index=False)
    return len(rows)


def export_zip(assets, rows: List[dict], out_path: str,
               include_images: bool = True, with_manifest: bool = True) -> int:
    """ZIP 包：metadata.jsonl + manifest.json + (可选)原图 copies/<asset_id>_<basename>"""
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="export_zip_")
    try:
        meta_path = os.path.join(tmpdir, "metadata.jsonl")
        export_jsonl(rows, meta_path)
        file_list = ["metadata.jsonl"]
        if include_images:
            img_dir = os.path.join(tmpdir, "images")
            os.makedirs(img_dir, exist_ok=True)
            copied = 0
            for a in assets:
                p = a.image_path
                if p and os.path.exists(p):
                    fn = f"{a.asset_id[:8]}_{os.path.basename(p)}"
                    try:
                        shutil.copy2(p, os.path.join(img_dir, fn))
                        file_list.append("images/" + fn)
                        copied += 1
                    except Exception:
                        pass
            print(f"[export] zip 含 {copied} 张原图")
        if with_manifest:
            manifest = build_manifest(rows, include_images=include_images)
            with open(os.path.join(tmpdir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            file_list.append("manifest.json")
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in file_list:
                zf.write(os.path.join(tmpdir, rel), rel)
        return len(rows)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_manifest(rows: List[dict], include_images: bool = False) -> dict:
    """PRD 50：每批任务生成 manifest.json"""
    status_counter = {}
    dim_weather = {}
    for r in rows:
        st = r["final"].get("status", "UNKNOWN")
        status_counter[st] = status_counter.get(st, 0) + 1
        for w in r.get("final_tags", {}).get("weather", []):
            dim_weather[w] = dim_weather.get(w, 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_assets": len(rows),
        "status_distribution": status_counter,
        "weather_distribution": dim_weather,
        "include_images": include_images,
        "format_note": "jsonl: 一行一 Asset（PRD 48）",
    }


def export_project(assets, project_name: str, fmt: str, out_dir: str,
                   include_images: bool = True) -> Dict:
    """
    统一入口：assets = DB Asset 查询结果；fmt in jsonl/csv/parquet/zip
    返回 {path, format, count, manifest}
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rows = [serialize_asset(a) for a in assets]
    fmt = (fmt or "jsonl").lower()

    if fmt == "jsonl":
        path = os.path.join(out_dir, f"export_{ts}.jsonl")
        export_jsonl(rows, path)
    elif fmt == "json":
        # JSON 数组格式: 标准单文件, 可直接用 JSON 查看器/编辑器打开
        path = os.path.join(out_dir, f"export_{ts}.json")
        with open(path, "w", encoding="utf-8") as _jf:
            import json as _json
            _json.dump(rows, _jf, ensure_ascii=False, indent=2)
    elif fmt == "csv":
        path = os.path.join(out_dir, f"export_{ts}.csv")
        export_csv(rows, path)
    elif fmt == "parquet":
        path = os.path.join(out_dir, f"export_{ts}.parquet")
        export_parquet(rows, path)
    elif fmt == "zip":
        path = os.path.join(out_dir, f"export_{ts}.zip")
        export_zip(assets, rows, path, include_images=include_images)
    else:
        raise ValueError(f"未知导出格式: {fmt}")

    manifest = build_manifest(rows, include_images=include_images)
    return {"path": path, "format": fmt, "count": len(rows), "manifest": manifest}


if __name__ == "__main__":
    # 用 sync_test 风格假对象做冒烟：直接构造 dict 行
    rows = [
        {
            "asset_id": "AST1", "source_id": "SRC1", "image_path": "/x/a.jpg",
            "source": {"relative_path": "CityA/a.jpg", "directory_chain": ["CityA"], "source_root": "/nas"},
            "frame": {"frame_index": 12, "timestamp": 1.2},
            "business": {"vehicle": "G91", "resolution": "8M"},
            "ai_tags": {"weather": ["雨天"]},
            "human_tags": {}, "final_tags": {"weather": ["雨天"]},
            "final": {"status": "APPROVED"},
            "detections": {"yolo_count": 1, "dino_count": 0},
        },
        {
            "asset_id": "AST2", "source_id": "SRC1", "image_path": "/x/b.jpg",
            "source": {"relative_path": "CityA/b.jpg", "directory_chain": ["CityA"], "source_root": "/nas"},
            "frame": {"frame_index": 30, "timestamp": 3.0},
            "business": {"vehicle": "G91", "resolution": "8M"},
            "ai_tags": {"weather": ["晴天"], "road": ["高速/高架"]},
            "human_tags": {}, "final_tags": {"weather": ["晴天"]},
            "final": {"status": "APPROVED"},
            "detections": {"yolo_count": 2, "dino_count": 0},
        },
    ]
    import tempfile
    td = tempfile.mkdtemp(prefix="exp_")
    export_jsonl(rows, os.path.join(td, "t.jsonl"))
    export_csv(rows, os.path.join(td, "t.csv"))
    try:
        export_parquet(rows, os.path.join(td, "t.parquet"))
        print("parquet OK")
    except Exception as e:
        print("parquet skip:", e)
    with open(os.path.join(td, "t.jsonl"), encoding="utf-8") as f:
        n = sum(1 for _ in f)
    assert n == 2, n
    print("exporter smoke OK ->", td)
