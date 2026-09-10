# -*- coding: utf-8 -*-
"""
Benchmark 评测引擎 - PRD 24-25
- Benchmark 数据集：结构化 GT（时间/天气/道路/目标/事件/风险）
- 评测模型组合：VLM / YOLO / DINO / Fusion（各组合）
- 指标：每维度 Precision / Recall / F1 + Macro 平均
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# 维度定义（与 Ontology 对齐）
TAG_DIMS = ["time", "weather", "road", "objects", "events", "risk"]

# YOLO COCO 80 类 -> 平台本体 objects 映射（自动驾驶相关子集）
_YOLO_TO_ONTOLOGY = {
    "person": "行人", "bicycle": "两轮车", "motorbike": "两轮车",
    "car": "小车", "truck": "大车", "bus": "大车", "train": "大车",
    "traffic light": "交通设施", "stop sign": "交通设施",
    "traffic_sign": "交通设施", "fire hydrant": "交通设施",
    "bench": "交通设施", "parking meter": "交通设施",
}
_YOLO_IGNORE = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                "donut", "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster",
                "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush", "airplane", "boat"}


def _norm_tag_names(tags_obj: Optional[Dict], dim: str) -> set:
    """从 {dim: [{tag,confidence,...}|str]} 提取标准标签名集合"""
    out = set()
    if not tags_obj:
        return out
    vals = tags_obj.get(dim) or []
    for v in vals:
        if isinstance(v, dict):
            t = v.get("tag")
        else:
            t = v
        if isinstance(t, str) and t.strip():
            out.add(t.strip())
    return out


def yolo_labels_to_objects(labels: List[str]) -> List[str]:
    """YOLO COCO 标签 -> 平台本体 objects 标签"""
    out = []
    for lbl in labels:
        key = (lbl or "").strip().lower()
        if key in _YOLO_TO_ONTOLOGY:
            mapped = _YOLO_TO_ONTOLOGY[key]
            if mapped not in out:
                out.append(mapped)
    return out


def _tag_names_from_detections(dets: List[Dict]) -> List[str]:
    """检测记录 [{label:..}, ...] -> 标签名列表"""
    out = []
    for d in dets or []:
        lbl = (d.get("label") or "").strip()
        if lbl and lbl not in out:
            out.append(lbl)
    return out


@dataclass
class DimMetrics:
    dimension: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def compute(self):
        self.precision = (self.tp / (self.tp + self.fp)) if (self.tp + self.fp) else 0.0
        self.recall = (self.tp / (self.tp + self.fn)) if (self.tp + self.fn) else 0.0
        self.f1 = (2 * self.precision * self.recall / (self.precision + self.recall)) if (self.precision + self.recall) else 0.0
        return self

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def evaluate_asset(gt_tags: Dict, pred_tags: Dict, dims: List[str] = None) -> dict:
    """
    单样本评测：gt/pred 均为 {dim: [tag...] 或 [tag-dict...]} 结构化标签。
    返回 {per_dim: [...], macro: {precision, recall, f1}, sample_metrics}
    """
    dims = dims or TAG_DIMS
    per_dim = []
    total_tp = total_fp = total_fn = 0
    for dim in dims:
        gt = _norm_tag_names(gt_tags, dim)
        pred = _norm_tag_names(pred_tags, dim)
        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        m = DimMetrics(dim, tp, fp, fn).compute()
        per_dim.append(m.to_dict())
        total_tp += tp
        total_fp += fp
        total_fn += fn
    mp = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    mr = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
    mf = (2 * mp * mr / (mp + mr)) if (mp + mr) else 0.0
    return {
        "per_dim": per_dim,
        "macro": {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf, 4)},
        "hits": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }


def _merge_preds(dicts: List[Optional[Dict]]) -> Dict[str, List[str]]:
    """多来源标签 union 合并：{dim: [names]}（用于 fusion 组合）"""
    merged: Dict[str, set] = {}
    for d in dicts:
        if not d:
            continue
        for dim, names in d.items():
            merged.setdefault(dim, set()).update(names)
    return {k: sorted(v) for k, v in merged.items()}


def extract_prediction_sources(asset) -> Dict[str, Optional[Dict]]:
    """
    从 DB Asset 提取各模型来源的结构化标签（离线可用；有真实模型时由上层在线推理替换）：
      vlm    -> asset.ai_tags（VLM 全维度）
      yolo   -> detections.yolo 的 objects（COCO 映射）
      dino   -> detections.dino 的 objects（标签直用）
      final  -> asset.final_tags（人工/Fusion 确认结果）
      fusion -> ai_tags ∪ yolo/dino objects
    返回 {source: {dim: [names]} or None}
    """
    result: Dict[str, Optional[Dict]] = {"vlm": None, "yolo": None, "dino": None, "final": None, "fusion": None}

    ai_tags = asset.ai_tags or {}
    result["vlm"] = {d: sorted(_norm_tag_names(ai_tags, d)) for d in TAG_DIMS if _norm_tag_names(ai_tags, d)} or None

    dets = asset.detections or {}
    yolo_dets = [d for d in (dets.get("yolo") or []) if isinstance(d, dict)]
    dino_dets = [d for d in (dets.get("dino") or []) if isinstance(d, dict)]
    if yolo_dets:
        yolo_objects = yolo_labels_to_objects(_tag_names_from_detections(yolo_dets))
        if yolo_objects:
            result["yolo"] = {"objects": sorted(set(yolo_objects))}
    if dino_dets:
        dino_objects = _tag_names_from_detections(dino_dets)
        if dino_objects:
            result["dino"] = {"objects": sorted(set(dino_objects))}

    final_tags = asset.final_tags or {}
    if final_tags:
        result["final"] = {d: sorted(_norm_tag_names(final_tags, d)) for d in TAG_DIMS if _norm_tag_names(final_tags, d)} or None

    # fusion: ai_tags 各维 ∪ yolo/dino objects
    fusion = {}
    for d in TAG_DIMS:
        names = _norm_tag_names(ai_tags, d)
        if names:
            fusion[d] = sorted(names)
    objs = set()
    if result["yolo"]:
        objs |= set(result["yolo"]["objects"])
    if result["dino"]:
        objs |= set(result["dino"]["objects"])
    if objs:
        fusion["objects"] = sorted(set(fusion.get("objects", [])) | objs)
    if fusion:
        result["fusion"] = fusion

    return result


def evaluate_asset_all_sources(asset, gt_tags: Dict) -> Dict[str, dict]:
    """对单个 asset 跑全部可用来源的评测"""
    sources = extract_prediction_sources(asset)
    out = {}
    gt_norm = {d: sorted(_norm_tag_names(gt_tags, d)) for d in TAG_DIMS if _norm_tag_names(gt_tags, d)}
    for src, preds in sources.items():
        if not preds:
            continue
        out[src] = evaluate_asset(gt_norm, preds)
    return out


def aggregate_evaluations(per_asset: List[Tuple[str, Dict[str, dict]]]) -> Dict[str, dict]:
    """
    聚合多样本评测：
    per_asset: [(asset_id, {source: metrics_dict}), ...]
    返回 {source: {avg_macro, avg_f1, n, per_dim_avg}}
    """
    from collections import defaultdict
    by_src: Dict[str, List[dict]] = defaultdict(list)
    for _aid, src_map in per_asset:
        for src, met in src_map.items():
            by_src[src].append(met)

    result = {}
    for src, mets in by_src.items():
        n = len(mets)
        mp = sum(m["macro"]["precision"] for m in mets) / n
        mr = sum(m["macro"]["recall"] for m in mets) / n
        mf = sum(m["macro"]["f1"] for m in mets) / n
        # per-dim 平均
        dim_agg = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
        for m in mets:
            for dm in m["per_dim"]:
                a = dim_agg[dm["dimension"]]
                a["precision"].append(dm["precision"])
                a["recall"].append(dm["recall"])
                a["f1"].append(dm["f1"])
        per_dim = {}
        for d, arr in dim_agg.items():
            per_dim[d] = {
                "precision": round(sum(arr["precision"]) / len(arr["precision"]), 4),
                "recall": round(sum(arr["recall"]) / len(arr["recall"]), 4),
                "f1": round(sum(arr["f1"]) / len(arr["f1"]), 4),
            }
        result[src] = {
            "samples": n,
            "macro": {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf, 4)},
            "per_dim": per_dim,
        }
    return result


if __name__ == "__main__":
    # 单元测试
    gt = {"weather": ["雨天"], "road": ["城市道路"], "time": ["夜晚"],
          "events": ["行人横穿"], "objects": ["行人", "小车"], "risk": ["中风险"]}
    pred = {"weather": ["雨天", "阴天"], "road": ["城市道路"], "time": ["夜晚"],
            "events": [], "objects": ["行人"], "risk": ["中风险"]}
    r = evaluate_asset(gt, pred)
    print("per_dim:", r["per_dim"])
    print("macro:", r["macro"])
    # TP: 雨天,城市道路,夜晚,行人,中风险 =5; FP: 阴天=1; FN: 小车,行人横穿=2
    assert r["hits"] == {"tp": 5, "fp": 1, "fn": 2}, r["hits"]
    assert abs(r["macro"]["precision"] - 5 / 6) < 1e-3, r["macro"]
    assert abs(r["macro"]["recall"] - 5 / 7) < 1e-3, r["macro"]

    # yolo 映射测试
    assert yolo_labels_to_objects(["car", "truck", "person", "cat"]) == ["小车", "大车", "行人"]
    print("unit tests OK")
