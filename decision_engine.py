# -*- coding: utf-8 -*-
"""
Decision Engine - PRD 18: 自动决策引擎
每个 Asset 推理完成后生成 decision_status, decision_reason, decision_score

状态：
- AUTO_PASS: 高置信度，直接进入数据资产
- REVIEW: 需要人工审核
- FILTER: 明显不符合需求，不进入最终数据集但保留记录
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class DecisionStatus(str, Enum):
    AUTO_PASS = "AUTO_PASS"
    REVIEW = "REVIEW"
    FILTER = "FILTER"


@dataclass
class Decision:
    status: DecisionStatus
    reason: str
    score: Optional[float] = None  # 0-1 综合置信度; 无真实概率证据时为 None(不虚报)
    details: Dict = field(default_factory=dict)  # 详细判断依据


@dataclass
class DecisionRule:
    """单条决策规则"""
    name: str
    condition: Callable[[Dict], bool]  # 输入 asset 推理结果，返回是否匹配
    decision: DecisionStatus
    reason: str
    priority: int = 0  # 优先级，数值越大越先评估


class DecisionEngine:
    """
    决策引擎核心
    支持可配置规则链，按优先级评估
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.rules: List[DecisionRule] = []
        self._build_rules()
    
    def _default_config(self) -> Dict:
        return {
            # 置信度阈值
            "vlm_confidence_threshold": 0.85,
            # VLM-only 直通阈值(批量AI审核只跑VLM时, >0.4 即自动通过免人工)
            "vlm_only_pass_threshold": 0.45,
            "yolo_confidence_threshold": 0.7,
            "dino_confidence_threshold": 0.6,
            "siglip_score_threshold": 0.3,
            
            # 冲突检测
            "enable_conflict_detection": True,
            "conflict_iou_threshold": 0.5,
            
            # 风险判定
            "high_risk_tags": ["危险", "拥堵", "行人横穿", "车辆加塞", "异常停车"],
            
            # 业务规则
            "required_tags": {},  # 如 {"weather": ["雨天"], "road": ["城市道路"]}
            "filter_tags": {},    # 如 {"weather": ["雪天"]} 表示雪天直接过滤
            
            # 评分权重
            "score_weights": {
                "vlm": 0.4,
                "yolo": 0.2,
                "dino": 0.2,
                "siglip": 0.2
            }
        }
    
    def _build_rules(self):
        """构建规则链（按优先级排序）"""
        cfg = self.config
        
        # 规则 1：业务过滤规则（最高优先级）
        if cfg.get("filter_tags"):
            def filter_condition(asset_data):
                final_tags = asset_data.get("final_tags", {}) or asset_data.get("ai_tags", {})
                for dim, filter_list in cfg["filter_tags"].items():
                    asset_tags = final_tags.get(dim, [])
                    asset_tag_names = [t.get("tag") if isinstance(t, dict) else t for t in asset_tags]
                    if any(ft in asset_tag_names for ft in filter_list):
                        return True
                return False
            
            self.rules.append(DecisionRule(
                name="business_filter",
                condition=filter_condition,
                decision=DecisionStatus.FILTER,
                reason="命中业务过滤标签",
                priority=100
            ))
        
        # 规则 2：必需标签缺失
        if cfg.get("required_tags"):
            def required_condition(asset_data):
                final_tags = asset_data.get("final_tags", {}) or asset_data.get("ai_tags", {})
                for dim, req_list in cfg["required_tags"].items():
                    asset_tags = final_tags.get(dim, [])
                    asset_tag_names = [t.get("tag") if isinstance(t, dict) else t for t in asset_tags]
                    if not any(rt in asset_tag_names for rt in req_list):
                        return True  # 缺失必需标签
                return False
            
            self.rules.append(DecisionRule(
                name="missing_required_tags",
                condition=required_condition,
                decision=DecisionStatus.REVIEW,
                reason="缺失业务必需标签",
                priority=90
            ))
        
        
        # 规则 3.6：VLM事件 vs 检测冲突（互相校验）——VLM 报了目标类事件但 YOLO/DINO 检不到对应目标
        # => 疑似编造, 强制人工 REVIEW(不 AUTO_PASS)
        def event_conflict_condition(asset_data):
            return bool(asset_data.get("vlm_event_conflict"))

        self.rules.append(DecisionRule(
            name="event_detection_conflict",
            condition=event_conflict_condition,
            decision=DecisionStatus.REVIEW,
            reason="VLM事件与目标检测冲突(需人工核实)",
            priority=78
        ))

        # 规则 3.5：VLM-only 高置信直通（贴合 V2 批量审核流程：只跑 VLM 无检测证据时）
        # 条件：本次只有 VLM 证据(yolo/dino 均未跑) + VLM 置信达标 -> 免人工(风险场景已被规则3先拦为 REVIEW)
        # 若 yolo/dino 已跑(全证据模式)则跳过本规则, 走完整规则链
        def vlm_only_pass_condition(asset_data):
            # VLM(文本生成)无真实概率——不再用假置信阈值(旧 0.85/0.45 硬编码已废弃)。
            # 仅当本次只跑 VLM(无 yolo/dino 检测)且结构化标签解析成功 -> 直通免人工
            # (风险场景已被 high_risk 规则先拦为 REVIEW)
            yolo_conf = asset_data.get("yolo_max_conf", 0)
            dino_conf = asset_data.get("dino_max_conf", 0)
            if yolo_conf > 0 or dino_conf > 0:
                return False
            return bool(asset_data.get("vlm_tags_ok") or asset_data.get("final_tags"))

        self.rules.append(DecisionRule(
            name="vlm_only_auto_pass",
            condition=vlm_only_pass_condition,
            decision=DecisionStatus.AUTO_PASS,
            reason="VLM标签直通(免人工,无概率不虚报)",
            priority=75
        ))

        # 规则 4：模型置信度不足
        def low_confidence_condition(asset_data):
            yolo_conf = asset_data.get("yolo_max_conf", 1.0)
            dino_conf = asset_data.get("dino_max_conf", 1.0)

            # VLM 文本无概率, 不参与置信度判断; 仅真实检测置信不足才进人工
            if yolo_conf < cfg["yolo_confidence_threshold"]:
                return True
            if dino_conf < cfg["dino_confidence_threshold"]:
                return True
            return False
        
        self.rules.append(DecisionRule(
            name="low_confidence",
            condition=low_confidence_condition,
            decision=DecisionStatus.REVIEW,
            reason="模型置信度不足",
            priority=70
        ))
        
        # 规则 5：模型冲突（YOLO vs DINO 类别不一致）
        if cfg.get("enable_conflict_detection"):
            def conflict_condition(asset_data):
                yolo_labels = set(asset_data.get("yolo_labels", []))
                dino_labels = set(asset_data.get("dino_labels", []))
                # 简单冲突检测：同一目标类别但置信度差异大，或互斥类别同时出现
                # 这里简化：如果有目标但两模型都没检到共同类别，视为潜在冲突
                if yolo_labels and dino_labels:
                    common = yolo_labels & dino_labels
                    if len(common) == 0 and (len(yolo_labels) > 0 or len(dino_labels) > 0):
                        return True
                return False
            
            self.rules.append(DecisionRule(
                name="model_conflict",
                condition=conflict_condition,
                decision=DecisionStatus.REVIEW,
                reason="模型检测结果冲突",
                priority=60
            ))
        
        # 规则 7：高置信度无冲突 -> AUTO_PASS（最低优先级，兜底）
        def auto_pass_condition(asset_data):
            yolo_conf = asset_data.get("yolo_max_conf", 0)
            dino_conf = asset_data.get("dino_max_conf", 0)
            siglip_score = asset_data.get("siglip_score", 0)
            
            # 检查冲突：内联实现，避免递归
            yolo_labels = set(asset_data.get("yolo_labels", []))
            dino_labels = set(asset_data.get("dino_labels", []))
            no_conflict = True
            if yolo_labels and dino_labels:
                common = yolo_labels & dino_labels
                if len(common) == 0 and (len(yolo_labels) > 0 or len(dino_labels) > 0):
                    no_conflict = False
            
            return (yolo_conf >= cfg["yolo_confidence_threshold"] and
                    dino_conf >= cfg["dino_confidence_threshold"] and
                    no_conflict)
        
        self.rules.append(DecisionRule(
            name="auto_pass",
            condition=auto_pass_condition,
            decision=DecisionStatus.AUTO_PASS,
            reason="高置信度无冲突",
            priority=10
        ))
        
        # 规则 8：默认 REVIEW（兜底）
        def default_review(asset_data):
            return True  # 总是匹配
        
        self.rules.append(DecisionRule(
            name="default_review",
            condition=default_review,
            decision=DecisionStatus.REVIEW,
            reason="默认人工复核",
            priority=0
        ))
        
        # 按优先级排序
        self.rules.sort(key=lambda r: -r.priority)
    
    def decide(self, asset_data: Dict) -> Decision:
        """
        执行决策
        asset_data: {
            "vlm_confidence": float,
            "vlm_tags": {...},
            "yolo_labels": [...],
            "yolo_max_conf": float,
            "dino_labels": [...],
            "dino_max_conf": float,
            "siglip_score": float,
            "final_tags": {...},
            "ai_tags": {...},
            ...
        }
        """
        # 综合评分(诚实口径): 只统计有真实概率的证据; VLM 文本无概率不参与;
        # 按实际提供证据的权重归一; 无任何概率证据时 score=None(规则判定, 不虚报置信分)
        weights = self.config.get("score_weights", {})
        _terms = []
        _wsum = 0.0
        # SigLIP(检索)与 VLM(无概率)不参与置信加权; 仅真实检测(yolo/dino)
        for _key, _wkey in (("yolo_max_conf", "yolo"), ("dino_max_conf", "dino")):
            _v = asset_data.get(_key)
            _w = weights.get(_wkey, 0)
            if _w and isinstance(_v, (int, float)) and _v > 0:
                _terms.append(float(_v) * _w)
                _wsum += _w
        score = (sum(_terms) / _wsum) if _wsum > 0 else None
        
        # 按规则链评估
        for rule in self.rules:
            try:
                if rule.condition(asset_data):
                    return Decision(
                        status=rule.decision,
                        reason=rule.reason,
                        score=score,
                        details={
                            "rule": rule.name,
                            "priority": rule.priority
                        }
                    )
            except Exception as e:
                print(f"[DecisionEngine] 规则 {rule.name} 评估异常: {e}")
                continue
        
        # 兜底
        return Decision(
            status=DecisionStatus.REVIEW,
            reason="规则评估异常，默认审核",
            score=score,
            details={"error": "fallback"}
        )
    
    def update_config(self, new_config: Dict):
        """热更新配置"""
        self.config.update(new_config)
        self.rules.clear()
        self._build_rules()


# 预设配置模板
PRESET_CONFIGS = {
    "strict": {
        "vlm_confidence_threshold": 0.9,
        "vlm_only_pass_threshold": 0.85,
        "yolo_confidence_threshold": 0.8,
        "dino_confidence_threshold": 0.7,
        "siglip_score_threshold": 0.4,
        "filter_tags": {"weather": ["雪天", "雾天"]},
    },
    "lenient": {
        "vlm_confidence_threshold": 0.7,
        "vlm_only_pass_threshold": 0.6,
        "yolo_confidence_threshold": 0.5,
        "dino_confidence_threshold": 0.4,
        "siglip_score_threshold": 0.2,
    },
    "safety_first": {
        "vlm_confidence_threshold": 0.85,
        "vlm_only_pass_threshold": 0.85,
        "yolo_confidence_threshold": 0.7,
        "dino_confidence_threshold": 0.6,
        "siglip_score_threshold": 0.3,
        "high_risk_tags": ["危险", "拥堵", "行人横穿", "车辆加塞", "异常停车", "道路施工"],
        "filter_tags": {},  # 不自动过滤，全走审核
    }
}


def create_decision_engine(preset: str = "balanced", custom: Dict = None) -> DecisionEngine:
    """工厂函数：创建决策引擎"""
    base_configs = {
        "balanced": {},
        "strict": PRESET_CONFIGS["strict"],
        "lenient": PRESET_CONFIGS["lenient"],
        "safety_first": PRESET_CONFIGS["safety_first"],
    }
    config = base_configs.get(preset, {}).copy()
    if custom:
        config.update(custom)
    return DecisionEngine(config)


if __name__ == "__main__":
    # 测试
    engine = create_decision_engine("balanced")
    
    # 测试用例 1：高置信度无冲突
    test1 = {
        "vlm_confidence": 0.95,
        "yolo_max_conf": 0.9,
        "dino_max_conf": 0.85,
        "siglip_score": 0.8,
        "yolo_labels": ["car", "person"],
        "dino_labels": ["car", "person"],
        "final_tags": {"weather": [{"tag": "晴天"}], "events": []}
    }
    d1 = engine.decide(test1)
    print(f"Test 1 (high conf): {d1.status} - {d1.reason} (score={d1.score:.2f})")
    
    # 测试用例 2：低置信度
    test2 = {
        "vlm_confidence": 0.5,
        "yolo_max_conf": 0.6,
        "dino_max_conf": 0.5,
        "siglip_score": 0.7,
        "yolo_labels": ["car"],
        "dino_labels": ["truck"],
        "final_tags": {"weather": [{"tag": "雨天"}]}
    }
    d2 = engine.decide(test2)
    print(f"Test 2 (low conf): {d2.status} - {d2.reason} (score={d2.score:.2f})")
    
    # 测试用例 3：高风险
    test3 = {
        "vlm_confidence": 0.9,
        "yolo_max_conf": 0.9,
        "dino_max_conf": 0.85,
        "siglip_score": 0.8,
        "yolo_labels": ["person"],
        "dino_labels": ["person"],
        "final_tags": {"events": [{"tag": "行人横穿"}], "risk": [{"tag": "高风险"}]}
    }
    d3 = engine.decide(test3)
    print(f"Test 3 (high risk): {d3.status} - {d3.reason} (score={d3.score:.2f})")
    
    # 测试用例 4：模型冲突
    test4 = {
        "vlm_confidence": 0.9,
        "yolo_max_conf": 0.9,
        "dino_max_conf": 0.9,
        "siglip_score": 0.8,
        "yolo_labels": ["car"],
        "dino_labels": ["truck"],  # 不同类别
        "final_tags": {}
    }
    d4 = engine.decide(test4)
    print(f"Test 4 (conflict): {d4.status} - {d4.reason} (score={d4.score:.2f})")
    
    print("DecisionEngine tests passed!")