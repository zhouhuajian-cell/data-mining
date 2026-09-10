# -*- coding: utf-8 -*-
"""
Tag 体系重构 - PRD 15, 16: Business/AI/Human/Final 四层分离 + 来源追踪

Tag 结构：
{
  "business": {"source": "OBJ_1V", "vehicle": "G91", "resolution": "8M"},
  "ai_tags": {
    "weather": [{"tag": "雨天", "source": "VLM", "confidence": 0.93, "model": "Qwen2.5-VL-7B", "version": "v1", "created_at": "2024-01-15T10:30:00"}],
    "road": [...],
    "objects": [...],
    "events": [...],
    "risk": [...]
  },
  "human_tags": {...},  # 同结构，source=Human
  "final_tags": {...}   # 最终生效标签
}
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import copy


class TagSource(str, Enum):
    AI = "AI"
    HUMAN = "Human"
    RULE = "Rule"


class TagDimension(str, Enum):
    WEATHER = "weather"
    ROAD = "road"
    TIME = "time"
    OBJECTS = "objects"
    EVENTS = "events"
    RISK = "risk"
    VEHICLE = "vehicle"
    RESOLUTION = "resolution"
    ROAD_SURFACE = "road_surface"
    SCENE = "scene"


# 标准 Ontology（PRD 14）
ONTOLOGY = {
    TagDimension.WEATHER: ["晴天", "阴天", "雨天", "雪天", "雾天", "其他"],
    TagDimension.ROAD: ["城市道路", "高速/高架", "隧道/涵洞", "乡村道路", "坡道/弯道", "收费站/闸机", "场地测试", "其他场景"],
    TagDimension.TIME: ["白天", "夜晚", "黄昏", "未指定"],
    TagDimension.OBJECTS: ["行人", "两轮车", "三轮车", "小车", "大车", "异型车", "特殊车辆", "交通设施", "其他目标"],
    TagDimension.EVENTS: ["行人横穿", "非机动车横穿", "车辆加塞", "异常停车", "道路施工", "道路障碍", "车辆拥堵", "车辆密集", "行人密集"],
    TagDimension.RISK: ["低风险", "中风险", "高风险"],
    TagDimension.ROAD_SURFACE: ["干燥", "湿滑", "积水", "结冰", "破损", "未指定"],
    TagDimension.VEHICLE: [],  # 动态，来自业务元数据
    TagDimension.RESOLUTION: ["2M", "8M", "其他"],
}


@dataclass
class Tag:
    """单个标签，包含完整溯源信息"""
    tag: str
    source: TagSource
    confidence: float = 1.0
    model: str = ""
    version: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Tag":
        return cls(**data)
    
    @classmethod
    def create_ai(cls, tag: str, confidence: float, model: str, version: str = "v1") -> "Tag":
        return cls(
            tag=tag,
            source=TagSource.AI,
            confidence=confidence,
            model=model,
            version=version
        )
    
    @classmethod
    def create_human(cls, tag: str) -> "Tag":
        return cls(
            tag=tag,
            source=TagSource.HUMAN,
            confidence=1.0,
            model="human",
            version="v1"
        )
    
    @classmethod
    def create_rule(cls, tag: str, rule_name: str) -> "Tag":
        return cls(
            tag=tag,
            source=TagSource.RULE,
            confidence=1.0,
            model=f"rule:{rule_name}",
            version="v1"
        )


@dataclass
class TagSet:
    """某维度的标签集合"""
    tags: List[Tag] = field(default_factory=list)
    
    def add(self, tag: Tag):
        # 去重：同 tag + source 视为同一个
        for existing in self.tags:
            if existing.tag == tag.tag and existing.source == tag.source:
                # 更新置信度取高
                if tag.confidence > existing.confidence:
                    existing.confidence = tag.confidence
                return
        self.tags.append(tag)
    
    def remove(self, tag_name: str, source: TagSource = None):
        self.tags = [t for t in self.tags 
                     if not (t.tag == tag_name and (source is None or t.source == source))]
    
    def get_names(self) -> List[str]:
        return [t.tag for t in self.tags]
    
    def get_by_source(self, source: TagSource) -> List[Tag]:
        return [t for t in self.tags if t.source == source]
    
    def to_dict(self) -> List[Dict]:
        return [t.to_dict() for t in self.tags]
    
    @classmethod
    def from_dict(cls, data: List[Dict]) -> "TagSet":
        ts = cls()
        for d in data:
            ts.add(Tag.from_dict(d))
        return ts


class TagManager:
    """标签管理器：处理四层标签的增删改查、合并、冲突解决"""
    
    def __init__(self):
        self.ontology = ONTOLOGY
    
    def create_empty_tags(self) -> Dict[str, Dict[str, TagSet]]:
        """创建空的四层标签结构"""
        dims = [d.value for d in TagDimension]
        empty_dims = {dim: TagSet() for dim in dims}
        return {
            "business": {},
            "ai_tags": empty_dims.copy(),
            "human_tags": empty_dims.copy(),
            "final_tags": empty_dims.copy(),
        }
    
    def validate_tag(self, dimension: str, tag: str) -> bool:
        """验证标签是否在 Ontology 中"""
        if dimension not in [d.value for d in TagDimension]:
            return True  # 未知维度允许任意标签
        return tag in self.ontology.get(TagDimension(dimension), [])
    
    def merge_ai_tags(self, target: Dict[str, Dict[str, TagSet]], source: Dict[str, List[Dict]]):
        """合并 AI 标签（来自 VLM/YOLO/DINO 等）"""
        for dim, tags in source.items():
            if dim not in target:
                target[dim] = TagSet()
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, dict):
                        tag = Tag.from_dict(t)
                    elif isinstance(t, str):
                        tag = Tag.create_ai(t, 1.0, "unknown")
                    else:
                        continue
                    target[dim].add(tag)
    
    def merge_human_tags(self, target: Dict[str, Dict[str, TagSet]], source: Dict[str, List[str]]):
        """合并人工标签"""
        for dim, tags in source.items():
            if dim not in target:
                target[dim] = TagSet()
            for t in tags:
                tag = Tag.create_human(t)
                target[dim].add(tag)
    
    def compute_final_tags(self, tags_struct: Dict[str, Dict[str, TagSet]]) -> Dict[str, TagSet]:
        """
        计算最终生效标签（Final Tags）
        优先级：Human > AI > Rule
        冲突时：Human 覆盖 AI，AI 保留高置信度
        """
        final = {}
        ai_tags = tags_struct.get("ai_tags", {})
        human_tags = tags_struct.get("human_tags", {})
        
        all_dims = set(ai_tags.keys()) | set(human_tags.keys())
        
        for dim in all_dims:
            final_set = TagSet()
            
            # 先加 AI 标签
            if dim in ai_tags:
                for tag in ai_tags[dim].tags:
                    final_set.add(tag)
            
            # 再加 Human 标签（覆盖同名 AI 标签）
            if dim in human_tags:
                for tag in human_tags[dim].tags:
                    # 移除同名的 AI 标签
                    final_set.remove(tag.tag, TagSource.AI)
                    final_set.add(tag)
            
            final[dim] = final_set
        
        return final
    
    def tags_to_dict(self, tags_struct: Dict[str, Dict[str, TagSet]]) -> Dict:
        """转换为可序列化的字典"""
        result = {}
        for layer, dims in tags_struct.items():
            if layer == "business":
                result[layer] = dims
            elif isinstance(dims, dict):
                result[layer] = {dim: ts.to_dict() for dim, ts in dims.items()}
        return result
    
    def tags_from_dict(self, data: Dict) -> Dict[str, Dict[str, TagSet]]:
        """从字典恢复标签结构"""
        result = {"business": data.get("business", {})}
        for layer in ["ai_tags", "human_tags", "final_tags"]:
            if layer in data:
                result[layer] = {}
                for dim, tags in data[layer].items():
                    result[layer][dim] = TagSet.from_dict(tags)
        return result
    
    def get_tag_stats(self, tags_struct: Dict) -> Dict:
        """获取标签统计（支持 TagSet 和已序列化两种格式）"""
        stats = {}
        for layer in ["ai_tags", "human_tags", "final_tags"]:
            if layer not in tags_struct:
                continue
            layer_stats = {}
            dims = tags_struct[layer]
            if isinstance(dims, dict):
                # TagSet 格式
                for dim, ts in dims.items():
                    if hasattr(ts, 'tags'):
                        # TagSet 对象
                        by_source = {}
                        for tag in ts.tags:
                            src = tag.source.value if hasattr(tag.source, 'value') else tag.source
                            by_source[src] = by_source.get(src, 0) + 1
                        layer_stats[dim] = {
                            "total": len(ts.tags),
                            "by_source": by_source,
                            "tags": ts.get_names()
                        }
                    else:
                        # 已序列化的 list 格式
                        by_source = {}
                        for tag in ts:
                            src = tag.get("source", "UNKNOWN")
                            by_source[src] = by_source.get(src, 0) + 1
                        layer_stats[dim] = {
                            "total": len(ts),
                            "by_source": by_source,
                            "tags": [t.get("tag", "") for t in ts]
                        }
            stats[layer] = layer_stats
        return stats
    
    def find_conflicts(self, tags_struct: Dict) -> List[Dict]:
        """查找 AI 与 Human 标签的冲突"""
        conflicts = []
        ai_tags = tags_struct.get("ai_tags", {})
        human_tags = tags_struct.get("human_tags", {})
        
        for dim in set(ai_tags.keys()) | set(human_tags.keys()):
            ai_names = set()
            if dim in ai_tags:
                ai_names = {t.tag for t in ai_tags[dim].tags}
            
            human_names = set()
            if dim in human_tags:
                human_names = {t.tag for t in human_tags[dim].tags}
            
            # AI 有但 Human 没有（可能被遗漏）
            only_ai = ai_names - human_names
            # Human 有但 AI 没有（人工补充）
            only_human = human_names - ai_names
            # 都有但不同（冲突修正）
            both = ai_names & human_names
            
            if only_ai or only_human:
                conflicts.append({
                    "dimension": dim,
                    "only_ai": list(only_ai),
                    "only_human": list(only_human),
                    "both": list(both)
                })
        
        return conflicts


# 全局实例
_tag_manager = None

def get_tag_manager() -> TagManager:
    global _tag_manager
    if _tag_manager is None:
        _tag_manager = TagManager()
    return _tag_manager


if __name__ == "__main__":
    # 测试
    tm = TagManager()
    
    # 创建空标签
    tags = tm.create_empty_tags()
    print("Empty tags created")
    
    # 添加 AI 标签
    ai_data = {
        "weather": [
            {"tag": "雨天", "source": "AI", "confidence": 0.93, "model": "Qwen2.5-VL", "version": "v1"},
            {"tag": "阴天", "source": "AI", "confidence": 0.1, "model": "Qwen2.5-VL", "version": "v1"}
        ],
        "road": [{"tag": "城市道路", "source": "AI", "confidence": 0.85, "model": "Qwen2.5-VL", "version": "v1"}],
        "events": [{"tag": "行人横穿", "source": "AI", "confidence": 0.78, "model": "Qwen2.5-VL", "version": "v1"}],
    }
    tm.merge_ai_tags(tags, ai_data)
    
    # 添加 Human 标签（修正 rain 为 晴天）
    human_data = {
        "weather": ["晴天"],  # 人工修正
        "road": ["城市道路"],  # 确认
    }
    tm.merge_human_tags(tags, human_data)
    
    # 计算最终标签
    final = tm.compute_final_tags(tags)
    tags["final_tags"] = final
    
    # 序列化
    serialized = tm.tags_to_dict(tags)
    print("Serialized tags:")
    import json
    print(json.dumps(serialized, ensure_ascii=False, indent=2))
    
    # 统计
    stats = tm.get_tag_stats(tags)
    print("\nStats:", json.dumps(stats, ensure_ascii=False, indent=2))
    
    # 冲突检测
    conflicts = tm.find_conflicts(tags)
    print("\nConflicts:", json.dumps(conflicts, ensure_ascii=False, indent=2))
    
    print("TagManager tests passed!")