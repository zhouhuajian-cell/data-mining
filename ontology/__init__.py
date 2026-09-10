# -*- coding: utf-8 -*-
"""统一场景本体读取层（单一真源：scene.json）。

对外接口：
  load_ontology()      原始 dict
  dimensions()         统一维度表 {dim: {"cn":.., "values":[..], "aliases":[..]}}
  dim_cn(dim)          维度中文名
  alias_map()          旧维度名 -> 统一维度名（如 road_type -> road）
  values_of(dim)       某维度的允许取值（传旧名会自动映射）
  valid_tags()         所有合法标签集合（用于输出校验，不在集合里的直接丢弃）
  objects_coarse()     细类 -> 粗类（车辆/非机动车/行人…）用于跨链路聚合
  events_need()        事件 -> 需要检出的目标类（事件与检测互相校验用）
  scene_categories()   场景关注分类（原 scene_categories.json 已并入本体）
  scene_enum_str()     提示词用的枚举文本
"""
import os, json

_DIR = os.path.dirname(os.path.abspath(__file__))
# 旧部署里 scene_categories.json 与本目录同级；本体没写时回退读它
_SCENE_CATEGORIES_LEGACY = os.path.join(os.path.dirname(_DIR), "scene_categories.json")


def load_ontology():
    with open(os.path.join(_DIR, "scene.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def dimensions():
    """统一维度表；文件里没有 dimensions 时用 v1 的键兜底"""
    o = load_ontology()
    dims = o.get("dimensions")
    if isinstance(dims, dict) and dims:
        return dims
    out = {}
    for dim, vals in (o.get("scene") or {}).items():
        out[dim] = {"cn": dim, "values": vals}
    if o.get("ego_state"):
        out["ego_state"] = {"cn": "自车状态", "values": o["ego_state"]}
    if o.get("events"):
        out["events"] = {"cn": "事件", "values": o["events"]}
    return out


def dim_cn(dim):
    d = dimensions().get(dim)
    return (d or {}).get("cn") or dim if isinstance(d, dict) else dim


def alias_map():
    """旧维度名 -> 统一维度名（历史数据里 road_type/surface/area 等）"""
    m = {}
    for dim, d in dimensions().items():
        for a in (d or {}).get("aliases") or []:
            m[a] = dim
    return m


def values_of(dim):
    """某维度的允许取值；传旧名(如 road_type)会自动映射到统一名"""
    dims = dimensions()
    if dim not in dims:
        dim = alias_map().get(dim, dim)
    return list((dims.get(dim) or {}).get("values") or [])


def valid_tags():
    """所有合法标签集合（用于输出校验，不在集合里的直接丢弃）"""
    o = load_ontology()
    s = set()
    for d in dimensions().values():
        s.update((d or {}).get("values") or [])
    for vals in (o.get("scene") or {}).values():
        s.update(vals)
    s.update(o.get("ego_state") or [])
    osd = o.get("object_state") or {}
    s.update(osd.get("types") or [])
    s.update(osd.get("states") or [])
    s.update(o.get("events") or [])
    return s


def objects_coarse():
    return dict(load_ontology().get("objects_coarse") or {})


def events_need():
    """事件 -> 需要检出的目标类（事件与检测互相校验用）"""
    return dict(load_ontology().get("events_need") or {})


def scene_categories():
    """场景关注分类：优先本体；本体没有则回退旧的 scene_categories.json（兼容旧部署）"""
    cats = load_ontology().get("scene_categories")
    if isinstance(cats, list) and cats:
        return cats
    try:
        with open(_SCENE_CATEGORIES_LEGACY, "r", encoding="utf-8") as f:
            return json.load(f).get("scene_categories") or []
    except Exception:
        return []


def scene_enum_str():
    """供提示词使用的枚举文本（统一维度名）"""
    parts = []
    for dim, d in dimensions().items():
        parts.append("%s: %s" % (dim, "/".join((d or {}).get("values") or [])))
    return "\n".join(parts)
