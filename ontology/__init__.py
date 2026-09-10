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

词表维护（常态化更新标签，归口在「融合语义搜索」）：
  value_alias_map()    取值别名表 {别名: 正式值}（区别于 alias_map()，那是旧维度名）
  canonical_tag()      取值归一到正式值
  add_value()          候选标签升为正式取值
  add_value_alias()    候选标签归并到已有正式取值
"""
import os, json, shutil

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


# ============ 词表维护（常态化更新标签：候选 -> 正式标签 / 归并到已有标签）============
def scene_path():
    return os.path.join(_DIR, "scene.json")


def value_alias_map(dim=None):
    """值别名表：{别名: 正式值}；传 dim 只取该维度。
    与 alias_map() 不同——那个映射的是【旧维度名】(road_type->road)，这里映射【取值】
    (如 公交车切出 -> 车辆切出)。维护入口：融合语义搜索的「标签维护」区。"""
    out = {}
    for d, cfg in dimensions().items():
        if dim and d != dim:
            continue
        for a, v in ((cfg or {}).get("value_aliases") or {}).items():
            out[a] = v
    return out


def canonical_tag(dim, tag):
    """把取值归一到正式值：命中值别名的返回其正式值，否则原样返回"""
    return value_alias_map(dim).get(tag, tag)


def _write_ontology(o):
    """原子写回本体，并留一份 .bak（词表是判定标签合法性的唯一真源，改坏影响全局）"""
    p = scene_path()
    if os.path.exists(p):
        try:
            shutil.copyfile(p, p + ".bak")
        except Exception:
            pass
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def add_value(dim, value, cn=None):
    """把一个候选标签升为正式取值；已存在则直接返回。返回 (是否新增, 该维度当前取值)"""
    o = load_ontology()
    dims = o.setdefault("dimensions", {})
    cfg = dims.setdefault(dim, {"cn": cn or dim, "values": []})
    if cn and not cfg.get("cn"):
        cfg["cn"] = cn
    vals = cfg.setdefault("values", [])
    # 已经作为别名存在时，改为扶正（删掉别名条目，避免既别名又正式）
    al = cfg.setdefault("value_aliases", {})
    al.pop(value, None)
    if value in vals:
        _write_ontology(o)
        return False, list(vals)
    # 插到 unknown 前（unknown 约定放最后）
    idx = vals.index("unknown") if "unknown" in vals else len(vals)
    vals.insert(idx, value)
    _write_ontology(o)
    return True, list(vals)


def dismissed_map(dim=None):
    """被"放弃使用"的候选标签 {dim: [tag,...]}：不进本体、不再出现在候选池。
    与值别名不同——放弃只是不再提示，数据里已出现的原词原样保留。"""
    out = {}
    for d, cfg in dimensions().items():
        if dim and d != dim:
            continue
        vals = list((cfg or {}).get("dismissed") or [])
        if vals:
            out[d] = vals
    return out


def dismiss_value(dim, tag):
    """放弃使用某候选标签；返回 (是否变更, 该维度已放弃列表)"""
    o = load_ontology()
    cfg = o.setdefault("dimensions", {}).setdefault(dim, {"cn": dim, "values": []})
    lst = cfg.setdefault("dismissed", [])
    if tag in lst:
        return False, list(lst)
    lst.append(tag)
    _write_ontology(o)
    return True, list(lst)


def restore_value(dim, tag):
    """把已放弃的标签放回候选池；返回 (是否变更, 该维度已放弃列表)"""
    o = load_ontology()
    cfg = (o.get("dimensions") or {}).get(dim) or {}
    lst = list(cfg.get("dismissed") or [])
    if tag not in lst:
        return False, lst
    lst.remove(tag)
    cfg["dismissed"] = lst
    _write_ontology(o)
    return True, lst


def add_value_alias(dim, alias, canonical):
    """把候选标签归并到已有正式值：写 value_aliases[alias]=canonical。
    返回 (是否写入, 该维度正式取值)。canonical 必须已在本体里。"""
    o = load_ontology()
    dims = o.setdefault("dimensions", {})
    cfg = dims.setdefault(dim, {"cn": dim, "values": []})
    vals = cfg.setdefault("values", [])
    if canonical not in vals:
        raise ValueError("目标标签不在本体内: %s" % canonical)
    if alias in vals:
        raise ValueError("该标签已是正式取值，不能归并: %s" % alias)
    al = cfg.setdefault("value_aliases", {})
    if al.get(alias) == canonical:
        return False, list(vals)
    al[alias] = canonical
    _write_ontology(o)
    return True, list(vals)
