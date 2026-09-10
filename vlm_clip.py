# -*- coding: utf-8 -*-
# vlm_clip.py
import re
import os
import json
import torch

from ontology import load_ontology, valid_tags, dimensions, values_of
from sampling import sample_representative

VALID = valid_tags()
# 旧维度名 -> 统一名（历史输出兼容），与 ontology.alias_map() 同源
from ontology import alias_map as _alias_map
_DIM_ALIASES = {}
for _old, _new in _alias_map().items():
    _DIM_ALIASES.setdefault(_new, []).append(_old)


def build_clip_prompt():
    """枚举全部来自统一本体(scene.json)，维度名与帧级链路一致(road/road_surface/scene)，
    这样帧级与 Clip 级产出的标签可以直接放进同一张分布表聚合。"""
    o = load_ontology()
    dims = dimensions()

    def _vals(dim):
        return "/".join((dims.get(dim) or {}).get("values") or [])

    return (
        "你是自动驾驶场景分析专家。以下5张图片是从同一段视频按固定间隔抽取的帧"
        "（原始帧号递增，彼此可能相隔若干帧，不是相邻帧），"
        "按时间顺序为 Frame 1 → Frame 2 → Frame 3 → Frame 4 → Frame 5。\n"
        "请基于多帧之间的变化进行判断（不要只看单帧，位置/朝向/与自车距离的变化就是事件证据），"
        "严格输出以下JSON，"
        "不要任何其他文字、解释或代码块标记：\n"
        "{\n"
        f'  "scene": {{"time": ["<从枚举选>"], "weather": ["<从枚举选>"], '
        f'"road": ["<从枚举选>"], "road_surface": ["<从枚举选>"], "scene": ["<从枚举选>"], '
        f'"lighting": ["<从枚举选>"], "traffic_state": ["<从枚举选>"]}},\n'
        f'  "ego_vehicle": {{"state": ["<从枚举选>"], "confidence": <0.0~1.0真实数值>}},\n'
        f'  "objects": [{{"type": "<从枚举选>", "state": ["<从枚举选>"], "confidence": <0.0~1.0真实数值>}}],\n'
        f'  "events": [{{"type": "<从枚举选>", "confidence": <0.0~1.0真实数值>, '
        f'"evidence": ["F1:<第1帧的初始位置>", "F3:<第3帧的变化>", "F5:<第5帧的最终状态>"]}}]\n'
        "}\n"
        "可用枚举值：\n"
        f"scene.time: {_vals('time')}\n"
        f"scene.weather: {_vals('weather')}\n"
        f"scene.road: {_vals('road')}\n"
        f"scene.road_surface: {_vals('road_surface')}\n"
        f"scene.scene: {_vals('scene')}\n"
        f"scene.lighting: {_vals('lighting')}\n"
        f"scene.traffic_state: {_vals('traffic_state')}\n"
        f"ego_vehicle.state: {'/'.join(o['ego_state'])}\n"
        f"objects.type: {_vals('objects')}\n"
        f"objects.state: {'/'.join(o['object_state']['states'])}\n"
        f"events.type: {'/'.join(o['events'])}\n"
        "强制规则：\n"
        "1. 无法确认的维度填 [\"unknown\"]，禁止臆测；\n"
        "2. 禁止输出枚举之外的泛化词（如复杂城市交通/危险环境）；\n"
        "3. events 的 evidence 必须写明三帧证据：初始位置→中间变化→最终状态；\n"
        "4. 单帧静止的行人/车辆，若5帧间无位置变化，不要报横穿/切入等动态事件；\n"
        "5. objects 和 events 可以为空数组；\n"
        "6. 尖括号 <> 内只是占位说明，必须替换为你实际判断出的值，禁止原样照抄示例内容；\n"
        "7. confidence 必须是 0.0~1.0 的真实数值，反映你的确信程度，不得统一填 0.0 或照抄示例；\n"
        "8. 事件必须“帧间有变化”才报：逐帧确认目标位置，位置基本不变的一律不报动态事件；\n"
        "9. 不确定是否有事件时，events 输出空数组 []，宁可少报也不要套一个常见事件；\n"
        "10. 所有维度值必须是数组：即使只有一个值也要写成 [\"值\"]，禁止写成裸字符串；\n"
        "11. objects 和 events 最多各输出 5 条，取最关键的，不要重复罗列同类目标；\n"
        "12. 输出必须是一个完整闭合的 JSON 对象，最后一个字符是 }。"
    )


def _salvage_truncated(text):
    """VLM 输出被 max_new_tokens 截断时抢救：切掉最后一个不完整元素，补齐未闭合的括号。
    只影响"尾部不完整"的情况；本来就完整的 JSON 返回 None（交给正常解析）。"""
    i = text.find("{")
    if i < 0:
        return None
    s = text[i:]
    stack, in_str, esc, last_ok = [], False, False, None
    for k, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return None          # 括号闭合，原文完整
            last_ok = k
        elif ch == "," and stack:
            last_ok = k
    if last_ok is None:
        return None
    head, st, in_str, esc = s[:last_ok], [], False, False
    for ch in head:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            st.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if st:
                st.pop()
    return head + "".join(reversed(st))


def _extract_json_object(text):
    """取第一个 '{' 到与之配对的 '}' 之间的子串（跳过字符串内部的括号）。
    不能用 rfind('}')：模型常在完整 JSON 之后多吐一个 '}' 或说明文字，
    取最后一个右括号会把尾巴带进来，json.loads 直接报 Extra data 而丢掉整条结果。"""
    s = text.find("{")
    if s < 0:
        return None
    depth, in_str, esc = 0, False, False
    for k in range(s, len(text)):
        ch = text[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[s:k + 1]
    return None


def parse_and_validate(text):
    """解析VLM输出JSON + Ontology白名单校验（非法标签直接丢弃）"""
    text = (text or "").strip()
    cleaned = re.sub(r"```(?:json)?", "", text)
    obj = None
    sub = _extract_json_object(cleaned)
    if sub:
        try:
            obj = json.loads(sub)
        except Exception:
            obj = None
    if obj is None:
        # 括号没闭合（被 max_new_tokens 截断）→ 抢救
        fixed = _salvage_truncated(cleaned)
        if fixed:
            try:
                obj = json.loads(fixed)
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None

    def _filt(vals):
        """场景维度值：部分模型(如 7B)会输出裸字符串而非数组，先统一成列表再校验白名单，
        否则逐字符迭代会导致整个维度变成 unknown。"""
        if isinstance(vals, str):
            vals = [vals]
        return [v for v in (vals or []) if isinstance(v, str) and v in VALID] or ["unknown"]

    def _tag(v):
        """取单个合法标签。VLM 常把 type/state 输出成数组(如 ["静止"])，
        直接 v in VALID (set) 会抛 unhashable type: 'list'，必须先收敛成字符串。"""
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        return v if isinstance(v, str) and v in VALID else None

    def _conf(v):
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _evlist(v):
        if isinstance(v, str):
            return [v]
        return [str(x) for x in (v or [])]

    scene = obj.get("scene") or {}
    out_scene = {}
    # 统一维度名与旧名同时接受：新提示词输出 road/road_surface/scene，
    # 历史数据/旧模型输出可能是 road_type/surface/area，两者都归一化到统一名
    for dim in ("time", "weather", "road", "road_surface", "scene", "lighting", "traffic_state"):
        vals = scene.get(dim)
        if vals in (None, [], ""):
            for old in _DIM_ALIASES.get(dim, ()):
                if scene.get(old) not in (None, [], ""):
                    vals = scene.get(old)
                    break
        out_scene[dim] = _filt(vals)

    ego = obj.get("ego_vehicle") or {}
    events = []
    for ev in (obj.get("events") or []):
        t = _tag(ev.get("type")) if isinstance(ev, dict) else None
        if t:
            events.append({
                "type": t,
                "confidence": _conf(ev.get("confidence")),
                "evidence": _evlist(ev.get("evidence"))[:5],
            })
    objects = []
    for ob in (obj.get("objects") or []):
        t = _tag(ob.get("type")) if isinstance(ob, dict) else None
        if t:
            objects.append({
                "type": t,
                "state": _tag(ob.get("state")) or "unknown",
                "confidence": _conf(ob.get("confidence")),
            })

    return {
        "scene": out_scene,
        "ego_vehicle": {
            "state": _filt(ego.get("state")),
            "confidence": _conf(ego.get("confidence")),
        },
        "objects": objects,
        "events": events,
    }


def predict_clip(vlm_model, vlm_processor, frame_paths, device="cuda", max_pixels_imgsz=None, max_new_tokens=1024):
    """5帧Clip多图推理。返回 (validated_result, raw_text)。失败返回 (None, raw)"""
    from PIL import Image
    if max_pixels_imgsz is None:
        # 与 app.py 的处理器构造读同一个变量：这里传参会覆盖处理器上的设置，
        # 只改环境变量而不改这里的话，分辨率上限依然是 448
        max_pixels_imgsz = int(os.environ.get("AD_VLM_MAX_PX", "448"))
    # 这行才是真正的分辨率瓶颈：PIL 先把帧缩到 640 以内(1920x1080 -> 640x360 ≈ 23万像素)，
    # 远低于处理器上限，导致调 AD_VLM_MAX_PX 完全不生效。要提细节必须同时放大这里。
    _thumb = int(os.environ.get("AD_VLM_THUMB", "1024"))
    rep = sample_representative(frame_paths, 5)
    images = []
    for p in rep:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((_thumb, _thumb))
            images.append(im)
        except Exception:
            continue
    if not images:
        return None, ""

    conversation = [{
        "role": "user",
        "content": [{"type": "image"}] * len(images)
                   + [{"type": "text", "text": build_clip_prompt()}],
    }]
    text = vlm_processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)
    # min/max_pixels 限每帧448px防OOM。transformers<4.49 的 preprocess() 不接受这两个调用参数
    # (直接 TypeError)，只能设在 image_processor 属性上；新版才支持当参数传，故先试参数再回退。
    _px = {"min_pixels": 256 * 28 * 28, "max_pixels": max_pixels_imgsz * 28 * 28}
    try:
        inputs = vlm_processor(text=[text], images=images, return_tensors="pt", **_px).to(device)
    except TypeError:
        _ip = getattr(vlm_processor, "image_processor", None)
        if _ip is None:
            raise
        for _k, _v in _px.items():
            setattr(_ip, _k, _v)
        inputs = vlm_processor(text=[text], images=images, return_tensors="pt").to(device)

    with torch.inference_mode():
        # 只用温和的重复惩罚：no_repeat_ngram 对结构化 JSON 是灾难（JSON 本身高度重复，
        # 强行禁 8-gram 会把模型逼成全部输出 unknown）。截断风险改由解析端抢救。
        out = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=float(os.environ.get("AD_VLM_REP_PENALTY", "1.05")),
        )
    out = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    raw = vlm_processor.batch_decode(out, skip_special_tokens=True)[0]
    return parse_and_validate(raw), raw
