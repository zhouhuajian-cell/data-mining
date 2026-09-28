# -*- coding: utf-8 -*-
# vlm_clip.py
import re
import os
import json
import torch

from ontology import load_ontology, valid_tags, dimensions, values_of, canonical_tag
from sampling import sample_representative

VALID = valid_tags()
# 旧维度名 -> 统一名（历史输出兼容），与 ontology.alias_map() 同源
from ontology import alias_map as _alias_map
_DIM_ALIASES = {}
for _old, _new in _alias_map().items():
    _DIM_ALIASES.setdefault(_new, []).append(_old)


def _ev_hints() -> str:
    """易混事件判据（与帧级/联合打标共用本体真源，避免两处口径跑偏）。"""
    try:
        from ontology import event_hints_text
        return event_hints_text()
    except Exception:
        return ""


def build_clip_prompt(n_frames=None):
    """按**实际帧数**生成提示词。枚举全部来自统一本体(scene.json)，维度名与帧级链路一致。

    ⚠️ 2026-09-20 重写（实测根因，别再退回去）：
    ① 以前写死"5 帧 / F1→F3→F5"，而实际送 8 帧（AD_CLIP_PICK）——连"末帧"都指错了；
    ② 更糟的是示例里的 evidence 是个**填空句式**（"F1:<初始位置> F3:<变化> F5:<最终状态>"），
       模型直接当模板照抄：62 段「行人横穿」里 51 段是同一句的两个变体（只差"穿过/横穿"两字），
       而 YOLO 框实测这些段 89% 确实有行人横向位移 —— 即"结论对、依据假"。
    现在：帧号按实际帧数生成；evidence 只说"要写什么"、不再给任何句式；
    并明确禁止复述判据原文（判据只用于"判断是不是这个事件"）。"""
    o = load_ontology()
    dims = dimensions()
    n = int(n_frames or 0)
    if n <= 0:
        n = int(os.environ.get("AD_CLIP_PICK", "8"))
    _frames = " → ".join("Frame %d" % i for i in range(1, n + 1))
    _mid = max(2, n // 2)

    def _vals(dim, drop_unknown=False):
        vs = (dims.get(dim) or {}).get("values") or []
        if drop_unknown:
            # traffic_sign 有明确的否定值「无标识」了，枚举里再留 unknown 只会诱导模型写它
            vs = [v for v in vs if v != "unknown"]
        return "/".join(vs)

    return (
        f"你是自动驾驶场景分析专家。以下 {n} 张图片是从同一段视频按固定间隔抽取的帧"
        "（原始帧号递增，彼此可能相隔若干帧，不是相邻帧），"
        f"按时间顺序为 {_frames}。\n"
        "请基于多帧之间的变化进行判断（不要只看单帧，位置/朝向/与自车距离的变化就是事件证据），"
        "严格输出以下JSON，不要任何其他文字、解释或代码块标记：\n"
        "{\n"
        f'  "scene": {{"time": ["<从枚举选；没把握的维度直接省略>"], "weather": ["<同上>"], \n'
        f'"road_shape": ["<从枚举选>"], "lane_count": ["<从枚举选>"], \n'
        f'"junction": ["<从枚举选>"], "road_marking": ["<从枚举选>"], \n'
        f'"road_surface": ["<从枚举选>"], "scene": ["<从枚举选>"], \n'
        f'"lighting": ["<从枚举选>"], "traffic_state": ["<从枚举选>"], \n'
        f'"traffic_sign": ["<从枚举选>"]}},\n'
        f'  "risk": ["<从枚举选>"],\n'
        f'  "ego_vehicle": {{"state": ["<从枚举选>"], "confidence": <0.0~1.0真实数值>}},\n'
        f'  "events": [{{"type": "<从枚举选>", "confidence": <0.0~1.0真实数值>, \n'
        f'"anchor": "<画面里你据以判断的具体物，如 车道线/路缘/人行横道/停止线/前车尾灯>", \n'
        f'"reason": "<为什么判成这个事件：依据哪几帧的什么变化，为何不是最像的另一个事件>"}}],\n'
        f'  "objects": [{{"type": "<从枚举选>"}}]\n'
        "}\n"
        "可用枚举值：\n"
        f"scene.time: {_vals('time')}\n"
        f"scene.weather: {_vals('weather')}\n"
        f"scene.road_shape: {_vals('road_shape')}   （道路线形，单一轴：直的还是弯的）\n"
        f"scene.lane_count: {_vals('lane_count')}   （车道数/宽度，单一轴）\n"
        f"scene.junction: {_vals('junction')}   （路口形态，单一轴；**必须表态，禁止省略**）\n"
        "   · 怎么分：**三条**路交汇、缺一条腿 = T型路口；**四条**路交汇 = 十字路口；环形 = 环岛；\n"
        "     一侧有车道汇进来 = 汇入口；有分流岛 = 分流区；画面里根本没有路口 = 无路口。\n"
        "   · 人工实测提醒（2026-09-21）：这一段数据集里**大量被误判成「十字路口」，其实是 T 型路口**，\n"
        "     所以看到「像路口」但要二选一时，先数清是三条路还是四条路（只看得到三条就填 T型路口）。\n"
        "   · 这一轴**不许省略**（省略 = 人工无法判断你是「没看」还是「看了没有」）：判断不了形态时，\n"
        "     看有没有汇入的车道线/路缘开口——有就按上面填具体值，实在没有就填 无路口。\n"
        f"scene.road_marking: {_vals('road_marking')}   （**路面标线**：画在地上的线，与立在路边的牌子不是一类）\n"
        f"scene.road_surface: {_vals('road_surface')}\n"
        f"scene.scene: {_vals('scene')}\n"
        f"scene.lighting: {_vals('lighting')}\n"
        f"scene.traffic_state: {_vals('traffic_state')}\n"
        f"scene.traffic_sign: {_vals('traffic_sign', True)}\n"
        f"risk: {_vals('risk')}\n"
        f"ego_vehicle.state: {'/'.join(o['ego_state'])}\n"
        f"objects.type: {_vals('objects')}\n"
        f"objects.state: {'/'.join(o['object_state']['states'])}\n"
        f"events.type: {'/'.join(o['events'])}\n"
        "强制规则：\n"
        "1. 无法确认的维度填 [\"unknown\"]，禁止臆测；\n"
        "2. 禁止输出枚举之外的泛化词（如复杂城市交通/危险环境）；\n"
        "3. events[].anchor 必填：写清你据以判断的**画面参照物**"
        "（车道线/路缘/人行道/人行横道/停止线/信号灯/隔离带/车头前方/被某车遮挡…）；写不出就不要报这个事件；\n"
        "3b. events[].reason 必须写成这种形式（不超过 50 字）：\n"
        "   「F<起>→F<止>：<目标>从<画面参照物A>移到/变成<画面参照物B>；排除「<最像的另一个事件>」因为<一句理由>」\n"
        "   必须同时出现**两个帧号**和**一个画面参照物**（同上）；只写「左侧/右侧/前方」这种纯方位词不算；\n"
        "   ⚠️ **有事件就要报**：不要因为「理由写不具体」就改成不报 —— 理由不具体由人工判断（界面会打标），"
        "**漏报没人能补救**（2026-09-21 实测：光加「必须写出两个物」的硬要求，事件数从 85 掉到 18，"
        "而套话一条没少）；\n"
        "   ⚠️ 但**同一次运动最多报 2 个事件类型**（例如「切入」和「切出」各报一次就够了）；\n"
        "   **绝对不要**把同一次位移贴上 5 个标签 —— 实测有一段同时报了 切入/切出/变道/加塞/急减速，\n"
        "   5 条理由还是逐字相同的同一句，人工完全没法标。判断不确定时：分不清切入还是切出 → 报「车辆变道」；\n"
        "   旁车强行插入自车前方、自车被迫减速 → 报「车辆加塞」（它就是切入里最危险的一种，不必再重复报切入）；\n"
        "4. **禁止**把事件判据的原文、或「位于道路一侧/移动到道路另一侧/完全穿过道路/切入本车道/强行加塞」这类结论式套话写进 reason。自检：这句话如果换到别的视频里也照样成立，它就是废话，不要写；\n"
        "5. 事件必须“帧间有变化”才报：逐帧确认目标位置，位置基本不变的一律不报动态事件；\n"
        "6. 场景与风险取值必须有画面依据，看不到就填 [\"unknown\"]（尤其不要默认晴天/干燥路面）：\n"
        "   - time / lighting：按画面亮度与光照判断；\n"
        "   - weather / road_surface：必须看到雨丝、积水反光、积雪、结冰等直接迹象；\n"
        "   - scene：不能因为“像城市”就填城市道路；填隧道要看到隧道结构（拱形/照明带），"
        "填高速要看到中央隔离带、匝道、服务区等特征；\n"
        "   - traffic_state：按画面中车辆/行人的密度判断；\n"
        "   - risk：按本段是否出现需要自车避让的目标行为判断，没有就填 [\"unknown\"]；\n"
        "7. objects 和 events 可以为空数组；不确定是否有事件时 events 输出 []，宁可少报也不要套一个常见事件；\n"
        "8. 尖括号 <> 内只是占位说明，必须替换为你实际判断出的值，禁止原样照抄示例内容；\n"
        "9. confidence 必须是 0.0~1.0 的真实数值，反映你的确信程度，不得统一填 0.0 或照抄示例；\n"
        "10. 所有维度值必须是数组：即使只有一个值也要写成 [\"值\"]，禁止写成裸字符串；\n"
        "11. objects **只写交通参与者与路面上的物**（同类只写一条，最多 8 条）：行人/两轮车/三轮车/小车/大车/公交车/异型车/特殊车辆/锥桶/围挡 —— 检测摘要里出现过的类，只要你也看到了就必须列；画面里确实没有目标才写空数组 []；\n"
        "    ⚠️ **红绿灯、标识牌、斑马线不要再写进 objects**（2026-09-21 标签规整）：红绿灯/标识牌属于 traffic_sign、斑马线属于 road_marking，写进 objects 会被程序搬到那些维度，等于同一件事写两遍；\n"
        "12. scene / risk 里没有把握的维度**直接省略**，不要输出 [\"unknown\"] 占位（看不清就省略，不扣分）；\n"
        "   ⚠️ 例外：**junction / road_marking / traffic_sign 必须表态、不得省略**（见 12b）。\n"
        "12b. **traffic_sign 与 road_marking 都是「必答维」，不适用上一条的『没把握就省略』**\n"
        "   （2026-09-21 标签规整后重写）：\n"
        "   ① traffic_sign：看到**路侧的标志牌或信号灯**就按本维枚举填（红绿灯→交通信号灯；\n"
        "      认不出是哪一种牌子 → 道路标识牌）；\n"
        "   ② road_marking：看到**地面上的线**就按本维枚举填（人行横道/停止线/导流线/网状线/\n"
        "      待行区/减速标线/可变导向车道/公交专用道）；\n"
        "   ③ 这两类**都不要写进 objects**（objects 只放参与者与锥桶/围挡）；\n"
        "   ④ 本维**不再用 unknown**（2026-09-21 用户明确）：确实没有标志牌/信号灯 → 写 [\"无标识\"]；\n"
        "      这两维都要**明确表态**，禁止省略、禁止写 unknown。\n"
        "12c. **有牌子但认不出是哪一种 → 填 [\"交通标识牌\"]，不要给无标识**（2026-09-21 用户明确）：\n"
        "   例如远远看到一块指示牌/警示牌、但看不清牌面图案与文字时，填 [\"交通标识牌\"] ——\n"
        "   这个取值的意思就是「本场景有标识牌」。**「无标识」只用于「确实没有标识 / 完全看不出」**，\n"
        "   不要用它代替「有牌子但认不出」。\n"
        "13. 输出必须是一个完整闭合的 JSON 对象，最后一个字符是 }；\n"
        "14. 下面的事件判据用于判断“是不是这个事件”，**不要**把它的文字抄进 reason：\n"
        + _ev_hints()
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


# ---- 按维度校验 / 跨维度归位 / 反模板（2026-09-20 加）----
# 背景：以前校验用的是 valid_tags() —— 那是**所有维度值的全局并集**（145 个），
# 于是 road 里塞「城市道路」（场景值）、events 里塞「晴天」都能通过。实测 25 段里 10 段
# 的 road 被填成了场景值。这里改成按维度校验，并把错位的值**归位**而不是丢弃。
_SCENE_DIMS = ("time", "weather", "road_shape", "lane_count", "junction", "road_marking",
                "road_surface", "scene", "lighting", "traffic_state")
_EMIT_DIMS = set(_SCENE_DIMS) | {"traffic_sign", "risk"}
_DIM_VALS = {}
_VAL2DIM = {}
_VAL_ALIAS = {}      # dim -> {别名值: 正式值}（如 高速道路 -> 高速公路）
_EVENT_VALS = set()
_OBJ_VALS = set()
_OBJ_STATE = set()
_EGO_VALS = set()

# 2026-09-21 标签规整：objects 里的「设施类」已经迁到别的维度（用户定：目标只留参与者 + 锥桶/围挡）。
# 模型按老习惯仍会把它们写进 objects，而别名机制只在**同一维度内**生效、`_VAL2DIM` 也认不出
# 迁走的词 —— 所以这里显式指路，把值**搬过去**而不是丢掉（丢 = 白丢一条模型真看到的证据）。
_OBJ_MOVE = {
    "红绿灯": ("traffic_sign", "交通信号灯"),
    "信号灯": ("traffic_sign", "交通信号灯"),
    "交通灯": ("traffic_sign", "交通信号灯"),
    "交通标识牌": ("traffic_sign", "交通标识牌"),
    "交通设施": ("traffic_sign", "交通标识牌"),
    "指示牌": ("traffic_sign", "交通标识牌"),
    "路牌": ("traffic_sign", "交通标识牌"),
    "标识牌": ("traffic_sign", "交通标识牌"),
    "警示牌": ("traffic_sign", "交通标识牌"),
    "标志牌": ("traffic_sign", "交通标识牌"),
    "斑马线": ("road_marking", "人行横道"),
    "人行横道": ("road_marking", "人行横道"),
}

# ---- 事件去重（2026-09-21 用户要求）：模型会给**同一次运动**贴多个互相矛盾的标签 ----
# 实测最夸张的一段报了 5 个「车辆切入」且理由逐字相同；另一段同时报 切入/切出/变道/加塞/急减速
# （同一次横向位移不可能同时是这四个）。人工打标时会看到互相打架的卡片，真值没法标。
# ⚠️ 必须是**确定性**处理（不依赖模型听话）：上一轮试过用措辞收紧，结果事件从 85 塌到 18 而套话没少。
# ⚠️ 车辆急减速**不在互斥族里**：前车侵入本来就可以同时让自车减速，两者不矛盾。
_MOVE_EXCL = ("车辆切入", "车辆切出", "车辆变道", "车辆加塞")
# 同时出现时留哪一条：加塞 ⊂ 切入（更具体也更危险），变道是"分不清时才选"的兜底 → 优先级最低
_MOVE_PRIO = {"车辆加塞": 3, "车辆切入": 2, "车辆切出": 1, "车辆变道": 0}


def dedup_events(events):
    """同一次运动只留一条事件（解析端与存量迁移**共用这一个实现**，避免口径漂移）。

    ① (类型, anchor, reason) 完全相同 → 只留置信度最高的一条
       （对应实测的「车辆切入 ×5、理由逐字相同」）
    ② 同一 anchor 下的互斥族（切入/切出/变道/加塞）→ 只留最贴切的一条
       （按 置信度 → 类型优先级；**anchor 为空时不做②**，宁可多留也不误删不同目标的事件）

    返回新列表（不修改入参）。"""
    def _c(e):
        try:
            return float(e.get("confidence") or 0)
        except Exception:
            return 0.0
    _seen, uniq = set(), []
    for e in (events or []):
        if not isinstance(e, dict) or not e.get("type"):
            continue
        k = (e.get("type"), str(e.get("anchor") or "").strip(), str(e.get("reason") or "").strip())
        if k in _seen:
            continue
        _seen.add(k)
        uniq.append(e)
    by_anchor = {}
    for e in uniq:
        a = str(e.get("anchor") or "").strip()
        if a and e.get("type") in _MOVE_EXCL:
            by_anchor.setdefault(a, []).append(e)
    drop = set()
    for lst in by_anchor.values():
        # ⚠️ **≥3 个才收敛**（2026-09-21 实测修正）：模型报「切入+切出」这一对是**常态写法**
        # （改动前 35 段都是这样，它把同一段横向位移的两个方向都描述了一遍），不是刷屏。
        # 早期版本把成对的也按优先级留一条 → 「车辆切出」从 43 段塌到 1 段：**静默删掉不等于判对**，
        # 该由人工点掉错的那条（那才是评测里 FP 的真值来源）。只把真正的刷屏（≥3 个）收敛掉。
        if len(lst) < 3:
            continue
        keep = max(lst, key=lambda e: (_c(e), _MOVE_PRIO.get(e.get("type"), 0)))
        for e in lst:
            if e is not keep:
                drop.add(id(e))
    return [e for e in uniq if id(e) not in drop]


def _init_dim_maps():
    """懒加载各维度词表 + 「值 → 所属维度」反查表（只跑一次）。"""
    global _EVENT_VALS, _OBJ_VALS, _OBJ_STATE, _EGO_VALS
    if _DIM_VALS:
        return
    for d in tuple(_EMIT_DIMS) + ("objects", "events"):
        # ⚠️ events 必须也在内：本体的 events.value_aliases（小车切出→车辆切出、自行车横穿→
        # 非机动车横穿…）以前**从来没生效过** —— 校验走的是顶层 events 列表的精确匹配，
        # 别名表建都没建（实测模型写「小车切出」会被整条丢掉，2026-09-21 标签规整时发现）。
        try:
            vals = set(values_of(d))
        except Exception:
            vals = set()
        _DIM_VALS[d] = vals
        for v in vals:
            _VAL2DIM.setdefault(v, set()).add(d)
        for a, v in ({k: v for k, v in ((cfg.get("value_aliases") or {}).items())}
                     if (cfg := (dimensions().get(d) or {})) else {}).items():
            _VAL_ALIAS.setdefault(d, {})[a] = v
    try:
        o = load_ontology()
        _EVENT_VALS = set(o.get("events") or [])
        _OBJ_VALS = set((dimensions().get("objects") or {}).get("values") or [])
        _OBJ_STATE = set((o.get("object_state") or {}).get("states") or [])
        _EGO_VALS = set(o.get("ego_state") or [])
    except Exception:
        pass


def _tag_in(v, allowed):
    """取单个合法标签，且必须属于指定维度词表。VLM 常把值输出成数组，先收敛成字符串。"""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return v if isinstance(v, str) and allowed and v in allowed else None


# 具体画面参照物：证据要算"描述了画面所见"，必须点到这类**具体参照物**。
# ⚠️ 只写「左侧/右侧/前方」这类纯方位词**不算** —— 实测模型会学会只补个方位词
# （"行人从左侧向右侧移动"），依然是结论式套话，换段视频照样成立。
# 判据原文抄出来的句子恰好都不含这类锚点（"位于道路一侧/移动到道路另一侧/完全穿过道路"），
# 所以这个判据能把"套话"和"真观察到的东西"分开。
_ANCHOR_RE = re.compile(
    r"画面|车头|车前|车尾|车道线|路缘|人行道|人行横道|停止线|导流线|网状线|待行区|减速标线|"
    r"隔离带|中央|反光|积水|积雪|"
    r"遮挡|尾灯|刹车灯|转向灯|车灯|信号灯|影子|倒影|护栏|锥桶|施工|"
    r"红|黄|蓝|绿|白|黑|灰|橙|紫|大型|小型|背包|撑伞|衣着|\d")
# 「物」类参照物（不含颜色/数字这类弱证据）：判 reason 有没有写出**两个能指出来的物**
_OBJ_ANCHOR_RE = re.compile(
    r"画面|车头|车前|车尾|车道线|路缘|人行道|人行横道|停止线|导流线|网状线|待行区|减速标线|"
    r"隔离带|护栏|锥桶|围挡|信号灯|尾灯|刹车灯|转向灯|车灯|施工|影子|倒影|反光|积水|积雪|"
    r"背包|撑伞")
# 判据同义词 / 结论式套话：换到任何一段视频里都成立的词（规则 3b/4 明令禁止）
_BOILER_RE = re.compile(
    r"马路对面|道路一侧|道路另一侧|另一侧|穿过道路|完全穿过|本车道|强行插入|移到路对面")


def _reason_flags(etype, reason, evidence=None, anchor=None):
    """推理逻辑质量标记（替代原来的证据链标记）：
      generic  —— reason 里没有任何**具体画面参照物**（疑似把判据换成空话）；
      boiler   —— 出现判据同义词/结论式套话，**且没写出两个能指出来的物**
                  （模板要求「从<物A>移到<物B>」；只写一个物或写"马路对面"都算没写依据）；
      criteria —— reason 与该事件判据原文大段雷同（最长公共子串占比 >=45%）；
      repeat   —— 残留 evidence 里有两条完全相同。
    只打标不删，前端展示标记。

    ⚠️ 2026-09-21 标签规整后补 `boiler`、并把新词表（人行横道/停止线/导流线…）加进锚点白名单：
    规整把「人行横道」写进了 prompt 枚举，模型随即拿它当参照物写
    「F1→F8：行人从人行横道移到马路对面」—— 白名单里只有「人行道/斑马线」，
    「人行横道」既不是"人行道"的子串也匹配不上，于是整批被判 generic 去触发定向重问。
    锚点白名单该补齐（它是**具体的物**），但这类句式确实是套话 —— 由 boiler 继续抓，
    不能靠改白名单把指标"改好看"。"""
    import re as _re
    # ⚠️ 先剥掉 "F3:" 这类帧号前缀再判锚点：否则前缀里的数字会被 \d 当成"具体参照物"，
    # 让所有照抄判据的套话都蒙混过关（实测踩过）。
    raw = " ".join([str(reason or "")] + [str(x) for x in (evidence or [])]).strip()
    # 两套判据用的文本不同：锚点/复述判据要**剥掉帧号**（否则 F1 里的数字会被 \d 当成参照物），
    # 而"有没有逐帧比对"判据要**在原文里**找帧号 —— 混用会互相抵消（实测踩到）。
    txt = _re.sub(r"F\d+\s*[:：]?\s*", "", raw).strip()
    flags = {}
    # 有独立的 anchor 字段时以它为准（更可靠）；没有就退回在文本里找参照物
    _atxt = "%s %s" % (str(anchor or ""), txt)
    if not _ANCHOR_RE.search(_atxt):
        flags["generic"] = True
    if _BOILER_RE.search(_atxt) and len(set(_OBJ_ANCHOR_RE.findall(_atxt))) < 2:
        flags["boiler"] = True
    # 新增压套话判据：reason 必须给出"帧号对比"（至少两个帧号），否则视为没在比对画面
    if len(_re.findall(r"F\d+", raw)) < 2:
        flags["weak"] = True
    try:
        from ontology import load_ontology
        crit = str(((load_ontology().get("event_hints") or {}).get(etype)) or "")
    except Exception:
        crit = ""
    if crit and txt:
        import difflib
        _m = difflib.SequenceMatcher(None, txt, crit).find_longest_match(0, len(txt), 0, len(crit))
        if _m.size / float(len(txt)) >= 0.45:
            flags["criteria"] = True
    if evidence and len(evidence) >= 2 and len(set(evidence)) < len(evidence):
        flags["repeat"] = True
    return flags


def _evidence_flags(etype, items):
    """证据质量标记（确定性、不依赖模型是否听话）：

      - `repeat`：段内两条证据去重后完全相同 → 不可能是逐帧观察；
      - `generic`：一半以上证据不含任何具体画面锚点 → 疑似把判据换成了结论式套话。

    只打标**不删**（前端展示 ⚠️），让人工一眼看出这条依据能不能拿去核对画面。"""
    outs = []
    for x in (items or []):
        t = re.sub(r"^F\d+\s*[:：]\s*", "", str(x or "")).strip()
        if t:
            outs.append(t)
    flags = {}
    if len(outs) >= 2 and len(set(outs)) < len(outs):
        flags["repeat"] = True
    if outs and sum(1 for t in outs if not _ANCHOR_RE.search(t)) * 2 > len(outs):
        flags["generic"] = True
    return flags


def parse_and_validate(text):
    """解析VLM输出JSON + **按维度**白名单校验（跨维度错填归位、其余丢弃）+ 证据质量标记。"""
    _init_dim_maps()
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

    def _filt_dim(vals, dim):
        """按**本维度**词表校验（不再用全局并集 —— 全局并集下 road 里塞「城市道路」也拦不住）。

        返回 (保留值, 归位表)：属于别的维度词表的值不丢弃，而是归位到正确的维度
        （模型把场景值填进道路字段是实测高频现象，直接丢掉会白丢信息）。"""
        if isinstance(vals, str):
            vals = [vals]
        keep, move = [], {}
        _own = _DIM_VALS.get(dim) or set()
        for v in (vals or []):
            if not isinstance(v, str) or not v:
                continue
            v = (_VAL_ALIAS.get(dim) or {}).get(v, v)   # 旧值/别名先归一到正式值
            if v in _own:
                if v not in keep:
                    keep.append(v)
                continue
            for tgt in sorted((_VAL2DIM.get(v) or set()) & _EMIT_DIMS):
                if tgt == dim:
                    continue
                move.setdefault(tgt, [])
                if v not in move[tgt]:
                    move[tgt].append(v)
                break
        return keep, move

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
    _moved = {}
    # 统一维度名与旧名同时接受：新提示词输出 road/road_surface/scene，
    # 历史数据/旧模型输出可能是 road_type/surface/area，两者都归一化到统一名
    for dim in _SCENE_DIMS:
        vals = scene.get(dim)
        if vals in (None, [], ""):
            for old in _DIM_ALIASES.get(dim, ()):
                if scene.get(old) not in (None, [], ""):
                    vals = scene.get(old)
                    break
        keep, move = _filt_dim(vals, dim)
        out_scene[dim] = keep or ["unknown"]
        for tgt, vs in move.items():
            _moved.setdefault(tgt, [])
            for v in vs:
                if v not in _moved[tgt]:
                    _moved[tgt].append(v)
    # 归位值并入目标维度（去掉该维原有的 unknown 占位）
    for tgt, vs in _moved.items():
        if tgt in out_scene:
            merged = []
            for v in [x for x in out_scene[tgt] if x != "unknown"] + vs:
                if v not in merged:      # 归位要**去重**：模型把场景值填进 road 时，
                    merged.append(v)     # 归位后会和 scene 原有的同值重复（实测出现过）
            out_scene[tgt] = merged or ["unknown"]

    # traffic_sign / risk 是顶层字段：把 scene 里错填的值一并收过来
    _sign_raw = [v for v in (scene.get("traffic_sign") or []) if isinstance(v, str)] \
        if isinstance(scene.get("traffic_sign"), list) else \
        ([scene.get("traffic_sign")] if isinstance(scene.get("traffic_sign"), str) else [])
    if not _sign_raw:
        for _src in (obj,):
            _v = _src.get("traffic_sign")
            if isinstance(_v, str):
                _sign_raw = [_v]
            elif isinstance(_v, list):
                _sign_raw = [x for x in _v if isinstance(x, str)]
    _sign, _ = _filt_dim(_sign_raw + _moved.get("traffic_sign", []), "traffic_sign")
    # 本维不用 unknown（用户 2026-09-21）：模型偶尔还是写 unknown（规则 1 的通用说法会诱导它），
    # 这里确定性归成「无标识」—— 本维每一段都必须有明确语义，人工真值才好比。
    _sign = ["无标识" if (not isinstance(v, str) or v == "unknown") else v for v in (_sign or [])]
    _risk, _ = _filt_dim((obj.get("risk") if isinstance(obj.get("risk"), list) else
                          ([obj["risk"]] if isinstance(obj.get("risk"), str) else []))
                         + _moved.get("risk", []), "risk")

    # 逐帧观察（证据链第一段）：模型先说看到什么，再说结论。旧版本没有这个字段。
    _frames = [str(x).strip() for x in (obj.get("frames") or [])
               if isinstance(x, (str, int, float)) and str(x).strip()][:16]

    ego = obj.get("ego_vehicle") or {}
    _ego_vals = _EGO_VALS
    _est = [ego.get("state")] if isinstance(ego.get("state"), str) else list(ego.get("state") or [])
    _est = [v for v in _est if isinstance(v, str) and v in _ego_vals] or ["unknown"]

    events = []
    for ev in (obj.get("events") or []):
        if not isinstance(ev, dict):
            continue
        _et = ev.get("type")
        if isinstance(_et, str):
            # 事件别名归一生效（小车切出→车辆切出…）：见 _init_dim_maps 里 events 入表的注释
            _et = (_VAL_ALIAS.get("events") or {}).get(_et, _et)
        t = _tag_in(_et, _EVENT_VALS)
        if not t:
            continue
        _rsn = str(ev.get("reason") or "").strip()[:200]
        _anc = str(ev.get("anchor") or "").strip()[:80]
        _evi = _evlist(ev.get("evidence"))[:3]      # 兼容：老提示词/老数据仍带 evidence
        events.append({
            "type": t,
            "confidence": _conf(ev.get("confidence")),
            "reason": _rsn,
            "anchor": _anc,
            "flags": _reason_flags(t, _rsn, _evi, _anc),
        })
    # ⚠️ 硬性截断：提示词写了"最多各 5 条"，但模型会无视 —— 实测有段输出了 **175 个 objects**
    # （8900 字符），把 token 预算烧光：单段 150 秒，而且 events 还没写到就截断了、事件整批丢失
    # （2026-09-20 查实，这是判定变慢和"有事件变没事件"的直接原因）。解析端必须兜住。
    objects = []
    _moved_obj = {}          # 目标维度 -> [值]（objects 里被搬走的设施类，见 _OBJ_MOVE）
    for ob in (obj.get("objects") or []):
        if not isinstance(ob, dict):
            continue
        _ot = ob.get("type")
        if isinstance(_ot, str):
            # 别名归一（2026-09-21 加）：objects 原来没有别名表、且这里走 _tag_in 的纯词表检查，
            # 于是模型写"路牌/信号灯"这类同义词会被**直接丢弃**（用户反馈 traffic_sign 该列仍 unknown）。
            _ot = (_VAL_ALIAS.get("objects") or {}).get(_ot, _ot)
            _mv = _OBJ_MOVE.get(_ot)
            if _mv:
                # 迁走的设施类：搬到 traffic_sign / road_marking，不进 objects
                # （否则同一件事在两个维度各记一遍，人工真值没法算）
                _moved_obj.setdefault(_mv[0], [])
                if _mv[1] not in _moved_obj[_mv[0]]:
                    _moved_obj[_mv[0]].append(_mv[1])
                continue
        t = _tag_in(_ot, _OBJ_VALS)
        if t:
            # ⚠️ 提示词已不要求 objects 写 state/confidence —— 模型没给就**不要编 0.0**：
            # 0.0 会被下游当有效值参与"取最大置信度"算分，于是所有段都变成 0 分（实测踩到）。
            _o = {"type": t}
            _st = _tag_in(ob.get("state"), _OBJ_STATE)
            if _st:
                _o["state"] = _st
            if ob.get("confidence") is not None:
                _o["confidence"] = _conf(ob.get("confidence"))
            objects.append(_o)
    objects.sort(key=lambda x: -(x.get("confidence") or 0))
    objects = objects[:8]                      # 上限放宽到 8（要列全，尤其交通设施类）
    events.sort(key=lambda x: -(x.get("confidence") or 0))
    # 去重必须放在截断**之前**：否则 5 条重复的「车辆切入」会把真正的事件挤出前 5 名
    events = dedup_events(events)[:5]
    # objects 里被搬走的设施类（_OBJ_MOVE）并入目标维度：目标维原有的 unknown 占位要去掉，
    # 否则会出现 [「unknown」,「交通信号灯」] 这种自相矛盾的取值（unknown 恒被下游当"没给"用）。
    for _tgt, _vs in _moved_obj.items():
        if _tgt == "traffic_sign":
            _cur = [v for v in (_sign or []) if v != "unknown"]
            _sign = _cur + [v for v in _vs if v not in _cur]
        else:
            _cur = [v for v in (out_scene.get(_tgt) or []) if v != "unknown"]
            out_scene[_tgt] = _cur + [v for v in _vs if v not in _cur]
    return {
        "scene": out_scene,
        "traffic_sign": _sign or ["无标识"],
        "risk": _risk or ["unknown"],
        "ego_vehicle": {"state": _est, "confidence": _conf(ego.get("confidence"))},
        "objects": objects,
        "frames": _frames,
        "events": events,
    }
def _json_balanced(text):
    """text 里是否已经出现**一个完整闭合的顶层 JSON 对象**（{...} 深度回到 0）。

    ⚠️ 必须跳过字符串内部与反斜杠转义：reason 文本里出现 `}` 或引号时不能误判。
    返回 True 表示"JSON 已经写完，后面的都是赘述"。"""
    depth, in_str, esc, seen = 0, False, False, False
    for ch in (text or ""):
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
            seen = True
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and seen:
                    return True
    return False


def _json_stop_criteria(prompt_len: int):
    """构造"JSON 一闭合就停"的 StoppingCriteria（拿不到 transformers 时返回 None）。

    为什么需要它（2026-09-21 实测）：贪心解码下模型常**吐完 JSON 后继续赘述/重复**，
    一路撞到 max_new_tokens。实测有段生成 3094 字、而解析只用到 688 字 —— 白烧约 26 秒；
    取帧与检测摘要只占 0.0 秒，**时间是 100% 花在这种无用的解码上**。
    加上这个判据后，同一段应在 JSON 闭合时立刻停。"""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _JsonDone(StoppingCriteria):
            def __init__(self, plen):
                self.plen = plen
                self._done = set()

            def __call__(self, input_ids, scores, **kwargs):
                try:
                    tok = getattr(self, "tok", None)
                    if tok is None:
                        return False
                    for i in range(input_ids.shape[0]):
                        if i in self._done:
                            continue
                        txt = tok.decode(input_ids[i][self.plen:], skip_special_tokens=True)
                        if _json_balanced(txt):
                            self._done.add(i)
                    return len(self._done) >= input_ids.shape[0]
                except Exception:
                    return False

        return StoppingCriteriaList([_JsonDone(prompt_len)])
    except Exception:
        return None


def predict_clip(vlm_model, vlm_processor, frame_paths, device="cuda", max_pixels_imgsz=None,
                 max_new_tokens=1024, det_by_path=None, extra_note=None):
    """5帧Clip多图推理。返回 (validated_result, raw_text)。失败返回 (None, raw)

    det_by_path: {帧路径: 检测摘要文本}（可选）—— 段内帧的 YOLO/DINO 检测结果注入提示词，
    让 VLM 的场景/事件判断以真实检测为依据（2026-09-19 重构：检测与 VLM 分工的落点）。"""
    from PIL import Image
    if max_pixels_imgsz is None:
        # 与 app.py 的处理器构造读同一个变量：这里传参会覆盖处理器上的设置，
        # 只改环境变量而不改这里的话，分辨率上限依然是 448
        max_pixels_imgsz = int(os.environ.get("AD_VLM_MAX_PX", "448"))
    # 这行才是真正的分辨率瓶颈：PIL 先把帧缩到 640 以内(1920x1080 -> 640x360 ≈ 23万像素)，
    # 远低于处理器上限，导致调 AD_VLM_MAX_PX 完全不生效。要提细节必须同时放大这里。
    # 默认 448（不是 1024）：生产靠 systemd 覆盖成 448，代码默认值必须是安全值 ——
    # 1024 + 多帧在 12G 卡上会 OOM，还可能连锁触发 VLM 静默降级到 2B（2026-09-21 实测）。
    _thumb = int(os.environ.get("AD_VLM_THUMB", "448"))
    _n = int(os.environ.get("AD_CLIP_PICK", "8"))   # 段内送 VLM 帧数（安全默认 8，配合 448px）
    rep = sample_representative(frame_paths, _n)
    images = []
    _kept = []   # 与 images 一一对应的原始路径（检测摘要按它对齐）
    for p in rep:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((_thumb, _thumb))
            images.append(im)
            _kept.append(p)
        except Exception:
            continue
    if not images:
        return None, ""

    _prompt = build_clip_prompt(len(images))
    if extra_note:      # 定向重问：上一轮的理由缺具体画面依据，带一句话再问一次
        _prompt += "\n\n【重写要求】" + str(extra_note) + "\n"   # 帧号必须按实际送进去的帧数生成（原来写死 5 帧）
    if det_by_path:
        _notes = []
        for _i, p in enumerate(_kept, 1):
            _n = (det_by_path.get(p) or "").strip()
            _notes.append("Frame %d: %s" % (_i, _n or "检测器未输出目标"))
        _prompt += (
            "\n\n附：目标检测器(YOLO)对上述各帧的输出：\n" + "\n".join(_notes) + "\n"
            "强制规则：objects 与 events 的判断必须与上述检测结果一致——\n"
            "某类目标在任何 Frame 都未被检测到时，禁止报告涉及该目标的事件；\n"
            "报告「行人横穿/非机动车横穿」要求至少两个 Frame 检测到该目标且画面位置明显移动；\n"
            "检测与你的观察冲突时，以检测为准并在 evidence 中说明。\n"
        )

    conversation = [{
        "role": "user",
        "content": [{"type": "image"}] * len(images)
                   + [{"type": "text", "text": _prompt}],
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
        _sc = _json_stop_criteria(int(inputs.input_ids.shape[1]))
        if _sc is not None:
            try:
                # 判据要用 tokenizer 解码已生成的文本 → 把 tokenizer 挂上去
                for _c in _sc:
                    _c.tok = vlm_processor.tokenizer
            except Exception:
                _sc = None
        _kw = {"stopping_criteria": _sc} if _sc is not None else {}
        out = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=float(os.environ.get("AD_VLM_REP_PENALTY", "1.05")),
            **_kw,
        )
    out = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    raw = vlm_processor.batch_decode(out, skip_special_tokens=True)[0]
    return parse_and_validate(raw), raw
