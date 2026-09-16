# -*- coding: utf-8 -*-
"""前端 HTML 的“安全压缩”脚本

用法：
    python make_compact_html.py <源.html> [目标.compact.html]
未给目标时，默认在源文件名后加 .compact。

安全规则：
- HTML 标签之间、CSS、JS 普通语句之间的空行：删除。
- 保留多行 <pre>/<textarea> 内容内部的空行（是内容）。
- 保留 <script> 内 JS “模板字符串(反引号)文字部分”内部的空行（是内容）。
- 注释(// 与 /* */)内部空行可安全删除，不影响语义。
不修改原文件，仅输出新文件。
"""
import re
import sys
import os


def resolve_paths(argv):
    base = argv[0] if len(argv) > 0 else r"C:\Users\zhouhuajian\Desktop\prod\front.html"
    if len(argv) > 1:
        dst = argv[1]
    else:
        dst = base.replace(".html", ".compact.html")
    return base, dst


def split_blocks(text):
    """把文件切成带标签类型的块: ('html'|'style'|'script'|'pre'|'textarea'|'comment', content)"""
    pattern = re.compile(
        r"(?is)(<!--.*?-->|<style\b.*?</style>|<script\b.*?</script>|<pre\b.*?</pre>|<textarea\b.*?</textarea>)"
    )
    blocks = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            blocks.append(("html", text[last:m.start()]))
        seg = m.group(1)
        tag = re.match(r"(?is)<!--", seg)
        if tag:
            blocks.append(("html_comment", seg))
        else:
            t = re.match(r"(?is)<(\w+)", seg).group(1).lower()
            blocks.append((t, seg))
        last = m.end()
    if last < len(text):
        blocks.append(("html", text[last:]))
    return blocks


def keep_blank_js_lines(js_text):
    """返回应保留空行的行号集合(1-based)。只保留模板字符串文字内部的空行。"""
    keep = set()
    i = 0
    n = len(js_text)
    line = 1
    # mode stack: each element is 'code' | 'tmpl'
    stack = ["code"]
    # track current template literal "text" state depth as boolean via stack top
    while i < n:
        ch = js_text[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        mode = stack[-1]
        if mode == "code":
            if ch == "`":
                stack.append("tmpl")
                i += 1
            elif ch == "'" or ch == '"':
                quote = ch
                i += 1
                while i < n:
                    c = js_text[i]
                    if c == "\\":
                        i += 2
                        continue
                    if c == "\n":  # 字符串不该跨行，保守处理
                        break
                    if c == quote:
                        i += 1
                        break
                    i += 1
            elif ch == "/" and i + 1 < n and js_text[i + 1] == "/":
                while i < n and js_text[i] != "\n":
                    i += 1
            elif ch == "/" and i + 1 < n and js_text[i + 1] == "*":
                i += 2
                while i < n and not (js_text[i] == "*" and i + 1 < n and js_text[i + 1] == "/"):
                    if js_text[i] == "\n":
                        line += 1
                    i += 1
                i += 2
            else:
                i += 1
        else:  # tmpl 文字
            if ch == "\\":
                i += 2
                continue
            if ch == "$" and i + 1 < n and js_text[i + 1] == "{":
                stack.append("code")
                i += 2
                continue
            if ch == "`":
                stack.pop()
                i += 1
                continue
            # 模板文字中的新行处理：当前行整体属于模板文字
            if ch == "\n":
                line += 1
            i += 1

    # 上面的扫描需要记录“处于模板文字时经过的整行”。改进：二次扫描记录模板文字覆盖的行区间。
    # 这里改用逐行状态法重新精确求 keep 集合：
    return keep  # 占位，见 compute 函数


def compute_keep_lines_js(js_text):
    """精确计算模板字符串文字内部应保留空行的行号集合(1-based)。

    用栈正确处理模板字符串的嵌套与 ${...} 表达式：
      - 栈元素 ("code", in_expr, depth) 或 ("text",)
      - in_expr: 该 code 位于某模板的 ${...} 表达式内
      - depth: 表达式内额外嵌套的 { } 深度（用于找到闭合表达式的那一个 }）
    """
    keep = set()
    i = 0
    n = len(js_text)
    line = 1
    stack = [("code", False, 0)]
    while i < n:
        ch = js_text[i]
        kind = stack[-1][0]
        if kind == "code":
            _, in_expr, depth = stack[-1]
            if ch == "`":
                stack.append(("text",))
                i += 1
            elif ch == "'" or ch == '"':
                quote = ch
                i += 1
                while i < n:
                    c = js_text[i]
                    if c == "\\":
                        i += 2
                        continue
                    if c == "\n":
                        break
                    if c == quote:
                        i += 1
                        break
                    i += 1
            elif ch == "/" and i + 1 < n and js_text[i + 1] == "/":
                while i < n and js_text[i] != "\n":
                    i += 1
            elif ch == "/" and i + 1 < n and js_text[i + 1] == "*":
                i += 2
                while i < n and not (
                    js_text[i] == "*" and i + 1 < n and js_text[i + 1] == "/"
                ):
                    if js_text[i] == "\n":
                        line += 1
                    i += 1
                i += 2
            elif ch == "\n":
                line += 1
                i += 1
            elif in_expr and ch == "{":
                stack[-1] = ("code", True, depth + 1)
                i += 1
            elif in_expr and ch == "}":
                if depth == 0:
                    # 该 } 闭合了 ${...} 表达式，回到外层模板文本
                    stack.pop()
                    i += 1
                else:
                    stack[-1] = ("code", True, depth - 1)
                    i += 1
            else:
                i += 1
        else:  # text：模板文字
            if ch == "\\":
                i += 2
                continue
            if ch == "$" and i + 1 < n and js_text[i + 1] == "{":
                stack.append(("code", True, 0))
                i += 2
                continue
            if ch == "`":
                stack.pop()
                i += 1
                continue
            if ch == "\n":
                # 当前行(line)属于模板文字 → 若为空行则保留（是内容）
                keep.add(line)
                line += 1
                i += 1
            else:
                i += 1
    return keep


def compress(text, is_js):
    """压缩一段文本的空行。is_js=True 时通过 compute_keep 保留模板文字空行。
    返回压缩后文本。"""
    if is_js:
        keep = compute_keep_lines_js(text)
        lines = text.split("\n")
        out = []
        for idx, ln in enumerate(lines, start=1):
            if ln.strip() == "" and idx not in keep:
                continue
            out.append(ln)
        return "\n".join(out)
    else:
        # HTML / CSS / pre / textarea / comment:
        # pre 与 textarea 由调用方决定是否保留内部空行
        lines = text.split("\n")
        out = [ln for ln in lines if ln.strip() != ""]
        return "\n".join(out)


def main():
    global SRC, DST
    SRC, DST = resolve_paths(sys.argv[1:])
    with open(SRC, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = split_blocks(text)
    out_parts = []
    for tag, seg in blocks:
        if tag == "script":
            out_parts.append(compress(seg, is_js=True))
        elif tag in ("pre", "textarea"):
            # 保留内部内容空行：按行压缩仅删除“标签行之外没有内容的连续空行”复杂，
            # 稳妥起见：pre/textarea 整段不动（它们通常很短或为运行时空）。
            out_parts.append(seg)
        elif tag in ("style", "html", "html_comment"):
            out_parts.append(compress(seg, is_js=False))
        else:
            out_parts.append(compress(seg, is_js=False))
    result = "".join(out_parts)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(result)
    old_blank = sum(1 for ln in text.split("\n") if ln.strip() == "")
    new_lines = result.split("\n")
    new_blank = sum(1 for ln in new_lines if ln.strip() == "")
    print(f"完成：{DST}")
    print(f"原总行数: {len(text.splitlines())}  压缩后总行数: {len(new_lines)}")
    print(f"原空行: {old_blank}  压缩后空行: {new_blank}  删除: {old_blank - new_blank}")


if __name__ == "__main__":
    main()
