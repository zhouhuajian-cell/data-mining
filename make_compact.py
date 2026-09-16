# -*- coding: utf-8 -*-
"""生成 app_latest.py 的“全面压紧”预览版 app_latest.compact.py。

规则：
- 删除所有“不在多行字符串内部”的空行（含 import 之间、函数/类之间）。
- 保留夹在 '''/\"\"\" 多行字符串内部的内容空行（那是字符串数据，删了会改变运行结果）。
- 不修改原文件，只输出新文件。
"""
import io
import tokenize

SRC = r"C:\Users\zhouhuajian\Desktop\prod\app_latest.py"
DST = r"C:\Users\zhouhuajian\Desktop\prod\app_latest.compact.py"


def in_string_lines(source):
    """返回集合：哪些行号(1-based)处于多行字符串(token 跨多行)内部，这些行的空行不能删。"""
    keep = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.STRING and "\n" in tok.string:
                start_row = tok.start[0]
                end_row = tok.end[0]
                # token 所在整段（含首尾行）都视作字符串内容
                for r in range(start_row, end_row + 1):
                    keep.add(r)
    except Exception:
        pass
    return keep


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.split("\n")
    keep_in_string = in_string_lines(source)

    out = []
    prev_blank = False
    for idx, line in enumerate(lines, start=1):
        is_blank = line.strip() == ""
        if is_blank:
            if idx in keep_in_string:
                # 字符串内部的空行：必须保留（属于内容）
                out.append(line)
            # 字符串外的空行：全部丢弃（prev_blank 逻辑不再需要，因为直接删）
            continue
        out.append(line)

    # 去掉可能残留的连续空行导致的文件开头/结尾多余换行，并保证末尾单换行
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()

    with open(DST, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"完成：{DST}")
    print(f"原文件行数: {len(lines)}  压缩后行数: {len(out)}")


if __name__ == "__main__":
    main()
