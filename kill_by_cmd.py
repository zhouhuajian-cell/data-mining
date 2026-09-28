#!/venv/bin/python
# -*- coding: utf-8 -*-
"""
用 python 查/杀进程：**避免 `pgrep -f xxx` 匹配到自己**（AGENTS.md 坑 #1，今晚已踩过两次）。
做法：遍历 ps，跳过自己与父进程，按"精确子串"匹配。
用法：/venv/bin/python _kill_by_cmd.py <子串> [子串2] ...；打印杀掉了哪些 PID。
"""
import os, signal, subprocess, sys

pats = sys.argv[1:]
if not pats:
    print("用法: _kill_by_cmd.py <子串>..."); sys.exit(2)
me = {os.getpid(), os.getppid()}
out = subprocess.check_output(["ps", "-eo", "pid,cmd"]).decode(errors="replace")
killed = []
for ln in out.splitlines()[1:]:
    parts = ln.strip().split(None, 1)
    if len(parts) < 2:
        continue
    try:
        pid = int(parts[0])
    except ValueError:
        continue
    cmd = parts[1]
    if pid in me:
        continue
    if "kill_by_cmd" in cmd or "ps -eo" in cmd:
        continue                      # 别去杀查询者自己
    if any(p in cmd for p in pats):
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append((pid, cmd[:110]))
        except Exception as e:
            print("  kill %d 失败: %s" % (pid, e))
for pid, cmd in killed:
    print("  kill %d  %s" % (pid, cmd))
print("共杀 %d 个" % len(killed))
