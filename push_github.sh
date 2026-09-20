#!/usr/bin/env bash
# 一键同步：本地开发目录(prod) -> GitHub 仓库(data_mining) -> push
#
# 背景：开发在 C:\Users\zhouhuajian\Desktop\prod，而 git 仓库是另一个目录
# C:\Users\zhouhuajian\Desktop\data_mining（public: zhouhuajian-cell/data-mining）。
# prod 不是 git 仓库，所以每次推送前都要先把文件同步过去 —— 这个脚本把这三步做掉。
#
# 用法：
#   ./push_github.sh                      # 自动生成提交信息（带时间戳）
#   ./push_github.sh "fix: 修复xxx"        # 自定义提交信息
#   GITHUB_REPO=/d/other/repo ./push_github.sh   # 指定别的仓库位置
#
# 只同步"核心代码/文档"与"生产脚本"，**绝不用 git add -A** —— prod 里有二十多个
# 一次性对照实验脚本（_ab*.py/_bench*.py/_layout*.py…），那些不该进仓库。
set -euo pipefail

PROD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${GITHUB_REPO:-/c/Users/zhouhuajian/Desktop/data_mining}"

if [ ! -d "$REPO/.git" ]; then
  echo "❌ 找不到 git 仓库: $REPO" >&2
  echo "   可用 GITHUB_REPO=/path/to/repo 指定" >&2
  exit 1
fi

# 核心文件（有改动才同步）
CORE_FILES=(
  app.py front.html
  db_service.py db_patch.py models.py gpu_patch.py
  vlm_clip.py clip_service.py
  settings.py sampling.py decision_engine.py tag_system.py
  dedup.py exporter.py benchmark_engine.py check_vlm_env.py
  storage.py nas_local.py
  AGENTS.md compress_projects.txt requirements.txt
)
# 生产脚本：本地 _xxx.py -> 仓库 xxx.py（下划线前缀是本地临时脚本的命名习惯）
SCRIPT_MAP=(
  "_auto_ingest.py:auto_ingest.py"
  "_daily_backup.py:daily_backup.py"
  "_restore_eu.py:restore_index_from_db.py"
  "push_github.sh:push_github.sh"
)

cd "$REPO"
echo "== 同步 $PROD -> $REPO =="
changed=0
for f in "${CORE_FILES[@]}"; do
  if [ -f "$PROD/$f" ]; then
    if [ ! -f "$f" ] || ! cmp -s "$PROD/$f" "$f"; then
      cp "$PROD/$f" "$f"; echo "  更新 $f"; changed=1
    fi
  fi
done
for m in "${SCRIPT_MAP[@]}"; do
  src="${m%%:*}"; dst="${m##*:}"
  if [ -f "$PROD/$src" ]; then
    if [ ! -f "$dst" ] || ! cmp -s "$PROD/$src" "$dst"; then
      cp "$PROD/$src" "$dst"; echo "  更新 $dst（来自 $src）"; changed=1
    fi
  fi
done

if [ "$changed" -eq 0 ]; then
  echo "✅ 没有需要同步的改动"
  git status -sb | head -3
  exit 0
fi

# 只暂存上面明确同步的文件（不用 -A）
for f in "${CORE_FILES[@]}"; do [ -f "$f" ] && git add "$f"; done
for m in "${SCRIPT_MAP[@]}"; do
  dst="${m##*:}"
  [ -f "$dst" ] && git add "$dst"
done

MSG="${1:-sync: 本地同步 $(date '+%Y-%m-%d %H:%M')}"
if git diff --cached --quiet; then
  echo "✅ 内容与仓库一致（无新提交）"
  exit 0
fi
git -c core.safecrlf=false commit -q -m "$MSG"
echo "→ 推送到 GitHub…"
git push origin main
echo "✅ 完成：$(git log --oneline -1)"
