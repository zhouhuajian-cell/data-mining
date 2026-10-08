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
  app.py front.html det_store.py
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
  "_merge_export_json.py:merge_export_json.py"
  "_daily_backup.py:daily_backup.py"
  "_restore_eu.py:restore_index_from_db.py"
  "_build_tsr_dataset.py:build_tsr_dataset.py"
  "_build_unified_dataset.py:build_unified_dataset.py"
  "_build_tsr7_v2.py:build_tsr7_v2.py"
  "_build_tsr7_v3.py:build_tsr7_v3.py"
  "_mine_ownframe_labels.py:mine_ownframe_labels.py"
  "_train_tsr.py:train_tsr.py"
  "_eval_checkpoints.py:eval_checkpoints.py"
  "_ab_models_ownframes.py:ab_models_ownframes.py"
  "_pr_curve_per_epoch.py:pr_curve_per_epoch.py"
  "_verify_sign_deploy.py:verify_sign_deploy.py"
  "_rerun_sea_all.py:rerun_project_all.py"
  "_clear_results_keep.py:clear_results_keep.py"
  "_dl_hf_repo_tree.py:dl_hf_repo_tree.py"
  "_fetch_ru_datasets.sh:fetch_ru_datasets.sh"
  "_extract_rtsd_official.sh:extract_rtsd_official.sh"
  "_unpack_rtsd_cleaned.py:unpack_rtsd_cleaned.py"
  "_cleanup_datasets.sh:cleanup_datasets.sh"
  "_stage_mapillary.sh:stage_mapillary.sh"
  "_shard_json_to_nas.py:shard_json_to_nas.py"
  "_kill_by_cmd.py:kill_by_cmd.py"
  "_t_frame_hot.py:t_frame_hot.py"
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
# ⚠️ 这台机器**直连 GitHub 会 SSL 握手失败**（2026-10-08 实测 schannel handshake 失败），
#    必须走本地代理；没装代理/换了端口时用 GIT_PROXY=... 覆盖，或设 GIT_PROXY= 关掉。
: "${GIT_PROXY:=http://127.0.0.1:7897}"
if [ -n "$GIT_PROXY" ]; then
  git -c http.proxy="$GIT_PROXY" -c https.proxy="$GIT_PROXY" push origin main
else
  git push origin main
fi
echo "✅ 完成：$(git log --oneline -1)"
