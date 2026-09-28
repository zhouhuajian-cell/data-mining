#!/bin/bash
# 干什么：下俄罗斯（俄标 GOST）交通标志数据集到 NAS 的 Data_Platform\Russia_test（用户指定路径）。用户 2026-09-24："去找俄罗斯数据集，下载下来"（优先级高）。
#         走 hf-mirror（本环境唯一可达的 HF 源）；单文件直链 + 断点续传。
# 怎么判定成功：每个包落盘大小 == HF 上声明的大小，末尾打印。
# ⚠️ 服务器 curl 是 7.68.0，**别加 --retry-all-errors**（7.71 才有，加了直接 rc=2 一个字节不下，已踩过）。
set -u
BASE=https://hf-mirror.com/datasets
DST=/mnt/Data_Platform/Russia_test   # 用户 2026-09-24 指定：92.168.20.206\Data_Platform\Russia_test
LOG=/opt/ad_mining/logs/ru_fetch.log
mkdir -p "$DST"
get() {  # get <url> <输出文件> <期望字节>
  local url="$1" out="$2" want="$3"
  local have=0; [ -f "$out" ] && have=$(stat -c %s "$out")
  if [ -n "$want" ] && [ "$have" = "$want" ]; then echo "  跳过 $(basename "$out")（已完整 $((want/1048576)) MB）" | tee -a "$LOG"; return; fi
  echo "  → $(basename "$out")（已 $((have/1048576)) MB / 期望 $((want/1048576)) MB）$(date '+%T')" | tee -a "$LOG"
  curl -L -C - --retry 6 --retry-delay 15 -m 21600 --speed-time 120 --speed-limit 10240 -o "$out" "$url" >> "$LOG" 2>&1
  local rc=$? got=$(stat -c %s "$out" 2>/dev/null || echo 0)
  echo "    rc=$rc 落盘 $((got/1048576)) MB $([ -n "$want" ] && [ "$got" = "$want" ] && echo OK || echo '✗ 与期望不符')" | tee -a "$LOG"
}
echo "=== 俄罗斯数据集下载开始 $(date '+%F %T') ===" | tee -a "$LOG"
# 顺序下（单连接）：并发去拉会被 hf-mirror 限流(429)，实测踩过
get "$BASE/StarLineResearch/Russian_Road_Signs_Dataset/resolve/main/road_signs_dataset.zip"      "$DST/starline_road_signs_dataset.zip"  7029097622
get "$BASE/StarLineResearch/Lane-Lines-Dataset/resolve/main/lane_lines_dataset.zip"             "$DST/starline_lane_lines_dataset.zip"  1435858117
get "$BASE/StarLineResearch/Roadwork_Cones_Dataset/resolve/main/roadwork_cones_dataset.zip"       "$DST/starline_roadwork_cones_dataset.zip" 5734504591
get "$BASE/StarLineResearch/Pillars-Dataset/resolve/main/pillars_dataset.zip"                     "$DST/starline_pillars_dataset.zip"     1720816967
get "$BASE/eleldar/rtsd_cleaned/resolve/main/data/train-00000-of-00001-1decd5882f482f23.parquet" "$DST/rtsd_cleaned.parquet"             58343345
echo "=== 完成 $(date '+%F %T') ===" | tee -a "$LOG"
du -sh "$DST" | tee -a "$LOG"; ls -la "$DST" | tee -a "$LOG"
