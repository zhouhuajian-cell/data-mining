#!/bin/bash
# 干什么：把官方 MTSD（Mapillary）全量原始图片从 5 个 zip 解到本地暂存目录，供平台「直导」进 mapillary 项目。
#         `-j` 去掉 zip 内的 images/ 前缀 → 每个 split 一个目录，平台按目录名分桶（与抽帧一致，不平铺：
#         平铺几万张会把 CIFS 的 listdir 拖到秒级）。
# 怎么跑：bash tools/_stage_mapillary.sh（nohup 后台）
# 怎么判定成功：每个 split 的落盘数与 zip 内 .jpg 条目数对上，且"合计"= 52453。
set -u
SRC=/opt/datasets/tsr_more/crimedetector_roadsign
DST=/opt/datasets/mapillary_stage
LOG=/opt/ad_mining/logs/stage_mapillary.log
mkdir -p "$DST"
echo "=== 解包开始 $(date '+%F %T') ===" | tee -a "$LOG"
tot=0
for s in test train.0 train.1 train.2 val; do
  z="$SRC/mtsd_fully_annotated_images.$s.zip"
  if [ ! -f "$z" ]; then echo "  缺 $z" | tee -a "$LOG"; continue; fi
  want=$(unzip -l "$z" | awk '/\.jpg$/{n++} END{print n+0}')
  mkdir -p "$DST/$s"
  unzip -o -q -j "$z" -d "$DST/$s"
  got=$(ls "$DST/$s" | wc -l)
  echo "  $s: zip 内 $want / 落盘 $got $([ "$want" = "$got" ] && echo OK || echo '✗ 不一致')" | tee -a "$LOG"
  tot=$((tot+got))
done
echo "  合计 $tot（应为 52453）" | tee -a "$LOG"
du -sh "$DST" | tee -a "$LOG"
echo "=== 解包完成 $(date '+%F %T') ===" | tee -a "$LOG"
