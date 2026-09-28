# 干什么：清理 /opt/datasets 下**确定没用**的数据包（用户 2026-09-28："服务器上没用数据包该清理就清理下"）。
#   判定口径（每条都先核对"有没有别的副本/还被谁引用"）：
#     ① TT100K_raw/nosign_*.zip（88G）—— 5 个"无标志背景图"包，图**已解压**在 /opt/datasets/tt100k_nosign（89G，已核 82,097 张）→ 包是冗余
#     ② TT100K_raw/data.zip（17.8G）—— TT100K **2016** 旧标注版，被 2021 版取代、从未使用
#     ③ unified（15G）—— 第一/二轮那个被否掉的 **18 类统一模型**数据集（实验废弃，权重另存）
#     ④ tsr7_v2（32G）—— 第三轮数据集；**权重保留**，数据集可由脚本 ~20 分钟重建（第四轮 tsr7_v3 是现役，保留）
#     ⑤ MTSD2（24G）—— 老的 MTSD 镜像；已被官方 MTSD 全量（/opt/datasets/tsr_more，构建器实际读的那份）取代
#   **不动**：tsr_more（官方 MTSD 包，构建器在读）、tsr_yolo、stage2、tsr7_v3（现役数据集）、tsr_runs（权重）、
#            images_export（导出给用户的副本）、TT100K_raw/tt100k_2021.zip（带标注的 TT100K 2021，唯一副本）、
#            negatives / VietnamSign / GermanSign / ownframes* / tsr_weights
# 怎么跑：bash tools/_cleanup_datasets.sh [--dry-run]
# 怎么判定成功：打印每项"删前大小 → 删后不存在"，末尾 df 对比；清单写到 backups/cleanup_<ts>.txt
set -u
D=/opt/datasets
BK=/opt/ad_mining/backups
TS=$(date +%Y%m%d_%H%M%S)
MAN=$BK/cleanup_$TS.txt
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="yes"
mkdir -p "$BK"
TARGETS="$D/TT100K_raw/nosign_1.zip $D/TT100K_raw/nosign_2.zip $D/TT100K_raw/nosign_3.zip \
$D/TT100K_raw/nosign_4.zip $D/TT100K_raw/nosign_5.zip $D/TT100K_raw/data.zip \
$D/unified $D/tsr7_v2 $D/MTSD2"
echo "=== 数据包清理 $(date '+%F %T') ${DRY:+（dry-run）} ===" | tee -a "$MAN"
BEFORE=$(df -h /opt | tail -1 | awk '{print $4}')
for t in $TARGETS; do
  if [ ! -e "$t" ]; then echo "  跳过（不存在）: $t" | tee -a "$MAN"; continue; fi
  sz=$(du -sh "$t" 2>/dev/null | cut -f1)
  n=$(find "$t" -type f 2>/dev/null | wc -l)
  if [ -n "$DRY" ]; then
    printf '  [dry-run] 将删 %-52s %-7s (%s 个文件)\n' "$t" "$sz" "$n" | tee -a "$MAN"
    continue
  fi
  rm -rf "$t" && printf '  已删 %-52s %-7s (%s 个文件)\n' "$t" "$sz" "$n" | tee -a "$MAN"
  [ -e "$t" ] && echo "    ✗ 删不掉，仍在！" | tee -a "$MAN"
done
AFTER=$(df -h /opt | tail -1 | awk '{print $4}')
echo "  可用空间: $BEFORE → $AFTER" | tee -a "$MAN"
echo "=== 清理结束 $(date '+%F %T') ===" | tee -a "$MAN"
echo "  清单: $MAN"
