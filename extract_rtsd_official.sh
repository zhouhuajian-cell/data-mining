#!/bin/bash
# 干什么：把 Yandex 官方 RTSD 的 .tar.lzma（detection d1/d2/d3 的 frames+gt、classification r1/r3）解包到同目录的子文件夹里，
#         供下一轮训练按 7 类映射使用（用户 2026-09-25："要不要把现在下载的数据再来一轮训练" → 需要先解出来）。
# 怎么跑：bash tools/_extract_rtsd_official.sh（nohup 后台；lzma 解压吃 CPU，不占 GPU，可与训练并行）
# 怎么判定成功：日志出现 "=== RTSD 解包完成 ==="，且各子目录里有 jpg 与 gt（csv/txt）。
# ⚠️ .tar.lzma 是**裸 LZMA**（不是 .xz），要用 `xz --format=lzma -dc` 或 unlzma；先探测有无工具。
set -u
R=/mnt/Data_Platform/Russia_test/rtsd_official
LOG=/opt/ad_mining/logs/rtsd_extract.log
cd "$R" || exit 1
TOOL=""
for t in xz unlzma lzma; do command -v "$t" >/dev/null 2>&1 && { TOOL="$t"; break; }; done
echo "=== RTSD 解包开始 $(date '+%F %T')，工具=$TOOL ===" | tee -a "$LOG"
[ -z "$TOOL" ] && { echo "  ✗ 没有 xz/unlzma/lzma，装包后再来" | tee -a "$LOG"; exit 2; }
dec() {  # dec <输入文件> <输出目录>
  local f="$1" d="$2"
  [ -f "$f" ] || { echo "  缺 $f" | tee -a "$LOG"; return; }
  mkdir -p "$d"
  if [ "$(find "$d" -type f 2>/dev/null | wc -l)" -gt 10 ]; then echo "  $d 已有内容，跳过" | tee -a "$LOG"; return; fi
  # ⚠️ 别信扩展名！官方文件叫 *.tar.lzma，但**实际是 gzip**（magic 1f 8b；xz 会报
  #    "File format not recognized"，2026-09-25 实测）。按文件头自动判型。
  local magic; magic=$(head -c 2 "$f" | od -An -tx1 | tr -d ' 
')
  echo "  → 解 $f（magic=$magic）$(date '+%T')" | tee -a "$LOG"
  case "$magic" in
    1f8b) gzip -dc "$f" | tar -x -C "$d" >> "$LOG" 2>&1 ;;
    5d00|fd37) xz --format=lzma -dc "$f" | tar -x -C "$d" >> "$LOG" 2>&1 ;;
    fd377a58) xz -dc "$f" | tar -x -C "$d" >> "$LOG" 2>&1 ;;
    *)    tar -xf "$f" -C "$d" >> "$LOG" 2>&1 ;;
  esac
  echo "     $(basename "$d"): $(find "$d" -type f | wc -l) 个文件，$(du -sh "$d" 2>/dev/null | cut -f1) $(date '+%T')" | tee -a "$LOG"
}
# ⚠️ 文件是**平铺**在 rtsd_official/ 下的（我的下载器按 basename 存），不在 detection/ 子目录里
dec rtsd-d1-frames.tar.lzma detection/d1_frames
dec rtsd-d1-gt.tar.lzma     detection/d1_gt
dec rtsd-d2-frames.tar.lzma detection/d2_frames
dec rtsd-d2-gt.tar.lzma     detection/d2_gt
dec rtsd-d3-frames.tar.lzma detection/d3_frames
dec rtsd-d3-gt.tar.lzma     detection/d3_gt
dec rtsd-r1.tar.lzma        classification/r1
dec rtsd-r3.tar.lzma        classification/r3
echo "=== RTSD 解包完成 $(date '+%F %T') ===" | tee -a "$LOG"
for d in detection/d1_frames detection/d1_gt detection/d2_frames detection/d2_gt detection/d3_frames detection/d3_gt classification/r1 classification/r3; do
  printf '  %-22s %6s 个文件\n' "$d" "$(find "$R/$d" -type f 2>/dev/null | wc -l)" | tee -a "$LOG"
done
