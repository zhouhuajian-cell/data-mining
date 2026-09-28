# 干什么：把 RTSD 清理版 parquet 里的裁切标志图导出成图片文件，**按标志类别分目录**（便于分类/看样）。
#         用户 2026-09-24："下的压缩包要解压，规整好"。
# 怎么跑：/venv/bin/python -u tools/_unpack_rtsd_cleaned.py <parquet> <输出目录>
# 怎么判定成功：打印 "导出 N 张 / parquet 内 M 行"，N==M；并统计类别目录数。
# ⚠️ 十万张小文件写到 CIFS：按类别分目录（不平铺），否则 listdir 会退化（坑 #47 同源）。
import os
import sys
import time

import pyarrow.parquet as pq


def safe(name):
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
    return keep[:48] or "unknown"


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    f = pq.ParquetFile(src)
    n = f.metadata.num_rows
    print("[parquet] %s：%d 行，列 %s" % (os.path.basename(src), n, f.schema_arrow.names), flush=True)
    done = skip = 0
    classes = {}
    t0 = time.time()
    for batch in f.iter_batches(batch_size=512):
        d = batch.to_pylist()
        for i, row in enumerate(d):
            img = row.get("image") or {}
            b = img.get("bytes") if isinstance(img, dict) else None
            if not b:
                skip += 1
                continue
            cls = safe(row.get("sign_class") or "unknown")
            sid = row.get("sign_id")
            path = img.get("path") if isinstance(img, dict) else None
            ext = os.path.splitext(path or "")[1] or ".jpg"
            d_dir = os.path.join(out, cls)
            os.makedirs(d_dir, exist_ok=True)
            name = "%s_%s%s" % (cls, sid, ext)
            dst = os.path.join(d_dir, name)
            if os.path.exists(dst):
                skip += 1
            else:
                with open(dst, "wb") as fh:
                    fh.write(b)
                done += 1
            classes[cls] = classes.get(cls, 0) + 1
            if done % 10000 == 0 and done:
                print("      已导出 %d（%.0f 张/秒）" % (done, done / max(time.time() - t0, 1e-6)), flush=True)
    print("[导出] 新写 %d / 跳过(已存在或无图) %d，共 %d 张，类别目录 %d 个，耗时 %.1f 分钟" % (
        done, skip, n, len(classes), (time.time() - t0) / 60))
    with open(os.path.join(out, "_classes_count.txt"), "w", encoding="utf-8") as fh:
        for k in sorted(classes):
            fh.write("%-46s %d\n" % (k, classes[k]))
    print("[清单] 已写 %s\\_classes_count.txt" % out)


if __name__ == "__main__":
    main()
