# 干什么：把官方标注包里的 JSON 按 id 首字符分片解到目标目录（服务端跑，写的是 NAS 挂载点）。
#   为什么要分片：几万个 JSON 挤一个目录会让 CIFS 的 listdir 退化到秒级（坑 #47 同源）。
# 怎么跑：/venv/bin/python -u /opt/ad_mining/tools/_shard_json_to_nas.py <标注zip> <目标目录> [--dry-run]
# 怎么判定成功：打印 "JSON 落盘 N 个（zip 内 M 个）"，且 N==M（已存在的跳过不算失败）。
import os
import sys
import time
import zipfile

SHARDS = "-_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def size_of(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return None


def main():
    zp, out = sys.argv[1], sys.argv[2]
    dry = "--dry-run" in sys.argv
    with zipfile.ZipFile(zp) as z:
        infos = [i for i in z.infolist() if i.filename.endswith(".json")]
        print("[json] %s：zip 内 %d 个" % (os.path.basename(zp), len(infos)))
        if dry:
            return
        if not os.path.isdir(out):
            print("✗ 目标不存在: %s" % out); sys.exit(2)
        for c in SHARDS:                      # 预先建好分片目录，避免每文件一次 mkdir 往返
            os.makedirs(os.path.join(out, c), exist_ok=True)
        n_new = n_skip = 0
        t0 = time.time()
        for i, info in enumerate(infos):
            fid = os.path.splitext(os.path.basename(info.filename))[0]
            dst = os.path.join(out, (fid[0] if fid else "_"), fid + ".json")
            if size_of(dst) == info.file_size:
                n_skip += 1
                continue
            with z.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            n_new += 1
            if (i + 1) % 5000 == 0:
                el = time.time() - t0
                print("      进度 %d/%d，新写 %d，跳过 %d，%.0fs" % (
                    i + 1, len(infos), n_new, n_skip, el), flush=True)
        print("[json] 落盘 %d 个（新写 %d / 跳过 %d），耗时 %.0fs" % (n_new + n_skip, n_new, n_skip, time.time() - t0))


if __name__ == "__main__":
    main()
