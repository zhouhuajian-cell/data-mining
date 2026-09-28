# 干什么：按 HF 仓库的文件清单**逐文件并发**下载（几万~十万个小文件的仓库：huggingface-cli 的 tqdm 会崩、单线程太慢）。
#         走 hf-mirror；支持断点续传、按字节校验、失败重试。
# 怎么跑：/venv/bin/python -u tools/_dl_hf_repo_tree.py <repo_id> <目标目录> [--workers 24] [--type dataset]
# 怎么判定成功：末尾打印 "完成：成功 N / 跳过 M / 失败 K"。
# ⚠️ 两个踩过的坑：
#   ① **必须带浏览器 UA**：hf-mirror 对 Python-urllib 默认 UA 直接 403（同一接口 curl 访问是 200，2026-09-24 实测）；
#   ② 列清单**必须翻页**：tree 接口一次只回 ~1000 条，10 万文件的仓库不翻页会"看起来只有几千个"。
#      优先用 huggingface_hub（自己处理分页），失败再手工翻页。
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import http.client
import ssl

_TLS = threading.local()


def _conn(host="hf-mirror.com"):
    """按（线程, 主机）复用 HTTPS 长连接：十万文件时，省掉的 TLS 握手就是主要提速来源。
    ⚠️ mirror 对文件请求会 302 跳到 CDN，所以连接要按主机分别缓存，并且必须**手动跟随重定向**
    （http.client 不像 urllib 那样自动跟随 —— 2026-09-24 实测：不跟随的话全部报 HTTP 302、一个都没下）。"""
    pool = getattr(_TLS, "pool", None)
    if pool is None:
        pool = {}
        _TLS.pool = pool
    c = pool.get(host)
    if c is None:
        c = http.client.HTTPSConnection(host, timeout=120, context=ssl.create_default_context())
        pool[host] = c
    return c


def _drop_conn(host=None):
    pool = getattr(_TLS, "pool", None) or {}
    for h in ([host] if host else list(pool)):
        c = pool.pop(h, None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
from concurrent.futures import ThreadPoolExecutor

MIRROR = "https://hf-mirror.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
PACE = float(os.environ.get("AD_DL_PACE", "0.15"))   # 每文件后的间隔秒数（防 429）
STAT = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0}
LOCK = threading.Lock()


def size_of(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return None


def list_files(repo, rtype):
    kind = "dataset" if rtype == "dataset" else "model"
    # ⚠️ 别用 HfApi(endpoint=MIRROR)：实测它仍去连 huggingface.co 并进入 20 次重试（本环境连不上，
    #    白等几分钟）。直接手工翻页，UA 必须像浏览器（Python 默认 UA 会被镜像 403）。
    # ⚠️ 必须带 recursive=true：不带的话接口只回**顶层**条目（images/、labels/ 只当目录一条），
    #    10 万文件的仓库会被静默漏成几个（2026-09-24 实测：顶层只有 8 条）。翻页靠 Link 头。
    out, url = [], "%s/api/%s/%s/tree/main?recursive=true" % (MIRROR, "datasets" if kind == "dataset" else "models", repo)
    for _page in range(2000):
        # ⚠️ 列清单也要重试：镜像/DNS 会偶发抽风（实测 "Temporary failure in name resolution" 直接把进程打死）
        data, link = None, ""
        for _try in range(6):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.loads(r.read())
                    link = r.headers.get("Link", "")
                break
            except Exception as e:
                print("      列清单失败（第 %d 次）: %s" % (_try + 1, str(e)[:70]), flush=True)
                time.sleep(10 * (_try + 1))
        if data is None:
            raise SystemExit("列清单反复失败，退出（可稍后重跑，已下文件会跳过）")
        out += [x for x in data if x.get("type") == "file"]
        m = [s for s in link.split(",") if 'rel="next"' in s]
        if not m:
            break
        url = m[0].split(";")[0].strip().strip("<>")
        # ⚠️ 镜像返回的 Link 头指向 huggingface.co（本环境连不上）→ 改写回镜像域名，否则第二页就断
        url = url.replace("https://huggingface.co", MIRROR).replace("http://huggingface.co", MIRROR)
        if len(out) % 20000 < 1000:
            print("      已列 %d 条…" % len(out), flush=True)
    return out


def fetch(repo, rtype, path, dst, sizes):
    if path.startswith(".") or path.endswith("/"):
        return
    want = sizes.get(path)
    cur = size_of(dst)
    if want is not None and cur == want:
        with LOCK:
            STAT["skip"] += 1
        return
    # 清单没给大小时（entry 里无 size）退化为"已存在且非空即跳过"，免得重下六万张
    if want is None and cur and cur > 0:
        with LOCK:
            STAT["skip"] += 1
        return
    host = "hf-mirror.com"
    rel = "/%s/%s/resolve/main/%s" % ("datasets" if rtype == "dataset" else "models", repo,
                                      urllib.parse.quote(path))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(6):
        try:
            cn = _conn(host)                   # 长连接：不重新握手
            cn.request("GET", rel, headers=UA)
            r = cn.getresponse()
            hops = 0
            while r.status in (301, 302, 303, 307, 308) and hops < 5:   # 手动跟随（含跳到 CDN）
                loc = r.getheader("Location") or ""
                u = urllib.parse.urlsplit(loc)
                if u.netloc:
                    host = u.netloc
                rel = u.path + (("?" + u.query) if u.query else "")
                cn = _conn(host)
                cn.request("GET", rel, headers=UA)
                r = cn.getresponse()
                hops += 1
            if r.status == 429:
                raise IOError("429 Too Many Requests")
            if r.status != 200:
                raise IOError("HTTP %d" % r.status)
            with open(dst, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            got = size_of(dst)
            if want is not None and got != want:
                raise IOError("大小不符 got=%s want=%s" % (got, want))
            with LOCK:
                STAT["ok"] += 1
                STAT["bytes"] += got or 0
            time.sleep(PACE)          # 节流：不睡就会把镜像每秒配额烧光 → 429
            return
        except Exception as e:
            _drop_conn()
            is429 = "429" in str(e)
            if attempt == 5:
                with LOCK:
                    STAT["fail"] += 1
                print("  ✗ %s: %s" % (path[:70], str(e)[:80]), flush=True)
            else:
                # ⚠️ 429=镜像限流：必须长退避（实测 24 并发 × 十万请求会被限流打死），
                #    短退避（3s）等于继续撞墙。
                time.sleep((30 if is429 else 3) * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=6, help="默认 6：太高会被 hf-mirror 限流(429)")
    ap.add_argument("--type", default="dataset", choices=["dataset", "model"])
    a = ap.parse_args()
    print("[清单] 拉取 %s 的文件列表…" % a.repo, flush=True)
    files = list_files(a.repo, a.type)
    sizes = {f["path"]: f.get("size") for f in files}
    total = sum(s for s in sizes.values() if s)
    print("[清单] %d 个文件%s" % (len(files), "，已知合计 %.2f GB" % (total / 1e9) if total else ""), flush=True)
    if not files:
        print("✗ 清单为空，停止"); sys.exit(2)
    t0 = time.time()
    done = [0]

    def task(f):
        fetch(a.repo, a.type, f["path"], os.path.join(a.dst, f["path"]), sizes)
        with LOCK:
            done[0] += 1
            if done[0] % 2000 == 0:
                el = time.time() - t0
                print("      %d/%d，成功 %d 跳过 %d 失败 %d，%.1f GB，%.0f 文件/秒，已 %.1f 分钟" % (
                    done[0], len(files), STAT["ok"], STAT["skip"], STAT["fail"],
                    STAT["bytes"] / 1e9, done[0] / max(el, 1e-6), el / 60), flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(task, files))
    print("\n完成：成功 %d / 跳过 %d / 失败 %d，合计 %.2f GB，耗时 %.1f 分钟" % (
        STAT["ok"], STAT["skip"], STAT["fail"], STAT["bytes"] / 1e9, (time.time() - t0) / 60))
    sys.exit(1 if STAT["fail"] else 0)


if __name__ == "__main__":
    main()
