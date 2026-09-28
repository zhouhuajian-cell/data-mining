# 干什么：回归验证 app.py 的帧图本地热缓存（_hot_path / _hot_prune）——它必须"只让读变快，绝不让读失败"。
# 怎么跑（**在 Linux/服务器上跑**，因为缓存路径的语义是 POSIX 绝对路径）：
#   cd /tmp && /venv/bin/python _t_frame_hot.py [假的网盘根，默认 /mnt]
#   配套：同目录放 _hot_block.py（从 app.py 抽出的热缓存代码块；本地用 tools 里的生成器产出）
# 怎么判定成功：末尾打印 "全部通过"。
# ⚠️ Windows 上跑不了缓存那几条：缓存根 + 绝对路径在 Windows 会带盘符（这正是被包含性校验拦住的场景），
#    所以本文件在 nt 上只跑"关闭/透传/回退"三条，缓存与淘汰三条显式跳过。
import io, os, sys, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt"
NT = os.name == "nt"

blk = os.path.join(HERE, "_hot_block.py")
if os.path.exists(blk):
    block = io.open(blk, encoding="utf-8").read()
    src_desc = "_hot_block.py"
else:
    src = io.open(os.path.join(HERE, "app.py"), encoding="utf-8").read().split("\n")
    i0 = next(i for i, l in enumerate(src) if l.startswith("# 帧图本地热缓存"))
    i1 = next(i for i, l in enumerate(src) if l.startswith("def _hot_path"))
    i2 = next(i for i, l in enumerate(src[i1 + 1:], i1 + 1) if l and not l[0].isspace())
    block = "\n".join(src[i0:i2])
    src_desc = "app.py 第 %d~%d 行" % (i0 + 1, i2)
print("代码块来源: %s（%d 行）｜ 假网盘根: %s｜平台: %s" % (src_desc, len(block.split("\n")), ROOT, os.name))

fails = []
def chk(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)

tmp = tempfile.mkdtemp(prefix="hottest_")
nas = os.path.join(tmp, "nas")                 # 假网盘根（POSIX 下就是 ROOT 同形态）
os.makedirs(os.path.join(nas, "Data_Platform", "projects", "p1", "images", "bucketA"))
f1 = os.path.join(nas, "Data_Platform", "projects", "p1", "images", "bucketA", "a.jpg")
f2 = os.path.join(nas, "Data_Platform", "projects", "p1", "images", "bucketA", "b.jpg")
for p, tag in ((f1, b"X"), (f2, b"Y")):
    with open(p, "wb") as f:
        f.write(tag * 2048)
cache = os.path.join(tmp, "cache")

def load(env_on, gb="200"):
    ns = {"os": os, "shutil": shutil}
    if env_on:
        os.environ["AD_FRAME_HOT"] = cache
    else:
        os.environ.pop("AD_FRAME_HOT", None)
    os.environ["AD_FRAME_HOT_GB"] = gb
    exec(compile(block, "app.py(hot)", "exec"), ns)
    ns["_FRAME_HOT_PREFIX"] = (nas + os.sep,)
    return ns

print("\n[1] 关闭时（默认）：必须与改动前完全一致")
ns = load(False)
chk(ns["_hot_path"](f1) == f1, "未设 AD_FRAME_HOT → 原样返回")
chk(not os.path.isdir(cache), "未设 AD_FRAME_HOT → 不创建缓存目录")

print("\n[2] 非网盘路径：原样返回（本地路径不该被搬来搬去）")
ns = load(True)
local = os.path.join(tmp, "local.jpg")
open(local, "wb").write(b"Z" * 16)
chk(ns["_hot_path"](local) == local, "本地路径原样返回")

print("\n[3] 源文件不存在：必须回退原路径，不能抛异常")
missing = os.path.join(nas, "Data_Platform", "nope.jpg")
try:
    chk(ns["_hot_path"](missing) == missing, "读不到时回退原路径（不抛异常）")
except Exception as e:
    chk(False, "抛异常了：%r" % (e,))

if NT:
    print("\n[4][5][6] Windows 下跳过（缓存根 + 绝对路径带盘符，需在 Linux 上验证）")
else:
    print("\n[4] 开启时：未命中先搬一份，内容一致、源文件不动")
    r1 = ns["_hot_path"](f1)
    chk(r1 != f1, "返回缓存路径（不是原路径）")
    chk(os.path.abspath(r1).startswith(os.path.abspath(cache) + os.sep), "缓存路径落在缓存根内（无逃逸）")
    chk(os.path.exists(r1) and open(r1, "rb").read() == open(f1, "rb").read(), "缓存内容与原文件逐字节一致")
    chk(os.path.exists(f1), "源文件仍在（缓存只读不删源）")
    chk(not any(".tmp" in x for x in os.listdir(os.path.dirname(r1))), "临时文件已清理（os.replace 生效）")

    print("\n[5] 命中时：直接给本地路径，源删了也读得到")
    os.rename(f1, f1 + ".gone")
    chk(ns["_hot_path"](f1) == r1, "第二次仍返回同一缓存路径（源已不在也命中）")
    os.rename(f1 + ".gone", f1)

    print("\n[6] 淘汰：超预算按最久未访问删，预算生效，且不碰源")
    ns = load(True, gb="0.000001")            # 约 1KB，装不下 2KB×2
    ns["_hot_path"](f1)
    ns["_hot_path"](f2)
    ns["_HOT_STATE"]["added"] = 3 * 1024 ** 3  # 越过 2GB 阈值才触发统计
    ns["_hot_prune"]()
    kept = [os.path.join(r, x) for r, _d, fs in os.walk(cache) for x in fs]
    total = sum(os.path.getsize(p) for p in kept)
    chk(total <= 1024 ** 2, "淘汰后总量回到预算附近（实际 %d 字节）" % total)
    chk(os.path.exists(f1) and os.path.exists(f2), "淘汰只删缓存，源文件不受影响")

shutil.rmtree(tmp, ignore_errors=True)
print("\n%s" % ("全部通过" if not fails else "有 %d 条 FAIL：%s" % (len(fails), fails)))
sys.exit(0 if not fails else 1)
