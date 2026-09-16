import os
import glob
from PIL import Image
import concurrent.futures
import time

# 兼容旧版 Pillow：9.1 之前用 Image.LANCZOS，新版本用 Image.Resampling.LANCZOS
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS

# 只压缩列表配置文件：每行一个"项目/目录关键字"（可含 # 开头注释行）。
# 为空 = 压缩全部；非空 = 只压缩路径里包含任一关键字的项目(整批)，其余全部跳过。
# 用于：只想压指定的归档项目、跳过其它(活跃检测)项目。
INCLUDE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compress_projects.txt")


def load_include_keywords():
    """读取持久化"只压缩"关键字列表"""
    if not os.path.exists(INCLUDE_FILE):
        return []
    try:
        with open(INCLUDE_FILE, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f
                    if ln.strip() and not ln.strip().startswith("#")]
    except Exception as e:
        print(f"[!] 读取只压缩列表失败: {e}")
        return []


def save_include_keywords(keywords):
    """把只压缩关键字列表写回配置文件"""
    try:
        with open(INCLUDE_FILE, "w", encoding="utf-8") as f:
            f.write("# 每行一个要压缩的项目/目录关键字，可随时编辑此文件\n")
            f.write("# 为空 = 压缩全部；非空 = 只压缩包含这些关键字的项目\n")
            for k in keywords:
                f.write(k + "\n")
    except Exception as e:
        print(f"[!] 写入只压缩列表失败: {e}")

# ================= 默认配置 =================
DEFAULT_INPUT_DIR = "./raw_images"         # 原始高分辨率图片所在文件夹
DEFAULT_OUTPUT_DIR = "./compressed_images" # 压缩后图片的保存路径
DEFAULT_MAX_EDGE = 1280                    # 长边最大限制 (兼容 SigLIP，保证 DINO 检测精度)
DEFAULT_JPEG_QUALITY = 82                  # JPEG 保存质量 (82 是画质与极小体积的完美平衡点)
DEFAULT_MAX_WORKERS = 8                    # 并发线程数 (根据你的电脑 CPU 核心数调整)
# ===========================================

def ask_with_default(prompt, default):
    """交互式输入，直接回车则使用默认值"""
    value = input(f"{prompt} [默认: {default}]: ").strip()
    return value if value != "" else default


def setup_config():
    """通过交互式问答收集本次运行所需的配置"""
    print("\n===== 图片压缩配置向导 =====")
    print("(直接按回车可接受方括号中的默认值)\n")

    input_dir = ask_with_default("请输入原始图片文件夹路径", DEFAULT_INPUT_DIR)

    # 先问是否原地覆盖；仅当不覆盖(输出到独立目录)时才询问输出路径
    inplace = False
    q = input("是否原地覆盖源图片? (y=直接覆盖原图 / 回车=输出到独立目录): ").strip().lower()
    if q in ("y", "yes", "是"):
        inplace = True

    output_dir = DEFAULT_OUTPUT_DIR
    if not inplace:
        output_dir = ask_with_default("请输入压缩图片保存路径", DEFAULT_OUTPUT_DIR)

    # 逐个让用户确认数值型参数（回车使用默认）
    max_edge = DEFAULT_MAX_EDGE
    q = input(f"请输入长边最大限制(像素) [默认: {DEFAULT_MAX_EDGE}]: ").strip()
    if q:
        max_edge = int(q)

    jpeg_quality = DEFAULT_JPEG_QUALITY
    q = input(f"请输入 JPEG 保存质量(1-100) [默认: {DEFAULT_JPEG_QUALITY}]: ").strip()
    if q:
        jpeg_quality = int(q)

    max_workers = DEFAULT_MAX_WORKERS
    q = input(f"请输入并发线程数 [默认: {DEFAULT_MAX_WORKERS}]: ").strip()
    if q:
        max_workers = int(q)

    # ---- 只压缩项目管理（持久化到 compress_projects.txt）----
    include_kws = load_include_keywords()
    if include_kws:
        print(f"  当前只压缩的项目/目录关键字: {', '.join(include_kws)}（其它全部跳过）")
    else:
        print("  当前未设只压缩列表 → 将压缩 input_dir 下全部图片")
    q = input("要压缩的项目/目录关键字(只压这些)，多个用英文逗号分隔；\n"
              "  直接回车=保持不变；输入 清空 可改为压缩全部: ").strip()
    if q.strip().lower() in ("清空", "clear", "空", "none"):
        include_kws = []
        save_include_keywords([])
        print("  ✔ 已清空只压缩列表 → 将压缩全部图片")
    elif q:
        new_kws = [k.strip() for k in q.replace('，', ',').split(',') if k.strip()]
        # 与已有关键字合并去重后持久化
        include_kws = list(dict.fromkeys(include_kws + new_kws))
        save_include_keywords(include_kws)
        print(f"  ✔ 已更新只压缩列表 -> {INCLUDE_FILE}")

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "max_edge": max_edge,
        "jpeg_quality": jpeg_quality,
        "max_workers": max_workers,
        "inplace": inplace,
        "include_kws": include_kws,
    }


def ask_continue():
    """询问用户是否继续处理下一批"""
    answer = input("\n是否继续处理下一批图片? (y/n): ").strip().lower()
    return answer in ("y", "yes", "是", "")

def process_image(args):
    img_path, out_path, config = args
    max_edge = config["max_edge"]
    jpeg_quality = config["jpeg_quality"]
    try:
        # 若目标子目录不存在则自动创建（用于保留原目录结构）
        # exist_ok=True：多线程并发时同一子目录可能被多个线程同时尝试创建，避免 [Errno 17] File exists
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # 增量处理：如果压缩过的文件已存在，则自动跳过 (支持随时中断和断点续传)
        if os.path.exists(out_path):
            return True

        with Image.open(img_path) as img:
            # 统一转换为 RGB，防止读取到 PNG/RGBA 等格式时报错
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size

            # 核心逻辑：仅在图片最长边超过限制时才进行等比缩小
            if max(width, height) > max_edge:
                scale = max_edge / float(max(width, height))
                new_width = int(width * scale)
                new_height = int(height * scale)

                # 必须强制使用 LANCZOS 高质量滤波抗锯齿，保留远端小目标（如锥桶、文字）边缘不模糊
                img = img.resize((new_width, new_height), _LANCZOS)

            # 开启 optimize=True 进行极限瘦身，丢弃文件里的冗余头信息
            img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)

        return True
    except Exception as e:
        print(f"处理文件 {img_path} 失败: {e}")
        return False


def process_image_inplace(args):
    """原地覆盖模式：把长边超过 max_edge 的 JPEG 源图直接压缩并覆盖原文件。
    安全策略：
      - 只对 .jpg/.jpeg 做原地覆盖（同格式重写才安全）；
        PNG/BMP/WebP 会改变格式，直接覆盖会损坏文件，故返回 None 跳过。
      - 先写同目录临时文件，成功后再用 os.replace 原子替换原文件，
        万一中途失败/出错，原图仍在，不会丢失高清底片。
    返回：True=成功/无需压缩, False=失败, None=不适用(非JPEG，跳过)
    """
    img_path, _out_path, config = args
    max_edge = config["max_edge"]
    jpeg_quality = config["jpeg_quality"]

    ext = os.path.splitext(img_path)[1].lower()
    if ext not in ('.jpg', '.jpeg'):
        return None  # 非 JPEG，跳过（避免把 PNG/BMP 错写成 .jpg 内容）

    tmp_path = img_path + ".tmp.jpg"
    try:
        with Image.open(img_path) as img:
            # 统一转换为 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            # 长边未超限：无需缩放，原地再压只会损失画质，直接跳过
            if max(width, height) <= max_edge:
                return True

            scale = max_edge / float(max(width, height))
            new_width = int(width * scale)
            new_height = int(height * scale)
            # LANCZOS 高质量滤波抗锯齿，保留远端小目标（锥桶、文字）边缘不模糊
            img = img.resize((new_width, new_height), _LANCZOS)
            img.save(tmp_path, format="JPEG", quality=jpeg_quality, optimize=True)

        # with 块退出后原文件句柄已关闭，再用原子替换覆盖原文件
        os.replace(tmp_path, img_path)
        return True
    except Exception as e:
        print(f"处理文件 {img_path} 失败: {e}")
        # 出错时清理可能残留的临时文件，避免污染目录
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def main():
    # 运行结束后询问是否继续处理下一批，形成可循环的交互式流程
    while True:
        config = setup_config()

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        max_workers = config["max_workers"]
        inplace = config["inplace"]

        # 输入目录不存在则提示并重新配置
        if not os.path.exists(input_dir):
            print(f"❌ 输入目录不存在: {input_dir}，请检查路径后重试。")
            if not ask_continue():
                break
            continue

        # 独立目录模式下：输出文件夹不存在则自动创建
        if not inplace and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if inplace:
            print(f"🔁 原地覆盖模式已开启：直接覆盖源 JPEG（仅长边超过 {config['max_edge']}px 的才压缩；PNG/BMP/WebP 自动跳过，原图保留）")
        else:
            print(f"📤 独立目录模式：压缩结果保存到 {output_dir}（原图保持不变）")

        # 递归扫描 input_dir 下所有子文件夹中的常见图片格式
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        include_kws = config.get("include_kws", [])
        image_files = []
        filtered_count = 0
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                # 匹配任意大小写的图片后缀
                if os.path.splitext(f)[1].lower() in extensions:
                    img_path = os.path.join(root, f)
                    # 只压缩模式：非空时仅压缩路径含任一关键字的项目，其余全部跳过
                    if include_kws and not any(k in img_path for k in include_kws):
                        filtered_count += 1
                        continue
                    if inplace:
                        # 原地覆盖：目标就是原文件本身
                        out_path = img_path
                    else:
                        # 在输出目录中保留相对子目录结构，避免同名文件互相覆盖
                        rel_dir = os.path.relpath(root, input_dir)
                        name, _ = os.path.splitext(f)
                        if rel_dir == ".":
                            out_path = os.path.join(output_dir, f"{name}.jpg")
                        else:
                            out_path = os.path.join(output_dir, rel_dir, f"{name}.jpg")
                    image_files.append((img_path, out_path))

        total_files = len(image_files)
        if include_kws:
            print(f"🎯 只压缩模式：仅处理含关键字({', '.join(include_kws)})的图片，已跳过 {filtered_count} 张")
        if total_files == 0:
            print(f"⚠️ 在 {input_dir} 目录(含所有子目录)下没有找到需压缩的图片。")
            if not ask_continue():
                break
            continue

        print(f"\n✅ 递归找到 {total_files} 张待处理图片，开始执行极限压缩 (Downscaling)...")
        start_time = time.time()

        # 启用多线程池加速批处理
        processed_count = 0
        worker = process_image_inplace if inplace else process_image
        args_list = [(img_path, out_path, config) for img_path, out_path in image_files]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 将所有图片提交给线程池
            futures = [executor.submit(worker, a) for a in args_list]

            # 实时监控进度
            for future in concurrent.futures.as_completed(futures):
                if future.result() is True:
                    processed_count += 1
                # 每处理 100 张打印一次进度，防止控制台刷屏
                if processed_count % 100 == 0:
                    print(f"➡️ 进度: {processed_count} / {total_files}")

        end_time = time.time()
        print(f"\n🎉 批量压缩完成！共成功瘦身 {processed_count} 张图片。")
        print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒。")
        if inplace:
            print(f"📍 已在原目录内完成覆盖（处理失败/被跳过的请查看上方提示）。")
        else:
            print(f"📁 请前往 {output_dir} 文件夹查看压缩后的数据。")

        if not ask_continue():
            break

    print("👋 已退出，感谢使用！")


if __name__ == "__main__":
    main()