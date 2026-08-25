import cv2
import os
import glob
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_single_video(video_path, output_base_dir, fps_interval=1.0, diff_threshold=25.0):
    """
    处理单个视频的抽帧逻辑（双重触发机制）
    """
    # 为每个视频建一个独立的子文件夹，防止几万张图片堆在一起难以管理
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_path, 0, "Failed to open video"

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30.0 # 容错兜底
        
    frame_interval = int(video_fps * fps_interval)
    
    frame_count = 0
    saved_count = 0
    last_saved_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        save_this_frame = False

        # 1. 强制时间间隔抽帧 (兜底基础场景)
        if frame_count % frame_interval == 0:
            save_this_frame = True
        
        # 2. 突发场景动态抽帧 (捕捉高优 Corner Cases)
        elif last_saved_frame is not None:
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.cvtColor(last_saved_frame, cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(gray_curr, gray_last))
            
            if diff > diff_threshold:
                save_this_frame = True

        if save_this_frame:
            # 文件名带上视频源和毫秒时间戳，方便后续问题回溯
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            out_name = f"{video_name}_f{frame_count:06d}_t{timestamp_ms}ms.jpg"
            out_path = os.path.join(output_dir, out_name)
            
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            last_saved_frame = frame.copy()
            saved_count += 1

        frame_count += 1

    cap.release()
    return video_path, saved_count, "Success"

def batch_process_videos(input_dir, output_dir, max_workers=None):
    """
    多进程调度器
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 扫描目录下常见的视频格式
    video_files = []
    for ext in ('*.mp4', '*.avi', '*.mkv', '*.mov'):
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not video_files:
        print(f"在 {input_dir} 下没有找到视频文件！")
        return

    print(f"总共找到 {len(video_files)} 个视频文件，准备启动多进程抽帧...")
    start_time = time.time()
    total_extracted = 0

    # 使用 ProcessPoolExecutor 压榨多核性能
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_single_video, vf, output_dir): vf 
            for vf in video_files
        }
        
        # 实时打印子进程进度
        for future in as_completed(futures):
            v_path, count, status = future.result()
            v_name = os.path.basename(v_path)
            if status == "Success":
                print(f"✅ [完成] {v_name} -> 抽取 {count} 帧")
                total_extracted += count
            else:
                print(f"❌ [失败] {v_name} -> {status}")

    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"🎉 全部处理完毕！耗时: {elapsed:.2f} 秒")
    print(f"📊 总共提取图片: {total_extracted} 张")
    print(f"📁 存放目录: {output_dir}")

if __name__ == "__main__":
    # 配置好输入输出路径即可运行
    INPUT_VIDEOS_FOLDER = "./raw_videos"      # 把原始视频全扔进这个文件夹
    OUTPUT_FRAMES_FOLDER = "./dataset_frames" # 生成的图片会存到这里
    
    # max_workers 留空会自动使用机器所有的 CPU 核心
    batch_process_videos(INPUT_VIDEOS_FOLDER, OUTPUT_FRAMES_FOLDER)