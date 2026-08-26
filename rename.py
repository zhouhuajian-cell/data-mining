import os
import json

WORKSPACE_DIR = "./workspace"
IMG_DIR = os.path.join(WORKSPACE_DIR, "images")
META_PATH = os.path.join(WORKSPACE_DIR, "metadata.json")

def migrate_filenames():
    print("开始扫描底库...")
    if not os.path.exists(META_PATH):
        print(f"❌ 找不到 {META_PATH}，请确认当前路径是否正确。")
        return

    with open(META_PATH, "r", encoding="utf-8") as f:
        image_database = json.load(f)

    success_count = 0
    missing_count = 0

    for item in image_database:
        pure_filename = item["filename"]  # 原本就是干净的名字 (如: 1781258772.022.jpg)
        old_path = item["path"]           # 带有数字前缀的旧路径 (如: ./workspace/images/125_1781258772.022.jpg)
        
        # 构造纯净的新路径
        new_path = os.path.join(IMG_DIR, pure_filename)
        
        # 1. 物理重命名硬盘上的文件
        if old_path != new_path:
            if os.path.exists(old_path):
                # 为了防止多次运行报错，如果新文件已存在就先删掉
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(old_path, new_path)
                success_count += 1
            elif not os.path.exists(new_path):
                missing_count += 1
        else:
            # 已经没有前缀了
            success_count += 1
            
        # 2. 更新数据库中的路径记录，统一使用 / 斜杠
        item["path"] = new_path.replace("\\", "/")

    # 3. 覆盖保存 JSON 数据库
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(image_database, f, ensure_ascii=False)

    print("="*50)
    print("🎉 迁移大功告成！")
    print(f"✅ 成功去除前缀/更新路径: {success_count} 张")
    if missing_count > 0:
        print(f"⚠️ 物理文件已丢失: {missing_count} 张 (建议后续在网页端跑一次去重清理即可)")
    print("="*50)
    print("现在你可以使用最新的 app.py 启动系统了！")

if __name__ == "__main__":
    migrate_filenames()