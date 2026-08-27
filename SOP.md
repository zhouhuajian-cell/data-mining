# SOP：从输入图片到交付场景（自动驾驶场景挖掘与标注）

## 一、整体链路概览

```
[视频] --视频抽帧.py--> [原始图片]
                        │
        ┌───────────────┴─────────────────┐
   方式A: 网页上传(前端.html+app.py)   方式B: 离线建库(build_index.py)
        └───────────────┬─────────────────┘
                  [workspace/ 三件套]
        index.faiss + metadata.json + features.npy
                        │
        app.py 引擎: 去重 → 语义检索 → 目标检测
                        │
                  [标注导出 JSON / ZIP]
                        │
            copy_valid.py → 有效/Clip初筛  (交付场景)
```

核心依赖：`CLIP ViT-L/14`（语义/去重）、`Grounding DINO tiny`（开放词汇检测）、
`YOLOv8x`（COCO 检测）、`FAISS`（向量库）。
运行环境见 `requirements.txt`，`yolov8x.pt` 需放在根目录。

---

## 二、分步 SOP

### 阶段 0 — 环境准备（一次性）
1. 安装依赖：`pip install -r requirements.txt`
2. 确认根目录有 `yolov8x.pt`（缺失时首次启动会拉取/报错）。
3. 首次启动 `app.py` 会从 HuggingFace（已设 `hf-mirror.com` 镜像）缓存 CLIP 与 DINO 权重。
4. 双击 `start.bat` 或 `python app.py`，浏览器开 `http://localhost:8008`。

### 阶段 1 — 输入图片生产（可选，从视频来）
- 脚本：`视频挖掘处理/视频抽帧.py`
- 把原始视频丢进 `./raw_videos`，运行 → 抽帧存到 `./dataset_frames/<视频名>/`。
- 双重触发：① 按 `fps_interval=1.0s` 定时抽帧；② 帧间灰度差 `>25` 时动态补抽（捕捉 Corner Case）。
- 产物命名：`视频名_f000123_t1234ms.jpg`。

### 阶段 2 — 图片入库（建向量底库）
两种方式二选一：

**方式 A（网页上传，适合小批量/交互）**
- `前端.html` 多选图片 → 50 张/批 `POST /api/upload_batch`。
- 后端逐张：解码 → JPEG q90 落盘 `workspace/images/` → CLIP 提 768 维特征(L2归一化)
  → `index.add()` → 写元数据 → 落盘三件套。超过 30000 张拒绝。

**方式 B（离线建库，适合 3 万+ 大批量，纯 CPU）**
- 把图片直接放入 `workspace/images/`，运行 `build_index.py`。
- 用 DataLoader 批处理(CPU BATCH_SIZE=8)，输出 `index.faiss / metadata.json / features.npy`。
- 之后启动 `app.py` 会自动检测到三件套并恢复底库（**无需重新上传**）。

> 辅助脚本：`rename.py`（去除旧 `id_` 文件名前缀并同步 metadata）；
> `sync_db.py`（按磁盘实体文件核对，清理丢失项的索引，重建三件套）。

### 阶段 3 — 去重分析
- 前端「去重分析」→ `POST /api/dedup_stats(sim_threshold=0.95)`。
- 全量特征矩阵乘得两两余弦相似度 → 并查集聚类 → 重复簇（每簇留首张）。
- 可导出冗余 ID 清单(TXT)；或「永久删除冗余并同步底库」`/api/delete_and_sync`
  （物理删图+重建 FAISS+落盘）。

### 阶段 4 — 语义挖掘/检索（找目标场景）
- 文本检索：`POST /api/search`（CLIP 文本编码→FAISS 内积近邻，前端按匹配度滑块≥0.20 过滤）。
- 其它：`/api/search_by_filename`（文件名子串）、`/api/search_by_image`（库内以图搜图）、
  `/api/search_by_external_image`（外部图搜底库）。
- 分页浏览：`/api/list_all`（`currentMode` = browse/search/filename/dedup）。

### 阶段 5 — 目标检测（打框）
- 选定目标集（当前页 / 检索全集 / 重复簇）→ 逐张 `POST /api/yolo_detect` 或 `/api/ground_detect`。
- YOLO：COCO 80 类；DINO：自然语言提示词开放词汇（自动补句号）。
- 前端按置信度滑块过滤 → 自动生成标签（如 `car 3`）→ 含有效目标的卡片自动勾选。

### 阶段 6 — 标注导出 / 交付（交付场景）
- 勾选目标卡片 → `exportData`：组装 JSON
  `{image_id, filename, file_path, score, annotations{boxes/labels/scores}, final_tags}` 浏览器下载。
- 或 `/api/export_images`：按 ID 打包所选原图为 ZIP（临时文件即用即删）。
- **交付归集**：`copy_valid.py` 读取导出的筛选 JSON（如 `AD_CLIP_RawSearch_...json`），
  在 `workspace/images` 匹配并复制到 `有效/Clip初筛/`，作为最终交付素材。

---

## 三、关键注意事项
- 不要手动增删 `workspace/images/` 内文件，否则 ID 错位；删图一律走「永久删除并同步」。
- 入库上限 30000 帧；单任务失败不中断整批。
- `build_index.py` 是纯 CPU 护肝模式；`app.py` 会自动用 GPU(FP16)/CPU(FP32)。
- `视频抽帧.py` 默认输入 `./raw_videos`、输出 `./dataset_frames`，需自行改路径或把视频放对应目录。

## 四、文件职责速查
| 文件 | 作用 |
|------|------|
| `视频挖掘处理/视频抽帧.py` | 视频→帧（阶段1） |
| `app.py` | FastAPI 引擎：上传/检索/去重/检测/导出（阶段2A,3-6） |
| `前端.html` | 单页前端，由 app.py 托管 |
| `build_index.py` | 离线批量建 FAISS 底库（阶段2B） |
| `sync_db.py` | 磁盘-索引核对重建 |
| `rename.py` | 旧文件名前缀迁移 |
| `copy_valid.py` | 导出 JSON→复制到 `有效/Clip初筛`（交付） |
| `start.bat` | 激活 .venv 启动服务 |
