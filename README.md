# 自动驾驶场景挖掘与标注引擎

基于 **CLIP + Grounding DINO + YOLOv8x** 三模型协同的多模态图像检索、去重、目标检测与标注导出系统。后端 FastAPI,前端纯原生 HTML/JS 单页应用,支持本地持久化(FAISS 向量库),单任务容量上限 **30,000 帧**。

---

## 目录

- [系统总览](#系统总览)
- [环境依赖](#环境依赖)
- [快速启动](#快速启动)
- [模型架构](#模型架构)
- [数据流处理](#数据流处理)
- [API 接口一览](#api-接口一览)
- [前端模块设计](#前端模块设计)
- [持久化与目录结构](#持久化与目录结构)

---

## 系统总览

```
┌─────────────────────────────────────────────────────────────┐
│                     前端.html (浏览器)                        │
│   上传入库 │ 去重分析 │ 多模态检索 │ 批量检测 │ 标注导出        │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (FormData / JSON)
┌──────────────────────────▼──────────────────────────────────┐
│                FastAPI 后端 (localhost:8008)                 │
│                                                              │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐   │
│  │ CLIP        │  │ Grounding DINO   │  │ YOLOv8x       │   │
│  │ ViT-L/14    │  │ tiny (FP16)      │  │ (COCO 80类)   │   │
│  └──────┬──────┘  └──────────────────┘  └───────────────┘   │
│         │                                                     │
│  ┌──────▼──────────────────────────────────────────────┐     │
│  │ FAISS IndexFlatIP (768维内积) + metadata.json       │     │
│  └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 环境依赖

| 组件 | 版本/说明 |
|------|-----------|
| Python | 3.10+ |
| torch | CUDA 可用时自动启用 GPU,否则回退 CPU |
| transformers | 加载 CLIP 与 Grounding DINO |
| ultralytics | YOLOv8x 推理 |
| faiss-cpu / faiss-gpu | 向量索引 |
| Pillow | 图像解码与落盘 |

安装:

```bash
pip install -r requirements.txt
```

> `yolov8x.pt` 权重需放置于项目根目录。

## 快速启动

```bash
python app.py
# 或双击 start.bat
```

浏览器访问 `http://localhost:8008`(后端直接托管 `前端.html`)。

首次启动会从 HuggingFace 拉取并缓存:
- `openai/clip-vit-large-patch14`
- `IDEA-Research/grounding-dino-tiny`

---

## 模型架构

### 1. CLIP ViT-L/14 —— 多模态语义骨干

| 项目 | 配置 |
|------|------|
| 模型 | `openai/clip-vit-large-patch14` |
| 嵌入维度 | **768** |
| 精度 | GPU 上 FP16 半精度(防爆显存),CPU 上 FP32 |
| 输出 | L2 归一化后的图像/文本特征向量 |

职责:
- **图像特征提取**:上传入库时对每帧计算图像 embedding;
- **文本→图像检索**:将查询文本编码为同空间向量,与 FAISS 库做内积(即余弦相似度)检索;
- **以图搜图**:对外部图片或库内图片提取特征后近邻检索;
- **去重**:基于库内已缓存的图像特征矩阵计算两两相似度。

### 2. Grounding DINO tiny —— 开放词汇检测

| 项目 | 配置 |
|------|------|
| 模型 | `IDEA-Research/grounding-dino-tiny` |
| 输入 | 图像 + 自然语言提示词(如 `"road sign."`,自动补句号) |
| 输出 | 边框坐标 `[x1,y1,x2,y2]`、类别文本标签、置信度 |

职责:无需训练即可按任意英文短语定位目标,用于长尾/自定义类别的场景打捞。

### 3. YOLOv8x —— 高精度通用检测

| 项目 | 配置 |
|------|------|
| 模型 | 本地 `yolov8x.pt` |
| 类别 | COCO 80 类(person/car/truck/bus/traffic light...) |
| 输出 | xyxy 边框、类别名、置信度 |

职责:对常见交通参与要素做批量快速检测,配合前端置信度滑块过滤。

### 硬件适配策略

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(8)          # CPU 限 8 线程,兼顾办公
weight_dtype = fp16 (GPU) / fp32  # 半精度防显存溢出卡死
```

---

## 数据流处理

### 流程 A:批量上传入库

```
用户选择 N 张图 → 前端按 50 张/批切片 FormData
   → POST /api/upload_batch
      逐张: bytes 解码 → PIL RGB 转换
           → JPEG q=90 落盘 workspace/images/
           → CLIP 提取 768 维特征(L2 归一化)
           → index.add() 入 FAISS
           → image_database 追加元数据
           → asyncio.sleep(0.01) 让出事件循环(保持服务响应)
   → save_persistence(): FAISS索引 + JSON元数据 + 特征npy 三件套写盘
   → 前端更新进度条,循环下一批
```

容错:单张失败仅打印日志不中断整批;超过 30000 张上限拒绝入库;网络中断时已处理部分已安全落盘。

### 流程 B:多模态语义检索

```
文本查询 "a photo of vehicles on the street"
   → POST /api/search
   → CLIP text encoder → 768 维归一化向量
   → FAISS IndexFlatIP.search(top_k) 内积近邻
   → 返回 [{id, filename, score}]
   → 前端按 CLIP 匹配度滑块(默认≥0.20)二次过滤
   → 前端分页渲染(30 张/页)
```

同构流程:`search_by_image`(库内以图搜同类)、`search_by_external_image`(外部图搜底库)、`search_by_filename`(纯字符串子串匹配,不走模型)。

### 流程 C:全量去重分析

```
POST /api/dedup_stats(sim_threshold=0.95)
   1. 载入缓存特征矩阵 feats ∈ R^{n×768}
   2. sim_matrix = feats · featsᵀ  (一次矩阵乘得全量两两余弦相似度)
   3. 并查集(Union-Find):
        for i<j: sim[i,j] ≥ 阈值 → union(i,j)
   4. 连通分量即"重复簇",每簇保留首张,其余记为冗余
   → 返回统计(总数/唯一帧/冗余数/冗余率) + 重复簇明细
```

前端拿到结果后可:
- 导出冗余 ID 清单(TXT);
- `delete_and_sync`:物理删除冗余图片文件 → 重建 image_database / 特征列表 → 新建 FAISS 索引重新 add → 全量落盘同步。

### 流程 D:批量目标检测

```
前端确定目标集(browse=当前页 / search=过阈值全集 / dedup=重复簇)
   → 用户确认 → 逐张串行请求:
      POST /api/yolo_detect 或 /api/ground_detect
   → 后端推理返回 boxes/labels/scores + 原始宽高
   → 前端:
      · 按当前置信度滑块过滤有效框
      · 统计各类别数量生成标签 ("car 3")
      · 含有效目标的卡片自动勾选
      · 每张间隔 10ms 防止 UI 卡死(后台均衡负荷)
```

### 流程 E:标注导出

```
exportData(checked=true/false)
   → 收集当前页勾选/未勾选的 checkbox ID
   → 组装 JSON:
      { image_id, filename, file_path,
        score,                       # CLIP 匹配度或 1.0
        annotations: {boxes/labels/scores},  # 已按阈值过滤
        final_tags }                 # 检测标签+手动标签
   → 浏览器端触发 .json 文件下载
```

另有 `/api/export_images` 支持按 ID 列表打包所选原图为 ZIP 下载(临时文件用后即删)。

---

## API 接口一览

| 方法 | 路径 | 功能 |
|------|------|------|
| GET  | `/` | 返回前端页面 |
| POST | `/api/upload_batch` | 批量上传入库(CLIP 编码 + FAISS 入库) |
| GET  | `/api/list_all?page=&size=` | 分页浏览全库 |
| POST | `/api/search` | 文本语义检索(top_k) |
| POST | `/api/search_by_filename` | 文件名子串检索 |
| POST | `/api/search_by_external_image` | 外部图以图搜图 |
| POST | `/api/search_by_image` | 库内图以图搜同类 |
| GET  | `/api/image/{image_id}` | 取原图 |
| POST | `/api/ground_detect` | DINO 开放词汇检测 |
| POST | `/api/yolo_detect` | YOLOv8x 检测 |
| POST | `/api/dedup_stats` | 特征去重聚类分析 |
| POST | `/api/export_images` | 打包导出所选图片 ZIP |
| POST | `/api/delete_and_sync` | 物理删除冗余并重建索引 |

> 后端通过自定义中间件放宽了 multipart 解析限制(`max_files/max_fields=100000`),以支撑大批量分片上传。

---

## 前端模块设计

| 模块 | 说明 |
|------|------|
| 1. 数据池挂载 | 文件多选 → 分批上传 → 进度条反馈 |
| 2. 去重分析 | 阈值可调(0.5~1.0)→ 统计面板 + 冗余清单导出 + 一键粉碎同步 |
| 3. 语义挖掘 | 文本检索 / 文件名精捞 / 以图搜图 / 分页浏览(浏览模式走后端分页,搜索模式前端分页) |
| 检测子系统 | YOLO/DINO 批量执行、双滑块阈值(CLIP 匹配度线 + 检测置信度)、画布叠加检测框、点击放大查看 |
| 标注体系 | 检测结果自动生成标签、手动 Add Tag、勾选状态实时统计、勾选/未勾选 JSON 导出 |

关键交互细节:
- 检测框按原始宽高换算为百分比坐标叠加,窗口缩放自适应;
- 阈值滑块调整即时重渲染,无需重新请求后端;
- `currentMode`(`browse/search/filename/dedup`)决定翻页策略与目标集合范围。

---

## 持久化与目录结构

```
数据挖掘/
├── app.py                  # FastAPI 后端入口
├── 前端.html               # 前端单页(由后端托管)
├── yolov8x.pt              # YOLO 权重
├── requirements.txt
├── start.bat
├── workspace/              # 工作区(自动创建)
│   ├── images/             # 图片文件(JPEG q90)
│   ├── index.faiss         # FAISS 向量索引
│   ├── metadata.json       # id → filename/path 映射
│   └── features.npy        # 全量特征缓存(供去重复用)
└── README.md
```

重启恢复机制:启动时检测三件套(index.faiss / metadata.json / features.npy)是否齐全,齐全则完整还原向量库与元数据,**无需重新上传**。

注意事项:
- 请勿手动增删 `workspace/images/` 内文件,否则会导致 ID 错位;
- 删除操作请统一走「永久删除冗余并同步底库」,保证索引与磁盘一致。
