Maxieye High-Precision Scene Mining Terminal（新版）
本文档对应新版 app.py（端口 8009）与配套 前端.html（暗色科幻风）。 若目录内 app.py 仍为旧版（8008/CLIP），请以实际代码为准；下文按新版设计描述。

基于 SigLIP + Grounding DINO-Base + YOLOv8x 三模型协同的自动驾驶场景挖掘终端： 支持多项目物理隔离、自动即时向量化入库、文本搜图/以图搜图、高维特征去重、开放词汇目标预标与标注导出。 后端 FastAPI，前端为纯原生 HTML/JS 单页（由后端托管）。

目录
系统总览
环境依赖
快速启动
模型架构
多项目空间
数据流处理
API 接口一览
前端模块设计
持久化与目录结构
系统总览
┌──────────────────────────────────────────────────────────────┐
│                    前端.html（浏览器，暗色科幻风格）            │
│  引导页选项目 → 上传入库 │ 去重分析 │ 语义检索 │ 批量检测 │ 导出 │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP (FormData / JSON)
┌───────────────────────────▼──────────────────────────────────┐
│            FastAPI 后端 (0.0.0.0:8009)                        │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐   │
│  │ SigLIP SO400M   │  │ Grounding DINO   │  │ YOLOv8x    │   │
│  │ (384px,1152维)  │  │ -Base 开放词汇   │  │ (COCO 80类)│   │
│  └───────┬─────────┘  └──────────────────┘  └────────────┘   │
│          │                                                     │
│  ┌───────▼───────────────────────────────────────────────┐    │
│  │ FAISS IndexFlatIP (1152维) + metadata.json             │    │
│  │ 按项目隔离：workspace/projects/<项目>/                   │    │
│  └────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
主要能力：

多项目空间：每个项目独立目录（图片 + FAISS 索引 + 元数据），物理隔离互不干扰。
自动向量化：上传图片/文件夹/服务器路径后自动提取 SigLIP 特征入库，或手动「增量向量化」。
场景语义检索：自然语言文本搜图、文件名检索、以图搜图。
高维特征去重：相似度阈值可调，冗余帧一键粉碎并重建索引。
目标预标：Grounding DINO 开放词汇 + YOLOv8x 批量检测，自动打标签。
环境依赖
组件	版本/说明
Python	3.12+（本机经 uv 管理的 .venv）
torch / torchvision	CUDA 可用时自动用 GPU，否则回退 CPU
transformers	加载 SigLIP 与 Grounding DINO
ultralytics	YOLOv8x 推理
faiss-cpu	向量索引（IndexFlatIP）
Pillow	图像解码与预处理
python-multipart	FastAPI 表单解析
安装（在 .venv 激活状态下）：

pip install -r requirements.txt
yolov8x.pt 需放置于项目根目录。

快速启动
# Windows
start.bat
# 或直接
.venv\Scripts\activate
python app.py

# Linux
bash start_linux.sh
浏览器访问：http://localhost:8009

首次启动会在 startup 阶段依次装载三个模型（已配置 hf-mirror.com 镜像以绕过 HuggingFace 网络问题）：

Google SigLIP-SO400M（google/siglip-so400m-patch14-384）
Grounding DINO-Base（IDEA-Research/grounding-dino-base）
YOLOv8x（本地 yolov8x.pt）
模型加载需一定时间（尤其 SigLIP/DINO 首次下载），期间端口暂未监听，属正常现象。

模型架构
1. SigLIP SO400M —— 多模态语义骨干
项目	配置
模型	google/siglip-so400m-patch14-384
输入	图像 Resize 至 384×384（BICUBIC）
嵌入维度	1152
精度	GPU 上 FP16（torch.cuda.amp.autocast），CPU 上 FP32
输出	L2 归一化的图像/文本特征向量
职责：

图像特征提取：SigLIPImageDataset + DataLoader(batch=128) 批量提取入库；
文本→图像检索：/api/search 将自然语言编码后与 FAISS 做内积（余弦）检索；
以图搜图：/api/search_by_external_image 多图联合（OR）检索；
去重：基于库内特征矩阵计算两两相似度。
2. Grounding DINO-Base —— 开放词汇检测
项目	配置
模型	IDEA-Research/grounding-dino-base
输入	图像 + 自然语言提示词（自动补 .）
输出	边框 [x1,y1,x2,y2]、类别文本标签、置信度
阈值	box_threshold=0.20, text_threshold=0.20
职责：无需训练即可按任意英文短语定位目标（/api/ground_detect），用于长尾/自定义类别场景打捞。

3. YOLOv8x —— 高精度通用检测
项目	配置
模型	本地 yolov8x.pt
类别	COCO 80 类
输出	xyxy 边框、类别名、置信度
职责：对常见交通参与要素批量快速检测（/api/yolo_detect）。

硬件适配
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# GPU 用 FP16 + autocast，CPU 用 FP32；RTX 4070 Ti 可流畅运行旗舰配置
多项目空间
数据按项目隔离存放于 workspace/projects/<项目名>/：

workspace/projects/
└── <项目名>/
    ├── images/        # 该项目的原始图片
    ├── index.faiss    # 该项目独立的 FAISS 索引
    └── metadata.json  # 该项目 id → filename/path/url 映射
项目名经 sanitize_project_name 清洗（保留字母数字下划线中文）。
每个项目有独立的 project_cache 上下文（索引 + 元数据），互不干扰。
提供创建/重命名/删除项目 API；default 为系统保留项目，不可删除。
项目级操作（导入、检索、去重、检测、导出）均需携带 project 参数。
数据流处理
流程 A：图片导入入库（三种方式）
方式 1 — 前端上传（图片/文件夹）：POST /api/upload_batch

前端按 300 张/批分片 FormData 上传（含 zip 解包逻辑）；
后端逐张解码 → 存入 images/ → SigLIP DataLoader 批量提取特征 → 入 FAISS → 落盘。
方式 2 — 服务器/挂载路径直导：POST /api/import_from_path

传入服务器绝对路径（如挂载网盘），后端递归 os.walk 复制新图片并向量化。
方式 3 — 增量向量化：POST /api/build_index_online

对 images/ 中尚未索引的图片补提取特征，常用于上传后手动触发。
流程 B：多模态语义检索
文本 "rainy night highway" → /api/search?query=&top_k=
  → SigLIP text encoder → 1152 维归一化向量
  → FAISS IndexFlatIP.search(top_k) → [{id, filename, score, url}]
  → 前端按匹配度滑块（默认≥0.20）过滤 + 分页渲染
同构：/api/search_by_filename（字符串子串）、/api/search_by_external_image（多图 OR 以图搜图）。

流程 C：高维特征去重
POST /api/dedup_stats(sim_threshold=0.95)
  1. index.reconstruct_n 重建全量特征矩阵 feats ∈ R^{n×1152}
  2. sim = feats·featsᵀ（一次矩阵乘得两两余弦相似度）
  3. 逐行聚类：相似度≥阈值归为一簇，每簇保留首张，其余记为冗余
  → 返回 {total, unique, duplicate_count, dedup_rate, clusters}
冗余可导出 ID 清单（TXT），或 POST /api/delete_and_sync 物理粉碎冗余图片并重建索引同步。

流程 D：批量目标检测
选定目标集（当前页/检索全集/去重簇）→ 逐张:
  POST /api/yolo_detect      # YOLO COCO 检测
  POST /api/ground_detect    # DINO 开放词汇（需 text_prompt）
  → 前端按置信度滑块过滤 → 自动生成标签（如 "car 3"）→ 有效目标自动勾选
流程 E：标注导出（交付）
exportData(checked=true/false)：导出当前页勾选/未勾选的卡片 JSON {id, filename, url, score, annotations{boxes/labels/scores}, userTags}。
exportSearchResults()：导出粗筛结果 JSON（SigLIP）。
手动 Add Tag：可为单张卡片追加标签（标记为已勾选）。
API 接口一览
方法	路径	功能
GET	/	返回 前端.html
GET	/api/projects	列出所有项目
POST	/api/projects/create	新建项目
POST	/api/projects/rename	重命名项目
POST	/api/projects/delete	删除项目（default 除外）
GET	/api/db_stats?project=	项目数据大盘（raw/processed/pending）
POST	/api/upload_batch	批量上传并向量化（支持 zip）
POST	/api/import_from_path	从服务器/挂载路径递归导入
POST	/api/build_index_online	对未索引图片增量向量化
GET	/api/list_all?project=&page=&size=	分页浏览全库
GET	/api/search?project=&query=&top_k=	文本语义检索
POST	/api/search_by_filename	文件名子串检索
POST	/api/search_by_external_image	外部多图以图搜图
GET	/api/image/{project}/{image_id}	取原图
POST	/api/dedup_stats	特征去重聚类分析
POST	/api/delete_and_sync	物理删除冗余并重建索引
POST	/api/ground_detect	DINO 开放词汇检测
POST	/api/yolo_detect	YOLOv8x 检测
前端模块设计
模块	说明
引导页（landing）	深色背景图 + 选择项目 + 「启动挖掘引擎」按钮
数据大盘	原始数据池 / 已处理（SigLIP-FAISS）/ 待处理 三指标
导入子系统	文件多选、文件夹选择、服务器路径直导、增量向量化、进度条
高维降重	相似度阈值滑块 → 统计面板 + 冗余导出 + 一键粉碎同步
语义挖掘	文本搜图 / 文件名精捞 / 以图搜图 / 分页浏览
检测子系统	YOLO/DINO 批量执行、双阈值滑块、画布叠加检测框、点击放大
标注体系	检测结果自动生成标签、手动 Add Tag、勾选状态统计、JSON 导出
持久化与目录结构
data_mining/
├── app.py                  # FastAPI 后端入口（端口 8009）
├── 前端.html               # 前端单页（由后端托管）
├── yolov8x.pt              # YOLO 权重
├── requirements.txt
├── start.bat / start_linux.sh
├── workspace/
│   └── projects/
│       └── <项目名>/        # 每项目独立
│           ├── images/      # 原始图片
│           ├── index.faiss  # FAISS 向量索引
│           └── metadata.json
└── README.md
重启恢复：启动时自动加载各项目的 index.faiss + metadata.json，无需重新上传。


请勿手动增删 workspace/projects/<项目>/images/ 内文件，否则会导致 ID 错位；
删除图片请走「删除冗余并同步」或项目删除流程，保证索引与磁盘一致；
前端引用的 /static_root/TT.jpg 与目录内 TT.png 资源名可能不一致——若页面加载静态资源异常，请将 logo 文件命名为 TT.jpg 或调整 前端.html
