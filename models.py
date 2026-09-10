# -*- coding: utf-8 -*-
"""
SQLAlchemy 数据库模型 - Maxieye Scene Mining Terminal V2
对应 PRD V2.0 核心实体：Projects, Sources, Assets, Frames, Jobs, Detections, Tags, Reviews, Benchmark, Models, InferenceCache
"""

import os
import json
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, Boolean,
    ForeignKey, Index, UniqueConstraint, Enum as SQLEnum, LargeBinary, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
# 标准 QueuePool（多线程多会话安全）
from settings import data_root, db_path as _resolve_db_path

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = data_root()
# SQLite 库：AD_DB_PATH / config.db_path 可独立指到本地 SSD（网盘 SMB 上 SQLite 锁性能差）
DB_PATH = _resolve_db_path()
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
    echo=False,  # 调试时改 True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============================================================
# 枚举定义
# ============================================================
class SourceType(PyEnum):
    IMAGE = "image"
    VIDEO = "video"


class AssetStatus(PyEnum):
    SOURCE = "SOURCE"
    EXTRACTED = "EXTRACTED"
    EMBEDDED = "EMBEDDED"
    DETECTED = "DETECTED"
    UNDERSTOOD = "UNDERSTOOD"
    TAGGED = "TAGGED"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    FILTERED = "FILTERED"
    EXPORTED = "EXPORTED"


class JobType(PyEnum):
    SCAN = "SCAN"
    EXTRACT = "EXTRACT"
    HASH = "HASH"
    EMBED = "EMBED"
    YOLO = "YOLO"
    DINO = "DINO"
    VLM = "VLM"
    TAG = "TAG"
    FUSION = "FUSION"
    REVIEW = "REVIEW"
    EXPORT = "EXPORT"


class JobStatus(PyEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class TagSource(PyEnum):
    AI = "AI"
    HUMAN = "Human"
    RULE = "Rule"


class DecisionStatus(PyEnum):
    AUTO_PASS = "AUTO_PASS"
    REVIEW = "REVIEW"
    FILTER = "FILTER"


class ReviewAction(PyEnum):
    CONFIRM = "CONFIRM"
    MODIFY = "MODIFY"
    ADD = "ADD"
    DELETE = "DELETE"
    REJECT = "REJECT"
    SKIP = "SKIP"


# ============================================================
# 核心表定义
# ============================================================

class Project(Base):
    """项目表 - 物理隔离的数据空间"""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    display_name = Column(String(256))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关联
    sources = relationship("Source", back_populates="project", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")
    benchmark_set = relationship("Benchmark", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project {self.name}>"


class Source(Base):
    """源文件注册表 - PRD 5.2: 每个原始文件建立唯一 Source"""
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(64), unique=True, nullable=False, index=True)  # UUID
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    source_type = Column(SQLEnum(SourceType), nullable=False)
    source_root = Column(String(1024), nullable=False)  # NAS/挂载根目录
    relative_path = Column(String(1024), nullable=False)  # 相对路径
    directory_chain = Column(SQLiteJSON, default=list)  # ["City_A", "Scene_001", "Rain"]
    file_name = Column(String(256), nullable=False)
    extension = Column(String(16))
    file_size = Column(Integer, default=0)
    file_hash = Column(String(64), index=True)  # SHA256 - 一级去重基础
    phash = Column(String(64), index=True)  # 感知哈希 - 二级去重
    created_time = Column(DateTime, default=datetime.utcnow)
    scan_status = Column(String(32), default="pending")  # pending/completed/failed
    meta = Column(SQLiteJSON, default=dict)  # 扩展字段：fps, duration, resolution 等

    # 关联
    project = relationship("Project", back_populates="sources")
    assets = relationship("Asset", back_populates="source", cascade="all, delete-orphan")
    frames = relationship("Frame", back_populates="source", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("project_id", "relative_path", name="uq_source_project_path"),
        Index("ix_source_hash", "file_hash"),
        Index("ix_source_phash", "phash"),
    )

    def __repr__(self):
        return f"<Source {self.source_id} {self.file_name}>"


class Asset(Base):
    """统一 Asset 表 - PRD 第 7 节：图片和视频抽帧统一转换成 Asset"""
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), unique=True, nullable=False, index=True)  # UUID
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    source_type = Column(SQLEnum(SourceType), nullable=False)  # image / video_frame
    image_path = Column(String(1024), nullable=False)  # 入库后的图片路径
    frame_index = Column(Integer, default=0)  # 视频帧序号，图片为 0
    sample_index = Column(Integer, default=0)  # 抽样序号
    timestamp = Column(Float, default=0.0)  # 视频时间戳(秒)
    total_frames = Column(Integer, default=0)
    fps = Column(Float, default=0.0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)

    # 结构化元数据 JSON
    asset_metadata = Column(SQLiteJSON, default=dict)  # {vehicle, resolution, sensor, ...}

    # 向量索引映射
    vector_id = Column(Integer, default=-1, index=True)  # FAISS 索引中的位置

    # 检测结果 JSON
    detections = Column(SQLiteJSON, default=dict)  # {yolo:[], dino:[]}

    # 场景标签 JSON - 四层分离
    ai_tags = Column(SQLiteJSON, default=dict)      # {weather:[{tag,conf,model,ver}], road:[], objects:[], events:[]}
    human_tags = Column(SQLiteJSON, default=dict)   # 同结构，source=Human
    final_tags = Column(SQLiteJSON, default=dict)   # 最终生效标签

    # 审核相关
    review = Column(SQLiteJSON, default=dict)       # {status, reviewer, comments, decided_at}
    final_result = Column(SQLiteJSON, default=dict) # {status, tags, score}

    # 状态机
    status = Column(SQLEnum(AssetStatus), default=AssetStatus.SOURCE, index=True)

    # 决策引擎结果
    decision_status = Column(SQLEnum(DecisionStatus), nullable=True, index=True)
    decision_reason = Column(Text)
    decision_score = Column(Float)

    # 模型版本追踪
    model_versions = Column(SQLiteJSON, default=dict)  # {siglip, yolo, dino, vlm, fusion, ontology, threshold}

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关联
    project = relationship("Project", back_populates="assets")
    source = relationship("Source", back_populates="assets")
    frames = relationship("Frame", back_populates="asset", cascade="all, delete-orphan")
    review_records = relationship("ReviewRecord", back_populates="asset", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_asset_status_project", "status", "project_id"),
        Index("ix_asset_decision", "decision_status"),
    )

    def __repr__(self):
        return f"<Asset {self.asset_id} {self.status.value}>"


class Frame(Base):
    """视频帧明细表 - PRD 6.3: 每帧必须记录完整元数据"""
    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(String(64), unique=True, nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False)  # 视频中的绝对帧号
    sample_index = Column(Integer, default=0)      # 抽样序号
    timestamp = Column(Float, default=0.0)         # 秒
    image_path = Column(String(1024))              # 抽帧保存路径

    # 关联
    asset = relationship("Asset", back_populates="frames")
    source = relationship("Source", back_populates="frames")

    __table_args__ = (
        Index("ix_frame_source_idx", "source_id", "frame_index"),
    )

    def __repr__(self):
        return f"<Frame {self.frame_id} idx={self.frame_index}>"


class Job(Base):
    """统一任务系统 - PRD 第 39-42 节"""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)  # UUID
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    job_type = Column(SQLEnum(JobType), nullable=False, index=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    progress = Column(Float, default=0.0)  # 0-100
    current_stage = Column(String(128))
    payload = Column(SQLiteJSON, default=dict)  # 输入参数
    result = Column(SQLiteJSON, default=dict)   # 输出结果
    error = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    parent_job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)  # 子任务链
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 关联
    project = relationship("Project", back_populates="jobs")
    children = relationship("Job", backref="parent", remote_side=[id])

    __table_args__ = (
        Index("ix_job_status_type", "status", "job_type"),
        Index("ix_job_project_status", "project_id", "status"),
    )

    def __repr__(self):
        return f"<Job {self.job_id} {self.job_type.value} {self.status.value}>"


class DetectionCache(Base):
    """目标检测持久化缓存 - 替代原 detections_cache.json"""
    __tablename__ = "detection_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(String(64), nullable=False, index=True)  # Asset.asset_id
    engine = Column(String(32), nullable=False)  # yolo / dino
    model_version = Column(String(64))
    prompt = Column(Text)  # DINO 的 prompt
    result = Column(SQLiteJSON, nullable=False)  # 完整检测结果
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("project_id", "asset_id", "engine", "model_version", "prompt", name="uq_detection_cache"),
    )


class InferenceCache(Base):
    """推理缓存 - PRD 44: asset_id + model_version + input_hash -> 缓存结果"""
    __tablename__ = "inference_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)  # siglip/yolo/dino/vlm
    model_version = Column(String(64), nullable=False)
    input_hash = Column(String(64), nullable=False, index=True)  # 图片内容哈希 or prompt哈希
    prompt_version = Column(String(64))  # DINO/VLM prompt 版本
    output = Column(SQLiteJSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("asset_id", "model_name", "model_version", "input_hash", "prompt_version", name="uq_inference_cache"),
    )


class ModelRegistry(Base):
    """模型版本管理 - PRD 45"""
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, index=True)  # siglip/yolo/dino/vlm
    version = Column(String(64), nullable=False)
    model_path = Column(String(512))  # 本地路径或 HF repo
    config = Column(SQLiteJSON, default=dict)  # 参数配置
    is_active = Column(Boolean, default=False)
    registered_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_version"),
    )


class Benchmark(Base):
    """Benchmark 数据集 - PRD 24-25"""
    __tablename__ = "benchmark"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(String(64), nullable=False, index=True)  # 关联 Asset
    split = Column(String(16), default="val")  # train/val/test
    gt_tags = Column(SQLiteJSON, nullable=False)  # Ground Truth: {weather, road, objects, events, risk}
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="benchmark_set")

    __table_args__ = (
        Index("ix_benchmark_split", "project_id", "split"),
    )


class BenchmarkEvaluation(Base):
    """Benchmark 评测记录"""
    __tablename__ = "benchmark_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_id = Column(String(64), unique=True, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_combo = Column(String(128))  # 如 "siglip+yolo+dino+vlm"
    model_versions = Column(SQLiteJSON)  # 各模型版本
    metrics = Column(SQLiteJSON)  # {precision, recall, f1, recall@k, precision@k}
    per_class = Column(SQLiteJSON)  # 各类别指标
    created_at = Column(DateTime, default=datetime.utcnow)


class HardCase(Base):
    """Hard Case 库 - PRD 23: AI ≠ Human 进入 Hard Case"""
    __tablename__ = "hard_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(String(64), nullable=False, index=True)
    ai_tags = Column(SQLiteJSON)
    human_tags = Column(SQLiteJSON)
    final_tags = Column(SQLiteJSON)
    diff = Column(SQLiteJSON)  # 差异分析
    status = Column(String(32), default="open")  # open/analyzed/resolved
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class ReviewRecord(Base):
    """人工审核记录 - PRD 19-21"""
    __tablename__ = "review_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    reviewer = Column(String(64))
    action = Column(SQLEnum(ReviewAction), nullable=False)
    ai_tags_snapshot = Column(SQLiteJSON)  # 审核时的 AI 标签快照
    human_tags = Column(SQLiteJSON)        # 人工修改的标签
    final_tags = Column(SQLiteJSON)        # 最终生效标签
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    asset = relationship("Asset", back_populates="review_records")


class ExportRecord(Base):
    """导出记录"""
    __tablename__ = "export_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    export_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    format = Column(String(16))  # jsonl/parquet/csv/zip
    filter_criteria = Column(SQLiteJSON)  # 导出筛选条件
    asset_count = Column(Integer, default=0)
    output_path = Column(String(1024))
    manifest = Column(SQLiteJSON)  # manifest.json 内容
    status = Column(String(32), default="pending")  # pending/running/completed/failed
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


# ============================================================
# 数据库初始化与工具函数
# ============================================================

def _enable_wal():
    """SQLite 开启 WAL + busy_timeout，支持读写并发"""
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA busy_timeout=30000")
            conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
    except Exception:
        pass


def init_db():
    """创建所有表"""
    _enable_wal()
    Base.metadata.create_all(bind=engine)
    print(f"✅ 数据库初始化完成: {DB_PATH}")


def get_db() -> Session:
    """获取数据库会话（FastAPI Depends 用）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """直接获取会话（非生成器上下文）"""
    return SessionLocal()


# ============================================================
# 便捷查询函数
# ============================================================

def get_or_create_project(db: Session, name: str, display_name: str = None) -> Project:
    """获取或创建项目"""
    project = db.query(Project).filter(Project.name == name).first()
    if not project:
        project = Project(name=name, display_name=display_name or name)
        db.add(project)
        db.commit()
        db.refresh(project)
    return project


def create_source(db: Session, project_id: int, source_root: str, relative_path: str,
                  source_type: SourceType, file_name: str, file_size: int = 0,
                  file_hash: str = None, phash: str = None, directory_chain: list = None,
                  meta: dict = None) -> Source:
    """创建 Source 记录"""
    source_id = str(uuid.uuid4())[:16]
    source = Source(
        source_id=source_id,
        project_id=project_id,
        source_type=source_type,
        source_root=source_root,
        relative_path=relative_path,
        directory_chain=directory_chain or [],
        file_name=file_name,
        extension=os.path.splitext(file_name)[1].lower(),
        file_size=file_size,
        file_hash=file_hash,
        phash=phash,
        meta=meta or {},
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


def create_asset(db: Session, project_id: int, source_id: int, source_type: SourceType,
                 image_path: str, frame_index: int = 0, sample_index: int = 0,
                 timestamp: float = 0.0, total_frames: int = 0, fps: float = 0.0,
                 width: int = 0, height: int = 0, metadata: dict = None) -> Asset:
    """创建 Asset 记录"""
    asset_id = str(uuid.uuid4())[:16]
    asset = Asset(
        asset_id=asset_id,
        project_id=project_id,
        source_id=source_id,
        source_type=source_type,
        image_path=image_path,
        frame_index=frame_index,
        sample_index=sample_index,
        timestamp=timestamp,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        asset_metadata=metadata or {},
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def create_job(db: Session, project_id: int, job_type: JobType, payload: dict = None,
               parent_job_id: int = None) -> Job:
    """创建任务记录"""
    job_id = str(uuid.uuid4())[:16]
    job = Job(
        job_id=job_id,
        project_id=project_id,
        job_type=job_type,
        payload=payload or {},
        parent_job_id=parent_job_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_job_status(db: Session, job_id: str, status: JobStatus,
                      progress: float = None, current_stage: str = None,
                      result: dict = None, error: str = None):
    """更新任务状态"""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if job:
        job.status = status
        if progress is not None:
            job.progress = progress
        if current_stage:
            job.current_stage = current_stage
        if result:
            job.result = result
        if error:
            job.error = error
        if status == JobStatus.RUNNING and not job.started_at:
            job.started_at = datetime.utcnow()
        if status in (JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.CANCELLED):
            job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()


# ============================================================
# 迁移辅助：从旧 JSON 导入
# ============================================================

def migrate_from_json(db: Session, projects_root: str):
    """
    从旧版 workspace/projects/<项目>/metadata.json 迁移数据
    同时迁移 detections_cache.json
    """
    import glob
    
    # 1. 迁移项目和 Assets
    for meta_path in glob.glob(os.path.join(projects_root, "*", "metadata.json")):
        project_name = os.path.basename(os.path.dirname(meta_path))
        project = get_or_create_project(db, project_name)

        # 幂等：该项目已有资产则跳过（增量入库由 sync hooks 处理，勿每次启动全量重扫）
        existing = db.query(Asset).filter(Asset.project_id == project.id).count()
        if existing:
            print(f"    - {project_name}: 已迁移 {existing} 条资产，跳过")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        
        for idx, item in enumerate(metadata_list):
            # 创建 Source（如果不存在）
            src_path = item.get("path", "")
            if src_path and os.path.exists(src_path):
                file_hash = None
                try:
                    import hashlib
                    with open(src_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                except Exception:
                    pass
                
                source = db.query(Source).filter(
                    Source.project_id == project.id,
                    Source.relative_path == os.path.basename(src_path)
                ).first()
                
                if not source:
                    source = create_source(
                        db, project.id,
                        source_root=os.path.dirname(src_path) or ".",
                        relative_path=os.path.basename(src_path),
                        source_type=SourceType.IMAGE,
                        file_name=item.get("filename", os.path.basename(src_path)),
                        file_size=os.path.getsize(src_path) if os.path.exists(src_path) else 0,
                        file_hash=file_hash,
                    )
                
                # 创建 Asset
                asset = db.query(Asset).filter(
                    Asset.project_id == project.id,
                    Asset.source_id == source.id,
                    Asset.frame_index == 0
                ).first()
                
                if not asset:
                    asset = create_asset(
                        db, project.id, source.id, SourceType.IMAGE,
                        image_path=src_path,
                        frame_index=0,
                        metadata={"original_id": item.get("id")},
                    )
                
                # 更新 vector_id
                if "id" in item:
                    asset.vector_id = item["id"]
                    db.commit()
    
    # 2. 迁移检测缓存
    cache_path = os.path.join(DATA_ROOT, "detections_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            detections_cache = json.load(f)
        
        for proj_name, proj_cache in detections_cache.items():
            project = db.query(Project).filter(Project.name == proj_name).first()
            if not project:
                continue
            for asset_id_str, det_result in proj_cache.items():
                if isinstance(det_result, dict):
                    engine = det_result.get("engine", "unknown")
                    model_version = "legacy"
                    prompt = det_result.get("prompt", "")
                    # 幂等：复合唯一键已存在则跳过（勿每次启动重复 INSERT 撞 UNIQUE）
                    exists = db.query(DetectionCache).filter(
                        DetectionCache.project_id == project.id,
                        DetectionCache.asset_id == asset_id_str,
                        DetectionCache.engine == engine,
                        DetectionCache.model_version == model_version,
                        DetectionCache.prompt == prompt,
                    ).first()
                    if exists:
                        continue
                    dc = DetectionCache(
                        project_id=project.id,
                        asset_id=asset_id_str,
                        engine=engine,
                        model_version=model_version,
                        prompt=prompt,
                        result=det_result,
                    )
                    db.merge(dc)  # upsert
        db.commit()
    
    print("✅ JSON 迁移完成")


if __name__ == "__main__":
    # 直接运行此文件可初始化数据库
    init_db()
    
    # 如果有旧数据，自动迁移
    db = get_db_session()
    try:
        projects_root = os.path.join(DATA_ROOT, "projects")
        if os.path.exists(projects_root):
            migrate_from_json(db, projects_root)
    finally:
        db.close()