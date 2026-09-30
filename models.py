# -*- coding: utf-8 -*-
"""
SQLAlchemy 数据库模型 - Maxieye Scene Mining Terminal V2
支持 SQLite（默认）和 PostgreSQL（AD_DB_TYPE=postgresql）
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
from sqlalchemy.dialects.postgresql import JSONB
from settings import data_root, db_path, db_type, pg_config as _pg_config

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = data_root()
DB_PATH = db_path()
DB_TYPE = db_type()

if DB_TYPE == "postgresql":
    _pg = _pg_config()
    DATABASE_URL = f"postgresql://{_pg['user']}:{_pg['password']}@{_pg['host']}:{_pg['port']}/{_pg['db']}"
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
else:
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=True,
        echo=False,
    )

# SQLite 性能 pragmas（仅 SQLite 时生效）
from sqlalchemy import event as _sa_event

@_sa_event.listens_for(engine, "connect")
def _db_pragmas(dbapi_conn, _rec):
    if DB_TYPE == "sqlite":
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA cache_size=-262144")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA mmap_size=1073741824")
        cur.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JSON 类型选择
JSONType = JSONB if DB_TYPE == "postgresql" else SQLiteJSON


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
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    display_name = Column(String(256))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sources = relationship("Source", back_populates="project", cascade="all, delete-orphan")
    assets = relationship("Asset", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")
    benchmark_set = relationship("Benchmark", back_populates="project", cascade="all, delete-orphan")


class Source(Base):
    __tablename__ = "sources"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    source_type = Column(SQLEnum(SourceType), nullable=False)
    source_root = Column(String(1024), nullable=False)
    relative_path = Column(String(1024), nullable=False)
    directory_chain = Column(JSONType, default=list)
    file_name = Column(String(256), nullable=False)
    extension = Column(String(16))
    file_size = Column(Integer, default=0)
    file_hash = Column(String(64), index=True)
    phash = Column(String(64), index=True)
    created_time = Column(DateTime, default=datetime.utcnow)
    scan_status = Column(String(32), default="pending")
    meta = Column(JSONType, default=dict)
    project = relationship("Project", back_populates="sources")
    assets = relationship("Asset", back_populates="source", cascade="all, delete-orphan")
    frames = relationship("Frame", back_populates="source", cascade="all, delete-orphan")
    __table_args__ = (
        UniqueConstraint("project_id", "relative_path", name="uq_source_project_path"),
        Index("ix_source_hash", "file_hash"),
        Index("ix_source_phash", "phash"),
    )


class Asset(Base):
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    source_type = Column(SQLEnum(SourceType), nullable=False)
    image_path = Column(String(1024), nullable=False)
    frame_index = Column(Integer, default=0)
    sample_index = Column(Integer, default=0)
    timestamp = Column(Float, default=0.0)
    total_frames = Column(Integer, default=0)
    fps = Column(Float, default=0.0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    asset_metadata = Column(JSONType, default=dict)
    vector_id = Column(Integer, default=-1, index=True)
    detections = Column(JSONType, default=dict)
    ai_tags = Column(JSONType, default=dict)
    human_tags = Column(JSONType, default=dict)
    final_tags = Column(JSONType, default=dict)
    review = Column(JSONType, default=dict)
    final_result = Column(JSONType, default=dict)
    status = Column(SQLEnum(AssetStatus), default=AssetStatus.SOURCE, index=True)
    decision_status = Column(SQLEnum(DecisionStatus), nullable=True, index=True)
    decision_reason = Column(Text)
    decision_score = Column(Float)
    model_versions = Column(JSONType, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    project = relationship("Project", back_populates="assets")
    source = relationship("Source", back_populates="assets")
    frames = relationship("Frame", back_populates="asset", cascade="all, delete-orphan")
    review_records = relationship("ReviewRecord", back_populates="asset", cascade="all, delete-orphan")
    __table_args__ = (
        Index("ix_asset_status_project", "status", "project_id"),
        Index("ix_asset_decision", "decision_status"),
        Index("ix_assets_proj_status_decision", "project_id", "status", "decision_status"),
    )


class Frame(Base):
    __tablename__ = "frames"
    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(String(64), unique=True, nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False)
    sample_index = Column(Integer, default=0)
    timestamp = Column(Float, default=0.0)
    image_path = Column(String(1024))
    asset = relationship("Asset", back_populates="frames")
    source = relationship("Source", back_populates="frames")
    __table_args__ = (
        Index("ix_frame_source_idx", "source_id", "frame_index"),
    )


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    job_type = Column(SQLEnum(JobType), nullable=False, index=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    progress = Column(Float, default=0.0)
    current_stage = Column(String(128))
    payload = Column(JSONType, default=dict)
    result = Column(JSONType, default=dict)
    error = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    parent_job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    project = relationship("Project", back_populates="jobs")
    children = relationship("Job", backref="parent", remote_side=[id])
    __table_args__ = (
        Index("ix_job_status_type", "status", "job_type"),
        Index("ix_job_project_status", "project_id", "status"),
    )


class DetectionCache(Base):
    __tablename__ = "detection_cache"
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(String(64), nullable=False, index=True)
    engine = Column(String(32), nullable=False)
    model_version = Column(String(64))
    prompt = Column(Text)
    result = Column(JSONType, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint("project_id", "asset_id", "engine", "model_version", "prompt", name="uq_detection_cache"),
    )


class InferenceCache(Base):
    __tablename__ = "inference_cache"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String(64), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=False)
    input_hash = Column(String(64), nullable=False, index=True)
    prompt_version = Column(String(64))
    output = Column(JSONType, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint("asset_id", "model_name", "model_version", "input_hash", "prompt_version", name="uq_inference_cache"),
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, index=True)
    version = Column(String(64), nullable=False)
    model_path = Column(String(512))
    config = Column(JSONType, default=dict)
    is_active = Column(Boolean, default=False)
    registered_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_version"),
    )


class Benchmark(Base):
    __tablename__ = "benchmark"
    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(String(64), nullable=False, index=True)
    split = Column(String(16), default="val")
    gt_tags = Column(JSONType, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    project = relationship("Project", back_populates="benchmark_set")
    __table_args__ = (
        Index("ix_benchmark_split", "project_id", "split"),
    )


class BenchmarkEvaluation(Base):
    __tablename__ = "benchmark_evaluations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_id = Column(String(64), unique=True, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_combo = Column(String(128))
    model_versions = Column(JSONType)
    metrics = Column(JSONType)
    per_class = Column(JSONType)
    created_at = Column(DateTime, default=datetime.utcnow)


class HardCase(Base):
    __tablename__ = "hard_cases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(String(64), nullable=False, index=True)
    ai_tags = Column(JSONType)
    human_tags = Column(JSONType)
    final_tags = Column(JSONType)
    diff = Column(JSONType)
    status = Column(String(32), default="open")
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class ReviewRecord(Base):
    __tablename__ = "review_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    reviewer = Column(String(64))
    action = Column(SQLEnum(ReviewAction), nullable=False)
    ai_tags_snapshot = Column(JSONType)
    human_tags = Column(JSONType)
    final_tags = Column(JSONType)
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    asset = relationship("Asset", back_populates="review_records")


class ExportRecord(Base):
    __tablename__ = "export_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    export_id = Column(String(64), unique=True, nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    format = Column(String(16))
    filter_criteria = Column(JSONType)
    asset_count = Column(Integer, default=0)
    output_path = Column(String(1024))
    manifest = Column(JSONType)
    status = Column(String(32), default="pending")
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


# ============================================================
# 数据库初始化与工具函数
# ============================================================

def _enable_wal():
    """SQLite 开启 WAL（仅 SQLite 时生效）"""
    if DB_TYPE != "sqlite":
        return
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
    print(f"✅ 数据库初始化完成: {DB_TYPE} @ {DATABASE_URL if DB_TYPE == 'postgresql' else DB_PATH}")


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


if __name__ == "__main__":
    init_db()
