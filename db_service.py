# -*- coding: utf-8 -*-
"""
数据库服务层 - 封装所有数据库操作
替代原有的 JSON 文件读写、project_cache、detections_cache 等
"""

import os
import json
import hashlib
import uuid
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from models import (
    Base, engine, SessionLocal, get_db_session,
    Project, Source, Asset, Frame, Job, DetectionCache,
    InferenceCache, ModelRegistry, Benchmark, BenchmarkEvaluation,
    HardCase, ReviewRecord, ExportRecord,
    SourceType, AssetStatus, JobType, JobStatus, TagSource,
    DecisionStatus, ReviewAction,
    init_db, migrate_from_json,
)


# ============================================================
# 会话管理
# ============================================================

@contextmanager
def db_session():
    """数据库会话上下文管理器"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ============================================================
# Project 服务
# ============================================================

def create_project(db: Session, name: str, display_name: str = None, description: str = "") -> Project:
    """创建项目"""
    project = Project(
        name=name,
        display_name=display_name or name,
        description=description,
    )
    db.add(project)
    db.flush()
    return project


def get_project(db: Session, name: str) -> Optional[Project]:
    """按名称获取项目"""
    return db.query(Project).filter(Project.name == name).first()


def get_project_by_id(db: Session, project_id: int) -> Optional[Project]:
    """按 ID 获取项目"""
    return db.query(Project).filter(Project.id == project_id).first()


def list_projects(db: Session) -> List[Project]:
    """列出所有项目"""
    return db.query(Project).order_by(Project.created_at.desc()).all()


def rename_project(db: Session, old_name: str, new_name: str) -> Optional[Project]:
    """重命名项目"""
    project = get_project(db, old_name)
    if not project:
        return None
    if get_project(db, new_name):
        raise ValueError(f"项目名已存在: {new_name}")
    project.name = new_name
    project.updated_at = datetime.utcnow()
    db.flush()
    return project


def delete_project(db: Session, name: str) -> bool:
    """删除项目（级联删除关联数据）

    SQLite 默认关闭外键，db.delete(project) 不会级联删子表——
    必须手工按 project_id 清理全部关联表，否则产生孤儿数据（项目删了但资产残留）。
    """
    from models import (Asset, Source, Frame, DetectionCache, InferenceCache,
                        Job, HardCase, ReviewRecord, ExportRecord)
    project = get_project(db, name)
    if not project:
        return False
    if name == "default":
        raise ValueError("default 是系统保留项目，无法删除")
    # 先清全部关联子表（避免孤儿行）
    for cls in (Asset, Source, Frame, DetectionCache, InferenceCache,
                Job, HardCase, ReviewRecord, ExportRecord):
        if hasattr(cls, "project_id"):
            db.query(cls).filter(cls.project_id == project.id).delete(synchronize_session=False)
    db.delete(project)
    db.commit()
    return True


def get_project_stats(db: Session, project_id: int) -> Dict[str, int]:
    """获取项目统计"""
    raw_count = db.query(Source).filter(Source.project_id == project_id).count()
    asset_count = db.query(Asset).filter(Asset.project_id == project_id).count()
    embedded_count = db.query(Asset).filter(
        Asset.project_id == project_id,
        Asset.status.in_([AssetStatus.EMBEDDED, AssetStatus.DETECTED, AssetStatus.UNDERSTOOD, 
                          AssetStatus.TAGGED, AssetStatus.REVIEW, AssetStatus.APPROVED])
    ).count()
    return {
        "raw_count": raw_count,
        "processed_count": embedded_count,
        "pending_count": max(0, raw_count - embedded_count),
    }


# ============================================================
# Source 服务
# ============================================================

def compute_file_hash(filepath: str) -> str:
    """计算文件 SHA256"""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(filepath: str) -> Optional[str]:
    """计算感知哈希 (pHash) - 需要 PIL"""
    try:
        from PIL import Image
        import imagehash
        img = Image.open(filepath).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return None


def create_source(
    db: Session,
    project_id: int,
    source_root: str,
    relative_path: str,
    source_type: SourceType,
    file_name: str,
    file_size: int = 0,
    file_hash: str = None,
    phash: str = None,
    directory_chain: List[str] = None,
    meta: Dict = None,
) -> Source:
    """创建 Source 记录（去重检查）"""
    # 检查是否已存在（同项目同相对路径）
    existing = db.query(Source).filter(
        Source.project_id == project_id,
        Source.relative_path == relative_path,
    ).first()
    if existing:
        return existing
    
    source = Source(
        source_id=str(uuid.uuid4())[:16],
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
        scan_status="completed",
    )
    db.add(source)
    db.flush()
    return source


def get_source_by_hash(db: Session, project_id: int, file_hash: str) -> Optional[Source]:
    """按哈希查找 Source（一级去重）"""
    if not file_hash:
        return None
    return db.query(Source).filter(
        Source.project_id == project_id,
        Source.file_hash == file_hash,
    ).first()


def find_similar_sources(db: Session, project_id: int, phash: str, max_distance: int = 5) -> List[Source]:
    """按 pHash 查找近似源（二级去重）"""
    if not phash:
        return []
    # 简单实现：全量扫描比较汉明距离
    # 生产环境建议用专用索引或数据库函数
    sources = db.query(Source).filter(
        Source.project_id == project_id,
        Source.phash.isnot(None),
    ).all()
    similar = []
    for src in sources:
        try:
            dist = sum(c1 != c2 for c1, c2 in zip(phash, src.phash))
            if dist <= max_distance:
                similar.append(src)
        except Exception:
            continue
    return similar


def list_sources(db: Session, project_id: int, source_type: SourceType = None) -> List[Source]:
    """列出项目的源文件"""
    query = db.query(Source).filter(Source.project_id == project_id)
    if source_type:
        query = query.filter(Source.source_type == source_type)
    return query.order_by(Source.created_time.desc()).all()


# ============================================================
# Asset 服务
# ============================================================

def create_asset(
    db: Session,
    project_id: int,
    source_id: int,
    source_type: SourceType,
    image_path: str,
    frame_index: int = 0,
    sample_index: int = 0,
    timestamp: float = 0.0,
    total_frames: int = 0,
    fps: float = 0.0,
    width: int = 0,
    height: int = 0,
    asset_metadata: Dict = None,
) -> Asset:
    """创建 Asset 记录"""
    # 检查是否已存在（同 source 同 frame_index）
    existing = db.query(Asset).filter(
        Asset.project_id == project_id,
        Asset.source_id == source_id,
        Asset.frame_index == frame_index,
    ).first()
    if existing:
        # 更新路径等信息
        existing.image_path = image_path
        existing.width = width
        existing.height = height
        db.flush()
        return existing
    
    asset = Asset(
        asset_id=str(uuid.uuid4())[:16],
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
        asset_metadata=asset_metadata or {},
        status=AssetStatus.SOURCE,
    )
    db.add(asset)
    db.flush()
    return asset


def get_asset(db: Session, project_id: int, asset_id: str) -> Optional[Asset]:
    """按 asset_id 获取 Asset"""
    return db.query(Asset).filter(
        Asset.project_id == project_id,
        Asset.asset_id == asset_id,
    ).first()


def get_asset_by_vector_id(db: Session, project_id: int, vector_id: int) -> Optional[Asset]:
    """按 FAISS vector_id 获取 Asset"""
    return db.query(Asset).filter(
        Asset.project_id == project_id,
        Asset.vector_id == vector_id,
    ).first()


def list_assets(
    db: Session,
    project_id: int,
    status: AssetStatus = None,
    page: int = 1,
    size: int = 30,
) -> Tuple[List[Asset], int]:
    """分页列出 Asset"""
    query = db.query(Asset).filter(Asset.project_id == project_id)
    if status:
        query = query.filter(Asset.status == status)
    total = query.count()
    assets = query.order_by(Asset.created_at.desc()).offset((page - 1) * size).limit(size).all()
    return assets, total


def update_asset_status(db: Session, asset_id: str, status: AssetStatus):
    """更新 Asset 状态"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if asset:
        asset.status = status
        asset.updated_at = datetime.utcnow()


def update_asset_vector_id(db: Session, asset_id: str, vector_id: int):
    """更新 FAISS vector_id 映射"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if asset:
        asset.vector_id = vector_id
        asset.status = AssetStatus.EMBEDDED
        asset.updated_at = datetime.utcnow()


def update_asset_detections(db: Session, asset_id: str, detections: Dict):
    """更新检测结果"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if asset:
        asset.detections = detections
        asset.status = AssetStatus.DETECTED
        asset.updated_at = datetime.utcnow()


def update_asset_tags(
    db: Session,
    asset_id: str,
    ai_tags: Dict = None,
    human_tags: Dict = None,
    final_tags: Dict = None,
):
    """更新标签（四层分离）"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if not asset:
        return
    if ai_tags is not None:
        asset.ai_tags = ai_tags
    if human_tags is not None:
        asset.human_tags = human_tags
    if final_tags is not None:
        asset.final_tags = final_tags
    asset.updated_at = datetime.utcnow()
    if final_tags is not None:
        asset.status = AssetStatus.TAGGED


def update_asset_decision(
    db: Session,
    asset_id: str,
    decision_status,
    decision_reason: str,
    decision_score: float,
):
    """更新决策引擎结果（decision_status 兼容枚举或字符串）"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if asset:
        # 兼容字符串输入（如 "AUTO_PASS"）
        if isinstance(decision_status, str):
            try:
                decision_status = DecisionStatus(decision_status)
            except ValueError:
                decision_status = DecisionStatus.REVIEW
        asset.decision_status = decision_status
        asset.decision_reason = decision_reason
        asset.decision_score = decision_score
        if decision_status == DecisionStatus.AUTO_PASS:
            asset.status = AssetStatus.APPROVED
        elif decision_status == DecisionStatus.FILTER:
            asset.status = AssetStatus.FILTERED
        else:
            asset.status = AssetStatus.REVIEW
        asset.updated_at = datetime.utcnow()


def update_asset_review(db: Session, asset_id: str, review: Dict):
    """更新审核记录"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if asset:
        asset.review = review
        asset.updated_at = datetime.utcnow()


def update_asset_final_result(db: Session, asset_id: str, final_result: Dict):
    """更新最终结果"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if asset:
        asset.final_result = final_result
        if final_result.get("status") == "APPROVED":
            asset.status = AssetStatus.APPROVED
        elif final_result.get("status") == "FILTERED":
            asset.status = AssetStatus.FILTERED
        asset.updated_at = datetime.utcnow()


def update_asset_model_versions(db: Session, asset_id: str, versions: Dict):
    """更新模型版本追踪"""
    asset = db.query(Asset).filter(Asset.asset_id == asset_id).first()
    if asset:
        asset.model_versions = versions
        asset.updated_at = datetime.utcnow()


# ============================================================
# Frame 服务
# ============================================================

def create_frame(
    db: Session,
    asset_id: int,
    source_id: int,
    frame_index: int,
    sample_index: int,
    timestamp: float,
    image_path: str,
) -> Frame:
    """创建帧记录"""
    frame = Frame(
        frame_id=str(uuid.uuid4())[:16],
        asset_id=asset_id,
        source_id=source_id,
        frame_index=frame_index,
        sample_index=sample_index,
        timestamp=timestamp,
        image_path=image_path,
    )
    db.add(frame)
    db.flush()
    return frame


def get_frames_by_source(db: Session, source_id: int) -> List[Frame]:
    """获取源视频的所有帧"""
    return db.query(Frame).filter(Frame.source_id == source_id).order_by(Frame.frame_index).all()


# ============================================================
# Job 服务 (统一任务系统)
# ============================================================

def create_job(
    db: Session,
    project_id: int,
    job_type: JobType,
    payload: Dict = None,
    parent_job_id: int = None,
) -> Job:
    """创建任务"""
    job = Job(
        job_id=str(uuid.uuid4())[:16],
        project_id=project_id,
        job_type=job_type,
        payload=payload or {},
        parent_job_id=parent_job_id,
    )
    db.add(job)
    db.flush()
    return job


def update_job_status(
    db: Session,
    job_id: str,
    status: JobStatus,
    progress: float = None,
    current_stage: str = None,
    result: Dict = None,
    error: str = None,
):
    """更新任务状态"""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        return
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


def get_job(db: Session, job_id: str) -> Optional[Job]:
    """获取任务"""
    return db.query(Job).filter(Job.job_id == job_id).first()


def list_jobs(
    db: Session,
    project_id: int = None,
    job_type: JobType = None,
    status: JobStatus = None,
) -> List[Job]:
    """列出任务"""
    query = db.query(Job)
    if project_id:
        query = query.filter(Job.project_id == project_id)
    if job_type:
        query = query.filter(Job.job_type == job_type)
    if status:
        query = query.filter(Job.status == status)
    return query.order_by(Job.created_at.desc()).all()


def get_running_jobs(db: Session, project_id: int = None) -> List[Job]:
    """获取运行中的任务"""
    query = db.query(Job).filter(Job.status == JobStatus.RUNNING)
    if project_id:
        query = query.filter(Job.project_id == project_id)
    return query.all()


# ============================================================
# DetectionCache 服务
# ============================================================

def get_detection_cache(
    db: Session,
    project_id: int,
    asset_id: str,
    engine: str,
    model_version: str = "latest",
    prompt: str = "",
) -> Optional[DetectionCache]:
    """获取检测缓存"""
    return db.query(DetectionCache).filter(
        DetectionCache.project_id == project_id,
        DetectionCache.asset_id == asset_id,
        DetectionCache.engine == engine,
        DetectionCache.model_version == model_version,
        DetectionCache.prompt == prompt,
    ).first()


def save_detection_cache(
    db: Session,
    project_id: int,
    asset_id: str,
    engine: str,
    model_version: str,
    prompt: str,
    result: Dict,
) -> DetectionCache:
    """保存检测缓存（upsert）"""
    cache = get_detection_cache(db, project_id, asset_id, engine, model_version, prompt)
    if cache:
        cache.result = result
        cache.created_at = datetime.utcnow()
    else:
        cache = DetectionCache(
            project_id=project_id,
            asset_id=asset_id,
            engine=engine,
            model_version=model_version,
            prompt=prompt,
            result=result,
        )
        db.add(cache)
    db.flush()
    return cache


def get_all_detections(db: Session, project_id: int) -> Dict[str, Dict]:
    """获取项目所有检测缓存（兼容旧接口）"""
    caches = db.query(DetectionCache).filter(DetectionCache.project_id == project_id).all()
    result = {}
    for c in caches:
        result.setdefault(c.asset_id, {})[c.engine] = c.result
    return result


# ============================================================
# InferenceCache 服务
# ============================================================

def get_inference_cache(
    db: Session,
    asset_id: str,
    model_name: str,
    model_version: str,
    input_hash: str,
    prompt_version: str = None,
) -> Optional[InferenceCache]:
    """获取推理缓存"""
    query = db.query(InferenceCache).filter(
        InferenceCache.asset_id == asset_id,
        InferenceCache.model_name == model_name,
        InferenceCache.model_version == model_version,
        InferenceCache.input_hash == input_hash,
    )
    if prompt_version:
        query = query.filter(InferenceCache.prompt_version == prompt_version)
    return query.first()


def save_inference_cache(
    db: Session,
    asset_id: str,
    model_name: str,
    model_version: str,
    input_hash: str,
    output: Dict,
    prompt_version: str = None,
) -> InferenceCache:
    """保存推理缓存"""
    cache = get_inference_cache(db, asset_id, model_name, model_version, input_hash, prompt_version)
    if cache:
        cache.output = output
        cache.created_at = datetime.utcnow()
    else:
        cache = InferenceCache(
            asset_id=asset_id,
            model_name=model_name,
            model_version=model_version,
            input_hash=input_hash,
            prompt_version=prompt_version,
            output=output,
        )
        db.add(cache)
    db.flush()
    return cache


# ============================================================
# ModelRegistry 服务
# ============================================================

def register_model(
    db: Session,
    name: str,
    version: str,
    model_path: str = None,
    config: Dict = None,
    is_active: bool = False,
) -> ModelRegistry:
    """注册模型版本"""
    model = db.query(ModelRegistry).filter(
        ModelRegistry.name == name,
        ModelRegistry.version == version,
    ).first()
    if model:
        model.model_path = model_path
        model.config = config or {}
        model.is_active = is_active
    else:
        model = ModelRegistry(
            name=name,
            version=version,
            model_path=model_path,
            config=config or {},
            is_active=is_active,
        )
        db.add(model)
    db.flush()
    return model


def get_active_model(db: Session, name: str) -> Optional[ModelRegistry]:
    """获取当前激活的模型版本"""
    return db.query(ModelRegistry).filter(
        ModelRegistry.name == name,
        ModelRegistry.is_active == True,
    ).first()


def set_active_model(db: Session, name: str, version: str):
    """设置激活模型版本"""
    db.query(ModelRegistry).filter(ModelRegistry.name == name).update({ModelRegistry.is_active: False})
    model = db.query(ModelRegistry).filter(
        ModelRegistry.name == name,
        ModelRegistry.version == version,
    ).first()
    if model:
        model.is_active = True


# ============================================================
# Benchmark & HardCase 服务
# ============================================================

def add_benchmark_sample(
    db: Session,
    project_id: int,
    asset_id: str,
    gt_tags: Dict,
    split: str = "val",
) -> Benchmark:
    """添加 Benchmark 样本"""
    bm = Benchmark(
        benchmark_id=str(uuid.uuid4())[:16],
        project_id=project_id,
        asset_id=asset_id,
        split=split,
        gt_tags=gt_tags,
    )
    db.add(bm)
    db.flush()
    return bm


def get_benchmark_samples(db: Session, project_id: int, split: str = None) -> List[Benchmark]:
    """获取 Benchmark 样本"""
    query = db.query(Benchmark).filter(Benchmark.project_id == project_id)
    if split:
        query = query.filter(Benchmark.split == split)
    return query.all()


def save_benchmark_evaluation(
    db: Session,
    project_id: int,
    model_combo: str,
    model_versions: Dict,
    metrics: Dict,
    per_class: Dict = None,
) -> BenchmarkEvaluation:
    """保存评测结果"""
    eval = BenchmarkEvaluation(
        eval_id=str(uuid.uuid4())[:16],
        project_id=project_id,
        model_combo=model_combo,
        model_versions=model_versions,
        metrics=metrics,
        per_class=per_class or {},
    )
    db.add(eval)
    db.flush()
    return eval


def create_hard_case(
    db: Session,
    project_id: int,
    asset_id: str,
    ai_tags: Dict,
    human_tags: Dict,
    final_tags: Dict,
    diff: Dict,
) -> HardCase:
    """创建 Hard Case"""
    hc = HardCase(
        case_id=str(uuid.uuid4())[:16],
        project_id=project_id,
        asset_id=asset_id,
        ai_tags=ai_tags,
        human_tags=human_tags,
        final_tags=final_tags,
        diff=diff,
    )
    db.add(hc)
    db.flush()
    return hc


def list_hard_cases(db: Session, project_id: int, status: str = None) -> List[HardCase]:
    """列出 Hard Case"""
    query = db.query(HardCase).filter(HardCase.project_id == project_id)
    if status:
        query = query.filter(HardCase.status == status)
    return query.order_by(HardCase.created_at.desc()).all()


# ============================================================
# ReviewRecord 服务
# ============================================================

def create_review_record(
    db: Session,
    project_id: int,
    asset_id: int,
    reviewer: str,
    action: ReviewAction,
    ai_tags_snapshot: Dict,
    human_tags: Dict,
    final_tags: Dict,
    comments: str = "",
) -> ReviewRecord:
    """创建审核记录"""
    rr = ReviewRecord(
        review_id=str(uuid.uuid4())[:16],
        project_id=project_id,
        asset_id=asset_id,
        reviewer=reviewer,
        action=action,
        ai_tags_snapshot=ai_tags_snapshot,
        human_tags=human_tags,
        final_tags=final_tags,
        comments=comments,
    )
    db.add(rr)
    db.flush()
    return rr


def get_review_queue(
    db: Session,
    project_id: int,
    filters: Dict = None,
    page: int = 1,
    size: int = 20,
) -> Tuple[List[Asset], int]:
    """获取审核队列（状态为 REVIEW 的 Asset）"""
    query = db.query(Asset).filter(
        Asset.project_id == project_id,
        Asset.status == AssetStatus.REVIEW,
    )
    if filters:
        if filters.get("high_risk"):
            query = query.filter(Asset.final_result["risk"] == "high")
        if filters.get("model_conflict"):
            query = query.filter(Asset.decision_reason.like("%冲突%"))
    total = query.count()
    assets = query.order_by(Asset.updated_at.asc()).offset((page - 1) * size).limit(size).all()
    return assets, total


# ============================================================
# ExportRecord 服务
# ============================================================

def create_export_record(
    db: Session,
    project_id: int,
    format: str,
    filter_criteria: Dict,
    asset_count: int,
    output_path: str,
    manifest: Dict,
) -> ExportRecord:
    """创建导出记录"""
    exp = ExportRecord(
        export_id=str(uuid.uuid4())[:16],
        project_id=project_id,
        format=format,
        filter_criteria=filter_criteria,
        asset_count=asset_count,
        output_path=output_path,
        manifest=manifest,
    )
    db.add(exp)
    db.flush()
    return exp


# ============================================================
# 搜索与分析服务
# ============================================================

def search_assets_by_tags(
    db: Session,
    project_id: int,
    tag_filters: Dict[str, List[str]],  # {weather: ["雨天"], road: ["城市道路"]}
    status: AssetStatus = None,
    page: int = 1,
    size: int = 50,
) -> Tuple[List[Asset], int]:
    """按标签筛选 Asset（结构化搜索，Python 侧匹配 final_tags/ai_tags）
    维度值支持 dict 溯源标签（{tag:...,confidence:...}）或纯字符串。"""
    query = db.query(Asset).filter(Asset.project_id == project_id)
    if status:
        query = query.filter(Asset.status == status)

    def _dim_names(tags_obj, dim):
        vals = (tags_obj or {}).get(dim) or []
        names = set()
        for v in vals:
            if isinstance(v, dict):
                names.add(v.get("tag", ""))
            elif isinstance(v, str):
                names.add(v)
        names.discard("")
        return names

    all_assets = query.all()  # 数据量适中；超大规模需换 SQL JSON path 或倒排
    filtered = []
    for a in all_assets:
        ok = True
        for dim, want_tags in tag_filters.items():
            if not want_tags:
                continue
            have = _dim_names(a.final_tags, dim) | _dim_names(a.ai_tags, dim)
            if not have.intersection(want_tags):
                ok = False
                break
        if ok:
            filtered.append(a)

    total = len(filtered)
    start = (page - 1) * size
    assets = filtered[start:start + size]
    return assets, total


def get_analytics_overview(db: Session, project_id: int) -> Dict[str, Any]:
    """获取仪表盘概览数据"""
    # 基础统计
    total_assets = db.query(Asset).filter(Asset.project_id == project_id).count()
    approved = db.query(Asset).filter(
        Asset.project_id == project_id,
        Asset.status == AssetStatus.APPROVED,
    ).count()
    review_count = db.query(Asset).filter(
        Asset.project_id == project_id,
        Asset.status == AssetStatus.REVIEW,
    ).count()
    
    # 维度分布
    def get_distribution(dim: str, tag_source: str = "final_tags") -> Dict[str, int]:
        assets = db.query(Asset).filter(
            Asset.project_id == project_id,
            Asset.status == AssetStatus.APPROVED,
        ).all()
        dist = {}
        for a in assets:
            tags = getattr(a, tag_source) or {}
            for t in tags.get(dim, []):
                tag_name = t.get("tag") if isinstance(t, dict) else t
                dist[tag_name] = dist.get(tag_name, 0) + 1
        return dist
    
    return {
        "total_assets": total_assets,
        "approved_count": approved,
        "review_count": review_count,
        "ai_coverage": f"{(approved + review_count) / max(1, total_assets) * 100:.1f}%",
        "distributions": {
            "weather": get_distribution("weather"),
            "road": get_distribution("road"),
            "time": get_distribution("time"),
            "objects": get_distribution("objects"),
            "events": get_distribution("events"),
            "risk": get_distribution("risk"),
        },
    }


def get_coverage_analysis(
    db: Session,
    project_id: int,
    requirements: List[Dict],  # [{weather: ["夜晚"], road: ["城市道路"], objects: ["行人"], events: ["行人横穿"]}]
) -> Dict[str, Any]:
    """需求覆盖度分析"""
    results = []
    for req in requirements:
        assets, count = search_assets_by_tags(db, project_id, req, status=AssetStatus.APPROVED, size=10000)
        target = req.get("target", 10000)
        results.append({
            "requirement": req,
            "target": target,
            "current": count,
            "coverage": f"{count / max(1, target) * 100:.1f}%",
            "gap": max(0, target - count),
            "sources": list(set(a.source_id for a in assets)),
        })
    return {"results": results}


# ============================================================
# 迁移入口
# ============================================================

# ============================================================
# 入库链路 DB 同步（双写适配器）：把 JSON metadata 记录同步进 Source/Asset 表
# 幂等：按 (project_id, vector_id) 判重，已存在则只补路径更新，不重复建源
# ============================================================

def sync_asset_records_to_db(project_name: str, records: List[dict],
                             img_root: str = None) -> Dict[str, int]:
    """
    把一批入库记录（含 id/filename/path/src_dir）同步为 DB Source + Asset。
    - Source：以 image_path 目录为 source_root 简化登记（file_hash 由后续 SCAN 任务补齐）
    - Asset：vector_id = 记录 id；已存在则跳过（保持幂等），仅缺 asset 时创建
    返回 {"assets": n, "sources": n, "skipped": n}
    """
    if not records:
        return {"assets": 0, "sources": 0, "skipped": 0}
    from models import get_or_create_project
    db = get_db_session()
    try:
        proj = get_or_create_project(db, project_name)
        existing_ids = {v for (v,) in db.query(Asset.vector_id).filter(Asset.project_id == proj.id).all()}

        new_assets = 0
        new_sources = 0
        skipped = 0
        for rec in records:
            vid = rec.get("id")
            if vid is None:
                continue
            try:
                vid = int(vid)
            except Exception:
                continue
            if vid in existing_ids:
                skipped += 1
                continue
            image_path = rec.get("path") or ""
            if not image_path:
                skipped += 1
                continue
            filename = rec.get("filename") or os.path.basename(image_path)

            # 源登记：同目录同文件名视为同一源（幂等）
            source_root = os.path.dirname(image_path) or (img_root or ".")
            src = db.query(Source).filter(
                Source.project_id == proj.id,
                Source.relative_path == os.path.basename(image_path),
            ).first()
            if src is None:
                src = Source(
                    source_id=str(uuid.uuid4())[:16],
                    project_id=proj.id,
                    source_type=SourceType.IMAGE,
                    source_root=source_root,
                    relative_path=os.path.basename(image_path),
                    directory_chain=[],
                    file_name=filename,
                    extension=os.path.splitext(filename)[1].lower(),
                    file_size=os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                    meta={"src_dir": rec.get("src_dir")} if rec.get("src_dir") else {},
                    scan_status="synced",
                )
                db.add(src)
                new_sources += 1
                db.flush()
            # Asset 登记
            w = h = 0
            if os.path.exists(image_path):
                try:
                    from PIL import Image as _PILImage
                    with _PILImage.open(image_path) as im:
                        w, h = im.size
                except Exception:
                    pass
            asset = Asset(
                asset_id=str(uuid.uuid4())[:16],
                project_id=proj.id,
                source_id=src.id,
                source_type=SourceType.IMAGE,
                image_path=image_path,
                frame_index=0,
                vector_id=vid,
                width=w, height=h,
                status=AssetStatus.EXTRACTED,
            )
            db.add(asset)
            new_assets += 1
        db.commit()
        return {"assets": new_assets, "sources": new_sources, "skipped": skipped}
    finally:
        db.close()


def count_db_assets(project_name: str) -> int:
    """统计某项目 DB Asset 数量（供同步进度核对）"""
    from models import get_or_create_project
    db = get_db_session()
    try:
        proj = get_or_create_project(db, project_name)
        return db.query(Asset).filter(Asset.project_id == proj.id).count()
    finally:
        db.close()


def ensure_sync_hook(project_name: str, metadata_slice: List[dict]):
    """供 backend 各入库 worker 在 save_project_context 后调用（吞异常，不阻塞主流程）"""
    try:
        return sync_asset_records_to_db(project_name, metadata_slice)
    except Exception as e:
        print(f"[sync] 入库 DB 同步失败(不阻塞): {e}")
        return {"assets": 0, "sources": 0, "skipped": 0, "error": str(e)}



def run_migration():
    """运行完整迁移：初始化表 + 导入旧 JSON 数据"""
    print("[Migration] 初始化数据库...")
    init_db()

    print("[Migration] 导入旧 JSON 数据...")
    db = get_db_session()
    try:
        from settings import data_root as _dr
        projects_root = os.path.join(_dr(), "projects")
        if os.path.exists(projects_root):
            migrate_from_json(db, projects_root)
        print("[Migration] 完成")
    finally:
        db.close()


if __name__ == "__main__":
    run_migration()