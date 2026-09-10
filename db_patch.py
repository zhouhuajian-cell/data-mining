# -*- coding: utf-8 -*-
"""
数据库集成补丁 - 替换 backend.py 中的 JSON 文件操作
使用方式：在 backend.py 顶部 import 后调用 apply_db_patches()
"""

import os
import json
import threading
from typing import List, Optional, Dict, Any

# 导入数据库服务
from db_service import (
    db_session, get_db_session,
    Project, Source, Asset, Frame, Job, DetectionCache, InferenceCache,
    SourceType, AssetStatus, JobType, JobStatus, DecisionStatus, ReviewAction,
    create_project, get_project, list_projects, rename_project, delete_project,
    get_project_stats,
    create_source, get_source_by_hash, find_similar_sources, list_sources,
    create_asset, get_asset, get_asset_by_vector_id, list_assets,
    update_asset_status, update_asset_vector_id, update_asset_detections,
    update_asset_tags, update_asset_decision, update_asset_review,
    update_asset_final_result, update_asset_model_versions,
    create_frame, get_frames_by_source,
    create_job, update_job_status, get_job, list_jobs, get_running_jobs,
    get_detection_cache, save_detection_cache, get_all_detections,
    get_inference_cache, save_inference_cache,
    register_model, get_active_model, set_active_model,
    add_benchmark_sample, get_benchmark_samples,
    save_benchmark_evaluation, create_hard_case, list_hard_cases,
    create_review_record, get_review_queue,
    create_export_record,
    search_assets_by_tags, get_analytics_overview, get_coverage_analysis,
    run_migration,
)


# ============================================================
# 兼容层：模拟原有的 project_cache 行为
# ============================================================

class _DBProjectContext:
    """兼容原有 project_cache 字典结构"""
    def __init__(self, project_name: str):
        self.name = project_name
        self.project = None
        self._load_project()
    
    def _load_project(self):
        db = get_db_session()
        try:
            self.project = get_project(db, self.name)
            if not self.project:
                self.project = create_project(db, self.name)
        finally:
            db.close()
    
    @property
    def dir(self):
        from settings import data_root as _dr
        return os.path.join(_dr(), "projects", self.name)
    
    @property
    def img_dir(self):
        return os.path.join(self.dir, "images")
    
    @property
    def idx_path(self):
        return os.path.join(self.dir, "index.faiss")
    
    @property
    def meta_path(self):
        return os.path.join(self.dir, "metadata.json")
    
    @property
    def index(self):
        # 返回一个兼容对象，实际 FAISS 索引仍由后端管理
        class _FakeIndex:
            def __init__(self):
                self.d = 1152
                self.ntotal = 0
            def add(self, feats):
                try:
                    self.ntotal += int(feats.shape[0])
                except Exception:
                    pass
            def search(self, q, k):
                import numpy as np
                n = getattr(q, "shape", [1])[0]
                return (np.zeros((n, k), dtype=np.float32), np.zeros((n, k), dtype=np.int64))
        return _FakeIndex()
    
    @property
    def metadata(self):
        """兼容：返回列表格式的 metadata"""
        db = get_db_session()
        try:
            assets, _ = list_assets(db, self.project.id, size=10000)
            result = []
            for a in assets:
                result.append({
                    "id": a.vector_id,
                    "filename": os.path.basename(a.image_path),
                    "path": a.image_path,
                    "url": f"/api/image/{self.name}/{a.vector_id}",
                    "asset_id": a.asset_id,
                })
            return result
        finally:
            db.close()


# 全局缓存（保持原有接口兼容）
project_cache = {}
_detections_cache = {}
_detections_lock = threading.Lock()


def load_project_context(project_name: str) -> Dict:
    """兼容原有 load_project_context"""
    if project_name in project_cache:
        return project_cache[project_name]
    
    ctx = _DBProjectContext(project_name)
    project_cache[project_name] = ctx
    return ctx


def save_project_context(ctx):
    """兼容原有 save_project_context - 现在主要同步 FAISS 索引文件"""
    # 元数据已在数据库，只需保存 FAISS 索引到磁盘
    pass


# ============================================================
# 检测缓存兼容层
# ============================================================

DETECTIONS_CACHE_FILE = os.path.join("workspace", "detections_cache.json")


def _det_cache_path():
    """检测缓存路径跟随数据根（settings.data_root）"""
    from settings import data_root as _dr
    return os.path.join(_dr(), "detections_cache.json")

def _persist_detections():
    """兼容：保存检测缓存到文件（保留向后兼容）"""
    try:
        with open(DETECTIONS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_detections_cache, f, ensure_ascii=False)
    except Exception:
        pass


def save_detection_record(project: str, image_id: int, result: dict):
    """兼容原有 save_detection_record - 双写数据库和内存"""
    global _detections_cache
    proj = project or "default"
    
    # 写数据库
    db = get_db_session()
    try:
        proj_obj = get_project(db, proj)
        if proj_obj:
            # 获取 asset_id
            assets, _ = list_assets(db, proj_obj.id, size=10000)
            if 0 <= image_id < len(assets):
                asset = assets[image_id]
                save_detection_cache(
                    db, proj_obj.id, asset.asset_id,
                    result.get("engine", "unknown"),
                    result.get("model_version", "latest"),
                    result.get("prompt", ""),
                    result,
                )
    finally:
        db.close()
    
    # 写内存缓存（兼容旧代码）
    with _detections_lock:
        sub = _detections_cache.setdefault(proj, {})
        sub[str(image_id)] = result
    
    # 异步持久化到文件
    _persist_detections()


# 加载旧缓存文件（启动时）
if os.path.exists(DETECTIONS_CACHE_FILE):
    try:
        with open(DETECTIONS_CACHE_FILE, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
        _detections_cache = _loaded if isinstance(_loaded, dict) else {}
        _total = sum(len(v) for v in _detections_cache.values() if isinstance(v, dict))
        print(f"✅ 成功恢复 {_total} 帧历史目标检测记录")
    except Exception:
        _detections_cache = {}


def get_all_detections_compat(project: str = "default") -> Dict:
    """兼容原有 get_all_detections 接口"""
    db = get_db_session()
    try:
        proj_obj = get_project(db, project)
        if proj_obj:
            return get_all_detections(db, proj_obj.id)
    finally:
        db.close()
    return _detections_cache.get(project, {})


# ============================================================
# 任务状态兼容层
# ============================================================

# 原有的 task_status, delivery_status, extract_task_pool 保持不变
# 新增：数据库 Job 表作为权威来源

def sync_task_status_to_db(task_status_dict: dict, project: str = "default"):
    """将内存 task_status 同步到数据库 Job 表"""
    if not task_status_dict.get("is_running"):
        return
    
    db = get_db_session()
    try:
        proj_obj = get_project(db, project)
        if not proj_obj:
            return
        
        job_type_map = {
            "import": JobType.EXTRACT,
            "delivery": JobType.EXPORT,
            "extract": JobType.EXTRACT,
        }
        job_type = job_type_map.get(task_status_dict.get("task_type", ""), JobType.EXTRACT)
        
        # 查找或创建 Job
        job = db.query(Job).filter(
            Job.project_id == proj_obj.id,
            Job.job_type == job_type,
            Job.status == JobStatus.RUNNING,
        ).first()
        
        if not job:
            job = create_job(db, proj_obj.id, job_type, {"path": task_status_dict.get("current_path", "")})
        
        update_job_status(
            db, job.job_id,
            JobStatus.RUNNING,
            progress=task_status_dict.get("processed_count", 0) / max(1, task_status_dict.get("total_count", 1)) * 100,
            current_stage=task_status_dict.get("msg", ""),
        )
    finally:
        db.close()


def sync_db_jobs_to_memory():
    """从数据库恢复运行中的任务到内存（启动时调用）"""
    db = get_db_session()
    try:
        running_jobs = get_running_jobs(db)
        for job in running_jobs:
            # 这里可以恢复到 task_status/delivery_status/extract_task_pool
            # 具体逻辑视后端需求而定
            pass
    finally:
        db.close()


# ============================================================
# 应用补丁函数
# ============================================================

def apply_db_patches():
    """
    在 backend.py 启动时调用此函数，完成数据库初始化和迁移
    """
    print("[DB Patch] 初始化数据库...")
    run_migration()
    print("[DB Patch] 数据库补丁已应用")


# ============================================================
# 新增 API 所需的服务函数（供后续 Phase 2+ 使用）
# ============================================================

def get_project_db_stats(project_name: str) -> Dict[str, int]:
    """获取项目统计（供 /api/db_stats 使用）"""
    db = get_db_session()
    try:
        proj = get_project(db, project_name)
        if not proj:
            return {"raw_count": 0, "processed_count": 0, "pending_count": 0}
        return get_project_stats(db, proj.id)
    finally:
        db.close()


def list_all_projects() -> List[str]:
    """列出所有项目名（供 /api/projects 使用）"""
    db = get_db_session()
    try:
        projects = list_projects(db)
        names = [p.name for p in projects]
        if "default" not in names:
            names.insert(0, "default")
        return sorted(names)
    finally:
        db.close()


def create_project_db(project_name: str) -> str:
    """创建项目（供 /api/projects/create 使用）"""
    db = get_db_session()
    try:
        proj = create_project(db, project_name)
        return proj.name
    finally:
        db.close()


def rename_project_db(old_name: str, new_name: str) -> str:
    """重命名项目（供 /api/projects/rename 使用）"""
    db = get_db_session()
    try:
        proj = rename_project(db, old_name, new_name)
        if proj:
            return proj.name
        raise ValueError("项目不存在或新名称已存在")
    finally:
        db.close()


def delete_project_db(project_name: str) -> bool:
    """删除项目（供 /api/projects/delete 使用）"""
    db = get_db_session()
    try:
        return delete_project(db, project_name)
    finally:
        db.close()


if __name__ == "__main__":
    apply_db_patches()