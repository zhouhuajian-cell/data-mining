# -*- coding: utf-8 -*-
"""
三级去重系统 - PRD 8.1-8.4
- 一级：SHA256 完全一致去重（入库时拦截）
- 二级：pHash 感知哈希近重复检测（汉明距离）
- 三级：SigLIP Cosine 语义相似度（标记供人工确认，禁止自动删除）
"""

import os
import hashlib
import threading
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from PIL import Image
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False
    print("[Dedup] imagehash 不可用，二级去重将跳过")

from db_service import get_db_session, Source, Asset, SourceType
from models import engine


@dataclass
class DuplicateGroup:
    """重复组"""
    group_id: str
    level: int  # 1=完全一致, 2=感知哈希, 3=语义相似
    assets: List[Dict]  # [{asset_id, image_path, score}]
    representative: str  # 保留的 asset_id


class DedupEngine:
    """三级去重引擎"""
    
    def __init__(self, 
                 phash_threshold: int = 5,      # pHash 汉明距离阈值
                 siglip_threshold: float = 0.95, # SigLIP 余弦相似度阈值
                 batch_size: int = 1000):
        self.phash_threshold = phash_threshold
        self.siglip_threshold = siglip_threshold
        self.batch_size = batch_size
        self._lock = threading.RLock()
    
    # ==================== 一级去重：SHA256 ====================
    
    @staticmethod
    def compute_sha256(filepath: str) -> Optional[str]:
        """计算文件 SHA256"""
        try:
            hasher = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def check_exact_duplicate(self, project_id: int, file_hash: str) -> Optional[Source]:
        """检查是否已存在相同哈希的源文件（一级去重）"""
        if not file_hash:
            return None
        db = get_db_session()
        try:
            return db.query(Source).filter(
                Source.project_id == project_id,
                Source.file_hash == file_hash
            ).first()
        finally:
            db.close()
    
    # ==================== 二级去重：pHash ====================
    
    @staticmethod
    def compute_phash(filepath: str) -> Optional[str]:
        """计算感知哈希"""
        if not PHASH_AVAILABLE:
            return None
        try:
            img = Image.open(filepath).convert("RGB")
            return str(imagehash.phash(img))
        except Exception:
            return None
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """计算汉明距离"""
        if len(hash1) != len(hash2):
            return 999
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def find_phash_duplicates(self, project_id: int, phash: str, 
                              exclude_asset_id: str = None) -> List[Asset]:
        """查找 pHash 近似的资产（二级去重）"""
        if not phash or not PHASH_AVAILABLE:
            return []
        
        db = get_db_session()
        try:
            # 获取所有有 phash 的资产
            assets = db.query(Asset).join(Source).filter(
                Source.project_id == project_id,
                Source.phash.isnot(None)
            ).all()
            
            if exclude_asset_id:
                assets = [a for a in assets if a.asset_id != exclude_asset_id]
            
            duplicates = []
            for asset in assets:
                src_phash = asset.source.phash if asset.source else None
                if src_phash:
                    dist = self.hamming_distance(phash, src_phash)
                    if dist <= self.phash_threshold:
                        duplicates.append(asset)
            
            return duplicates
        finally:
            db.close()
    
    # ==================== 三级去重：SigLIP ====================
    
    def find_siglip_duplicates(self, project_id: int, vector_id: int,
                               exclude_asset_id: str = None) -> List[Tuple[Asset, float]]:
        """查找 SigLIP 语义相似的资产（三级去重，需 FAISS 索引）"""
        # 这个需要访问项目的 FAISS 索引，由上层调用
        # 这里返回接口定义，实际实现依赖 backend 中的 extract_and_index_project 上下文
        return []
    
    # ==================== 综合去重入口 ====================
    
    def process_new_asset(self, project_id: int, asset: Asset, 
                          filepath: str) -> Dict[str, any]:
        """
        新资产入库时执行三级去重检查
        返回: {
            'exact_duplicate': Source or None,
            'phash_duplicates': [Asset],
            'siglip_duplicates': [(Asset, score)],
            'action': 'skip' | 'mark_duplicate' | 'keep'
        }
        """
        result = {
            'exact_duplicate': None,
            'phash_duplicates': [],
            'siglip_duplicates': [],
            'action': 'keep',
            'duplicate_group_id': None,
        }
        
        # 1. 一级去重：SHA256
        file_hash = asset.source.file_hash if asset.source else None
        if not file_hash and os.path.exists(filepath):
            file_hash = self.compute_sha256(filepath)
            if file_hash and asset.source:
                # 更新 source 的 file_hash
                db = get_db_session()
                try:
                    src = db.query(Source).filter(Source.id == asset.source_id).first()
                    if src:
                        src.file_hash = file_hash
                        db.commit()
                finally:
                    db.close()
        
        if file_hash:
            exact_dup = self.check_exact_duplicate(project_id, file_hash)
            if exact_dup and exact_dup.id != asset.source_id:
                result['exact_duplicate'] = exact_dup
                result['action'] = 'skip'  # 完全一致直接跳过入库
                return result
        
        # 2. 二级去重：pHash
        phash = asset.source.phash if asset.source else None
        if not phash and os.path.exists(filepath):
            phash = self.compute_phash(filepath)
            if phash and asset.source:
                db = get_db_session()
                try:
                    src = db.query(Source).filter(Source.id == asset.source_id).first()
                    if src:
                        src.phash = phash
                        db.commit()
                finally:
                    db.close()
        
        if phash:
            phash_dups = self.find_phash_duplicates(project_id, phash, asset.asset_id)
            if phash_dups:
                result['phash_duplicates'] = phash_dups
                result['action'] = 'mark_duplicate'
                # 创建/分配 duplicate_group_id
                group_id = f"phash_{phash[:8]}"
                result['duplicate_group_id'] = group_id
                # 标记所有相关资产
                self._mark_duplicate_group(project_id, [asset] + phash_dups, group_id, level=2)
        
        return result
    
    def _mark_duplicate_group(self, project_id: int, assets: List[Asset], 
                              group_id: str, level: int):
        """标记重复组"""
        db = get_db_session()
        try:
            for asset in assets:
                meta = asset.asset_metadata or {}
                dup_groups = meta.get('duplicate_groups', [])
                if group_id not in dup_groups:
                    dup_groups.append(group_id)
                meta['duplicate_groups'] = dup_groups
                meta['duplicate_level'] = level
                asset.asset_metadata = meta
            db.commit()
        finally:
            db.close()
    
    # ==================== 批量扫描去重 ====================
    
    def scan_full_dedup(self, project_id: int, 
                        siglip_index=None, siglip_metadata=None) -> Dict:
        """
        全量扫描三级去重（用于 /api/dedup_stats）
        返回: 与原有 dedup_stats 兼容的格式
        """
        db = get_db_session()
        try:
            assets = db.query(Asset).join(Source).filter(
                Source.project_id == project_id
            ).all()
            
            total = len(assets)
            if total < 2:
                return {
                    "total_images": total,
                    "unique_images": total,
                    "duplicate_count": 0,
                    "dedup_rate": "0.0%",
                    "clusters": []
                }
            
            # 一级去重：按 SHA256 分组
            hash_groups = {}
            for asset in assets:
                if asset.source and asset.source.file_hash:
                    h = asset.source.file_hash
                    hash_groups.setdefault(h, []).append(asset)
            
            # 二级去重：按 pHash 聚类
            phash_groups = {}
            for asset in assets:
                if asset.source and asset.source.phash:
                    ph = asset.source.phash
                    phash_groups.setdefault(ph, []).append(asset)
            
            # 合并一二级去重结果
            visited = set()
            clusters = []
            duplicate_count = 0
            
            # 处理 SHA256 完全一致组
            for h, group in hash_groups.items():
                if len(group) > 1:
                    cluster_items = []
                    for asset in group:
                        visited.add(asset.asset_id)
                        cluster_items.append({
                            "asset_id": asset.asset_id,
                            "filename": os.path.basename(asset.image_path),
                            "path": asset.image_path,
                            "level": 1,
                            "score": 1.0
                        })
                        duplicate_count += 1
                    if cluster_items:
                        clusters.append({
                            "type": "exact",
                            "hash": h,
                            "items": cluster_items
                        })
            
            # 处理 pHash 近似组（排除已在一级去重中的）
            for ph, group in phash_groups.items():
                unvisited = [a for a in group if a.asset_id not in visited]
                if len(unvisited) > 1:
                    cluster_items = []
                    for asset in unvisited:
                        visited.add(asset.asset_id)
                        cluster_items.append({
                            "asset_id": asset.asset_id,
                            "filename": os.path.basename(asset.image_path),
                            "path": asset.image_path,
                            "level": 2,
                            "score": 1.0 - (self.phash_threshold / 64.0)  # 近似分数
                        })
                        duplicate_count += 1
                    if cluster_items:
                        clusters.append({
                            "type": "phash",
                            "phash": ph,
                            "items": cluster_items
                        })
            
            # 三级去重：SigLIP（如果提供了索引）
            if siglip_index is not None and siglip_metadata is not None:
                siglip_clusters = self._siglip_dedup(
                    siglip_index, siglip_metadata, visited
                )
                clusters.extend(siglip_clusters)
                for c in siglip_clusters:
                    duplicate_count += len(c['items']) - 1
            
            unique_images = total - duplicate_count
            rate = f"{(duplicate_count / total * 100):.2f}%" if total > 0 else "0.0%"
            
            return {
                "total_images": total,
                "unique_images": unique_images,
                "duplicate_count": duplicate_count,
                "dedup_rate": rate,
                "clusters": clusters
            }
        finally:
            db.close()
    
    def _siglip_dedup(self, index, metadata: List, visited: Set) -> List[Dict]:
        """SigLIP 语义去重"""
        total = index.ntotal
        if total < 2:
            return []
        
        # 重建特征矩阵
        feats = index.reconstruct_n(0, total)
        sim_matrix = np.dot(feats, feats.T)
        np.fill_diagonal(sim_matrix, 0)
        
        clusters = []
        for i in range(total):
            asset_id = metadata[i].get('asset_id') or metadata[i].get('id')
            if asset_id in visited:
                continue
            
            sim_indices = np.where(sim_matrix[i] >= self.siglip_threshold)[0]
            cluster_items = [{
                "asset_id": asset_id,
                "filename": metadata[i].get('filename', ''),
                "path": metadata[i].get('path', ''),
                "level": 3,
                "score": 1.0
            }]
            
            for idx in sim_indices:
                other_asset_id = metadata[idx].get('asset_id') or metadata[idx].get('id')
                if other_asset_id not in visited:
                    visited.add(other_asset_id)
                    cluster_items.append({
                        "asset_id": other_asset_id,
                        "filename": metadata[idx].get('filename', ''),
                        "path": metadata[idx].get('path', ''),
                        "level": 3,
                        "score": float(sim_matrix[i][idx])
                    })
            
            if len(cluster_items) > 1:
                clusters.append({
                    "type": "siglip",
                    "items": cluster_items
                })
        
        return clusters


# 全局实例
_dedup_engine = None
_dedup_lock = threading.Lock()

def get_dedup_engine() -> DedupEngine:
    global _dedup_engine
    with _dedup_lock:
        if _dedup_engine is None:
            _dedup_engine = DedupEngine()
        return _dedup_engine


if __name__ == "__main__":
    # 简单测试
    engine = DedupEngine()
    
    # 测试 SHA256
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        f.write(b'test content')
        tmp_path = f.name
    
    hash1 = engine.compute_sha256(tmp_path)
    hash2 = engine.compute_sha256(tmp_path)
    print(f"SHA256: {hash1} == {hash2}: {hash1 == hash2}")
    
    # 测试 pHash
    if PHASH_AVAILABLE:
        phash = engine.compute_phash(tmp_path)
        print(f"pHash: {phash}")
    
    os.unlink(tmp_path)
    print("DedupEngine test passed!")