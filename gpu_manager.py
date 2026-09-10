# -*- coding: utf-8 -*-
"""
GPU Model Manager - 统一显存调度
PRD 54: GPU 资源管理 - 加载/释放/任务排队/Batch/显存检查统一控制
"""

import os
import torch
import threading
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from enum import Enum


class ModelName(Enum):
    SIGLIP = "siglip"
    YOLO = "yolo"
    DINO = "dino"
    VLM = "vlm"


class GPUModelManager:
    """
    统一 GPU 模型管理器
    - 维护模型引用计数
    - 自动 LRU 释放显存
    - 显存预检防止 OOM
    - 线程安全
    """
    
    def __init__(self, device: str = None, max_vram_usage_gb: float = 11.0):
        """
        Args:
            device: cuda/cpu，默认自动检测
            max_vram_usage_gb: 最大显存使用阈值(GB)，留 1G 余量给系统
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_vram_bytes = int(max_vram_usage_gb * 1024 ** 3)
        
        # 模型状态: {model_name: {"model": obj, "processor": obj, "ref_count": int, "last_used": float, "loaded": bool}}
        self._models: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # 加载器映射：model_name -> loader_fn
        self._loaders: Dict[str, Callable] = {}
        
        print(f"[GPUManager] 初始化完成，设备: {self.device}, 显存上限: {max_vram_usage_gb}GB")
    
    def register_loader(self, model_name: str, loader_fn: Callable):
        """注册模型加载函数
        loader_fn 签名: () -> (model, processor) 或 () -> model
        """
        with self._lock:
            self._loaders[model_name] = loader_fn
    
    def _get_vram_info(self) -> tuple:
        """获取显存信息 (free, total) bytes"""
        if self.device == "cuda" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free, total
        return (float('inf'), float('inf'))
    
    def _check_vram(self, required_gb: float = 2.0) -> bool:
        """检查显存是否足够"""
        if self.device != "cuda":
            return True
        free, _ = self._get_vram_info()
        return free >= required_gb * 1024 ** 3
    
    def _evict_lru(self, exclude: set = None, required_gb: float = 2.0):
        """按 LRU 释放模型，直到能腾出 required_gb（此前用固定 2GB 判断，
        加载 10G 级模型时会出现“判断够用、真加载就 OOM”）"""
        exclude = exclude or set()
        with self._lock:
            # 按最后使用时间排序
            candidates = [
                (name, info["last_used"]) 
                for name, info in self._models.items() 
                if info["loaded"] and name not in exclude and info["ref_count"] == 0
            ]
            candidates.sort(key=lambda x: x[1])  # 最旧的在前
            
            for name, _ in candidates:
                if self._check_vram(required_gb):  # 已腾够
                    break
                self._unload_model(name)
            self._check_vram(required_gb)
    
    def _unload_model(self, model_name: str):
        """卸载单个模型"""
        if model_name not in self._models:
            return
        
        info = self._models[model_name]
        if info["model"] is not None:
            try:
                del info["model"]
            except Exception:
                pass
            info["model"] = None
        if info["processor"] is not None:
            try:
                del info["processor"]
            except Exception:
                pass
            info["processor"] = None
        info["loaded"] = False
        
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        print(f"[GPUManager] 已卸载 {model_name}")
    
    def acquire(self, model_name: str, required_gb: float = 2.0) -> tuple:
        """
        获取模型（加载或复用）
        Returns: (model, processor)
        """
        with self._lock:
            # 显存预检
            if not self._check_vram(required_gb):
                print(f"[GPUManager] 显存不足(需 {required_gb}GB)，尝试释放 LRU 模型...")
                self._evict_lru(exclude={model_name}, required_gb=required_gb)
                if not self._check_vram(required_gb):
                    free, _ = self._get_vram_info()
                    raise RuntimeError(
                        f"显存不足，无法加载 {model_name} (需要 {required_gb}GB，实际可用 {free / 1024**3:.2f}GB)")
            
            # 初始化模型状态
            if model_name not in self._models:
                self._models[model_name] = {
                    "model": None,
                    "processor": None,
                    "ref_count": 0,
                    "last_used": 0.0,
                    "loaded": False,
                }
            
            info = self._models[model_name]
            
            # 如果已加载，直接复用
            if info["loaded"] and info["model"] is not None:
                info["ref_count"] += 1
                # 注意：不要用未 record 的 torch.cuda.Event 求 elapsed_time（会抛
                # "Both events must be recorded"），时间戳一律用 time.time()
                import time
                info["last_used"] = time.time()
                return info["model"], info["processor"]
            
            # 需要加载
            if model_name not in self._loaders:
                raise ValueError(f"未注册加载器: {model_name}")
            
            print(f"[GPUManager] 正在加载 {model_name}...")
            model, processor = self._loaders[model_name]()
            
            info["model"] = model
            info["processor"] = processor
            info["ref_count"] = 1
            info["loaded"] = True
            import time
            info["last_used"] = time.time()
            
            print(f"[GPUManager] {model_name} 加载完成")
            return model, processor
    
    def release(self, model_name: str):
        """释放模型引用（引用计数归零时可被 LRU 回收）"""
        with self._lock:
            if model_name not in self._models:
                return
            info = self._models[model_name]
            info["ref_count"] = max(0, info["ref_count"] - 1)
            import time
            info["last_used"] = time.time()
    
    @contextmanager
    def use_model(self, model_name: str, required_gb: float = 2.0):
        """上下文管理器：自动 acquire/release"""
        model, processor = self.acquire(model_name, required_gb)
        try:
            yield model, processor
        finally:
            self.release(model_name)
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """获取已加载模型列表"""
        with self._lock:
            return {name: info["loaded"] for name, info in self._models.items()}
    
    def force_unload(self, model_name: str):
        """强制卸载（忽略引用计数）"""
        with self._lock:
            if model_name in self._models:
                self._models[model_name]["ref_count"] = 0
                self._unload_model(model_name)
    
    def unload_all(self):
        """卸载所有模型"""
        with self._lock:
            for name in list(self._models.keys()):
                self._unload_model(name)
    
    def get_vram_usage(self) -> Dict[str, float]:
        """获取显存使用情况 (GB)"""
        if self.device != "cuda":
            return {"free": float('inf'), "total": float('inf'), "used": 0.0}
        free, total = self._get_vram_info()
        return {
            "free": free / 1024**3,
            "total": total / 1024**3,
            "used": (total - free) / 1024**3,
        }


# ============================================================
# 全局单例
# ============================================================
_gpu_manager: Optional[GPUModelManager] = None
_gpu_manager_lock = threading.Lock()


def get_gpu_manager() -> GPUModelManager:
    """获取全局 GPUManager 实例"""
    global _gpu_manager
    with _gpu_manager_lock:
        if _gpu_manager is None:
            _gpu_manager = GPUModelManager()
        return _gpu_manager


def init_gpu_manager(device: str = None, max_vram_gb: float = 11.0) -> GPUModelManager:
    """显式初始化（启动时调用）"""
    global _gpu_manager
    with _gpu_manager_lock:
        _gpu_manager = GPUModelManager(device=device, max_vram_usage_gb=max_vram_gb)
        return _gpu_manager


# ============================================================
# 便捷装饰器
# ============================================================

def with_model(model_name: str, required_gb: float = 2.0):
    """
    装饰器：自动管理模型生命周期
    @with_model("dino", required_gb=4.0)
    def my_dino_func(dino_model, dino_processor, ...):
        ...
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            mgr = get_gpu_manager()
            with mgr.use_model(model_name, required_gb) as (model, processor):
                return fn(model, processor, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # 简单测试
    mgr = GPUModelManager(device="cpu")
    
    def dummy_loader():
        class DummyModel:
            def __call__(self, x): return x
        return DummyModel(), None
    
    mgr.register_loader("test", dummy_loader)
    
    with mgr.use_model("test") as (model, proc):
        print(f"Got model: {model}")
    
    print(f"Loaded: {mgr.get_loaded_models()}")
    print(f"VRAM: {mgr.get_vram_usage()}")