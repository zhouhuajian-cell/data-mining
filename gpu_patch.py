# -*- coding: utf-8 -*-
"""
GPU Model Manager 集成补丁 - 替换 backend.py 中分散的模型加载/释放逻辑
"""

from gpu_manager import GPUModelManager, get_gpu_manager, init_gpu_manager, with_model, ModelName


# ============================================================
# 注册现有模型加载器
# ============================================================

def register_siglip_loader(device: str):
    """注册 SigLIP 加载器"""
    from gpu_manager import get_gpu_manager
    mgr = get_gpu_manager()
    
    def loader():
        from transformers import AutoProcessor, AutoModel
        import torch
        
        SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME, local_files_only=True)
        model = AutoModel.from_pretrained(
            SIGLIP_MODEL_NAME, 
            torch_dtype=dtype, 
            local_files_only=True
        ).to(device)
        model.eval()
        return model, processor
    
    mgr.register_loader(ModelName.SIGLIP.value, loader)


def register_yolo_loader(device: str):
    """注册 YOLO 加载器"""
    from gpu_manager import get_gpu_manager
    import os
    
    mgr = get_gpu_manager()
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    YOLO_MODEL_NAME = os.path.join(PROJECT_DIR, "yolov8x.pt")
    
    def loader():
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_NAME)
        return model, None
    
    mgr.register_loader(ModelName.YOLO.value, loader)


def register_dino_loader(device: str):
    """注册 DINO 加载器"""
    from gpu_manager import get_gpu_manager
    
    mgr = get_gpu_manager()
    DINO_MODEL_NAME = "IDEA-Research/grounding-dino-base"
    
    def loader():
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(DINO_MODEL_NAME, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_MODEL_NAME, 
            torch_dtype=dtype, 
            local_files_only=True
        ).to(device)
        model.eval()
        return model, processor
    
    mgr.register_loader(ModelName.DINO.value, loader)


def register_vlm_loader(device: str):
    """注册 VLM 加载器"""
    from gpu_manager import get_gpu_manager
    import os
    
    mgr = get_gpu_manager()
    VLM_MODEL_NAME = os.environ.get("AD_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    
    def loader():
        from transformers import AutoProcessor
        import torch
        
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # 优先 Qwen2.5-VL
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
            model = ModelCls.from_pretrained(VLM_MODEL_NAME, torch_dtype=dtype, device_map="auto")
            try:
                processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, min_pixels=256*28*28, max_pixels=1280*28*28)
            except Exception:
                processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
            return model, processor
        except Exception as e:
            print(f"[GPUManager] Qwen2.5-VL 加载失败({e})，回退 Qwen2-VL-2B...")
            from transformers import Qwen2VLForConditionalGeneration
            fallback = "Qwen/Qwen2-VL-2B-Instruct"
            model = Qwen2VLForConditionalGeneration.from_pretrained(fallback, torch_dtype=dtype, device_map="auto")
            processor = AutoProcessor.from_pretrained(fallback)
            return model, processor
    
    mgr.register_loader(ModelName.VLM.value, loader)


# ============================================================
# 初始化函数（在 backend.py startup_event 中调用）
# ============================================================

def init_gpu_managers(device: str = None, max_vram_gb: float = 11.0):
    """初始化 GPU Manager 并注册所有加载器"""
    mgr = init_gpu_manager(device=device, max_vram_gb=max_vram_gb)
    
    # 注册所有模型加载器
    register_siglip_loader(mgr.device)
    register_yolo_loader(mgr.device)
    register_dino_loader(mgr.device)
    register_vlm_loader(mgr.device)
    
    print(f"[GPUManager] 所有模型加载器已注册，设备: {mgr.device}")
    return mgr


# ============================================================
# 替换原有的 _ensure_*/_free_* 函数
# ============================================================

def ensure_siglip():
    """替代原 _ensure_siglip - 返回 (model, processor) 或 (None, None)"""
    mgr = get_gpu_manager()
    try:
        return mgr.acquire(ModelName.SIGLIP.value, required_gb=3.0)
    except Exception as e:
        print(f"[!] SigLIP 加载失败: {e}")
        return None, None


def free_siglip():
    """替代原 _free_siglip - 真卸载(显存互斥用): 仅 release 不减引用, 必须把模型移出显存"""
    mgr = get_gpu_manager()
    try:
        _models = getattr(mgr, "_models", None)
        if _models:
            info = _models.get(ModelName.SIGLIP.value)
            if info and info.get("model") is not None:
                try:
                    info["model"] = info["model"].cpu()
                except Exception:
                    pass
                info["model"] = None
                info["processor"] = None
                info["loaded"] = False
                info["ref_count"] = 0
        import torch
        torch.cuda.empty_cache()
        print("[GPU] SigLIP 已卸载释放显存", flush=True)
    except Exception as e:
        print(f"[GPU] SigLIP 卸载异常: {e}", flush=True)
    finally:
        mgr.release(ModelName.SIGLIP.value)


def ensure_yolo():
    """替代原 _ensure_yolo - 返回 model 或 None"""
    mgr = get_gpu_manager()
    try:
        model, _ = mgr.acquire(ModelName.YOLO.value, required_gb=2.0)
        return model
    except Exception as e:
        print(f"[!] YOLO 加载失败: {e}")
        return None


def free_yolo():
    """替代原 _free_yolo"""
    mgr = get_gpu_manager()
    mgr.release(ModelName.YOLO.value)


def ensure_dino():
    """替代原 _ensure_dino - 返回 (model, processor) 或 (None, None)"""
    mgr = get_gpu_manager()
    # DINO 与 VLM/YOLO 互斥：先释放它们
    mgr.release(ModelName.VLM.value)
    mgr.release(ModelName.YOLO.value)
    try:
        return mgr.acquire(ModelName.DINO.value, required_gb=4.0)
    except Exception as e:
        print(f"[!] DINO 加载失败: {e}")
        return None, None


def free_dino():
    """替代原 _free_dino"""
    mgr = get_gpu_manager()
    mgr.release(ModelName.DINO.value)


def ensure_vlm():
    """替代原 init_vlm_local - 返回 (model, processor) 或 (None, None)"""
    mgr = get_gpu_manager()
    # VLM 与 DINO/YOLO 互斥
    mgr.release(ModelName.DINO.value)
    mgr.release(ModelName.YOLO.value)
    try:
        return mgr.acquire(ModelName.VLM.value, required_gb=6.0)
    except Exception as e:
        print(f"[!] VLM 加载失败: {e}")
        return None, None


def free_vlm():
    """替代原 _free_vlm"""
    mgr = get_gpu_manager()
    mgr.release(ModelName.VLM.value)


# ============================================================
# 装饰器版本（推荐新代码使用）
# ============================================================

# @with_model("siglip", required_gb=3.0)
# def siglip_infer(model, processor, images):
#     ...

# @with_model("yolo", required_gb=2.0)
# def yolo_infer(model, processor, images):
#     ...

# @with_model("dino", required_gb=4.0)
# def dino_infer(model, processor, images, prompt):
#     ...

# @with_model("vlm", required_gb=6.0)
# def vlm_infer(model, processor, image, prompt):
#     ...


# ============================================================
# 显存监控工具
# ============================================================

from typing import Dict

def get_vram_usage() -> Dict[str, float]:
    """获取显存使用情况"""
    mgr = get_gpu_manager()
    return mgr.get_vram_usage()


def print_vram_status():
    """打印显存状态"""
    usage = get_vram_usage()
    loaded = get_gpu_manager().get_loaded_models()
    print(f"[VRAM] 空闲: {usage['free']:.2f}GB / 总计: {usage['total']:.2f}GB / 已用: {usage['used']:.2f}GB")
    print(f"[VRAM] 已加载模型: {loaded}")


if __name__ == "__main__":
    # 测试
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mgr = init_gpu_managers(device=device)
    print_vram_status()