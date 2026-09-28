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


def _unload_real(model_name: str, label: str):
    """**真卸载**：把权重移出显存并归还，而不是只减引用计数。

    ⚠️ 2026-09-21 修（这是 OOM 循环的根因，别再退回去）：
    原来 free_vlm / free_dino / free_yolo 都只调 `mgr.release()`（引用计数 -1），
    权重仍留在显存里 —— 而 app 层还会把自己的全局引用置空，于是"看起来已释放、
    显存其实没还"。后果是 make_room_for 声称"腾出空间"却腾不动：
    批量检测跑完后 YOLO 的约 6G 一直占着，7B VLM（约 8.8G）加载不进去 →
    **每逢检测之后做段级判定就整段整段 OOM**（实测：一次 1,687 帧的重检跑完，
    紧接着判定 10 段成功后就连续 OOM，重启服务才恢复；也是 2026-09-20
    "10 帧联合推理 OOM / 8 帧是安全边界"那天量到的同一个现象）。
    只有 free_siglip 从一开始就是真卸载（作者注释写着"必须把模型移出显存"），
    这里把另外三个统一成同一套做法：先 `.cpu()` 强制把张量搬出显存（否则 app 层
    持有的引用会让 `del` 释放不掉），再清 loaded/ref_count，最后 empty_cache。"""
    mgr = get_gpu_manager()
    try:
        _models = getattr(mgr, "_models", None)
        _did = False
        if _models:
            info = _models.get(model_name)
            if info and info.get("model") is not None:
                try:
                    info["model"] = info["model"].cpu()
                except Exception:
                    pass
                info["model"] = None
                info["processor"] = None
                info["loaded"] = False
                info["ref_count"] = 0
                _did = True
        import torch
        if _did:
            # ⚠️ 只在**真的卸下来了**才打这行。原来无条件打印，于是空操作也会留日志，
            # 排查时会被当成"发生过驱逐"的证据（2026-09-20 已被这类日志误导过一次）。
            print("[GPU] %s 已卸载释放显存" % label, flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception as e:
        print(f"[GPU] {label} 卸载异常: {e}", flush=True)
    finally:
        mgr.release(model_name)


def free_siglip():
    """替代原 _free_siglip - 真卸载(显存互斥用): 仅 release 不减引用, 必须把模型移出显存"""
    _unload_real(ModelName.SIGLIP.value, "SigLIP")


def ensure_yolo():
    """替代原 _ensure_yolo - 返回 model 或 None。
    ultralytics 偶发 'Both events must be recorded' 计时 bug -> 失败重试多次(清显存后重新加载)。"""
    import torch, time as _t
    mgr = get_gpu_manager()
    for attempt in range(1, 4):
        try:
            model, _ = mgr.acquire(ModelName.YOLO.value)
            return model
        except Exception as e:
            print(f"[!] YOLO 加载失败(第{attempt}次): {e}")
            try:
                mgr.release(ModelName.YOLO.value)
                torch.cuda.empty_cache()
            except Exception:
                pass
            _t.sleep(2)
    import traceback
    traceback.print_exc()
    return None

def free_yolo():
    """替代原 _free_yolo - 真卸载（见 _unload_real 的说明：只 release 不还显存，
    是"检测跑完→判定 OOM"的根因）"""
    _unload_real(ModelName.YOLO.value, "YOLO")


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
    """替代原 _free_dino - 真卸载（同上）"""
    _unload_real(ModelName.DINO.value, "DINO")


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
    """替代原 _free_vlm - 真卸载（同上）"""
    _unload_real(ModelName.VLM.value, "VLM")


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