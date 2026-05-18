# app/models.py
import logging
import os
import threading
import urllib.request
import gc
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    BitsAndBytesConfig,
    BlipForConditionalGeneration,
    BlipProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


def _patch_transformers_for_ram() -> None:
    """RAM's vendored BERT still imports helpers removed/moved in transformers v5."""
    import transformers.modeling_utils as modeling_utils
    from transformers import pytorch_utils

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = pytorch_utils.apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


_patch_transformers_for_ram()

from insightface.app import FaceAnalysis
from ram import get_transform, inference_ram
from ram.models import ram_plus
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_URLS = {
    "ram_plus": "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth",
}

MODEL_IDS = {
    "emotion": "trpakov/vit-face-expression",
    "scene": "Salesforce/blip-image-captioning-large",
    "embed": "sentence-transformers/all-MiniLM-L6-v2",
    "sarvam": "sarvamai/sarvam-translate",
    "qwen_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}

MAIN_MODEL_SOURCE_ENV = {
    "emotion": "LOCAL_EMOTION_DIR",
    "scene": "LOCAL_BLIP_DIR",
    "embed": "LOCAL_EMBED_DIR",
    "sarvam": "LOCAL_SARVAM_DIR",
    "qwen_vl": "LOCAL_QWEN_VL_DIR",
}



PRETRAINED_DIR = Path(__file__).parent / "pretrained"
DEFAULT_MODEL_CACHE_DIR = Path(__file__).parent / "models-local"

MODELS = {}
MODEL_LOCKS = {}
MODELS_LOCK = threading.RLock()
GPU_LOCK = threading.Lock()

QWEN_DEFAULT_MAX_PIXELS = 1024 * 1024
QWEN_DEFAULT_MIN_PIXELS = 256 * 256
QWEN_MAX_PIXELS = int(os.getenv("QWEN_MAX_PIXELS", str(QWEN_DEFAULT_MAX_PIXELS)))
QWEN_MIN_PIXELS = int(os.getenv("QWEN_MIN_PIXELS", str(QWEN_DEFAULT_MIN_PIXELS)))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_QWEN_4BIT = _bool_env("ENABLE_QWEN_4BIT", True)
ENABLE_SARVAM_4BIT = _bool_env("ENABLE_SARVAM_4BIT", True)
if QWEN_MIN_PIXELS > QWEN_MAX_PIXELS:
    QWEN_MIN_PIXELS = QWEN_MAX_PIXELS


def _is_offline_mode() -> bool:
    return _bool_env("HF_HUB_OFFLINE", False) or _bool_env("TRANSFORMERS_OFFLINE", False)


def _allow_hub_fallback() -> bool:
    if _is_offline_mode():
        return False
    return _bool_env("ALLOW_HF_FALLBACK", True)


def _local_dir_for(model_key: str, service: str) -> Optional[Path]:
    env_map = MAIN_MODEL_SOURCE_ENV
    env_name = env_map.get(model_key)
    if env_name and os.getenv(env_name):
        return Path(os.getenv(env_name)).expanduser()

    default_dir = DEFAULT_MODEL_CACHE_DIR / service / model_key
    return default_dir if default_dir.exists() else None


def _resolve_model_source(model_key: str, service: str) -> tuple[str, bool]:
    local_dir = _local_dir_for(model_key, service)
    if local_dir and local_dir.exists():
        return str(local_dir), True

    if _allow_hub_fallback():
        return MODEL_IDS[model_key], False

    env_map = MAIN_MODEL_SOURCE_ENV
    env_name = env_map.get(model_key, "MODEL_PATH")
    raise RuntimeError(
        f"Offline mode active and local model path missing for '{model_key}'. "
        f"Set {env_name} or place files at '{DEFAULT_MODEL_CACHE_DIR / service / model_key}'."
    )


def _hf_kwargs() -> dict:
    return {"local_files_only": _is_offline_mode()}


def _log_model_source(model_key: str, source: str, is_local: bool) -> None:
    source_type = "local path" if is_local else "hub id"
    logger.info(f"📍 {model_key}: using {source_type} '{source}'")


def _patch_tokenizer_compat() -> None:
    def _install_patch(cls) -> None:
        existing = getattr(cls, "additional_special_tokens_ids", None)
        if isinstance(existing, property):
            return

        def _getter(self):
            tokens = getattr(self, "additional_special_tokens", None) or []
            token_ids = [self.convert_tokens_to_ids(tok) for tok in tokens]
            token_ids = [tok_id for tok_id in token_ids if tok_id is not None and tok_id >= 0]
            if token_ids:
                return token_ids

            # RAM++ expects at least one special token id on older tokenizer API behavior.
            for attr_name in ("cls_token_id", "sep_token_id", "eos_token_id"):
                fallback_id = getattr(self, attr_name, None)
                if isinstance(fallback_id, int) and fallback_id >= 0:
                    return [fallback_id]
            return []

        cls.additional_special_tokens_ids = property(_getter)

    _install_patch(BertTokenizer)
    _install_patch(BertTokenizerFast)


def _patch_ram_bert_compat() -> None:
    try:
        from ram.models import bert as ram_bert  # type: ignore

        if hasattr(ram_bert.BertModel, "all_tied_weights_keys"):
            return

        @property
        def all_tied_weights_keys(self):
            return {}

        ram_bert.BertModel.all_tied_weights_keys = all_tied_weights_keys

        if not hasattr(ram_bert.BertModel, "get_head_mask"):
            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                return head_mask

            ram_bert.BertModel.get_head_mask = get_head_mask
    except Exception:
        # Best-effort compatibility patch only.
        pass


def get_model_lock(model_key: str) -> threading.Lock:
    with MODELS_LOCK:
        if model_key not in MODEL_LOCKS:
            MODEL_LOCKS[model_key] = threading.Lock()
        return MODEL_LOCKS[model_key]


def get_gpu_lock() -> threading.Lock:
    return GPU_LOCK


def clear_cuda_memory(reason: str = "") -> None:
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if reason:
            logger.info(f"🧹 Memory cleanup complete ({reason})")
    except Exception as e:
        logger.warning(f"Memory cleanup warning: {e}")


def _build_4bit_config(enabled: bool, device: str) -> tuple[Optional[BitsAndBytesConfig], bool]:
    if not enabled or device != "cuda":
        return None, False
    return (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        True,
    )


def get_device_info():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if device == "cuda" and hasattr(torch.backends, "cuda"):
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        except Exception as e:
            logger.warning(f"Unable to set SDPA backend preferences: {e}")

    return device, torch_dtype


def load_main_models():
    """Load models for the main inference service (non-ASR)."""
    device, torch_dtype = get_device_info()

    with MODELS_LOCK:
        try:
            if "face" not in MODELS:
                if not torch.cuda.is_available():
                    logger.error(
                        "❌ InsightFace not loaded: GPU-only mode requires CUDA "
                        "(torch.cuda.is_available() is False)"
                    )
                else:
                    logger.info("📦 Loading InsightFace (GPU-only)...")
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    try:
                        face_app = FaceAnalysis(name="buffalo_l", providers=providers)
                    except TypeError:
                        # Older insightface builds may not accept 'providers='.
                        face_app = FaceAnalysis(name="buffalo_l")
                    face_app.prepare(ctx_id=0, det_size=(640, 640))
                    MODELS["face"] = face_app
                    logger.info("✅ InsightFace loaded on cuda (GPU-only)")
        except Exception as e:
            logger.error(f"❌ Failed to load InsightFace: {e}")

        try:
            if "emotion" not in MODELS:
                logger.info("📦 Loading Emotion model...")
                source, is_local = _resolve_model_source("emotion", service="main")
                _log_model_source("emotion", source, is_local)
                emo_processor = AutoImageProcessor.from_pretrained(source, **_hf_kwargs())
                emo_model = AutoModelForImageClassification.from_pretrained(source, **_hf_kwargs()).to(device)
                MODELS["emotion"] = {"processor": emo_processor, "model": emo_model}
        except Exception as e:
            logger.error(f"❌ Failed to load Emotion model: {e}")

        try:
            if "scene" not in MODELS:
                logger.info("📦 Loading BLIP scene model...")
                source, is_local = _resolve_model_source("scene", service="main")
                _log_model_source("scene", source, is_local)
                blip_processor = BlipProcessor.from_pretrained(source, **_hf_kwargs())
                blip_model = BlipForConditionalGeneration.from_pretrained(source, **_hf_kwargs()).to(device)
                MODELS["scene"] = {"processor": blip_processor, "model": blip_model}
        except Exception as e:
            logger.error(f"❌ Failed to load BLIP model: {e}")

        try:
            if "ram_plus" not in MODELS:
                logger.info("📦 Loading RAM++ model (Object Detection)...")
                ram_plus_model = _load_ram_plus_model(device)
                ram_plus_transform = get_transform(image_size=384)
                MODELS["ram_plus"] = {
                    "model": ram_plus_model,
                    "transform": ram_plus_transform,
                    "device": device,
                }
        except Exception as e:
            logger.warning(f"⚠️ RAM++ model failed to load (skipping): {e}")

        try:
            if "embed" not in MODELS:
                logger.info("📦 Loading SentenceTransformer...")
                source, is_local = _resolve_model_source("embed", service="main")
                _log_model_source("embed", source, is_local)
                st_device = "cuda" if device == "cuda" else "cpu"
                MODELS["embed"] = SentenceTransformer(source, device=st_device)
                logger.info(f"✅ SentenceTransformer loaded on {st_device}")
        except Exception as e:
            logger.error(f"❌ Failed to load SentenceTransformer: {e}")

        try:
            if "sarvam" not in MODELS:
                logger.info("📦 Loading Sarvam translation model...")
                source, is_local = _resolve_model_source("sarvam", service="main")
                _log_model_source("sarvam", source, is_local)
                sarvam_tokenizer = AutoTokenizer.from_pretrained(source, **_hf_kwargs())
                sarvam_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch_dtype
                sarvam_qconf, sarvam_quant_enabled = _build_4bit_config(ENABLE_SARVAM_4BIT, device)
                sarvam_kwargs = dict(_hf_kwargs())
                sarvam_kwargs["torch_dtype"] = sarvam_dtype
                if sarvam_qconf is not None:
                    sarvam_kwargs["quantization_config"] = sarvam_qconf
                    sarvam_kwargs["device_map"] = "auto"
                try:
                    sarvam_model = AutoModelForCausalLM.from_pretrained(source, **sarvam_kwargs)
                except Exception as quant_err:
                    if sarvam_qconf is None:
                        raise
                    logger.warning(f"⚠️ Sarvam 4-bit load failed, falling back to non-quantized: {quant_err}")
                    fallback_kwargs = dict(_hf_kwargs())
                    fallback_kwargs["torch_dtype"] = sarvam_dtype
                    sarvam_model = AutoModelForCausalLM.from_pretrained(source, **fallback_kwargs).to(device)
                    sarvam_quant_enabled = False
                if not sarvam_quant_enabled:
                    sarvam_model = sarvam_model.to(device)
                sarvam_model.eval()
                MODELS["sarvam"] = {
                    "tokenizer": sarvam_tokenizer,
                    "model": sarvam_model,
                    "device": device,
                    "quantized_4bit": sarvam_quant_enabled,
                    "compute_dtype": str(sarvam_dtype),
                }
                logger.info(f"✅ Sarvam loaded on {device} (4bit={sarvam_quant_enabled}, compute_dtype={sarvam_dtype})")
        except Exception as e:
            logger.error(f"❌ Failed to load Sarvam model: {e}")

        try:
            if "qwen_vl" not in MODELS:
                logger.info("📦 Loading Qwen2.5-VL model...")
                source, is_local = _resolve_model_source("qwen_vl", service="main")
                _log_model_source("qwen_vl", source, is_local)
                qwen_processor = AutoProcessor.from_pretrained(source, **_hf_kwargs())
                qwen_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch_dtype
                qwen_qconf, qwen_quant_enabled = _build_4bit_config(ENABLE_QWEN_4BIT, device)
                qwen_kwargs = dict(_hf_kwargs())
                qwen_kwargs["torch_dtype"] = qwen_dtype
                if qwen_qconf is not None:
                    qwen_kwargs["quantization_config"] = qwen_qconf
                    qwen_kwargs["device_map"] = "auto"
                try:
                    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(source, **qwen_kwargs)
                except Exception as quant_err:
                    if qwen_qconf is None:
                        raise
                    logger.warning(f"⚠️ Qwen 4-bit load failed, falling back to non-quantized: {quant_err}")
                    fallback_kwargs = dict(_hf_kwargs())
                    fallback_kwargs["torch_dtype"] = qwen_dtype
                    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(source, **fallback_kwargs).to(device)
                    qwen_quant_enabled = False
                if not qwen_quant_enabled:
                    qwen_model = qwen_model.to(device)
                qwen_model.eval()
                MODELS["qwen_vl"] = {
                    "processor": qwen_processor,
                    "model": qwen_model,
                    "device": device,
                    "quantized_4bit": qwen_quant_enabled,
                    "compute_dtype": str(qwen_dtype),
                }
                logger.info(f"✅ Qwen2.5-VL loaded on {device} (4bit={qwen_quant_enabled}, compute_dtype={qwen_dtype})")
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen2.5-VL model: {e}")


    logger.info("✅ Main models loading sequence complete")





def _download_weights(model_name: str) -> Path:
    local_weight_override = os.getenv("LOCAL_RAM_PLUS_WEIGHTS")
    if local_weight_override:
        weight_path = Path(local_weight_override).expanduser()
        if not weight_path.exists():
            raise RuntimeError(f"Configured LOCAL_RAM_PLUS_WEIGHTS does not exist: {weight_path}")
        logger.info(f"📍 ram_plus: using local weights '{weight_path}'")
        return weight_path

    PRETRAINED_DIR.mkdir(exist_ok=True, parents=True)
    url = MODEL_URLS[model_name]
    filename = url.split("/")[-1]
    weight_path = PRETRAINED_DIR / filename
    if weight_path.exists():
        return weight_path

    if _is_offline_mode():
        raise RuntimeError(
            f"Offline mode active and RAM++ weights not found at '{weight_path}'. "
            "Set LOCAL_RAM_PLUS_WEIGHTS or pre-download into pretrained/."
        )

    logger.info(f"⬇️ Downloading weights for {model_name}...")
    urllib.request.urlretrieve(url, str(weight_path))
    return weight_path


def _load_ram_plus_model(device: str):
    _patch_tokenizer_compat()
    _patch_ram_bert_compat()
    weight_path = _download_weights("ram_plus")
    model = ram_plus(pretrained=str(weight_path), image_size=384, vit="swin_l")
    model.eval()
    model = model.to(device)
    return model


def get_runtime_model_status() -> dict:
    payload = {
        "offline_mode": _is_offline_mode(),
        "allow_hf_fallback": _allow_hub_fallback(),
        "loaded_models": list(MODELS.keys()),
        "ram_plus_available": "ram_plus" in MODELS,
        "qwen_vl_available": "qwen_vl" in MODELS,
    }
    if torch.cuda.is_available():
        try:
            payload.update(
                {
                    "cuda_device": torch.cuda.get_device_name(0),
                    "cuda_mem_allocated": int(torch.cuda.memory_allocated()),
                    "cuda_mem_reserved": int(torch.cuda.memory_reserved()),
                }
            )
        except Exception:
            pass
    return payload


def unload_models():
    with MODELS_LOCK:
        MODELS.clear()
        clear_cuda_memory(reason="shutdown")
        logger.info("🧹 All models unloaded")