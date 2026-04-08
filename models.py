# app/models.py
import torch
import threading
import urllib.request
from pathlib import Path
from insightface.app import FaceAnalysis
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
# from ram import get_transform, inference_ram
# from ram.models import ram_plus

# MODEL_URLS = {
#     "ram_plus": "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth",
# }
PRETRAINED_DIR = Path(__file__).parent / "pretrained"

MODELS = {}
MODELS_LOCK = threading.RLock()  # Reentrant lock for thread-safe access

def load_models():
    """Load all models in a thread-safe manner"""
    with MODELS_LOCK:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # ===============================
        # Face Recognition (InsightFace)
        # ===============================
        try:
            face_app = FaceAnalysis(name="buffalo_l")
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            MODELS["face"] = face_app
        except Exception as e:
            raise RuntimeError(f"Failed to load InsightFace: {e}")

        # ===============================
        # Emotion (ViT Face Expression)
        # ===============================
        try:
            emo_processor = AutoImageProcessor.from_pretrained(
                "trpakov/vit-face-expression"
            )
            emo_model = AutoModelForImageClassification.from_pretrained(
                "trpakov/vit-face-expression"
            ).to(device)

            MODELS["emotion"] = {
                "processor": emo_processor,
                "model": emo_model
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load Emotion model: {e}")

        # ===============================
        # Scene Description (BLIP-1 Large)
        # ===============================
        try:
            blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
            blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(device)

            MODELS["scene"] = {
                "processor": blip_processor,
                "model": blip_model
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP model: {e}")

        # ===============================
        # RAM++ Tags (Recognize Anything) - DISABLED
        # ===============================
        # try:
        #     ram_plus_model = _load_ram_plus_model(device)
        #     ram_plus_transform = get_transform(image_size=384)
        #     MODELS["ram_plus"] = {
        #         "model": ram_plus_model,
        #         "transform": ram_plus_transform,
        #         "device": device,
        #     }
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load RAM++ model: {e}")

        # ===============================
        # Transcription & Translation (Whisper Large V3 Turbo)
        # ===============================
        try:
            whisper_turbo_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch_dtype,
                device=device,
            )
            # Map both keys to the same turbo pipeline so you don't have to rewrite older parts of your code
            MODELS["whisper"] = whisper_turbo_pipe
            MODELS["whisper_large_v3"] = whisper_turbo_pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper V3 Turbo: {e}")

        # ===============================
        # Text Embeddings (SentenceTransformer)
        # ===============================
        try:
            MODELS["embed"] = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer: {e}")

        print("✅ All models loaded successfully")

def _download_weights(model_name: str) -> Path:
    """Download model weights from HuggingFace if not already cached."""
    PRETRAINED_DIR.mkdir(exist_ok=True, parents=True)
    url = MODEL_URLS[model_name]
    filename = url.split("/")[-1]
    weight_path = PRETRAINED_DIR / filename

    if weight_path.exists():
        return weight_path

    urllib.request.urlretrieve(url, str(weight_path))
    return weight_path


# def _load_ram_plus_model(device: str):
#     weight_path = _download_weights("ram_plus")
#     model = ram_plus(
#         pretrained=str(weight_path),
#         image_size=384,
#         vit="swin_l",
#     )
#     model.eval()
#     model = model.to(device)
#     return model


def unload_models():
    """Unload all models in a thread-safe manner"""
    with MODELS_LOCK:
        MODELS.clear()
        torch.cuda.empty_cache()
        print("🧹 All models unloaded")