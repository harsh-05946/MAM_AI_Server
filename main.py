# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import torch
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Local imports
from models import MODELS, load_main_models, unload_models, inference_ram, get_model_lock, get_runtime_model_status

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Main Inference Service")
    start = time.perf_counter()
    load_main_models()
    logger.info(f"✅ Main models loaded in {(time.perf_counter() - start):.2f}s")
    yield
    # Shutdown
    logger.info("🛑 Stopping Main Inference Service")
    unload_models()
    executor.shutdown(wait=True)

app = FastAPI(title="General Inference Server", lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    texts: List[str]

class SarvamTranslationRequest(BaseModel):
    text: str
    target_lang: str = "English"


SARVAM_DEFAULT_MAX_LENGTH = 1024
SARVAM_DEFAULT_TEMPERATURE = 0.01
SUPPORTED_LANGUAGES = {
    "assamese": "Assamese",
    "bengali": "Bengali",
    "bodo": "Bodo",
    "dogri": "Dogri",
    "gujarati": "Gujarati",
    "english": "English",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "kashmiri": "Kashmiri",
    "konkani": "Konkani",
    "maithili": "Maithili",
    "malayalam": "Malayalam",
    "manipuri": "Manipuri",
    "marathi": "Marathi",
    "nepali": "Nepali",
    "odia": "Odia",
    "punjabi": "Punjabi",
    "sanskrit": "Sanskrit",
    "santali": "Santali",
    "sindhi": "Sindhi",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "urdu": "Urdu",
}

# ===============================
# Helper for Locked Inference
# ===============================
def _lock_inference(lock, fn):
    with lock:
        return fn()


def _resolve_target_language(target_lang: str) -> str:
    normalized_target = target_lang.lower().strip()
    if normalized_target in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[normalized_target]
    if target_lang in SUPPORTED_LANGUAGES.values():
        return target_lang
    supported = ", ".join(sorted(SUPPORTED_LANGUAGES.values()))
    raise ValueError(f"Unsupported target language '{target_lang}'. Supported: {supported}")


def _mostly_latin(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    latin = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    return (latin / len(letters)) >= 0.70


def _sarvam_generate_once(tokenizer, model, messages: list[dict[str, str]], max_length: int, temperature: float) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=max(temperature, 0.01),
            num_return_sequences=1,
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _sarvam_translate_sync(text: str, target_lang: str) -> str:
    sarvam_bundle = MODELS.get("sarvam")
    if sarvam_bundle is None:
        raise RuntimeError("Sarvam model is not loaded")

    tokenizer = sarvam_bundle["tokenizer"]
    model = sarvam_bundle["model"]
    resolved_target = _resolve_target_language(target_lang)

    first_pass = [
        {"role": "system", "content": f"Translate the text below to {resolved_target}."},
        {"role": "user", "content": text},
    ]
    output = _sarvam_generate_once(
        tokenizer,
        model,
        first_pass,
        max_length=SARVAM_DEFAULT_MAX_LENGTH,
        temperature=SARVAM_DEFAULT_TEMPERATURE,
    )

    if resolved_target != "English" and _mostly_latin(output):
        second_pass = [
            {
                "role": "system",
                "content": (
                    f"Translate the text below to {resolved_target}. "
                    f"Return only {resolved_target} text, no notes."
                ),
            },
            {"role": "user", "content": output},
        ]
        output = _sarvam_generate_once(
            tokenizer,
            model,
            second_pass,
            max_length=SARVAM_DEFAULT_MAX_LENGTH,
            temperature=SARVAM_DEFAULT_TEMPERATURE,
        )

    return output

# ===============================
# Face Recognition
# ===============================
@app.post("/process/face")
async def face_recognition(file: UploadFile = File(...)):
    logger.info(f"🧑 Face recognition: {file.filename}")
    start = time.perf_counter()
    img = Image.open(file.file).convert("RGB")
    img_np = np.array(img)
    
    lock = get_model_lock("face")
    loop = asyncio.get_event_loop()
    faces = await loop.run_in_executor(executor, lambda: _lock_inference(lock, lambda: MODELS["face"].get(img_np)))

    duration = time.perf_counter() - start
    logger.info(f"🧑 Face recognition finished in {duration:.3f}s")
    return [
        {"bbox": face.bbox.tolist(), "embedding": face.normed_embedding.tolist()}
        for face in faces
    ]

# ===============================
# Emotion Detection
# ===============================
@app.post("/process/emotion")
async def emotion_detection(file: UploadFile = File(...)):
    logger.info(f"😊 Emotion detection: {file.filename}")
    start = time.perf_counter()
    image = Image.open(file.file).convert("RGB")
    
    lock = get_model_lock("emotion")
    def _sync():
        with lock:
            proc = MODELS["emotion"]["processor"]
            model = MODELS["emotion"]["model"]
            inputs = proc(images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)[0]
            idx = probs.argmax().item()
            return {"emotion": model.config.id2label[idx], "confidence": float(probs[idx])}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, _sync)
    duration = time.perf_counter() - start
    logger.info(f"😊 Emotion finished in {duration:.3f}s")
    return result

# ===============================
# Scene Description
# ===============================
@app.post("/process/scene")
async def scene_description(file: UploadFile = File(...)):
    logger.info(f"🖼️ Scene description: {file.filename}")
    start = time.perf_counter()
    image = Image.open(file.file).convert("RGB")
    
    lock = get_model_lock("scene")
    def _sync():
        with lock:
            proc = MODELS["scene"]["processor"]
            model = MODELS["scene"]["model"]
            inputs = proc(images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            return {"scene": proc.decode(out[0], skip_special_tokens=True)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, _sync)
    duration = time.perf_counter() - start
    logger.info(f"🖼️ Scene finished in {duration:.3f}s")
    return result

# ===============================
# Object Detection (RAM++)
# ===============================
@app.post("/process/object-detection")
async def object_detection(file: UploadFile = File(...)):
    logger.info(f"🏷️ RAM++ object detection: {file.filename}")
    
    if "ram_plus" not in MODELS:
        raise HTTPException(status_code=503, detail="RAM++ model not available")

    start = time.perf_counter()
    image = Image.open(file.file).convert("RGB")
    
    lock = get_model_lock("ram_plus")
    def _sync():
        with lock:
            m = MODELS["ram_plus"]
            image_tensor = m["transform"](image).unsqueeze(0).to(m["device"])
            with torch.no_grad():
                tags_en, tags_cn = inference_ram(image_tensor, m["model"])
            return {"tags_en": tags_en, "tags_cn": tags_cn}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, _sync)
    duration = time.perf_counter() - start
    logger.info(f"🏷️ RAM++ finished in {duration:.3f}s")
    return result

# ===============================
# Text Embeddings
# ===============================
@app.post("/process/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    logger.info(f"🔗 Embeddings request: {len(req.texts)} items")
    start = time.perf_counter()
    
    lock = get_model_lock("embed")
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(executor, lambda: _lock_inference(lock, lambda: MODELS["embed"].encode(req.texts)))
    
    duration = time.perf_counter() - start
    logger.info(f"🔗 Embeddings finished in {duration:.3f}s")
    return {"embeddings": [emb.tolist() for emb in embeddings], "count": len(embeddings)}

# ===============================
# Sarvam Translation
# ===============================
@app.post("/process/translation/sarvam")
async def sarvam_translation(req: SarvamTranslationRequest):
    logger.info(f"🇮🇳 Sarvam translation: {req.text[:30]}... -> {req.target_lang}")
    start = time.perf_counter()

    lock = get_model_lock("sarvam")
    def _sync():
        with lock:
            translated = _sarvam_translate_sync(text=req.text, target_lang=req.target_lang)
            return {"output": translated}

    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(executor, _sync)
        return {"translated_text": payload.get("output"), "target_lang": req.target_lang, "engine": "sarvam"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Sarvam failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info(f"🇮🇳 Sarvam finished in {time.perf_counter() - start:.3f}s")

@app.get("/health")
def health():
    status = get_runtime_model_status()
    return {"status": "ok", "service": "main-inference", **status}