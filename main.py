# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np
import torch
import time
import logging
import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Local imports
# from ram import inference_ram
from models import MODELS, load_models, unload_models

# ===============================
# Logging config
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ===============================
# Thread pool for CPU-intensive tasks
# ===============================
executor = ThreadPoolExecutor(max_workers=4)

# ===============================
# Lifespan context manager
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Server startup: loading models")
    start = time.perf_counter()
    load_models()
    logger.info(f"✅ Models loaded in {(time.perf_counter() - start):.2f}s")
    
    yield
    
    # Shutdown
    logger.info("🛑 Server shutdown: unloading models")
    start = time.perf_counter()
    unload_models()
    executor.shutdown(wait=True)
    logger.info(f"✅ Models unloaded in {(time.perf_counter() - start):.2f}s")

app = FastAPI(title="Unified AI Server", lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    texts: List[str]

# ===============================
# Face Recognition
# ===============================
async def _face_recognition_task(image_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: MODELS["face"].get(image_data)
    )

@app.post("/process/face")
async def face_recognition(file: UploadFile = File(...)):
    logger.info("🧑 Face recognition started")
    start = time.perf_counter()

    img = Image.open(file.file).convert("RGB")
    img_np = np.array(img)

    faces = await _face_recognition_task(img_np)

    duration = time.perf_counter() - start
    logger.info(f"🧑 Face recognition finished in {duration:.3f}s")

    return [
        {
            "bbox": face.bbox.tolist(),
            "embedding": face.normed_embedding.tolist()
        }
        for face in faces
    ]

# ===============================
# Emotion Detection
# ===============================
async def _emotion_detection_task(image):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: _emotion_detection_sync(image)
    )

def _emotion_detection_sync(image):
    processor = MODELS["emotion"]["processor"]
    model = MODELS["emotion"]["model"]

    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits.softmax(dim=-1)[0]
    idx = probs.argmax().item()

    return {
        "emotion": model.config.id2label[idx],
        "confidence": float(probs[idx])
    }

@app.post("/process/emotion")
async def emotion_detection(file: UploadFile = File(...)):
    logger.info("😊 Emotion detection started")
    start = time.perf_counter()

    image = Image.open(file.file).convert("RGB")
    result = await _emotion_detection_task(image)

    duration = time.perf_counter() - start
    logger.info(f"😊 Emotion detection finished in {duration:.3f}s")

    return result

# ===============================
# Scene Description
# ===============================
async def _scene_description_task(image):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: _scene_description_sync(image)
    )

def _scene_description_sync(image):
    processor = MODELS["scene"]["processor"]
    model = MODELS["scene"]["model"]

    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return {"scene": caption}

@app.post("/process/scene")
async def scene_description(file: UploadFile = File(...)):
    logger.info("🖼️ Scene description started")
    start = time.perf_counter()

    image = Image.open(file.file).convert("RGB")
    result = await _scene_description_task(image)

    duration = time.perf_counter() - start
    logger.info(f"🖼️ Scene description finished in {duration:.3f}s")

    return result

# ===============================
# RAM++ Image Tagging
# ===============================
@app.post("/process/ram-tags")
async def ram_tags(file: UploadFile = File(...), model: str = "ram_plus"):
    logger.info("🏷️ RAM++ image tagging (DISABLED)")
    raise HTTPException(
        status_code=503,
        detail="RAM++ tagging is currently disabled due to dependency conflicts with Transformers 5.x. Please use Scene Description (/process/scene) instead."
    )

# async def _ram_tags_task(image, model_name: str = "ram_plus"):
#     loop = asyncio.get_running_loop()
#     return await loop.run_in_executor(
#         executor,
#         lambda: _ram_tags_sync(image, model_name)
#     )

# def _ram_tags_sync(image, model_name: str = "ram_plus"):
#     model_info = MODELS.get(model_name)
#     if model_info is None:
#         raise RuntimeError(f"Model '{model_name}' is not loaded")

#     model = model_info["model"]
#     transform = model_info["transform"]
#     device = model_info["device"]

#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         tags_en, tags_cn = inference_ram(image_tensor, model)

#     return {
#         "model": model_name,
#         "tags_en": tags_en,
#         "tags_cn": tags_cn,
#     }

# ===============================
# Unified Whisper Logic (Transcription & Translation)
# ===============================
async def _whisper_task(audio_path, task="transcribe"):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: _whisper_sync(audio_path, task)
    )

def _whisper_sync(audio_path, task="transcribe"):
    pipe = MODELS.get("whisper")
    if pipe is None:
        raise RuntimeError("Whisper pipeline is not loaded")

    generate_kwargs = {
        "task": task,
    }

    result = pipe(
        audio_path, 
        return_timestamps=True, 
        generate_kwargs=generate_kwargs
    )

    segments = []
    chunks = result.get("chunks", [])
    
    for chunk in chunks:
        timestamps = chunk.get("timestamp")
        if not timestamps:
            timestamps = (0.0, 0.0)

        start_time = timestamps[0] if timestamps[0] is not None else 0.0
        end_time = timestamps[1] if timestamps[1] is not None else start_time + 1.0

        segments.append({
            "start": start_time,
            "end": end_time,
            "text": chunk.get("text", "")
        })

    return {
        "text": result.get("text", ""),
        "segments": segments,
        "task": task
    }

# ===============================
# Text Embeddings
# ===============================
async def _embedding_task(texts):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        lambda: _embedding_sync(texts)
    )

def _embedding_sync(texts):
    embedder = MODELS.get("embed")
    if embedder is None:
        raise RuntimeError("Embedding model not loaded")

    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]

@app.post("/process/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    logger.info("🔗 Embeddings creation started")
    start = time.perf_counter()

    texts = req.texts
    if not texts or not isinstance(texts, list):
        raise HTTPException(status_code=400, detail="`texts` must be a non-empty list of strings")

    try:
        embeddings = await _embedding_task(texts)
    except Exception as e:
        logger.exception("Error creating embeddings")
        raise HTTPException(status_code=500, detail=str(e))

    duration = time.perf_counter() - start
    logger.info(f"🔗 Embeddings created in {duration:.3f}s")

    return {"embeddings": embeddings, "count": len(embeddings)}


# ===============================
# API Endpoints for Audio
# ===============================
@app.post("/process/transcription")
async def transcription(file: UploadFile = File(...)):
    """
    Standard Transcription.
    Maintains original language spoken in the audio.
    """
    logger.info("🎙️ Transcription started")
    start = time.perf_counter()

    with tempfile.NamedTemporaryFile(
        prefix="transcription_",
        suffix=f"_{os.path.basename(file.filename)}",
        dir="/tmp",
        delete=False,
    ) as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    try:
        result = await _whisper_task(audio_path, task="transcribe")
    finally:
        try:
            os.remove(audio_path)
        except FileNotFoundError:
            pass

    duration = time.perf_counter() - start
    logger.info(f"🎙️ Transcription finished in {duration:.3f}s")

    return result


@app.post("/process/whisper-v3/translate")
async def whisper_v3_translate(file: UploadFile = File(...)):
    """
    Translation into English.
    Translates ANY supported foreign language audio into English text.
    """
    logger.info("🎧 Translation to English started")
    start = time.perf_counter()

    with tempfile.NamedTemporaryFile(
        prefix="whisper_v3_translate_",
        suffix=f"_{os.path.basename(file.filename)}",
        dir="/tmp",
        delete=False,
    ) as tmp:
        audio_path = tmp.name
        tmp.write(await file.read())

    try:
        result = await _whisper_task(audio_path, task="translate")
    finally:
        try:
            os.remove(audio_path)
        except FileNotFoundError:
            pass

    duration = time.perf_counter() - start
    logger.info(f"🎧 Translation finished in {duration:.3f}s")

    return result