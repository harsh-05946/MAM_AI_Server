# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from pydantic import BaseModel
from typing import List, Any, Union
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import torch
import time
import logging
import asyncio
import json
import io
import math
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Local imports
from models import (
    MODELS,
    load_main_models,
    unload_models,
    inference_ram,
    get_model_lock,
    get_runtime_model_status,
    get_gpu_lock,
    clear_cuda_memory,
    QWEN_MAX_PIXELS,
    QWEN_MIN_PIXELS,
)
try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=4)
GPU_LOCK = get_gpu_lock()
BATCHERS: dict[str, "MicroBatcher"] = {}
INFLIGHT_REQUESTS = 0
INSTANCE_NAME = os.getenv("INSTANCE_NAME", "main")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


BATCHING_ENABLED = _env_bool("BATCHING_ENABLED", True)


@dataclass
class BatchItem:
    payload: Any
    future: asyncio.Future


class MicroBatcher:
    def __init__(self, model_key: str, max_batch_size: int, max_wait_ms: int, process_fn):
        self.model_key = model_key
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(0, max_wait_ms)
        self.process_fn = process_fn
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name=f"{self.model_key}-batch-worker")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        await self.queue.put(None)
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(self, payload: Any):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self.queue.put(BatchItem(payload=payload, future=fut))
        return await fut

    async def _run(self):
        while True:
            first = await self.queue.get()
            if first is None:
                break

            batch = [first]
            start = time.perf_counter()
            wait_sec = self.max_wait_ms / 1000.0

            while len(batch) < self.max_batch_size:
                elapsed = time.perf_counter() - start
                remaining = wait_sec - elapsed
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if nxt is None:
                    await self.queue.put(None)
                    break
                batch.append(nxt)

            payloads = [item.payload for item in batch]
            try:
                if len(batch) > 1:
                    logger.info(
                        f"⚡ {self.model_key} micro-batch formed: size={len(batch)} "
                        f"(max={self.max_batch_size}, wait_ms={self.max_wait_ms})"
                    )
                results = await asyncio.get_running_loop().run_in_executor(
                    executor, lambda: self.process_fn(payloads)
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"{self.model_key} batch result size mismatch: "
                        f"expected {len(batch)}, got {len(results)}"
                    )
                for item, result in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as e:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)


def _process_emotion_batch(images: list[Image.Image]) -> list[dict[str, Any]]:
    lock = get_model_lock("emotion")
    with GPU_LOCK:
        with lock:
            proc = MODELS["emotion"]["processor"]
            model = MODELS["emotion"]["model"]
            inputs = proc(images=images, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            results = []
            for row in probs:
                idx = row.argmax().item()
                results.append({"emotion": model.config.id2label[idx], "confidence": float(row[idx])})
            return results


def _process_scene_batch(images: list[Image.Image]) -> list[dict[str, str]]:
    lock = get_model_lock("scene")
    with GPU_LOCK:
        with lock:
            proc = MODELS["scene"]["processor"]
            model = MODELS["scene"]["model"]
            inputs = proc(images=images, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=50)
            return [{"scene": proc.decode(seq, skip_special_tokens=True)} for seq in generated]


def _run_ram_single(m: dict[str, Any], image: Image.Image) -> tuple[Any, Any]:
    image_tensor = m["transform"](image).unsqueeze(0).to(m["device"])
    with torch.no_grad():
        tags_en, tags_cn = inference_ram(image_tensor, m["model"])
    return tags_en, tags_cn


def _process_ram_batch(images: list[Image.Image]) -> list[dict[str, Any]]:
    lock = get_model_lock("ram_plus")
    with GPU_LOCK:
        with lock:
            m = MODELS["ram_plus"]
            try:
                image_tensor = torch.stack([m["transform"](img) for img in images]).to(m["device"])
                with torch.no_grad():
                    tags_en, tags_cn = inference_ram(image_tensor, m["model"])
                if isinstance(tags_en, list) and isinstance(tags_cn, list):
                    if len(tags_en) == len(images) and len(tags_cn) == len(images):
                        return [{"tags_en": en, "tags_cn": cn} for en, cn in zip(tags_en, tags_cn)]
            except Exception:
                # Fallback to per-item inference if RAM++ runtime expects batch=1.
                pass

            out = []
            for img in images:
                en, cn = _run_ram_single(m, img)
                out.append({"tags_en": en, "tags_cn": cn})
            return out


def _process_face_batch(images_np: list[np.ndarray]) -> list[list[Any]]:
    lock = get_model_lock("face")
    with lock:
        model = MODELS["face"]
        return [model.get(img_np) for img_np in images_np]


def _process_embed_batch(texts: list[str]) -> list[list[float]]:
    """Run SentenceTransformer.encode for one or more strings; returns one vector per string (order preserved)."""
    if not texts:
        return []
    lock = get_model_lock("embed")
    with lock:
        embedder = MODELS["embed"]
        batch_size_kw: dict[str, int] = {}
        bs = _env_int("EMBED_ENCODE_BATCH_SIZE", 0)
        if bs > 0:
            batch_size_kw["batch_size"] = bs
        out: Union[np.ndarray, torch.Tensor, list] = embedder.encode(texts, **batch_size_kw)
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    if not isinstance(out, np.ndarray):
        raise TypeError(f"Unexpected embed encode output type: {type(out)}")
    if out.ndim == 1:
        if len(texts) != 1:
            raise RuntimeError("embed encode returned 1d array for multi-text batch")
        return [out.tolist()]
    if out.shape[0] != len(texts):
        raise RuntimeError(f"embed batch size mismatch: texts={len(texts)} rows={out.shape[0]}")
    return [out[i].tolist() for i in range(out.shape[0])]


def _process_embed_microbatch_payloads(texts: list[str]) -> list[list[float]]:
    return _process_embed_batch(texts)


def _init_batchers():
    if not BATCHING_ENABLED:
        logger.info("ℹ️ Internal micro-batching disabled (BATCHING_ENABLED=false)")
        return

    configs = {
        "emotion": (_env_int("BATCH_MAX_EMOTION", 16), _env_int("BATCH_WAIT_MS_EMOTION", 8), _process_emotion_batch),
        "scene": (_env_int("BATCH_MAX_SCENE", 8), _env_int("BATCH_WAIT_MS_SCENE", 10), _process_scene_batch),
        "ram_plus": (_env_int("BATCH_MAX_RAM_PLUS", 8), _env_int("BATCH_WAIT_MS_RAM_PLUS", 10), _process_ram_batch),
        "face": (_env_int("BATCH_MAX_FACE", 4), _env_int("BATCH_WAIT_MS_FACE", 8), _process_face_batch),
        "embed": (_env_int("BATCH_MAX_EMBED", 32), _env_int("BATCH_WAIT_MS_EMBED", 8), _process_embed_microbatch_payloads),
    }

    for model_key, (max_batch, wait_ms, fn) in configs.items():
        if model_key not in MODELS:
            continue
        batcher = MicroBatcher(model_key=model_key, max_batch_size=max_batch, max_wait_ms=wait_ms, process_fn=fn)
        batcher.start()
        BATCHERS[model_key] = batcher
        logger.info(f"⚡ Micro-batching enabled for {model_key} (max_batch={max_batch}, wait_ms={wait_ms})")


async def _shutdown_batchers():
    for model_key, batcher in list(BATCHERS.items()):
        try:
            await batcher.stop()
        except Exception as e:
            logger.warning(f"Failed stopping batcher {model_key}: {e}")
    BATCHERS.clear()


async def _submit_or_run(model_key: str, payload: Any, fallback_sync):
    batcher = BATCHERS.get(model_key)
    if batcher is not None:
        return await batcher.submit(payload)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fallback_sync)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Main Inference Service")
    start = time.perf_counter()
    load_main_models()
    _init_batchers()
    logger.info(f"✅ Main models loaded in {(time.perf_counter() - start):.2f}s")
    yield
    # Shutdown
    logger.info("🛑 Stopping Main Inference Service")
    await _shutdown_batchers()
    unload_models()
    executor.shutdown(wait=True)

app = FastAPI(title="General Inference Server", lifespan=lifespan)


@app.middleware("http")
async def track_inflight_requests(request: Request, call_next):
    global INFLIGHT_REQUESTS
    INFLIGHT_REQUESTS += 1
    try:
        return await call_next(request)
    finally:
        INFLIGHT_REQUESTS = max(0, INFLIGHT_REQUESTS - 1)

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
QWEN_DEFAULT_PROMPT = "Extract all text present in the image and return only that text, without any additional explanation or formatting. Preserve the original language exactly as it appears, and do not translate it."

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


def _qwen_caption_sync(image: Image.Image, prompt: str) -> str:
    qwen_bundle = MODELS.get("qwen_vl")
    if qwen_bundle is None:
        raise RuntimeError("Qwen model is not loaded")

    processor = qwen_bundle["processor"]
    model = qwen_bundle["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = [image], None

    processor_kwargs = dict(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        max_pixels=max(QWEN_MAX_PIXELS, 1),
        min_pixels=max(min(QWEN_MIN_PIXELS, QWEN_MAX_PIXELS), 1),
    )
    try:
        inputs = processor(**processor_kwargs).to(model.device)
    except TypeError:
        # Fallback for processor builds that do not expose max/min pixel kwargs.
        processor_kwargs.pop("max_pixels", None)
        processor_kwargs.pop("min_pixels", None)
        inputs = processor(**processor_kwargs).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    prompt_input_ids = getattr(inputs, "input_ids", None)
    if hasattr(prompt_input_ids, "shape"):
        prompt_len = prompt_input_ids.shape[1]
    elif isinstance(prompt_input_ids, list) and prompt_input_ids:
        prompt_len = len(prompt_input_ids[0])
    else:
        prompt_len = 0
    if isinstance(generated_ids, list):
        output_ids = [row[prompt_len:] for row in generated_ids]
    else:
        output_ids = generated_ids[:, prompt_len:]
    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not output_text:
        return ""
    return output_text[0].strip()


def _downsample_to_max_pixels(image: Image.Image, max_pixels: int) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    original = image.size
    if max_pixels <= 0:
        return image, original, original
    current_pixels = original[0] * original[1]
    if current_pixels <= max_pixels:
        return image, original, original

    scale = math.sqrt(max_pixels / float(current_pixels))
    new_w = max(1, int(original[0] * scale))
    new_h = max(1, int(original[1] * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, original, resized.size

# ===============================
# Face Recognition
# ===============================
@app.post("/process/face")
async def face_recognition(file: UploadFile = File(...)):
    logger.info(f"🧑 Face recognition: {file.filename}")
    if MODELS.get("face") is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition model not available (CUDA GPU required)",
        )
    start = time.perf_counter()
    img = Image.open(file.file).convert("RGB")
    img_np = np.array(img)
    
    lock = get_model_lock("face")
    loop = asyncio.get_event_loop()
    try:
        faces = await _submit_or_run(
            "face",
            img_np,
            lambda: _lock_inference(lock, lambda: MODELS["face"].get(img_np)),
        )
    finally:
        clear_cuda_memory("after face stage")

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
    
    try:
        result = await _submit_or_run(
            "emotion",
            image,
            lambda: _process_emotion_batch([image])[0],
        )
        return result
    finally:
        duration = time.perf_counter() - start
        logger.info(f"😊 Emotion finished in {duration:.3f}s")
        clear_cuda_memory("after emotion stage")


@app.post("/process/emotion/batch")
async def emotion_detection_batch(files: List[UploadFile] = File(..., description="Repeat form field 'files' for each image")):
    """Classify emotion for many face crops in one ViT forward (multipart, same field name 'files')."""
    max_n = max(1, _env_int("EMOTION_BATCH_MAX", 32))
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > max_n:
        raise HTTPException(status_code=400, detail=f"At most {max_n} images allowed (EMOTION_BATCH_MAX)")

    images: list[Image.Image] = []
    names: list[str] = []
    for uf in files:
        raw = await uf.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {uf.filename!r}")
        images.append(img)
        names.append(uf.filename or "")

    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            executor,
            lambda ims=list(images): _process_emotion_batch(ims),
        )
    finally:
        clear_cuda_memory("after emotion batch stage")
        logger.info(f"😊 Emotion batch finished: n={len(files)} in {(time.perf_counter() - start):.3f}s")

    out: list[dict[str, Any]] = []
    for name, row in zip(names, results):
        item = dict(row)
        if name:
            item["filename"] = name
        out.append(item)
    return out

# ===============================
# Scene Description
# ===============================
@app.post("/process/scene")
async def scene_description(file: UploadFile = File(...)):
    logger.info(f"🖼️ Scene description: {file.filename}")
    start = time.perf_counter()
    image = Image.open(file.file).convert("RGB")
    
    try:
        result = await _submit_or_run(
            "scene",
            image,
            lambda: _process_scene_batch([image])[0],
        )
        return result
    finally:
        duration = time.perf_counter() - start
        logger.info(f"🖼️ Scene finished in {duration:.3f}s")
        clear_cuda_memory("after scene stage")

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
    
    try:
        result = await _submit_or_run(
            "ram_plus",
            image,
            lambda: _process_ram_batch([image])[0],
        )
        return result
    finally:
        duration = time.perf_counter() - start
        logger.info(f"🏷️ RAM++ finished in {duration:.3f}s")
        clear_cuda_memory("after ram++")

# ===============================
# Text Embeddings
# ===============================
@app.post("/process/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    logger.info(f"🔗 Embeddings request: {len(req.texts)} items")
    start = time.perf_counter()
    loop = asyncio.get_event_loop()

    if len(req.texts) == 1:
        text = req.texts[0]
        batcher = BATCHERS.get("embed")
        if batcher is not None:
            vec = await batcher.submit(text)
        else:
            vec = await loop.run_in_executor(
                executor,
                lambda t=text: _process_embed_batch([t])[0],
            )
        embeddings = [vec]
    else:
        embeddings = await loop.run_in_executor(
            executor,
            lambda texts=list(req.texts): _process_embed_batch(texts),
        )

    duration = time.perf_counter() - start
    logger.info(f"🔗 Embeddings finished in {duration:.3f}s")
    return {"embeddings": embeddings, "count": len(embeddings)}

# ===============================
# Sarvam Translation
# ===============================
@app.post("/process/translation/sarvam")
async def sarvam_translation(req: SarvamTranslationRequest):
    logger.info(f"🇮🇳 Sarvam translation: {req.text[:30]}... -> {req.target_lang}")
    start = time.perf_counter()

    lock = get_model_lock("sarvam")
    def _sync():
        with GPU_LOCK:
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
        clear_cuda_memory("after sarvam")


@app.post("/process/caption/qwen")
async def qwen_caption(file: UploadFile = File(...), prompt: str = Form(QWEN_DEFAULT_PROMPT)):
    logger.info(f"🖼️ Qwen caption request: {file.filename}")
    if "qwen_vl" not in MODELS:
        raise HTTPException(status_code=503, detail="Qwen caption model not available")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image, original_size, processed_size = _downsample_to_max_pixels(image, max(QWEN_MAX_PIXELS, 1))
    if original_size != processed_size:
        logger.info(
            f"🧮 Qwen image downsampled from {original_size[0]}x{original_size[1]} "
            f"to {processed_size[0]}x{processed_size[1]} (max_pixels={QWEN_MAX_PIXELS})"
        )

    start = time.perf_counter()
    prompt_used = (prompt or "").strip() or QWEN_DEFAULT_PROMPT
    lock = get_model_lock("qwen_vl")
    loop = asyncio.get_event_loop()

    def _sync():
        with GPU_LOCK:
            with lock:
                caption = _qwen_caption_sync(image=image, prompt=prompt_used)
                return {"caption": caption}

    try:
        clear_cuda_memory("before qwen caption")
        payload = await loop.run_in_executor(executor, _sync)
        return {
            "caption": payload.get("caption", ""),
            "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "prompt_used": prompt_used,
            "duration_sec": round(time.perf_counter() - start, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Qwen captioning failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        clear_cuda_memory("after qwen caption")

@app.get("/health")
def health():
    status = get_runtime_model_status()
    return {
        "status": "ok",
        "service": "main-inference",
        "instance": INSTANCE_NAME,
        "inflight_requests": INFLIGHT_REQUESTS,
        **status,
    }