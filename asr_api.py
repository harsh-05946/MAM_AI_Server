# app/asr_api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import torch
import time
import logging
import asyncio
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import torchaudio
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=2)
VIBEVOICE_PROMPT = os.getenv("VIBEVOICE_PROMPT", "About Trump")
VIBEVOICE_MODEL_ID = os.getenv("VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-ASR-HF")

ASR_STATE_LOCK = threading.RLock()
INFER_LOCK = threading.Lock()
ASR_MODELS = {}


def load_vibevoice_models():
    with ASR_STATE_LOCK:
        if "vibevoice" in ASR_MODELS:
            return

        logger.info(f"📦 Loading VibeVoice from Hugging Face: {VIBEVOICE_MODEL_ID}")
        processor = AutoProcessor.from_pretrained(VIBEVOICE_MODEL_ID)
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            VIBEVOICE_MODEL_ID,
            device_map="auto",
        )
        ASR_MODELS["vibevoice"] = {"processor": processor, "model": model}


def unload_models():
    with ASR_STATE_LOCK:
        ASR_MODELS.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_runtime_model_status() -> dict:
    with ASR_STATE_LOCK:
        return {
            "loaded_models": list(ASR_MODELS.keys()),
            "vibevoice_model_id": VIBEVOICE_MODEL_ID,
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting ASR Service (VibeVoice)")
    start = time.perf_counter()
    load_vibevoice_models()
    logger.info(f"✅ VibeVoice loaded in {(time.perf_counter() - start):.2f}s")
    yield
    # Shutdown
    logger.info("🛑 Stopping ASR Service")
    unload_models()
    executor.shutdown(wait=True)

app = FastAPI(title="ASR Inference Server", lifespan=lifespan)

async def _vibevoice_task(audio_path):
    loop = asyncio.get_running_loop()
    lock = INFER_LOCK
    return await loop.run_in_executor(executor, lambda: _lock_inference(lock, lambda: _vibevoice_sync(audio_path)))

def _lock_inference(lock, fn):
    with lock:
        return fn()


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value


def _segment_output(start, end, text):
    start_f = _to_float(start, 0.0)
    end_f = _to_float(end, 0.0)
    text_v = str(text or "").strip()
    return {
        "start": start_f,
        "end": end_f,
        "text": text_v,
    }

def _vibevoice_sync(audio_path):
    vv = ASR_MODELS.get("vibevoice")
    if vv is None:
        raise RuntimeError("VibeVoice model is not loaded")

    processor = vv["processor"]
    model = vv["model"]

    # Align with the known-good standalone flow in testmicro.py.
    inputs = processor.apply_transcription_request(
        audio=audio_path,
        sampling_rate=24000,
    ).to(model.device, model.dtype)

    logger.info("🧩 Running same generate/decode path as testmicro.py")
    output_ids = model.generate(**inputs)
    if output_ids is None:
        raise RuntimeError("Model returned no output ids")
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise RuntimeError("Missing input_ids in processor output")

    generated_ids = output_ids[:, input_ids.shape[1] :]

    text = ""
    segments = []

    try:
        text_decoded = _to_first(processor.decode(generated_ids, return_format="transcription_only"))
        text = str(text_decoded or "").strip()
    except Exception as e:
        logger.warning(f"transcription_only decode failed: {e}")

    try:
        parsed_results = _to_first(processor.decode(generated_ids, return_format="parsed"))
        if isinstance(parsed_results, list):
            for item in parsed_results:
                if not isinstance(item, dict):
                    continue
                item_text = str(item.get("Content", "")).strip()
                if not item_text:
                    continue
                segments.append(_segment_output(
                    item.get("Start", 0.0),
                    item.get("End", 0.0),
                    item_text,
                ))
    except Exception as e:
        logger.warning(f"parsed decode failed: {e}")

    if not text and segments:
        text = " ".join(seg["text"] for seg in segments).strip()

    return {
        "text": text,
        "segments": segments,
        "task": "transcribe",
        "engine": "vibevoice"
    }

@app.post("/process/transcription/vibevoice")
async def transcription_vibevoice(file: UploadFile = File(...)):
    logger.info(f"🎙️ VibeVoice transcription request: {file.filename}")
    start = time.perf_counter()

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    safe_suffix = f"_{os.path.basename(file.filename or 'audio.wav')}"
    with tempfile.NamedTemporaryFile(prefix="asr_", suffix=safe_suffix, delete=False) as tmp:
        audio_path = tmp.name
        tmp.write(content)

    try:
        result = await _vibevoice_task(audio_path)
    except Exception as e:
        logger.exception("Error in VibeVoice transcription")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    duration = time.perf_counter() - start
    logger.info(f"🎙️ VibeVoice finished in {duration:.3f}s")
    return result

@app.get("/health")
def health():
    status = get_runtime_model_status()
    return {"status": "ok", "service": "vibevoice-asr", **status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
