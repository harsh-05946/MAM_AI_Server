import json
import logging
import re
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from app.config import config


# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_ID = "google/gemma-4-E2B-it"

DEFAULT_CATEGORIES = [
    "News",
    "Sports",
    "Entertainment",
    "Religious",
    "Documentary",
    "Lifestyle",
]

PROMPT_VERSION = "v2"

# =========================================================
# QUANTIZATION CONFIG
# =========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# =========================================================
# LOAD MODEL + PROCESSOR
# =========================================================

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.eval()


# =========================================================
# HELPERS
# =========================================================

def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    items = []

    for item in value:
        text = str(item or "").strip()

        if text:
            items.append(text)

    return list(dict.fromkeys(items))


def _remove_source_mentions(text: Any) -> str:
    value = str(text or "").strip()

    if not value:
        return ""

    patterns = [
        r"\b(?:according to|based on|from)\s+the\s+(?:transcript|transcription|source(?:\s+data)?|provided\s+text)\b[:,]?\s*",
        r"\b(?:the\s+)?(?:transcript|transcription|source(?:\s+data)?|provided\s+text)\s+(?:indicates|suggests|shows|states|mentions)\b[:,]?\s*",
        r"\b(?:as\s+per|per)\s+the\s+(?:transcript|transcription|source(?:\s+data)?)\b[:,]?\s*",
        r"\b(?:transcript|transcription|source\s+data)\b[:,]?\s*",
    ]

    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)

    value = re.sub(r"\s{2,}", " ", value).strip(" -,:;")

    return value


def _sanitize_string_list(value: Any) -> list[str]:
    cleaned = []

    for item in _clean_string_list(value):
        sanitized = _remove_source_mentions(item)

        if sanitized:
            cleaned.append(sanitized)

    return cleaned


def _normalize_payload(payload: dict) -> dict:
    return {
        "synopsis": _remove_source_mentions(payload.get("synopsis")),
        "description": _remove_source_mentions(payload.get("description")),
        "families": _sanitize_string_list(payload.get("families")),
        "categories": _clean_string_list(payload.get("categories")),
        "tags": _sanitize_string_list(payload.get("tags")),
    }


# =========================================================
# PROMPT BUILDER
# =========================================================

def _build_prompt(source_text: str, source_language_code: str | None) -> str:
    language_hint = (
        (source_language_code or "").strip()
        or "unknown / possibly multilingual"
    )

    categories = ", ".join(DEFAULT_CATEGORIES)

    return (
        "You are a media-content analyst.\n"
        "Generate JSON with keys: synopsis, description, families, categories, tags.\n"
        "\n"
        "Multilingual handling rules:\n"
        f"- Reported source language code (advisory only): {language_hint}.\n"
        "- The transcript may be in any language, may differ from the reported code,\n"
        "  and may contain multiple languages mixed together.\n"
        "- Internally interpret each segment in its native language.\n"
        "- Translate all non-English content internally to English.\n"
        "\n"
        "Output rules:\n"
        "1) synopsis: <= 80 words.\n"
        "2) description: <= 220 words.\n"
        "3) categories: choose one or more strictly from this fixed list: "
        f"[{categories}].\n"
        "4) families: short higher-level group labels.\n"
        "5) tags: 8-25 ontology-style keyword tags.\n"
        "6) Ground every claim in the content.\n"
        "7) ALL output must be in English.\n"
        "8) Output ONLY valid JSON.\n"
        "9) Do NOT mention transcript/source/provided text.\n"
        "\n"
        "Transcript:\n"
        f"{source_text}"
    )


# =========================================================
# MAIN FUNCTION
# =========================================================

def generate_content_enrichment(
    source_text: str,
    source_language_code: str | None = None,
) -> dict:

    text = (source_text or "").strip()

    if not text:
        raise RuntimeError("source_text is empty")

    # =====================================================
    # TRUNCATE LONG INPUTS
    # =====================================================

    max_chars = max(2000, int(config.CONTENT_ENRICHMENT_MAX_CHARS))

    if len(text) > max_chars:
        text = text[:max_chars]

    # =====================================================
    # BUILD CHAT MESSAGES
    # =====================================================

    messages = [
        {
            "role": "system",
            "content": (
                "Return ONLY valid JSON. "
                "No markdown. "
                "Use exactly these keys: "
                "synopsis, description, families, categories, tags. "
                "All output must be in English."
            ),
        },
        {
            "role": "user",
            "content": _build_prompt(text, source_language_code),
        },
    ]

    # =====================================================
    # APPLY CHAT TEMPLATE
    # =====================================================

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # =====================================================
    # TOKENIZE
    # =====================================================

    inputs = processor(
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]

    # =====================================================
    # GENERATE
    # =====================================================

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # =====================================================
    # DECODE GENERATED TOKENS ONLY
    # =====================================================

    response = processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
    )[0]

    # =====================================================
    # PARSE JSON
    # =====================================================

    try:
        parsed = json.loads(response)

    except json.JSONDecodeError:
        logging.warning(
            "Model returned invalid JSON: %s",
            response[:500],
        )

        raise RuntimeError(
            f"Model returned invalid JSON: {response[:500]}"
        )

    # =====================================================
    # NORMALIZE
    # =====================================================

    normalized = _normalize_payload(
        parsed if isinstance(parsed, dict) else {}
    )

    normalized["raw_response"] = response
    normalized["model"] = MODEL_ID
    normalized["provider"] = "local_gemma"
    normalized["prompt_version"] = PROMPT_VERSION
    normalized["source_language_code"] = source_language_code

    return normalized   