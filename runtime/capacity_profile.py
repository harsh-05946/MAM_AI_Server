"""Processing ↔ AI capacity contracts (scalable + six-lane rollback)."""
from __future__ import annotations

from typing import Any, Optional
import json
import os
from pathlib import Path

from runtime.model_registry import HTTP_BATCH_LIMIT_ENV

ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "configs" / "capacity"

PROFILE_SCALABLE = "l40s-scalable-lanes-v1"
PROFILE_SIX_LANE = "l40s-six-lane-v1"

PUBLIC_TO_INTERNAL = {
    "face": "insightface",
    "emotion": "emotion",
    "scene": "scene",
    "object": "ram_plus",
    "qwen": "qwen_vl",
    "sarvam": "sarvam",
    "embeddings": "embed",
}

INTERNAL_TO_PUBLIC = {v: k for k, v in PUBLIC_TO_INTERNAL.items()}

EXEC_CLASS = {
    "insightface": "visual",
    "emotion": "visual",
    "scene": "visual",
    "ram_plus": "visual",
    "embed": "visual",
    "qwen_vl": "qwen",
    "sarvam": "sarvam",
}

# Default batch ceilings (overridable via FACE_BATCH_MAX etc.).
DEFAULT_BATCHES = {
    "face": 16,
    "emotion": 64,
    "scene": 16,
    "object": 16,
    "qwen": 20,
    "sarvam": 20,
    "embeddings": 32,
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def active_profile_name() -> str:
    return os.getenv("AI_CAPACITY_PROFILE", PROFILE_SCALABLE).strip() or PROFILE_SCALABLE


def is_scalable_profile(name: Optional[str] = None) -> bool:
    return (name or active_profile_name()) == PROFILE_SCALABLE


def http_batch_max(internal_model: str) -> int:
    public = INTERNAL_TO_PUBLIC.get(internal_model)
    default = DEFAULT_BATCHES.get(public or "", 1)
    env_name, registry_default = HTTP_BATCH_LIMIT_ENV.get(internal_model, ("", default))
    if not env_name:
        return max(1, default)
    return max(1, _env_int(env_name, registry_default if registry_default else default))


def batch_for_public(public_name: str) -> int:
    internal = PUBLIC_TO_INTERNAL[public_name]
    return http_batch_max(internal)


def load_profile_json(profile: Optional[str] = None) -> dict[str, Any]:
    name = profile or active_profile_name()
    path = PROFILE_DIR / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text())
    # Synthesize if file missing.
    if name == PROFILE_SCALABLE:
        return synthesize_scalable_profile_json()
    return synthesize_six_lane_profile_json()


def synthesize_scalable_profile_json() -> dict[str, Any]:
    per_media = _env_int("PER_MEDIA_AI_INFLIGHT", 8)
    per_media_model = _env_int("PER_MEDIA_MODEL_INFLIGHT", 1)
    models = {
        name: {"max_concurrent": per_media_model, "batch": batch_for_public(name)}
        for name in PUBLIC_TO_INTERNAL
    }
    return {
        "profile": PROFILE_SCALABLE,
        "profile_version": 1,
        "scalable_fleet": True,
        "capacity_contract_required": True,
        "global_inflight": per_media,
        "per_media_inflight": per_media,
        "per_media_model_inflight": per_media_model,
        "models": models,
        "qwen_sarvam_parallel": _env_bool("AI_ENABLE_QWEN_SARVAM_OVERLAP", True),
    }


def synthesize_six_lane_profile_json() -> dict[str, Any]:
    models = {
        name: {"max_concurrent": 1, "batch": batch_for_public(name), "max_batch": batch_for_public(name)}
        for name in PUBLIC_TO_INTERNAL
    }
    return {
        "profile": PROFILE_SIX_LANE,
        "profile_version": 1,
        "scalable_fleet": False,
        "capacity_contract_required": True,
        "global_inflight": 6,
        "per_media_inflight": 6,
        "models": models,
        "qwen_sarvam_parallel": True,
    }


def scaling_knobs() -> dict[str, int]:
    """Env knobs shared with Processing (same derivation)."""
    return {
        "max_parallel_media": max(1, _env_int("MAX_PARALLEL_MEDIA", 2)),
        "per_media_ai_inflight": max(1, _env_int("PER_MEDIA_AI_INFLIGHT", 8)),
        "per_media_model_inflight": max(1, _env_int("PER_MEDIA_MODEL_INFLIGHT", 1)),
    }


def derive_admission(knobs: Optional[dict[str, int]] = None) -> dict[str, Any]:
    """fleet_global / fleet_model — must match Processing exactly."""
    k = knobs or scaling_knobs()
    max_parallel = k["max_parallel_media"]
    per_media = k["per_media_ai_inflight"]
    per_media_model = k["per_media_model_inflight"]
    fleet_global = per_media * max_parallel
    fleet_model = per_media_model * max_parallel
    models = {}
    for public in PUBLIC_TO_INTERNAL:
        models[public] = {
            "fleet": fleet_model,
            "per_media": per_media_model,
            "batch": batch_for_public(public),
        }
    return {
        "fleet_global": fleet_global,
        "per_media": per_media,
        "per_media_model": per_media_model,
        "fleet_model": fleet_model,
        "models": models,
    }


def execution_bounds() -> dict[str, Any]:
    """AI-local queue/GPU limits — independent of MAX_PARALLEL_MEDIA."""
    # Prefer AI_ACCEPTED_QUEUE_LIMIT; keep older aliases.
    accepted = _env_int(
        "AI_ACCEPTED_QUEUE_LIMIT",
        _env_int(
            "AI_MAX_ACCEPTED_INFERENCE_REQUESTS",
            _env_int("AI_INTERNAL_MAX_WAITING_REQUESTS", 6),
        ),
    )
    visual = _env_int("AI_GPU_SLOTS_VISUAL", _env_int("AI_VISUAL_EXECUTION_SLOTS", 1))
    qwen = _env_int("AI_GPU_SLOTS_QWEN", _env_int("AI_QWEN_EXECUTION_SLOTS", 1))
    sarvam = _env_int("AI_GPU_SLOTS_SARVAM", _env_int("AI_SARVAM_EXECUTION_SLOTS", 1))
    return {
        "accepted_queue_limit": max(1, accepted),
        "gpu_slots": {
            "visual": max(1, visual),
            "qwen": max(1, qwen),
            "sarvam": max(1, sarvam),
        },
        "visual_qwen_overlap": _env_bool("AI_ENABLE_VISUAL_QWEN_OVERLAP", False),
        "visual_sarvam_overlap": _env_bool("AI_ENABLE_VISUAL_SARVAM_OVERLAP", False),
        "qwen_sarvam_overlap": _env_bool("AI_ENABLE_QWEN_SARVAM_OVERLAP", True),
        "combined_generative_limit_enabled": _env_bool(
            "AI_COMBINED_GENERATIVE_LIMIT_ENABLED", False
        ),
        "max_active_qwen": _env_int("AI_MAX_ACTIVE_QWEN_REQUESTS", 1),
        "max_active_sarvam": _env_int("AI_MAX_ACTIVE_SARVAM_REQUESTS", 1),
    }


def model_fleet_concurrent(public_name: str) -> int:
    """AI-side per-model accepted concurrency (= fleet lane width)."""
    if is_scalable_profile():
        return derive_admission()["fleet_model"]
    return max(1, _env_int(f"AI_MODEL_MAX_CONCURRENT_{public_name.upper()}", 1))


# Back-compat alias used by gpu_scheduler.
def model_max_concurrent(public_name: str) -> int:
    return model_fleet_concurrent(public_name)


def capacity_limits() -> dict[str, Any]:
    """Internal summary used by scheduler init / legacy callers."""
    profile = active_profile_name()
    exec_b = execution_bounds()
    admission = derive_admission()
    if is_scalable_profile(profile):
        models = {
            public: {
                "max_concurrent": admission["models"][public]["fleet"],
                "max_batch": admission["models"][public]["batch"],
                "batch": admission["models"][public]["batch"],
                "internal_model": PUBLIC_TO_INTERNAL[public],
                "exec_class": EXEC_CLASS[PUBLIC_TO_INTERNAL[public]],
            }
            for public in PUBLIC_TO_INTERNAL
        }
        return {
            "profile": profile,
            "profile_version": 1,
            "scalable_fleet": True,
            "global_inflight": admission["fleet_global"],
            "hard_client_inflight": admission["fleet_global"],
            "recommended_client_inflight": admission["fleet_global"],
            "per_media_inflight": admission["per_media"],
            "accepted_queue_limit": exec_b["accepted_queue_limit"],
            "models": models,
            "qwen_sarvam_parallel": exec_b["qwen_sarvam_overlap"],
            "qwen_sarvam_parallel_allowed": exec_b["qwen_sarvam_overlap"],
            "execution": {
                "visual_slots": exec_b["gpu_slots"]["visual"],
                "qwen_slots": exec_b["gpu_slots"]["qwen"],
                "sarvam_slots": exec_b["gpu_slots"]["sarvam"],
                "visual_qwen_overlap": exec_b["visual_qwen_overlap"],
                "visual_sarvam_overlap": exec_b["visual_sarvam_overlap"],
                "qwen_sarvam_overlap": exec_b["qwen_sarvam_overlap"],
                "combined_generative_limit_enabled": exec_b[
                    "combined_generative_limit_enabled"
                ],
                "max_active_qwen": exec_b["max_active_qwen"],
                "max_active_sarvam": exec_b["max_active_sarvam"],
                "accepted_queue_limit": exec_b["accepted_queue_limit"],
            },
            "overload": {
                "model_retry_after_seconds": _env_int(
                    "AI_MODEL_OVERLOAD_RETRY_AFTER_SECONDS", 10
                ),
                "server_retry_after_seconds": _env_int(
                    "AI_OVERLOAD_RETRY_AFTER_SECONDS", 10
                ),
            },
            "admission": admission,
            "max_parallel_media": scaling_knobs()["max_parallel_media"],
        }

    # six-lane rollback
    models = {
        public: {
            "max_concurrent": 1,
            "max_batch": batch_for_public(public),
            "batch": batch_for_public(public),
            "internal_model": PUBLIC_TO_INTERNAL[public],
            "exec_class": EXEC_CLASS[PUBLIC_TO_INTERNAL[public]],
        }
        for public in PUBLIC_TO_INTERNAL
    }
    return {
        "profile": profile,
        "profile_version": 1,
        "scalable_fleet": False,
        "global_inflight": exec_b["accepted_queue_limit"],
        "hard_client_inflight": exec_b["accepted_queue_limit"],
        "recommended_client_inflight": exec_b["accepted_queue_limit"],
        "per_media_inflight": exec_b["accepted_queue_limit"],
        "accepted_queue_limit": exec_b["accepted_queue_limit"],
        "models": models,
        "qwen_sarvam_parallel": exec_b["qwen_sarvam_overlap"],
        "qwen_sarvam_parallel_allowed": exec_b["qwen_sarvam_overlap"],
        "execution": {
            "visual_slots": exec_b["gpu_slots"]["visual"],
            "qwen_slots": exec_b["gpu_slots"]["qwen"],
            "sarvam_slots": exec_b["gpu_slots"]["sarvam"],
            "visual_qwen_overlap": exec_b["visual_qwen_overlap"],
            "visual_sarvam_overlap": exec_b["visual_sarvam_overlap"],
            "qwen_sarvam_overlap": exec_b["qwen_sarvam_overlap"],
            "combined_generative_limit_enabled": exec_b[
                "combined_generative_limit_enabled"
            ],
            "max_active_qwen": exec_b["max_active_qwen"],
            "max_active_sarvam": exec_b["max_active_sarvam"],
            "accepted_queue_limit": exec_b["accepted_queue_limit"],
        },
        "overload": {
            "model_retry_after_seconds": _env_int(
                "AI_MODEL_OVERLOAD_RETRY_AFTER_SECONDS", 5
            ),
            "server_retry_after_seconds": _env_int("AI_OVERLOAD_RETRY_AFTER_SECONDS", 10),
        },
        "admission": {
            "fleet_global": exec_b["accepted_queue_limit"],
            "per_media": exec_b["accepted_queue_limit"],
            "models": {
                p: {"fleet": 1, "per_media": 1, "batch": batch_for_public(p)}
                for p in PUBLIC_TO_INTERNAL
            },
        },
        "max_parallel_media": 1,
    }


def build_capacity_response(
    *,
    ready: bool = True,
    scheduler_snapshot: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Exact schema Processing validates for l40s-scalable-lanes-v1."""
    profile = active_profile_name()
    knobs = scaling_knobs()
    admission = derive_admission(knobs)
    exec_b = execution_bounds()

    if is_scalable_profile(profile):
        body: dict[str, Any] = {
            "profile": PROFILE_SCALABLE,
            "profile_version": 1,
            "max_parallel_media": knobs["max_parallel_media"],
            "admission": {
                "fleet_global": admission["fleet_global"],
                "per_media": admission["per_media"],
                "models": admission["models"],
            },
            "execution": {
                "accepted_queue_limit": exec_b["accepted_queue_limit"],
                "gpu_slots": dict(exec_b["gpu_slots"]),
            },
        }
        return body

    # Rollback profile — keep previous six-lane fields Processing already knew.
    limits = capacity_limits()
    body = {
        "profile": profile,
        "profile_version": 1,
        "ready": ready,
        "recommended_client_inflight": limits["recommended_client_inflight"],
        "hard_client_inflight": limits["hard_client_inflight"],
        "global_inflight": limits["global_inflight"],
        "per_media_inflight": limits["per_media_inflight"],
        "models": {
            k: {
                "max_concurrent": v["max_concurrent"],
                "max_batch": v["batch"],
                "batch": v["batch"],
            }
            for k, v in limits["models"].items()
        },
        "qwen_sarvam_parallel": limits["qwen_sarvam_parallel"],
        "qwen_sarvam_parallel_allowed": limits["qwen_sarvam_parallel_allowed"],
        "execution": {
            "accepted_queue_limit": exec_b["accepted_queue_limit"],
            "gpu_slots": dict(exec_b["gpu_slots"]),
        },
    }
    if scheduler_snapshot is not None:
        body["live"] = {
            "active_by_model": scheduler_snapshot.get("active_by_model"),
            "waiting_by_model": scheduler_snapshot.get("waiting_by_model"),
            "rejected_by_model": scheduler_snapshot.get("rejected_by_model"),
            "exec_busy": scheduler_snapshot.get("exec_busy"),
        }
    return body


def public_model_name(internal: Optional[str]) -> Optional[str]:
    if not internal:
        return None
    return INTERNAL_TO_PUBLIC.get(internal, internal)


def exec_class_for_model(internal: str) -> str:
    return EXEC_CLASS.get(internal, "visual")
