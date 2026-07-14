from __future__ import annotations

from typing import Any


def detect_bottlenecks(agg: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    gpu = agg.get("gpu") or {}
    host = agg.get("host") or {}
    models = {r["model"]: r for r in agg.get("model_rows") or []}
    total_gpu = sum(float(r.get("total_gpu_time") or 0) for r in (agg.get("model_rows") or [])) or 1.0
    face = models.get("insightface") or {}
    qwen = models.get("qwen_vl") or {}
    face_share = 100.0 * float(face.get("total_gpu_time") or 0) / total_gpu
    qwen_share = 100.0 * float(qwen.get("total_gpu_time") or 0) / total_gpu
    avg_util = float(gpu.get("avg_util") or 0)
    occ = float((agg.get("scheduler") or {}).get("occupancy_percent") or 0)
    avg_cpu = float(host.get("avg_cpu") or 0)
    ram_fb = int((agg.get("counts") or {}).get("ram_fallbacks") or 0)
    completed = int((agg.get("counts") or {}).get("completed") or 0)
    providers = agg.get("providers") or []
    face_fail = any(
        (p.get("model") or "").startswith("insightface") and not p.get("validation_passed") for p in providers
    )

    if face_fail:
        notes.append("InsightFace CUDA provider validation FAILED. Stop interpreting throughput until CUDA EP is fixed.")
    if face_share > 40:
        notes.append(f"InsightFace currently consumes {face_share:.0f}% of recorded model GPU/ORT time.")
    if float(face.get("avg_sec_per_item") or 0) > 0.40:
        notes.append(
            f"InsightFace average {face.get('avg_sec_per_item')} sec/item exceeds 0.40s; three-media target may remain difficult."
        )
    if avg_util > 90 and occ > 90:
        notes.append("GPU average >90% and scheduler occupancy >90%: L40S appears compute-bound.")
    if avg_util < 65 and occ < 70:
        notes.append("GPU average <65% and scheduler occupancy <70%: GPU is underfed.")
    if avg_util < 65 and occ > 85:
        notes.append("GPU average <65% but scheduler occupancy >85%: scheduled section may be inefficient.")
    if avg_cpu > 85 and avg_util < 65:
        notes.append("CPU >85% while GPU <65%: likely host preprocessing bottleneck.")
    if qwen_share > 30:
        notes.append(f"Qwen occupies {qwen_share:.0f}% of model execution time.")
    if ram_fb > 0 and completed > 0 and (100.0 * ram_fb / max(completed, 1)) > 1:
        notes.append(f"RAM++ batch fallback count is {ram_fb}; fix batching before raising batch size.")
    if int((agg.get("timeout_risk") or {}).get("waiting_gt_600") or 0) > 0:
        notes.append("Requests exceeded 600s wait threshold; proxy timeout risk is elevated.")
    if int((agg.get("counts") or {}).get("oom") or 0) > 0:
        notes.append("CUDA OOM events observed; reduce batch size and inspect memory.")
    if not notes:
        notes.append("No deterministic bottleneck rule matched yet; continue collecting campaign data.")
    return notes
