#!/usr/bin/env python3
"""Export RAM++ tag logits to ONNX for Triton (Phase 3).

Exports the GPU scoring graph up to tag logits [B, num_class]. Thresholding and
string join stay in FastAPI (same as native generate_tag postprocess).

Writes:
  triton_models/ram_plus/1/model.onnx
  triton_models/ram_plus/config.pbtxt
  triton_models/ram_plus/meta.json   # thresholds, delete indices, tag lists paths
  triton_models/ram_plus/tag_list.txt / tag_list_chinese.txt (copied)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "triton_models" / "ram_plus"
VERSION_DIR = OUT_DIR / "1"


class RamLogitsWrapper(nn.Module):
    """Vectorized subset of RAM_plus.generate_tag → logits only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        m = self.m
        image_embeds = m.image_proj(m.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)

        image_cls_embeds = image_embeds[:, 0, :]
        bs = image_embeds.shape[0]
        des_per_class = int(m.label_embed.shape[0] / m.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
        reweight_scale = m.reweight_scale.exp()
        logits_per_image = reweight_scale * image_cls_embeds @ m.label_embed.t()
        logits_per_image = logits_per_image.view(bs, -1, des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        reshaped_value = m.label_embed.view(-1, des_per_class, 512)
        # [bs, num_class, des_per_class, 1] * [1, num_class, des_per_class, 512]
        product = weight_normalized.unsqueeze(-1) * reshaped_value.unsqueeze(0)
        label_embed_reweight = product.sum(dim=2)

        label_embed = torch.nn.functional.relu(m.wordvec_proj(label_embed_reweight))
        tagging_embed = m.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )
        logits = m.fc(tagging_embed[0]).squeeze(-1)
        return logits


def _config_pbtxt(num_class: int, max_batch: int = 16) -> str:
    return f"""name: "ram_plus"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch}
default_model_filename: "model.onnx"
input [
  {{
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 384, 384 ]
  }}
]
output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ {num_class} ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 5000
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RAM++ logits to ONNX")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-batch", type=int, default=16)
    args = parser.parse_args()

    from models import _load_ram_plus_model
    from ram import get_transform
    from PIL import Image

    print("Loading RAM++…", flush=True)
    model = _load_ram_plus_model(args.device)
    model.eval()
    wrapped = RamLogitsWrapper(model).to(args.device)
    wrapped.eval()

    transform = get_transform(image_size=384)
    img = Image.new("RGB", (384, 384), color=(40, 80, 120))
    dummy = transform(img).unsqueeze(0).to(args.device)

    with torch.inference_mode():
        ref = wrapped(dummy)
    num_class = int(ref.shape[-1])
    print(f"logits shape={tuple(ref.shape)} num_class={num_class}", flush=True)

    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = VERSION_DIR / "model.onnx"
    print(f"Exporting ONNX -> {onnx_path}", flush=True)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # Metadata for FastAPI postprocess
    tag_list = [str(x) for x in model.tag_list.tolist()] if hasattr(model.tag_list, "tolist") else list(map(str, model.tag_list))
    tag_list_cn = (
        [str(x) for x in model.tag_list_chinese.tolist()]
        if hasattr(model.tag_list_chinese, "tolist")
        else list(map(str, model.tag_list_chinese))
    )
    # ensure files
    (OUT_DIR / "tag_list.txt").write_text("\n".join(tag_list) + "\n", encoding="utf-8")
    (OUT_DIR / "tag_list_chinese.txt").write_text("\n".join(tag_list_cn) + "\n", encoding="utf-8")

    thr = model.class_threshold.detach().cpu().numpy().astype(np.float32)
    np.save(OUT_DIR / "class_threshold.npy", thr)
    delete_idx = [int(x) for x in list(getattr(model, "delete_tag_index", []) or [])]

    meta = {
        "num_class": num_class,
        "image_size": 384,
        "delete_tag_index": delete_idx,
        "threshold_shape": list(thr.shape),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "config.pbtxt").write_text(_config_pbtxt(num_class, args.max_batch), encoding="utf-8")

    # ORT smoke
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": dummy.detach().cpu().numpy()})[0]
        max_abs = float(np.max(np.abs(out - ref.detach().cpu().numpy())))
        print(f"ORT smoke OK shape={out.shape} max|Δ|={max_abs:.6g}")
    except Exception as exc:
        print(f"WARN ORT smoke: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
