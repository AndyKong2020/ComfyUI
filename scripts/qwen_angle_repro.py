#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw


DEFAULT_MODEL_DIR = os.environ.get("QWEN_IMAGE_EDIT_MODEL_DIR", "models/Qwen-Image-Edit-2511")
DEFAULT_ANGLE_LORA = os.environ.get(
    "QWEN_IMAGE_EDIT_ANGLE_LORA",
    "models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors",
)
DEFAULT_LIGHTNING_LORA = os.environ.get("QWEN_IMAGE_EDIT_LIGHTNING_LORA", "")
DEFAULT_INPUT_IMAGE = os.environ.get("QWEN_IMAGE_EDIT_INPUT", "input/example.png")
DEFAULT_OUT_DIR = os.environ.get("QWEN_IMAGE_EDIT_OUT_DIR", "output/qwen-angle-repro")


AZIMUTHS = [
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
]
ELEVATIONS = ["low-angle shot", "eye-level shot", "elevated shot", "high-angle shot"]
DISTANCES = ["close-up", "medium shot", "wide shot"]


def all_prompts():
    for distance in DISTANCES:
        for elevation in ELEVATIONS:
            for azimuth in AZIMUTHS:
                yield f"<sks> {azimuth} {elevation} {distance}"


def sample_prompts(limit):
    prompts = list(all_prompts())
    return prompts if limit is None else prompts[:limit]


def npu_hbm_mb():
    try:
        info = torch.npu.mem_get_info("npu:0")
        free_b, total_b = info
        return round((total_b - free_b) / 1024 / 1024, 1), round(total_b / 1024 / 1024, 1)
    except Exception:
        return None, None


def ensure_input(path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (768, 768), (230, 232, 224))
    draw = ImageDraw.Draw(im)
    draw.rectangle((260, 170, 508, 590), fill=(85, 105, 135), outline=(20, 25, 35), width=8)
    draw.ellipse((308, 75, 460, 227), fill=(220, 180, 140), outline=(20, 25, 35), width=8)
    draw.rectangle((300, 590, 355, 705), fill=(35, 38, 45))
    draw.rectangle((413, 590, 468, 705), fill=(35, 38, 45))
    draw.text((32, 32), "Qwen angle subject", fill=(20, 25, 35))
    im.save(path)
    return path


def load_pipe(args):
    pipe = DiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    angle_lora = Path(args.angle_lora)
    pipe.load_lora_weights(str(angle_lora.parent), weight_name=angle_lora.name, adapter_name="angle")
    adapter_names = ["angle"]
    adapter_weights = [args.angle_lora_scale]
    if args.lightning_lora:
        lightning_lora = Path(args.lightning_lora)
        pipe.load_lora_weights(str(lightning_lora.parent), weight_name=lightning_lora.name, adapter_name="lightning")
        adapter_names.append("lightning")
        adapter_weights.append(args.lightning_lora_scale)
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    pipe.to(args.device)
    return pipe


def run(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.npu.set_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = ensure_input(args.input)
    input_image = Image.open(input_path).convert("RGB")
    if args.input_size:
        input_image = input_image.resize((args.input_size, args.input_size), Image.LANCZOS)

    t0 = time.perf_counter()
    pipe = load_pipe(args)
    torch.npu.synchronize()
    load_s = time.perf_counter() - t0

    prompts = sample_prompts(args.limit)
    results_path = out_dir / "results.csv"
    rows = []

    if args.warmup and prompts:
        _ = pipe(
            image=input_image,
            prompt=prompts[0],
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
            max_sequence_length=args.max_sequence_length,
        ).images[0]
        torch.npu.synchronize()

    for idx, prompt in enumerate(prompts, 1):
        seed = args.seed + idx - 1
        hbm_before, hbm_total = npu_hbm_mb()
        start = time.perf_counter()
        image = pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            max_sequence_length=args.max_sequence_length,
        ).images[0]
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        hbm_after, _ = npu_hbm_mb()
        filename = f"{idx:03d}_{prompt.replace('<sks> ', '').replace(' ', '_').replace('-', '_')}.png"
        image.save(out_dir / filename)
        row = {
            "index": idx,
            "prompt": prompt,
            "seed": seed,
            "seconds": round(elapsed, 4),
            "hbm_before_mb": hbm_before,
            "hbm_after_mb": hbm_after,
            "hbm_total_mb": hbm_total,
            "output": filename,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_dir": args.model_dir,
        "angle_lora": args.angle_lora,
        "lightning_lora": args.lightning_lora,
        "device": args.device,
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "width": args.width,
        "height": args.height,
        "prompt_count": len(rows),
        "load_seconds": round(load_s, 4),
        "total_generation_seconds": round(sum(r["seconds"] for r in rows), 4),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 4) if rows else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--angle-lora", default=DEFAULT_ANGLE_LORA)
    parser.add_argument("--lightning-lora", default=DEFAULT_LIGHTNING_LORA)
    parser.add_argument("--angle-lora-scale", type=float, default=0.9)
    parser.add_argument("--lightning-lora-scale", type=float, default=1.0)
    parser.add_argument("--input", default=DEFAULT_INPUT_IMAGE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2511)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    args = parser.parse_args()
    if not args.lightning_lora:
        args.lightning_lora = None
    run(args)


if __name__ == "__main__":
    main()
