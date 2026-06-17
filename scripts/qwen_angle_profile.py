#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from PIL import Image

from qwen_angle_repro import ensure_input, load_pipe


DEFAULT_MODEL_DIR = os.environ.get("QWEN_IMAGE_EDIT_MODEL_DIR", "models/Qwen-Image-Edit-2511")
DEFAULT_ANGLE_LORA = os.environ.get(
    "QWEN_IMAGE_EDIT_ANGLE_LORA",
    "models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors",
)
DEFAULT_LIGHTNING_LORA = os.environ.get(
    "QWEN_IMAGE_EDIT_LIGHTNING_LORA",
    "models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
)
DEFAULT_INPUT_IMAGE = os.environ.get("QWEN_IMAGE_EDIT_INPUT", "input/example.png")
DEFAULT_OUT_DIR = os.environ.get("QWEN_IMAGE_EDIT_PROFILE_OUT_DIR", "output/qwen-angle-profile")


def npu_hbm_mb():
    free_b, total_b = torch.npu.mem_get_info("npu:0")
    return round((total_b - free_b) / 1024 / 1024, 1), round(total_b / 1024 / 1024, 1)


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
    parser.add_argument("--prompt", default="<sks> front view low-angle shot close-up")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2511)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.npu.set_device(args.device)

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "torch_npu_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(ensure_input(args.input)).convert("RGB")
    if args.input_size:
        input_image = input_image.resize((args.input_size, args.input_size), Image.LANCZOS)

    pipe = load_pipe(args)
    torch.npu.synchronize()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    _ = pipe(
        image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        generator=generator,
        max_sequence_length=args.max_sequence_length,
    ).images[0]
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        data_simplification=False,
    )
    schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
    handler = torch_npu.profiler.tensorboard_trace_handler(str(trace_dir))

    hbm_before, hbm_total = npu_hbm_mb()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        start = time.perf_counter()
        image = pipe(
            image=input_image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            generator=torch.Generator(device="cpu").manual_seed(args.seed + 1),
            max_sequence_length=args.max_sequence_length,
        ).images[0]
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        prof.step()
        prof.step()

    hbm_after, _ = npu_hbm_mb()
    image.save(out_dir / "profiled_output.png")
    summary = {
        "prompt": args.prompt,
        "steps": args.steps,
        "device": args.device,
        "seconds": round(elapsed, 4),
        "hbm_before_mb": hbm_before,
        "hbm_after_mb": hbm_after,
        "hbm_total_mb": hbm_total,
        "trace_dir": str(trace_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
