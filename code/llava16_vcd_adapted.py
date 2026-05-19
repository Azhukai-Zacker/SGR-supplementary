# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
)

import llava16_native_threebranch_3seeds as base


ROOT = Path(".")
MODEL_PATH = base.MODEL_PATH
JSONL_PATH = base.JSONL_PATH
IMAGE_FOLDER = base.IMAGE_FOLDER
QUESTION = base.QUESTION
COCO_ANN_PATH = "<PATH_TO_COCO>/annotations"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run adapted VCD baseline on LLaVA-NeXT under the same COCO/CHAIR setup."
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--jsonl-path", type=str, default=JSONL_PATH)
    parser.add_argument("--image-folder", type=str, default=IMAGE_FOLDER)
    parser.add_argument("--question", type=str, default=QUESTION)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--alpha", type=float, default=0.5, help="VCD contrast strength.")
    parser.add_argument("--beta", type=float, default=0.1, help="Adaptive plausibility threshold.")
    parser.add_argument("--disable-apc", action="store_true", help="Disable adaptive plausibility constraint.")
    parser.add_argument(
        "--negative-image",
        choices=["noise", "blur", "black"],
        default="noise",
        help="Distorted visual branch used as the VCD negative branch.",
    )
    parser.add_argument("--noise-strength", type=float, default=0.5)
    parser.add_argument("--blur-radius", type=float, default=8.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--tag", type=str, default="adapted")
    parser.add_argument("--run-chair", action="store_true")
    parser.add_argument("--chair-cache", type=str, default="chair.pkl")
    parser.add_argument("--coco-path", type=str, default=COCO_ANN_PATH)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def make_negative_image(image_pil, mode, seed, noise_strength=0.5, blur_radius=8.0):
    if mode == "black":
        return Image.new("RGB", image_pil.size, (0, 0, 0))
    if mode == "blur":
        return image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if mode == "noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(image_pil.convert("RGB"), dtype=np.float32)
        noise = rng.normal(loc=127.5, scale=127.5, size=arr.shape).astype(np.float32)
        mixed = (1.0 - noise_strength) * arr + noise_strength * noise
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed, mode="RGB")
    raise ValueError(f"Unsupported negative image mode: {mode}")


def format_float_for_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def update_vcd_scores(vis_logits, neg_logits, alpha, beta, use_apc):
    vis_logits = vis_logits.float()
    neg_logits = neg_logits.float().to(vis_logits.device)
    visual_top1_id = int(torch.argmax(vis_logits, dim=-1).item())
    negative_top1_id = int(torch.argmax(neg_logits, dim=-1).item())

    adjusted_scores = (1.0 + alpha) * vis_logits - alpha * neg_logits
    apc_filtered = 0
    if use_apc:
        probs_m = torch.softmax(vis_logits, dim=-1)
        max_prob = torch.max(probs_m, dim=-1, keepdim=True).values
        apc_mask = probs_m < (max_prob * beta)
        apc_filtered = int(apc_mask.sum().item())
        adjusted_scores = adjusted_scores.masked_fill(apc_mask, -float("inf"))

    adjusted_top1_id = int(torch.argmax(adjusted_scores, dim=-1).item())
    return adjusted_scores, {
        "same_top1": int(visual_top1_id == negative_top1_id),
        "effective": int(adjusted_top1_id != visual_top1_id),
        "apc_filtered": apc_filtered,
    }


def extend_decoding_state(input_ids, attention_mask, next_token):
    next_token = next_token.to(device=input_ids.device, dtype=input_ids.dtype).view(1, 1)
    next_mask = torch.ones(
        (attention_mask.shape[0], 1),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat([input_ids, next_token], dim=1), torch.cat([attention_mask, next_mask], dim=1)


def prepare_cached_step(model, input_ids, attention_mask, past_key_values, static_inputs):
    return model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        pixel_values=static_inputs.get("pixel_values"),
        image_sizes=static_inputs.get("image_sizes"),
        use_cache=True,
    )


@torch.inference_mode()
def run_vcd(model, processor, tokenizer, image_pil, args, input_device, negative_seed):
    mm_inputs = base.build_mm_inputs(processor, tokenizer, image_pil, args.question, input_device)
    negative_image = make_negative_image(
        image_pil,
        mode=args.negative_image,
        seed=negative_seed,
        noise_strength=args.noise_strength,
        blur_radius=args.blur_radius,
    )
    negative_inputs = base.build_mm_inputs(processor, tokenizer, negative_image, args.question, input_device)

    vis_outputs = model(
        **mm_inputs,
        use_cache=True,
        return_dict=True,
    )
    neg_outputs = model(
        **negative_inputs,
        use_cache=True,
        return_dict=True,
    )

    vis_input_ids = mm_inputs["input_ids"]
    neg_input_ids = negative_inputs["input_ids"]
    vis_attention_mask = mm_inputs["attention_mask"]
    neg_attention_mask = negative_inputs["attention_mask"]
    vis_past = vis_outputs.past_key_values
    neg_past = neg_outputs.past_key_values
    vis_logits = vis_outputs.logits[:, -1, :]
    neg_logits = neg_outputs.logits[:, -1, :]

    eos_ids = base.resolve_eos_ids(tokenizer)
    raw_ids = []
    diag_stats = {
        "vcd_calls": 0,
        "effective_interventions": 0,
        "negative_same_top1_cnt": 0,
        "apc_filtered_sum": 0,
    }

    for step_idx in range(args.max_new_tokens):
        adjusted_scores, step_stats = update_vcd_scores(
            vis_logits=vis_logits,
            neg_logits=neg_logits,
            alpha=args.alpha,
            beta=args.beta,
            use_apc=not args.disable_apc,
        )
        next_token = torch.argmax(adjusted_scores, dim=-1)
        next_token_id = int(next_token.item())
        raw_ids.append(next_token_id)

        diag_stats["vcd_calls"] += 1
        diag_stats["effective_interventions"] += step_stats["effective"]
        diag_stats["negative_same_top1_cnt"] += step_stats["same_top1"]
        diag_stats["apc_filtered_sum"] += step_stats["apc_filtered"]

        if next_token_id in eos_ids or step_idx == args.max_new_tokens - 1:
            break

        vis_input_ids, vis_attention_mask = extend_decoding_state(
            vis_input_ids,
            vis_attention_mask,
            next_token,
        )
        neg_input_ids, neg_attention_mask = extend_decoding_state(
            neg_input_ids,
            neg_attention_mask,
            next_token,
        )

        vis_step_inputs = prepare_cached_step(
            model,
            input_ids=vis_input_ids,
            attention_mask=vis_attention_mask,
            past_key_values=vis_past,
            static_inputs=mm_inputs,
        )
        neg_step_inputs = prepare_cached_step(
            model,
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            past_key_values=neg_past,
            static_inputs=negative_inputs,
        )
        vis_outputs = model(**vis_step_inputs, return_dict=True)
        neg_outputs = model(**neg_step_inputs, return_dict=True)
        vis_past = vis_outputs.past_key_values
        neg_past = neg_outputs.past_key_values
        vis_logits = vis_outputs.logits[:, -1, :]
        neg_logits = neg_outputs.logits[:, -1, :]

    trimmed_ids, saw_eos = base.trim_generated_ids(raw_ids, eos_ids)
    caption = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    return {
        "caption": caption,
        "generated_len": len(trimmed_ids),
        "truncated": not saw_eos and len(raw_ids) >= args.max_new_tokens,
        "diag_stats": diag_stats,
    }


def init_stats():
    return {
        "missing_image_cnt": 0,
        "error_cnt": 0,
        "truncated_cnt": 0,
        "success_cnt": 0,
        "avg_generated_len_sum": 0.0,
        "vcd_calls": 0,
        "effective_interventions": 0,
        "negative_same_top1_cnt": 0,
        "apc_filtered_sum": 0,
    }


def merge_stats(global_stats, local_stats):
    for key in global_stats:
        global_stats[key] += local_stats.get(key, 0)


def build_output_paths(args):
    tag_suffix = f"_{args.tag}" if args.tag else ""
    apc_tag = "noapc" if args.disable_apc else f"beta{format_float_for_name(args.beta)}"
    stem = (
        f"llava16_vcd_{args.negative_image}_alpha{format_float_for_name(args.alpha)}_"
        f"{apc_tag}_{args.max_samples}_seed{args.seed}_t{args.max_new_tokens}{tag_suffix}"
    )
    return {
        "output": ROOT / f"{stem}.json",
        "stats": ROOT / f"{stem}_stats.json",
        "chair": ROOT / f"chair_{stem}.json",
    }


def maybe_save(paths, results, stats):
    save_json(str(paths["output"]), {"annotations": results})
    save_json(str(paths["stats"]), stats)


def run_chair(paths, args):
    cmd = [
        sys.executable,
        "eval_pai_chair.py",
        "--cap_file",
        str(paths["output"]),
        "--image_id_key",
        "image_id",
        "--caption_key",
        "caption",
        "--cache",
        args.chair_cache,
        "--coco_path",
        args.coco_path,
        "--save_path",
        str(paths["chair"]),
    ]
    print("\nRunning CHAIR:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading LLaVA-NeXT from {args.model_path} ...")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("negative_image:", args.negative_image)
    print("alpha:", args.alpha)
    print("beta:", args.beta)
    print("use_apc:", not args.disable_apc)
    print("max_samples:", args.max_samples)
    print("max_new_tokens:", args.max_new_tokens)
    print("seed:", args.seed)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=base.resolve_torch_dtype(),
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    try:
        model.tie_weights()
    except Exception:
        pass

    image_processor = LlavaNextImageProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)
    tokenizer = processor.tokenizer
    input_device = base.get_input_device(model)
    print("model loaded")
    print("input device:", input_device)

    target_image_ids = base.load_target_image_ids(args.jsonl_path, args.max_samples)
    paths = build_output_paths(args)
    results = []
    stats = init_stats()

    for idx, img_id in enumerate(tqdm(target_image_ids, desc="LLaVA-NeXT adapted VCD"), start=1):
        img_filename = f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        image_path = os.path.join(args.image_folder, img_filename)

        if not os.path.exists(image_path):
            stats["missing_image_cnt"] += 1
            print(f"[Skip] image not found: {image_path}")
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")
            out = run_vcd(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_pil=raw_image,
                args=args,
                input_device=input_device,
                negative_seed=args.seed + idx,
            )
            results.append({"image_id": int(img_id), "caption": out["caption"]})
            stats["success_cnt"] += 1
            stats["truncated_cnt"] += int(out["truncated"])
            stats["avg_generated_len_sum"] += float(out["generated_len"])
            merge_stats(stats, out["diag_stats"])

            if idx % args.save_every == 0:
                maybe_save(paths, results, stats)

        except torch.cuda.OutOfMemoryError as exc:
            print(f"\n[OOM] image_id={img_id}: {repr(exc)}")
            stats["error_cnt"] += 1
            torch.cuda.empty_cache()
        except Exception:
            print(f"\n[Error] image_id={img_id}")
            stats["error_cnt"] += 1
            traceback.print_exc()
            break

    maybe_save(paths, results, stats)

    print("\nDone.")
    avg_len = stats["avg_generated_len_sum"] / max(stats["success_cnt"], 1)
    print(json.dumps(stats, indent=4, ensure_ascii=False))
    print(f"avg generated len: {avg_len:.2f}")
    print(f"saved captions -> {paths['output']}")
    print(f"saved stats -> {paths['stats']}")

    if args.run_chair:
        run_chair(paths, args)


if __name__ == "__main__":
    main()
