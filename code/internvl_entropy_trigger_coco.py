# -*- coding: utf-8 -*-
"""InternVL COCO SGR runs with entropy-based trigger schedules.

This file contains the shared implementation used by:
  - run_internvl_entropy_online_k8.py
  - run_internvl_entropy_oracle_topk.py

Both variants keep the SGR reranker unchanged and only replace the trigger
policy. The online variant uses only past entropy values; the oracle variant
uses a baseline entropy trace and is diagnostic only.
"""

import argparse
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

import run_mmhal_internvl as mm


ROOT = Path(".")
JSONL_PATH = Path("<PATH_TO_COCO_500_JSONL>")
IMAGE_FOLDER = Path("<PATH_TO_COCO>/val2014")
OUTPUT_DIR = ROOT / "intern"
QUESTION = "Please describe this image in detail."


def parse_args(default_mode="online"):
    parser = argparse.ArgumentParser(
        description="Run InternVL SGR with entropy-trigger diagnostics on COCO."
    )
    parser.add_argument("--trigger-mode", choices=["online", "oracle"], default=default_mode)
    parser.add_argument("--model-path", type=str, default=mm.MODEL_PATH)
    parser.add_argument("--jsonl-path", type=Path, default=JSONL_PATH)
    parser.add_argument("--image-folder", type=Path, default=IMAGE_FOLDER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--question", type=str, default=QUESTION)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-num-tiles", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--disable-flash-attn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-chair", action="store_true")
    parser.add_argument("--chair-cache", type=Path, default=ROOT / "chair.pkl")
    parser.add_argument("--coco-anno-path", type=Path, default=Path("<PATH_TO_COCO>/annotations"))

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--effective-max-step", type=int, default=220)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--text-alpha", type=float, default=0.20)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--black-prior-threshold", type=float, default=0.50)
    parser.add_argument("--black-visual-gap-threshold", type=float, default=0.30)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    parser.add_argument("--guard-mode", type=str, default="syntax", choices=["syntax", "none"])
    parser.add_argument("--contrast-branch", type=str, default="black", choices=["black", "text"])

    parser.add_argument("--entropy-warmup", type=int, default=5)
    parser.add_argument("--entropy-lambda", type=float, default=1.0)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def load_target_image_ids(jsonl_path, max_samples):
    image_ids = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            image_id = data.get("image_id", data.get("id"))
            if image_id is not None:
                image_ids.append(int(image_id))
    return image_ids[:max_samples]


def build_output_paths(args):
    stem = f"internvl_sgr_entropy_{args.trigger_mode}_k{args.k}_seed{args.seed}"
    if args.trigger_mode == "online":
        stem += f"_warm{args.entropy_warmup}_lam{str(args.entropy_lambda).replace('.', 'p')}"
    if args.tag:
        stem += f"_{args.tag}"
    return {
        "output": args.output_dir / f"{stem}.json",
        "stats": args.output_dir / f"{stem}_stats.json",
        "chair": args.output_dir / f"{stem}_chair.json",
    }


def init_stats(args):
    return {
        "method": f"internvl_sgr_entropy_{args.trigger_mode}",
        "success_cnt": 0,
        "missing_image_cnt": 0,
        "error_cnt": 0,
        "avg_generated_len_approx_sum": 0.0,
        "model_path": str(args.model_path),
        "jsonl_path": str(args.jsonl_path),
        "image_folder": str(args.image_folder),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_num_tiles": args.max_num_tiles,
        "trigger_mode": args.trigger_mode,
        "k": args.k,
        "effective_max_step": args.effective_max_step,
        "top_k": args.top_k,
        "black_alpha": args.black_alpha,
        "text_alpha": args.text_alpha,
        "apc_threshold": args.apc_threshold,
        "syntax_threshold": args.syntax_threshold,
        "syntax_margin": args.syntax_margin,
        "guard_mode": args.guard_mode,
        "contrast_branch": args.contrast_branch,
        "entropy_warmup": args.entropy_warmup,
        "entropy_lambda": args.entropy_lambda,
        "min_history": args.min_history,
        "planned_interventions": 0,
        "trigger_hits": 0,
        "total_interventions": 0,
        "effective_interventions": 0,
        "guardrail_blocks": 0,
        "guardrail_accept_cnt": 0,
        "guardrail_fallback_cnt": 0,
        "online_threshold_checks": 0,
        "online_threshold_passes": 0,
        "oracle_trace_tokens": 0,
    }


def add_derived_stats(stats):
    out = dict(stats)
    out["avg_generated_len_approx"] = out["avg_generated_len_approx_sum"] / max(out["success_cnt"], 1)
    out["interventions_per_success"] = out["total_interventions"] / max(out["success_cnt"], 1)
    out["effective_intervention_rate"] = out["effective_interventions"] / max(out["total_interventions"], 1)
    return out


def merge_numeric_stats(stats, local):
    for key, value in local.items():
        if isinstance(value, bool):
            stats[key] = stats.get(key, 0) + int(value)
        elif isinstance(value, (int, float)):
            stats[key] = stats.get(key, 0) + value
        elif key not in stats:
            stats[key] = value


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def entropy_from_logits(logits):
    probs = torch.softmax(logits, dim=-1)
    return float((-torch.sum(probs * torch.log(probs + 1e-10), dim=-1)).item()), probs


def running_threshold(history, lam):
    mean = sum(history) / len(history)
    var = sum((x - mean) ** 2 for x in history) / len(history)
    return mean + lam * math.sqrt(var)


@torch.inference_mode()
def baseline_entropy_trace(helper, model, tokenizer, image_path, question, args):
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    pixel_values = helper.load_image(str(image_path), max_num=args.max_num_tiles).to(torch.bfloat16).cuda()
    num_patches = pixel_values.shape[0]
    image_flags = torch.ones((num_patches, 1), dtype=torch.long, device=pixel_values.device)

    mm_text_inputs = helper.build_text_inputs_for_multimodal_chat(model, tokenizer, question, num_patches)
    next_input_ids = mm_text_inputs["input_ids"].cuda()
    next_attention_mask = mm_text_inputs["attention_mask"].cuda()
    past_key_values = None
    entropies = []

    for step in range(args.max_new_tokens):
        if step == 0:
            outputs = model(
                pixel_values=pixel_values,
                input_ids=next_input_ids,
                attention_mask=next_attention_mask,
                image_flags=image_flags,
                use_cache=True,
                return_dict=True,
            )
        else:
            outputs = model.language_model(
                input_ids=next_input_ids,
                attention_mask=next_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
        entropy, probs = entropy_from_logits(logits)
        token_id = int(torch.argmax(probs, dim=-1).item())
        entropies.append({"step": step, "entropy": entropy, "token_id": token_id})
        if token_id == eos_token_id:
            break
        next_input_ids = torch.tensor([[token_id]], device=next_input_ids.device, dtype=next_input_ids.dtype)
        next_attention_mask = torch.cat(
            [
                next_attention_mask,
                torch.ones((1, 1), device=next_attention_mask.device, dtype=next_attention_mask.dtype),
            ],
            dim=1,
        )
    return entropies


def oracle_steps_from_trace(entropies, args):
    active_end = max(1, min(args.max_new_tokens, args.effective_max_step))
    eligible = [item for item in entropies if item["step"] < active_end]
    eligible.sort(key=lambda item: item["entropy"], reverse=True)
    return set(sorted(item["step"] for item in eligible[: args.k]))


@torch.inference_mode()
def run_sgr_with_entropy_trigger(helper, model, tokenizer, image_path, question, args):
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    oracle_steps = set()
    oracle_trace = []
    if args.trigger_mode == "oracle":
        oracle_trace = baseline_entropy_trace(helper, model, tokenizer, image_path, question, args)
        oracle_steps = oracle_steps_from_trace(oracle_trace, args)

    pixel_values = helper.load_image(str(image_path), max_num=args.max_num_tiles).to(torch.bfloat16).cuda()
    pixel_black = torch.zeros_like(pixel_values)
    num_patches = pixel_values.shape[0]
    image_flags = torch.ones((num_patches, 1), dtype=torch.long, device=pixel_values.device)

    mm_text_inputs = helper.build_text_inputs_for_multimodal_chat(model, tokenizer, question, num_patches)
    current_mm_input_ids = mm_text_inputs["input_ids"].cuda()
    current_mm_attention_mask = mm_text_inputs["attention_mask"].cuda()

    current_black_input_ids = current_mm_input_ids.clone()
    current_black_attention_mask = current_mm_attention_mask.clone()

    text_inputs = helper._build_text_only_inputs(model, tokenizer, question)
    current_text_input_ids = text_inputs["input_ids"].cuda()
    current_text_attention_mask = text_inputs["attention_mask"].cuda()

    black_outputs = model(
        pixel_values=pixel_black,
        input_ids=current_black_input_ids,
        attention_mask=current_black_attention_mask,
        image_flags=image_flags,
        use_cache=True,
        return_dict=True,
    )
    black_past_key_values = black_outputs.past_key_values
    black_logits = black_outputs.logits[:, -1, :].float()
    black_pending_ids = []

    next_mm_input_ids = current_mm_input_ids
    next_mm_attention_mask = current_mm_attention_mask
    vis_past_key_values = None
    generated_tokens = []
    entropy_history = []
    trigger_steps = []
    diag = {
        "planned_interventions": args.k,
        "trigger_hits": 0,
        "total_interventions": 0,
        "effective_interventions": 0,
        "guardrail_blocks": 0,
        "guardrail_accept_cnt": 0,
        "guardrail_fallback_cnt": 0,
        "online_threshold_checks": 0,
        "online_threshold_passes": 0,
        "oracle_trace_tokens": len(oracle_trace),
    }
    active_end = max(1, min(args.max_new_tokens, args.effective_max_step))

    for step in range(args.max_new_tokens):
        if step == 0:
            vis_outputs = model(
                pixel_values=pixel_values,
                input_ids=next_mm_input_ids,
                attention_mask=next_mm_attention_mask,
                image_flags=image_flags,
                use_cache=True,
                return_dict=True,
            )
        else:
            vis_outputs = model.language_model(
                input_ids=next_mm_input_ids,
                attention_mask=next_mm_attention_mask,
                past_key_values=vis_past_key_values,
                use_cache=True,
                return_dict=True,
            )

        vis_past_key_values = vis_outputs.past_key_values
        vis_logits = vis_outputs.logits[:, -1, :].float()
        entropy, probs_m = entropy_from_logits(vis_logits)
        visual_top1_id = int(torch.argmax(probs_m, dim=-1).item())
        next_token_id = visual_top1_id

        should_intervene = False
        if step < active_end and diag["trigger_hits"] < args.k:
            if args.trigger_mode == "oracle":
                should_intervene = step in oracle_steps
            else:
                if step >= args.entropy_warmup and len(entropy_history) >= args.min_history:
                    diag["online_threshold_checks"] += 1
                    threshold = running_threshold(entropy_history, args.entropy_lambda)
                    should_intervene = entropy > threshold
                    if should_intervene:
                        diag["online_threshold_passes"] += 1

        if should_intervene:
            diag["trigger_hits"] += 1
            trigger_steps.append({"step": step, "entropy": entropy})

            if step > 0 and black_pending_ids:
                black_input_ids = torch.tensor(
                    [black_pending_ids],
                    device=current_mm_input_ids.device,
                    dtype=current_mm_input_ids.dtype,
                )
                black_outputs = model.language_model(
                    input_ids=black_input_ids,
                    attention_mask=current_black_attention_mask,
                    past_key_values=black_past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                black_past_key_values = black_outputs.past_key_values
                black_logits = black_outputs.logits[:, -1, :].float()
                black_pending_ids = []

            text_outputs = model.language_model(
                input_ids=current_text_input_ids,
                attention_mask=current_text_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            text_logits = text_outputs.logits[:, -1, :].float()

            diag["total_interventions"] += 1
            rerank_info = mm.rerank_threebranch(
                vis_logits=vis_logits,
                black_logits=black_logits,
                text_logits=text_logits,
                probs_m=probs_m,
                visual_top1_id=visual_top1_id,
                args=args,
            )
            next_token_id = rerank_info["token_id"]
            diag["guardrail_blocks"] += int(rerank_info["guardrail_blocks"])
            if rerank_info["fallback"]:
                diag["guardrail_fallback_cnt"] += 1
            else:
                diag["guardrail_accept_cnt"] += 1
            if next_token_id != visual_top1_id:
                diag["effective_interventions"] += 1

        entropy_history.append(entropy)

        if next_token_id == eos_token_id:
            break

        generated_tokens.append(next_token_id)
        black_pending_ids.append(next_token_id)
        next_token_tensor = torch.tensor(
            [[next_token_id]],
            device=current_mm_input_ids.device,
            dtype=current_mm_input_ids.dtype,
        )
        next_mm_input_ids = next_token_tensor
        next_mm_attention_mask = torch.cat(
            [
                next_mm_attention_mask,
                torch.ones((1, 1), device=next_mm_attention_mask.device, dtype=next_mm_attention_mask.dtype),
            ],
            dim=1,
        )
        current_black_attention_mask = torch.cat(
            [
                current_black_attention_mask,
                torch.ones((1, 1), device=current_black_attention_mask.device, dtype=current_black_attention_mask.dtype),
            ],
            dim=1,
        )
        current_text_input_ids = torch.cat([current_text_input_ids, next_token_tensor], dim=1)
        current_text_attention_mask = torch.cat(
            [
                current_text_attention_mask,
                torch.ones((1, 1), device=current_text_attention_mask.device, dtype=current_text_attention_mask.dtype),
            ],
            dim=1,
        )

    diag["trigger_steps"] = trigger_steps
    caption = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {
        "caption": caption,
        "generated_len": len(generated_tokens),
        "diag_stats": diag,
    }


def run_chair_eval(args, caption_path, chair_path):
    cmd = [
        sys.executable,
        str(ROOT / "eval_pai_chair.py"),
        "--cap_file",
        str(caption_path),
        "--image_id_key",
        "image_id",
        "--caption_key",
        "caption",
        "--cache",
        str(args.chair_cache),
        "--coco_path",
        str(args.coco_anno_path),
        "--save_path",
        str(chair_path),
    ]
    print(f"\n[CHAIR] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def dry_run(args):
    image_ids = load_target_image_ids(args.jsonl_path, args.max_samples)
    missing = []
    for img_id in image_ids:
        path = args.image_folder / f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        if not path.exists():
            missing.append(str(path))
    print(
        json.dumps(
            {
                "records": len(image_ids),
                "missing_images": len(missing),
                "first_missing": missing[:5],
                "trigger_mode": args.trigger_mode,
                "seed": args.seed,
                "k": args.k,
                "output_paths": {k: str(v) for k, v in build_output_paths(args).items()},
            },
            indent=4,
            ensure_ascii=False,
        )
    )


def main(default_mode="online"):
    args = parse_args(default_mode)
    if args.dry_run:
        dry_run(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL entropy-trigger runs.")

    paths = build_output_paths(args)
    helper = mm.load_internvl_helper()
    mm.configure_helper(helper, args)
    mm.set_seed(args.seed)
    model, tokenizer = mm.load_model(args)

    image_ids = load_target_image_ids(args.jsonl_path, args.max_samples)
    stats = init_stats(args)
    results = []

    print(f"[InternVL entropy trigger] mode={args.trigger_mode} seed={args.seed} k={args.k}")
    print(f"output: {paths['output']}")
    print(f"stats: {paths['stats']}")
    print(f"chair: {paths['chair']}")

    for idx, img_id in enumerate(tqdm(image_ids, desc=f"InternVL entropy {args.trigger_mode}"), start=1):
        image_path = args.image_folder / f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        if not image_path.exists():
            stats["missing_image_cnt"] += 1
            continue
        try:
            mm.set_seed(args.seed + idx)
            out = run_sgr_with_entropy_trigger(
                helper=helper,
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                question=args.question,
                args=args,
            )
            results.append({"image_id": int(img_id), "caption": out["caption"]})
            stats["success_cnt"] += 1
            stats["avg_generated_len_approx_sum"] += float(out["generated_len"])
            merge_numeric_stats(stats, out["diag_stats"])
        except torch.cuda.OutOfMemoryError as exc:
            stats["error_cnt"] += 1
            print(f"\n[OOM] image_id={img_id}: {repr(exc)}")
            torch.cuda.empty_cache()
        except Exception as exc:
            stats["error_cnt"] += 1
            print(f"\n[Error] image_id={img_id}: {repr(exc)}")
            traceback.print_exc()

        if idx % args.save_every == 0:
            save_json(paths["output"], {"annotations": results})
            save_json(paths["stats"], add_derived_stats(stats))

    save_json(paths["output"], {"annotations": results})
    save_json(paths["stats"], add_derived_stats(stats))
    print(json.dumps(add_derived_stats(stats), indent=4, ensure_ascii=False))
    if not args.no_chair:
        run_chair_eval(args, paths["output"], paths["chair"])


if __name__ == "__main__":
    main()
