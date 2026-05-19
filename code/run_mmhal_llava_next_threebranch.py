# -*- coding: utf-8 -*-
import argparse
import json
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
    LogitsProcessorList,
)

from llava16_native_threebranch_3seeds import (
    ThreeBranchProcessor,
    build_black_image,
    build_intervention_steps,
    build_mm_inputs,
    build_text_inputs,
    get_input_device,
    resolve_eos_ids,
    resolve_torch_dtype,
    set_seed,
    trim_generated_ids,
)


ROOT = Path(".")
MODEL_PATH = "<PATH_TO_LLAVA_NEXT>"
DATA_JSON_PATH = ROOT / "MMHal-Bench" / "response_template.json"
IMAGE_DIR = ROOT / "MMHal-Bench" / "images"
OUTPUT_JSON_PATH = ROOT / "mmhal_llava_next_threebranch_k10_soft_seed20260423_preds.json"
STATS_JSON_PATH = ROOT / "mmhal_llava_next_threebranch_k10_soft_seed20260423_stats.json"
QUESTION_FALLBACK = "Please describe this image in detail."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA-NeXT three-branch sparse decoding on MMHal-Bench."
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-json", type=Path, default=DATA_JSON_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON_PATH)
    parser.add_argument("--stats-output", type=Path, default=STATS_JSON_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--answer-key", type=str, default="model_answer")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/images without loading the model.")

    # Defaults match the strongest LLaVA COCO setting: threebranch soft k10 seed20260423.
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--schedule", type=str, default="global", choices=["global", "early_fixed"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--effective-max-step",
        type=int,
        default=500,
        help="Keep the intervention sampling range aligned with the COCO k10 run.",
    )
    parser.add_argument("--early-start", type=int, default=8)
    parser.add_argument("--early-end", type=int, default=56)
    parser.add_argument("--early-interval", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--black-gate-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--black-prior-threshold", type=float, default=0.50)
    parser.add_argument("--black-visual-gap-threshold", type=float, default=0.30)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    return parser.parse_args()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_dataset(data_json, max_samples):
    with open(data_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if max_samples is not None:
        dataset = dataset[: max_samples]
    return dataset


def resolve_image_path(item, image_dir):
    image_src = item.get("image_src", "")
    if not image_src:
        return None
    filename = image_src.split("/")[-1]
    return image_dir / filename


def init_stats(args):
    return {
        "success_cnt": 0,
        "missing_image_cnt": 0,
        "error_cnt": 0,
        "truncated_cnt": 0,
        "avg_generated_len_sum": 0.0,
        "planned_interventions": 0,
        "trigger_hits": 0,
        "black_same_top1_cnt": 0,
        "pass_black_prior_cnt": 0,
        "pass_black_gap_cnt": 0,
        "pass_black_both_cnt": 0,
        "black_gate_rejects": 0,
        "guardrail_blocks": 0,
        "guardrail_accept_cnt": 0,
        "guardrail_fallback_cnt": 0,
        "total_interventions": 0,
        "effective_interventions": 0,
        "model_path": str(args.model_path),
        "data_json": str(args.data_json),
        "image_dir": str(args.image_dir),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "seed_protocol": "set_seed(seed + 1-based sample index), same as llava16_native_threebranch_3seeds.py",
        "schedule": args.schedule,
        "k": args.k,
        "effective_max_step": args.effective_max_step,
        "black_gate_mode": args.black_gate_mode,
        "top_k": args.top_k,
        "black_alpha": args.black_alpha,
        "apc_threshold": args.apc_threshold,
        "syntax_threshold": args.syntax_threshold,
        "syntax_margin": args.syntax_margin,
    }


def add_derived_stats(stats):
    out = dict(stats)
    out["avg_generated_len"] = out["avg_generated_len_sum"] / max(out["success_cnt"], 1)
    out["effective_intervention_rate"] = out["effective_interventions"] / max(out["total_interventions"], 1)
    out["trigger_rate_per_success"] = out["trigger_hits"] / max(out["success_cnt"], 1)
    return out


def merge_diag_stats(stats, diag_stats):
    for key, value in diag_stats.items():
        stats[key] = stats.get(key, 0) + value


def save_checkpoint(output_path, stats_path, results, stats):
    save_json(output_path, results)
    save_json(stats_path, add_derived_stats(stats))


def dry_run(args):
    dataset = load_dataset(args.data_json, args.max_samples)
    missing = []
    for idx, item in enumerate(dataset, start=1):
        image_path = resolve_image_path(item, args.image_dir)
        if image_path is None or not image_path.exists():
            missing.append({"idx": idx, "image_path": str(image_path)})

    print(
        json.dumps(
            {
                "records": len(dataset),
                "missing_images": len(missing),
                "first_missing": missing[:5],
                "data_json": str(args.data_json),
                "image_dir": str(args.image_dir),
                "output": str(args.output),
                "stats_output": str(args.stats_output),
                "config": {
                    "seed": args.seed,
                    "schedule": args.schedule,
                    "k": args.k,
                    "black_gate_mode": args.black_gate_mode,
                    "max_new_tokens": args.max_new_tokens,
                    "effective_max_step": args.effective_max_step,
                },
            },
            indent=4,
            ensure_ascii=False,
        )
    )


@torch.inference_mode()
def generate_threebranch_answer(model, processor, tokenizer, image_pil, question, args, device):
    mm_inputs = build_mm_inputs(processor, tokenizer, image_pil, question, device)
    black_inputs = build_mm_inputs(processor, tokenizer, build_black_image(image_pil), question, device)
    text_inputs = build_text_inputs(processor, tokenizer, question, device)
    intervention_steps = build_intervention_steps(
        schedule=args.schedule,
        k=args.k,
        max_steps=args.max_new_tokens,
        effective_max_step=args.effective_max_step,
        early_start=args.early_start,
        early_end=args.early_end,
        early_interval=args.early_interval,
    )

    three_processor = ThreeBranchProcessor(
        model=model,
        base_text_input_ids=text_inputs["input_ids"],
        base_text_attention_mask=text_inputs["attention_mask"],
        base_black_inputs=black_inputs,
        intervention_steps=intervention_steps,
        top_k=args.top_k,
        black_alpha=args.black_alpha,
        apc_threshold=args.apc_threshold,
        black_gate_mode=args.black_gate_mode,
        black_prior_threshold=args.black_prior_threshold,
        black_visual_gap_threshold=args.black_visual_gap_threshold,
        syntax_threshold=args.syntax_threshold,
        syntax_margin=args.syntax_margin,
    )

    output = model.generate(
        **mm_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        logits_processor=LogitsProcessorList([three_processor]),
        return_dict_in_generate=True,
    )
    eos_ids = resolve_eos_ids(tokenizer)
    raw_ids = output.sequences[0, mm_inputs["input_ids"].shape[1]:].tolist()
    trimmed_ids, saw_eos = trim_generated_ids(raw_ids, eos_ids)
    answer = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "generated_len": len(trimmed_ids),
        "truncated": not saw_eos and len(raw_ids) >= args.max_new_tokens,
        "diag_stats": {
            "planned_interventions": three_processor.planned_interventions,
            "trigger_hits": three_processor.trigger_hits,
            "black_same_top1_cnt": three_processor.black_same_top1_cnt,
            "pass_black_prior_cnt": three_processor.pass_black_prior_cnt,
            "pass_black_gap_cnt": three_processor.pass_black_gap_cnt,
            "pass_black_both_cnt": three_processor.pass_black_both_cnt,
            "black_gate_rejects": three_processor.black_gate_rejects,
            "guardrail_blocks": three_processor.guardrail_blocks,
            "guardrail_accept_cnt": three_processor.guardrail_accept_cnt,
            "guardrail_fallback_cnt": three_processor.guardrail_fallback_cnt,
            "total_interventions": three_processor.total_interventions,
            "effective_interventions": three_processor.effective_interventions,
        },
    }


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return

    print(f"Loading LLaVA-NeXT from {args.model_path}")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("data:", args.data_json)
    print("image dir:", args.image_dir)
    print("output:", args.output)
    print("stats output:", args.stats_output)
    print("config:", {
        "seed": args.seed,
        "schedule": args.schedule,
        "k": args.k,
        "black_gate_mode": args.black_gate_mode,
        "max_new_tokens": args.max_new_tokens,
        "effective_max_step": args.effective_max_step,
    })

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=resolve_torch_dtype(),
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
    input_device = get_input_device(model)
    print("input device:", input_device)

    dataset = load_dataset(args.data_json, args.max_samples)
    results = []
    stats = init_stats(args)

    for idx, item in enumerate(tqdm(dataset, desc="MMHal LLaVA-NeXT threebranch"), start=1):
        out_item = dict(item)
        image_path = resolve_image_path(item, args.image_dir)
        if image_path is None or not image_path.exists():
            stats["missing_image_cnt"] += 1
            out_item[args.answer_key] = ""
            out_item["llava_next_threebranch_answer"] = ""
            out_item["error"] = f"missing image: {image_path}"
            results.append(out_item)
            continue

        question = item.get("question", QUESTION_FALLBACK)
        try:
            set_seed(args.seed + idx)
            image_pil = Image.open(image_path).convert("RGB")
            output = generate_threebranch_answer(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_pil=image_pil,
                question=question,
                args=args,
                device=input_device,
            )
            answer = output["answer"]
            out_item[args.answer_key] = answer
            out_item["llava_next_threebranch_answer"] = answer
            out_item["llava_next_threebranch_generated_len"] = output["generated_len"]
            out_item["llava_next_threebranch_truncated"] = output["truncated"]
            out_item["llava_next_threebranch_diag"] = output["diag_stats"]
            stats["success_cnt"] += 1
            stats["truncated_cnt"] += int(output["truncated"])
            stats["avg_generated_len_sum"] += float(output["generated_len"])
            merge_diag_stats(stats, output["diag_stats"])
        except Exception as exc:
            stats["error_cnt"] += 1
            out_item[args.answer_key] = ""
            out_item["llava_next_threebranch_answer"] = ""
            out_item["error"] = repr(exc)
            traceback.print_exc()

        results.append(out_item)
        if idx % args.save_every == 0:
            save_checkpoint(args.output, args.stats_output, results, stats)

    save_checkpoint(args.output, args.stats_output, results, stats)
    final_stats = add_derived_stats(stats)

    print("\nDone.")
    print(json.dumps(final_stats, indent=4, ensure_ascii=False))
    print(f"saved responses -> {args.output}")
    print(f"saved stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
