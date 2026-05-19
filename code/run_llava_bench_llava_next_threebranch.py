# -*- coding: utf-8 -*-
import argparse
import json
import random
import traceback
from pathlib import Path

import numpy as np
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

import llava16_native_threebranch_3seeds as base


ROOT = Path(".")
BENCH_DIR = ROOT / "llava-bench-in-the-wild"
MODEL_PATH = base.MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA-NeXT baseline / sparse three-branch on LLaVA-Bench in-the-wild."
    )
    parser.add_argument("--mode", choices=["baseline", "three", "both"], default="both")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--bench-dir", type=str, default=str(BENCH_DIR))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--schedule", type=str, default="global", choices=["global", "early_fixed"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--effective-max-step", type=int, default=500)
    parser.add_argument("--early-start", type=int, default=8)
    parser.add_argument("--early-end", type=int, default=56)
    parser.add_argument("--early-interval", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--black-gate-mode", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--black-prior-threshold", type=float, default=0.50)
    parser.add_argument("--black-visual-gap-threshold", type=float, default=0.30)
    parser.add_argument("--negative-image", type=str, default="black", choices=["black", "blur"])
    parser.add_argument("--blur-radius", type=float, default=12.0)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tag", type=str, default="llavabench")
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


def load_questions(bench_dir: Path):
    questions_path = bench_dir / "questions.jsonl"
    with open(questions_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_row(question, text, model_id):
    return {
        "question_id": int(question["question_id"]),
        "prompt": question["text"],
        "answer_id": f"{model_id}_{question['question_id']}",
        "model_id": model_id,
        "metadata": {"category": question.get("category", "")},
        "text": text,
    }


@torch.inference_mode()
def run_baseline(model, processor, tokenizer, image_pil, question, args, input_device):
    mm_inputs = base.build_mm_inputs(processor, tokenizer, image_pil, question, input_device)
    output = model.generate(
        **mm_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
    )
    eos_ids = base.resolve_eos_ids(tokenizer)
    raw_ids = output.sequences[0, mm_inputs["input_ids"].shape[1]:].tolist()
    trimmed_ids, saw_eos = base.trim_generated_ids(raw_ids, eos_ids)
    return {
        "text": tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip(),
        "generated_len": len(trimmed_ids),
        "truncated": not saw_eos and len(raw_ids) >= args.max_new_tokens,
    }


@torch.inference_mode()
def run_threebranch(model, processor, tokenizer, image_pil, question, args, input_device):
    mm_inputs = base.build_mm_inputs(processor, tokenizer, image_pil, question, input_device)
    prior_inputs = base.build_mm_inputs(
        processor,
        tokenizer,
        base.build_negative_image(image_pil, args),
        question,
        input_device,
    )
    text_inputs = base.build_text_inputs(processor, tokenizer, question, input_device)
    intervention_steps = base.build_intervention_steps(
        schedule=args.schedule,
        k=args.k,
        max_steps=args.max_new_tokens,
        effective_max_step=args.effective_max_step,
        early_start=args.early_start,
        early_end=args.early_end,
        early_interval=args.early_interval,
    )

    three_processor = base.ThreeBranchProcessor(
        model=model,
        base_text_input_ids=text_inputs["input_ids"],
        base_text_attention_mask=text_inputs["attention_mask"],
        base_black_inputs=prior_inputs,
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
    eos_ids = base.resolve_eos_ids(tokenizer)
    raw_ids = output.sequences[0, mm_inputs["input_ids"].shape[1]:].tolist()
    trimmed_ids, saw_eos = base.trim_generated_ids(raw_ids, eos_ids)
    return {
        "text": tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip(),
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


def init_stats(include_sparse=False):
    stats = {
        "missing_image_cnt": 0,
        "error_cnt": 0,
        "truncated_cnt": 0,
        "success_cnt": 0,
        "avg_generated_len_sum": 0.0,
    }
    if include_sparse:
        stats.update(
            {
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
            }
        )
    return stats


def merge_stats(stats, local):
    for key in stats:
        stats[key] += local.get(key, 0)


def build_paths(args):
    tag = f"_{args.tag}" if args.tag else ""
    paths = {}
    if args.mode in ("baseline", "both"):
        stem = f"llava_bench_llava16_baseline_t{args.max_new_tokens}{tag}"
        paths["baseline"] = ROOT / f"{stem}.jsonl"
        paths["baseline_stats"] = ROOT / f"{stem}_stats.json"
    if args.mode in ("three", "both"):
        neg_suffix = "" if args.negative_image == "black" else f"_{args.negative_image}neg_r{str(args.blur_radius).replace('.', 'p')}"
        stem = f"llava_bench_llava16_threebranch_k{args.k}_{args.schedule}_seed{args.seed}_t{args.max_new_tokens}{neg_suffix}{tag}"
        paths["three"] = ROOT / f"{stem}.jsonl"
        paths["three_stats"] = ROOT / f"{stem}_stats.json"
    return paths


def main():
    args = parse_args()
    set_seed(args.seed)

    bench_dir = Path(args.bench_dir)
    image_dir = bench_dir / "images"
    questions = load_questions(bench_dir)
    paths = build_paths(args)

    print(f"Loading LLaVA-1.6 from {args.model_path} ...")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("mode:", args.mode)
    print("questions:", len(questions))
    print("max_new_tokens:", args.max_new_tokens)
    print("seed:", args.seed)
    print("schedule:", args.schedule)
    print("k:", args.k)

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

    baseline_rows = []
    three_rows = []
    baseline_stats = init_stats(False)
    three_stats = init_stats(True)

    for idx, question in enumerate(tqdm(questions, desc="LLaVA-Bench LLaVA-NeXT"), start=1):
        image_path = image_dir / question["image"]
        if not image_path.exists():
            if args.mode in ("baseline", "both"):
                baseline_stats["missing_image_cnt"] += 1
            if args.mode in ("three", "both"):
                three_stats["missing_image_cnt"] += 1
            print(f"[Skip] missing image: {image_path}")
            continue

        try:
            image_pil = Image.open(image_path).convert("RGB")

            if args.mode in ("baseline", "both"):
                out = run_baseline(model, processor, tokenizer, image_pil, question["text"], args, input_device)
                baseline_rows.append(answer_row(question, out["text"], "llava16_baseline"))
                baseline_stats["success_cnt"] += 1
                baseline_stats["truncated_cnt"] += int(out["truncated"])
                baseline_stats["avg_generated_len_sum"] += float(out["generated_len"])

            if args.mode in ("three", "both"):
                set_seed(args.seed + idx)
                out = run_threebranch(model, processor, tokenizer, image_pil, question["text"], args, input_device)
                three_rows.append(answer_row(question, out["text"], f"llava16_threebranch_k{args.k}_{args.schedule}"))
                three_stats["success_cnt"] += 1
                three_stats["truncated_cnt"] += int(out["truncated"])
                three_stats["avg_generated_len_sum"] += float(out["generated_len"])
                merge_stats(three_stats, out["diag_stats"])

            if idx % args.save_every == 0:
                if args.mode in ("baseline", "both"):
                    save_jsonl(paths["baseline"], baseline_rows)
                    save_json(paths["baseline_stats"], baseline_stats)
                if args.mode in ("three", "both"):
                    save_jsonl(paths["three"], three_rows)
                    save_json(paths["three_stats"], three_stats)

        except torch.cuda.OutOfMemoryError as exc:
            print(f"\n[OOM] question_id={question['question_id']}: {repr(exc)}")
            if args.mode in ("baseline", "both"):
                baseline_stats["error_cnt"] += 1
            if args.mode in ("three", "both"):
                three_stats["error_cnt"] += 1
            torch.cuda.empty_cache()
        except Exception:
            print(f"\n[Error] question_id={question['question_id']}")
            if args.mode in ("baseline", "both"):
                baseline_stats["error_cnt"] += 1
            if args.mode in ("three", "both"):
                three_stats["error_cnt"] += 1
            traceback.print_exc()
            break

    if args.mode in ("baseline", "both"):
        save_jsonl(paths["baseline"], baseline_rows)
        save_json(paths["baseline_stats"], baseline_stats)
    if args.mode in ("three", "both"):
        save_jsonl(paths["three"], three_rows)
        save_json(paths["three_stats"], three_stats)

    print("\nDone.")
    if args.mode in ("baseline", "both"):
        avg_len = baseline_stats["avg_generated_len_sum"] / max(baseline_stats["success_cnt"], 1)
        print("Baseline stats:")
        print(json.dumps(baseline_stats, indent=2, ensure_ascii=False))
        print(f"avg generated len: {avg_len:.2f}")
        print(f"saved answers -> {paths['baseline']}")
    if args.mode in ("three", "both"):
        avg_len = three_stats["avg_generated_len_sum"] / max(three_stats["success_cnt"], 1)
        print("Threebranch stats:")
        print(json.dumps(three_stats, indent=2, ensure_ascii=False))
        print(f"avg generated len: {avg_len:.2f}")
        print(f"saved answers -> {paths['three']}")


if __name__ == "__main__":
    main()
