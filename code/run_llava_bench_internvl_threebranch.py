# -*- coding: utf-8 -*-
import argparse
import json
import random
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

import run_mmhal_internvl as mm


ROOT = Path(".")
BENCH_DIR = ROOT / "llava-bench-in-the-wild"
MODEL_PATH = mm.MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InternVL baseline / sparse three-branch on LLaVA-Bench in-the-wild."
    )
    parser.add_argument("--mode", choices=["baseline", "threebranch", "both"], default="both")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--bench-dir", type=str, default=str(BENCH_DIR))
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--max-num-tiles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tag", type=str, default="llavabench")
    parser.add_argument("--disable-flash-attn", action="store_true")

    parser.add_argument("--schedule", type=str, default="global", choices=["global", "early", "mid", "late", "stratified"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--effective-max-step", type=int, default=300)
    parser.add_argument("--early-range", type=int, nargs=2, default=[0, 88])
    parser.add_argument("--mid-range", type=int, nargs=2, default=[48, 144])
    parser.add_argument("--late-range", type=int, nargs=2, default=[88, 176])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--black-prior-threshold", type=float, default=0.50)
    parser.add_argument("--black-visual-gap-threshold", type=float, default=0.30)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    parser.add_argument("--guard-mode", type=str, default="syntax", choices=["syntax", "none"])
    parser.add_argument("--contrast-branch", type=str, default="black", choices=["black", "text"])
    parser.add_argument("--text-alpha", type=float, default=0.25)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_questions(bench_dir):
    with open(Path(bench_dir) / "questions.jsonl", "r", encoding="utf-8") as f:
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


def init_stats(include_sparse=False):
    stats = {
        "success_cnt": 0,
        "missing_image_cnt": 0,
        "error_cnt": 0,
        "avg_generated_len_approx_sum": 0.0,
    }
    if include_sparse:
        stats.update(mm.init_threebranch_stats(SimpleNamespace(
            model_path="",
            data_json="",
            image_dir="",
            max_new_tokens=0,
            max_num_tiles=0,
            seed=0,
            schedule="global",
            k=0,
            effective_max_step=0,
            top_k=0,
            black_alpha=0.0,
            apc_threshold=0.0,
            black_prior_threshold=0.0,
            black_visual_gap_threshold=0.0,
            syntax_threshold=0.0,
            syntax_margin=0.0,
            guard_mode="syntax",
            contrast_branch="black",
            text_alpha=0.0,
        )))
        for key in ("method", "model_path", "data_json", "image_dir", "max_new_tokens", "max_num_tiles", "seed", "schedule", "note"):
            stats.pop(key, None)
    return stats


def merge_numeric(stats, local):
    for key, value in local.items():
        if isinstance(value, (int, float)):
            stats[key] = stats.get(key, 0) + value


def build_paths(args):
    tag = f"_{args.tag}" if args.tag else ""
    paths = {}
    if args.mode in ("baseline", "both"):
        stem = f"llava_bench_internvl_baseline_t{args.max_new_tokens}{tag}"
        paths["baseline"] = ROOT / f"{stem}.jsonl"
        paths["baseline_stats"] = ROOT / f"{stem}_stats.json"
    if args.mode in ("threebranch", "both"):
        stem = f"llava_bench_internvl_threebranch_k{args.k}_{args.schedule}_seed{args.seed}_t{args.max_new_tokens}{tag}"
        paths["three"] = ROOT / f"{stem}.jsonl"
        paths["three_stats"] = ROOT / f"{stem}_stats.json"
    return paths


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("InternVL LLaVA-Bench inference requires CUDA.")

    set_seed(args.seed)
    bench_dir = Path(args.bench_dir)
    image_dir = bench_dir / "images"
    questions = load_questions(bench_dir)
    paths = build_paths(args)

    helper = mm.load_internvl_helper()
    mm.configure_helper(helper, args)

    print(f"Loading InternVL from {args.model_path}")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("mode:", args.mode)
    print("questions:", len(questions))
    print("max_new_tokens:", args.max_new_tokens)
    print("schedule:", args.schedule)
    print("k:", args.k)
    model, tokenizer = mm.load_model(args)
    print("model loaded")

    baseline_rows = []
    three_rows = []
    baseline_stats = init_stats(False)
    three_stats = init_stats(True)

    for idx, question in enumerate(tqdm(questions, desc="LLaVA-Bench InternVL"), start=1):
        image_path = image_dir / question["image"]
        if not image_path.exists():
            if args.mode in ("baseline", "both"):
                baseline_stats["missing_image_cnt"] += 1
            if args.mode in ("threebranch", "both"):
                three_stats["missing_image_cnt"] += 1
            print(f"[Skip] missing image: {image_path}")
            continue

        try:
            if args.mode in ("baseline", "both"):
                answer = mm.run_baseline_answer(helper, model, tokenizer, image_path, question["text"], args)
                baseline_rows.append(answer_row(question, answer, "internvl_baseline"))
                baseline_stats["success_cnt"] += 1
                baseline_stats["avg_generated_len_approx_sum"] += float(mm.token_len_approx(tokenizer, answer))

            if args.mode in ("threebranch", "both"):
                set_seed(args.seed + idx)
                out = mm.run_random_sparse_threebranch(helper, model, tokenizer, image_path, question["text"], args)
                three_rows.append(answer_row(question, out["caption"], f"internvl_threebranch_k{args.k}_{args.schedule}"))
                three_stats["success_cnt"] += 1
                three_stats["avg_generated_len_approx_sum"] += float(out["generated_len"])
                merge_numeric(three_stats, out["diag_stats"])

            if idx % args.save_every == 0:
                if args.mode in ("baseline", "both"):
                    save_jsonl(paths["baseline"], baseline_rows)
                    save_json(paths["baseline_stats"], baseline_stats)
                if args.mode in ("threebranch", "both"):
                    save_jsonl(paths["three"], three_rows)
                    save_json(paths["three_stats"], three_stats)

        except torch.cuda.OutOfMemoryError as exc:
            print(f"\n[OOM] question_id={question['question_id']}: {repr(exc)}")
            if args.mode in ("baseline", "both"):
                baseline_stats["error_cnt"] += 1
            if args.mode in ("threebranch", "both"):
                three_stats["error_cnt"] += 1
            torch.cuda.empty_cache()
        except Exception:
            print(f"\n[Error] question_id={question['question_id']}")
            if args.mode in ("baseline", "both"):
                baseline_stats["error_cnt"] += 1
            if args.mode in ("threebranch", "both"):
                three_stats["error_cnt"] += 1
            traceback.print_exc()
            break

    if args.mode in ("baseline", "both"):
        baseline_stats["avg_generated_len_approx"] = baseline_stats["avg_generated_len_approx_sum"] / max(baseline_stats["success_cnt"], 1)
        save_jsonl(paths["baseline"], baseline_rows)
        save_json(paths["baseline_stats"], baseline_stats)
    if args.mode in ("threebranch", "both"):
        three_stats["avg_generated_len_approx"] = three_stats["avg_generated_len_approx_sum"] / max(three_stats["success_cnt"], 1)
        save_jsonl(paths["three"], three_rows)
        save_json(paths["three_stats"], three_stats)

    print("\nDone.")
    if args.mode in ("baseline", "both"):
        print("Baseline stats:")
        print(json.dumps(baseline_stats, indent=2, ensure_ascii=False))
        print(f"saved answers -> {paths['baseline']}")
    if args.mode in ("threebranch", "both"):
        print("Threebranch stats:")
        print(json.dumps(three_stats, indent=2, ensure_ascii=False))
        print(f"saved answers -> {paths['three']}")


if __name__ == "__main__":
    main()
