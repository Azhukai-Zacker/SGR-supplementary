# -*- coding: utf-8 -*-
import argparse
import json
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InternVL random-sparse aggressiveness ablations on COCO for CHAIR."
    )
    parser.add_argument("--model-path", type=str, default=mm.MODEL_PATH)
    parser.add_argument("--jsonl-path", type=Path, default=JSONL_PATH)
    parser.add_argument("--image-folder", type=Path, default=IMAGE_FOLDER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--question", type=str, default=QUESTION)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full"],
        choices=["full", "loose_guard", "black_only", "text_contrast_only"],
    )
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-num-tiles", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260422, 20260423, 20260424])
    parser.add_argument("--disable-flash-attn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-chair", action="store_true", help="Only generate captions/stats; skip CHAIR evaluation.")
    parser.add_argument("--chair-cache", type=Path, default=ROOT / "chair.pkl")
    parser.add_argument("--coco-anno-path", type=Path, default=Path("<PATH_TO_COCO>/annotations"))

    parser.add_argument("--schedule", type=str, default="global", choices=["global", "early", "mid", "late", "stratified"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--effective-max-step", type=int, default=220)
    parser.add_argument("--early-range", type=int, nargs=2, default=[0, 88], metavar=("START", "END"))
    parser.add_argument("--mid-range", type=int, nargs=2, default=[48, 144], metavar=("START", "END"))
    parser.add_argument("--late-range", type=int, nargs=2, default=[88, 176], metavar=("START", "END"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--text-alpha", type=float, default=0.20)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--black-prior-threshold", type=float, default=0.50)
    parser.add_argument("--black-visual-gap-threshold", type=float, default=0.30)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    parser.add_argument("--trigger-mode", type=str, default="random", choices=["random", "object"])
    parser.add_argument("--object-top-k", type=int, default=10)
    parser.add_argument("--max-object-triggers", type=int, default=16)
    parser.add_argument("--object-trigger-cooldown", type=int, default=2)
    parser.add_argument("--object-require-prior-risk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def load_target_image_ids(jsonl_path, max_samples):
    target_image_ids = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            image_id = data.get("image_id", data.get("id"))
            if image_id is not None:
                target_image_ids.append(int(image_id))
    return target_image_ids[:max_samples]


def build_variant_args(args, variant):
    variant_args = argparse.Namespace(**vars(args))
    variant_args.data_json = args.jsonl_path
    variant_args.image_dir = args.image_folder
    variant_args.answer_key = "caption"
    variant_args.guard_mode = "syntax"
    variant_args.contrast_branch = "black"

    if variant == "full":
        pass
    elif variant == "loose_guard":
        variant_args.syntax_threshold = 0.0
        variant_args.syntax_margin = 0.0
    elif variant == "black_only":
        variant_args.guard_mode = "none"
        variant_args.syntax_threshold = 0.0
        variant_args.syntax_margin = 0.0
    elif variant == "text_contrast_only":
        variant_args.guard_mode = "none"
        variant_args.contrast_branch = "text"
        variant_args.syntax_threshold = 0.0
        variant_args.syntax_margin = 0.0

    return variant_args


def build_output_paths(args, variant, seed):
    if args.trigger_mode == "object":
        stem = (
            f"internvl_threebranch_{variant}_object_top{args.object_top_k}"
            f"_max{args.max_object_triggers}_cd{args.object_trigger_cooldown}_seed{seed}"
        )
        if args.object_require_prior_risk:
            stem = stem.replace("_seed", "_risk_seed")
    else:
        stem = f"internvl_threebranch_{variant}_k{args.k}_{args.schedule}_seed{seed}"
    if args.tag:
        stem = f"{stem}_{args.tag}"
    return {
        "output": args.output_dir / f"{stem}.json",
        "stats": args.output_dir / f"{stem}_stats.json",
        "chair": args.output_dir / f"{stem}_chair.json",
    }


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
                "variants": args.variants,
                "k": args.k,
                "schedule": args.schedule,
                "seeds": args.seeds,
                "max_new_tokens": args.max_new_tokens,
                "auto_chair": not args.no_chair,
                "outputs": {
                    f"{variant}_seed{seed}": {
                        "captions": str(build_output_paths(args, variant, seed)["output"]),
                        "stats": str(build_output_paths(args, variant, seed)["stats"]),
                        "chair": str(build_output_paths(args, variant, seed)["chair"]),
                    }
                    for seed in args.seeds
                    for variant in args.variants
                },
            },
            indent=4,
            ensure_ascii=False,
        )
    )


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


def run_one_variant(helper, model, tokenizer, args, variant, seed, image_ids):
    variant_args = build_variant_args(args, variant)
    variant_args.seed = seed
    paths = build_output_paths(args, variant, seed)
    stats = mm.init_threebranch_stats(variant_args)
    stats["variant"] = variant
    stats["seed"] = seed
    stats["question"] = args.question
    stats["caption_output"] = str(paths["output"])
    stats["chair_output"] = str(paths["chair"])

    results = []
    print(f"\n[InternVL COCO ablation] variant={variant} seed={seed}")
    print(f"output: {paths['output']}")
    print(f"stats: {paths['stats']}")
    print(f"chair: {paths['chair']}")

    for idx, img_id in enumerate(tqdm(image_ids, desc=f"InternVL {variant} seed={seed}"), start=1):
        image_path = args.image_folder / f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        if not image_path.exists():
            stats["missing_image_cnt"] += 1
            continue

        try:
            mm.set_seed(seed + idx)
            out = mm.run_random_sparse_threebranch(
                helper=helper,
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                question=args.question,
                args=variant_args,
            )
            results.append({"image_id": int(img_id), "caption": out["caption"]})
            stats["success_cnt"] += 1
            stats["avg_generated_len_approx_sum"] += float(out["generated_len"])
            mm.merge_numeric_stats(stats, out["diag_stats"])
        except torch.cuda.OutOfMemoryError as exc:
            stats["error_cnt"] += 1
            print(f"\n[OOM][{variant}] image_id={img_id}: {repr(exc)}")
            torch.cuda.empty_cache()
        except Exception as exc:
            stats["error_cnt"] += 1
            print(f"\n[Error][{variant}] image_id={img_id}: {repr(exc)}")
            traceback.print_exc()

        if idx % args.save_every == 0:
            mm.save_json(paths["output"], {"annotations": results})
            mm.save_json(paths["stats"], mm.add_derived_stats(stats))

    mm.save_json(paths["output"], {"annotations": results})
    mm.save_json(paths["stats"], mm.add_derived_stats(stats))
    print(json.dumps(mm.add_derived_stats(stats), indent=4, ensure_ascii=False))
    if not args.no_chair:
        run_chair_eval(args, paths["output"], paths["chair"])


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("InternVL COCO ablation requires CUDA. Run in the GPU environment.")

    helper = mm.load_internvl_helper()
    mm.configure_helper(helper, args)
    model, tokenizer = mm.load_model(args)

    image_ids = load_target_image_ids(args.jsonl_path, args.max_samples)
    print(f"Loaded {len(image_ids)} COCO image ids")
    print(f"seeds: {args.seeds}")
    print(f"variants: {args.variants}")
    for seed in args.seeds:
        for variant in args.variants:
            run_one_variant(helper, model, tokenizer, args, variant, seed, image_ids)


if __name__ == "__main__":
    main()
