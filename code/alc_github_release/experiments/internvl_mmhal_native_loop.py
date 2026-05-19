# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import os
import random
import traceback
from pathlib import Path

import torch
from tqdm import tqdm


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ALC_OUTPUT_DIR", RELEASE_ROOT / "outputs"))
INTERNVL_HELPER_SCRIPT = Path(os.environ.get("ALC_INTERNVL_HELPER", Path(__file__).resolve().parent / "internvl-2.py"))
MODEL_PATH = os.environ.get("ALC_INTERNVL_MODEL", "<PATH_TO_INTERNVL3_5_8B>")
DATA_JSON_PATH = Path(os.environ.get("ALC_MMHAL_JSON", RELEASE_ROOT / "data" / "MMHal-Bench" / "response_template.json"))
IMAGE_DIR = Path(os.environ.get("ALC_MMHAL_IMAGE_DIR", RELEASE_ROOT / "data" / "MMHal-Bench" / "images"))

BASELINE_OUTPUT_PATH = ROOT / "mmhal_internvl_baseline_preds.json"
BASELINE_STATS_PATH = ROOT / "mmhal_internvl_baseline_stats.json"
THREEBRANCH_OUTPUT_PATH = ROOT / "mmhal_internvl_threebranch_sparse_k8_global_seed20260423_preds.json"
THREEBRANCH_STATS_PATH = ROOT / "mmhal_internvl_threebranch_sparse_k8_global_seed20260423_stats.json"

QUESTION_FALLBACK = "Please describe this image in detail."


COCO_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

COMMON_OBJECT_WORDS = {
    "bag",
    "box",
    "camera",
    "computer",
    "container",
    "door",
    "food",
    "glass",
    "hat",
    "helmet",
    "jacket",
    "plate",
    "screen",
    "shelf",
    "shoe",
    "sign",
    "table",
    "towel",
    "tree",
    "window",
}


def build_object_trigger_vocab():
    vocab = set(COMMON_OBJECT_WORDS)
    for obj in COCO_OBJECTS:
        vocab.add(obj)
        for part in obj.split():
            vocab.add(part)
        if obj.endswith("s"):
            vocab.add(obj[:-1])
        else:
            vocab.add(obj + "s")
    return vocab


OBJECT_TRIGGER_VOCAB = build_object_trigger_vocab()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InternVL baseline and random-sparse three-branch decoding on MMHal-Bench."
    )
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "threebranch", "both"])
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-json", type=Path, default=DATA_JSON_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--baseline-output", type=Path, default=BASELINE_OUTPUT_PATH)
    parser.add_argument("--baseline-stats-output", type=Path, default=BASELINE_STATS_PATH)
    parser.add_argument("--threebranch-output", type=Path, default=THREEBRANCH_OUTPUT_PATH)
    parser.add_argument("--threebranch-stats-output", type=Path, default=THREEBRANCH_STATS_PATH)
    parser.add_argument("--answer-key", type=str, default="model_answer")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-num-tiles", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--dry-run", action="store_true", help="Validate MMHal files without loading InternVL.")
    parser.add_argument("--disable-flash-attn", action="store_true")

    # Current InternVL mainline: random sparse budget plus three branches.
    parser.add_argument("--schedule", type=str, default="global", choices=["global", "early", "mid", "late", "stratified"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--effective-max-step", type=int, default=220)
    parser.add_argument("--early-range", type=int, nargs=2, default=[0, 88], metavar=("START", "END"))
    parser.add_argument("--mid-range", type=int, nargs=2, default=[48, 144], metavar=("START", "END"))
    parser.add_argument("--late-range", type=int, nargs=2, default=[88, 176], metavar=("START", "END"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--black-prior-threshold", type=float, default=0.50)
    parser.add_argument("--black-visual-gap-threshold", type=float, default=0.30)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    parser.add_argument("--guard-mode", type=str, default="syntax", choices=["syntax", "none"])
    parser.add_argument("--trigger-mode", type=str, default="random", choices=["random", "object"])
    parser.add_argument(
        "--object-top-k",
        type=int,
        default=10,
        help="Check visual top-k decoded tokens for object-like candidates in object trigger mode.",
    )
    parser.add_argument(
        "--max-object-triggers",
        type=int,
        default=16,
        help="Maximum object-aware three-branch triggers per image.",
    )
    parser.add_argument(
        "--object-trigger-cooldown",
        type=int,
        default=2,
        help="Minimum generated-token gap after an object-aware trigger.",
    )
    parser.add_argument(
        "--object-require-prior-risk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In object mode, only rerank when black prior is confident and weakly supported by the visual branch.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_internvl_helper():
    spec = importlib.util.spec_from_file_location("internvl2_helper", INTERNVL_HELPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def configure_helper(helper, args):
    helper.model_path = args.model_path
    helper.MAX_NEW_TOKENS = args.max_new_tokens
    helper.MAX_NUM_TILES = args.max_num_tiles


def load_dataset(data_json, max_samples):
    dataset = load_json(data_json)
    if max_samples is not None:
        dataset = dataset[: max_samples]
    return dataset


def resolve_image_path(item, image_dir):
    image_src = item.get("image_src", "")
    if not image_src:
        return None
    return image_dir / image_src.split("/")[-1]


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
                "mode": args.mode,
                "model_path": args.model_path,
                "data_json": str(args.data_json),
                "image_dir": str(args.image_dir),
                "baseline_output": str(args.baseline_output),
                "threebranch_output": str(args.threebranch_output),
                "config": {
                    "method": "random_sparse_threebranch",
                    "trigger_mode": args.trigger_mode,
                    "schedule": args.schedule,
                    "k": args.k,
                    "object_top_k": args.object_top_k,
                    "max_object_triggers": args.max_object_triggers,
                    "seed": args.seed,
                    "max_new_tokens": args.max_new_tokens,
                    "effective_max_step": args.effective_max_step,
                    "max_num_tiles": args.max_num_tiles,
                },
            },
            indent=4,
            ensure_ascii=False,
        )
    )


def ensure_img_context_token_id(model, tokenizer):
    image_context_token = getattr(model, "img_context_token", "<IMG_CONTEXT>")
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(image_context_token)


def load_model(args):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=not args.disable_flash_attn,
        trust_remote_code=True,
    ).eval().cuda()
    ensure_img_context_token_id(model, tokenizer)
    return model, tokenizer


def init_base_stats(args, method):
    return {
        "method": method,
        "success_cnt": 0,
        "missing_image_cnt": 0,
        "error_cnt": 0,
        "avg_generated_len_approx_sum": 0.0,
        "model_path": str(args.model_path),
        "data_json": str(args.data_json),
        "image_dir": str(args.image_dir),
        "max_new_tokens": args.max_new_tokens,
        "max_num_tiles": args.max_num_tiles,
        "seed": args.seed,
    }


def init_threebranch_stats(args):
    trigger_mode = getattr(args, "trigger_mode", "random")
    method = "internvl_random_sparse_threebranch"
    if trigger_mode == "object":
        method = "internvl_object_aware_threebranch"
    stats = init_base_stats(args, method)
    stats.update(
        {
            "planned_interventions": 0,
            "trigger_hits": 0,
            "trigger_mode": trigger_mode,
            "object_candidate_hits": 0,
            "object_trigger_hits": 0,
            "object_trigger_budget_skips": 0,
            "object_trigger_cooldown_skips": 0,
            "object_prior_risk_skips": 0,
            "black_sync_calls": 0,
            "black_same_top1_cnt": 0,
            "pass_black_prior_cnt": 0,
            "pass_black_gap_cnt": 0,
            "pass_black_both_cnt": 0,
            "guardrail_blocks": 0,
            "guardrail_accept_cnt": 0,
            "guardrail_fallback_cnt": 0,
            "total_interventions": 0,
            "effective_interventions": 0,
            "total_tokens_generated": 0,
            "schedule": args.schedule,
            "k": args.k,
            "effective_max_step": args.effective_max_step,
            "top_k": args.top_k,
            "black_alpha": args.black_alpha,
            "text_alpha": getattr(args, "text_alpha", args.black_alpha),
            "apc_threshold": args.apc_threshold,
            "black_prior_threshold": args.black_prior_threshold,
            "black_visual_gap_threshold": args.black_visual_gap_threshold,
            "syntax_threshold": args.syntax_threshold,
            "syntax_margin": args.syntax_margin,
            "guard_mode": args.guard_mode,
            "contrast_branch": getattr(args, "contrast_branch", "black"),
            "object_top_k": getattr(args, "object_top_k", 10),
            "max_object_triggers": getattr(args, "max_object_triggers", 16),
            "object_trigger_cooldown": getattr(args, "object_trigger_cooldown", 2),
            "object_require_prior_risk": getattr(args, "object_require_prior_risk", True),
            "note": "Three branches are visual, black-image prior, and text-only syntax guard. trigger_mode=random uses a sampled sparse schedule; trigger_mode=object uses visual top-k object-candidate gating.",
        }
    )
    return stats


def add_derived_stats(stats):
    out = dict(stats)
    out["avg_generated_len_approx"] = out["avg_generated_len_approx_sum"] / max(out["success_cnt"], 1)
    if "total_interventions" in out:
        out["effective_intervention_rate"] = out["effective_interventions"] / max(out["total_interventions"], 1)
        out["interventions_per_success"] = out["total_interventions"] / max(out["success_cnt"], 1)
        out["trigger_rate_per_success"] = out["trigger_hits"] / max(out["success_cnt"], 1)
    return out


def merge_numeric_stats(stats, local_stats):
    for key, value in local_stats.items():
        if isinstance(value, bool):
            stats[key] = stats.get(key, 0) + int(value)
        elif isinstance(value, (int, float)):
            stats[key] = stats.get(key, 0) + value
        elif key not in stats:
            stats[key] = value


def token_len_approx(tokenizer, text):
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).get("input_ids", []))


def sample_unique_steps(start, end, k):
    if end <= start or k <= 0:
        return []
    pool = list(range(start, end))
    return sorted(random.sample(pool, min(k, len(pool))))


def sample_stratified_steps(max_steps, k):
    steps = []
    for idx in range(k):
        start = (idx * max_steps) // k
        end = ((idx + 1) * max_steps) // k
        if end <= start:
            end = min(start + 1, max_steps)
        if start >= max_steps:
            break
        steps.append(random.randrange(start, end))
    return sorted(steps)


def normalize_window(window, upper_bound):
    start, end = int(window[0]), int(window[1])
    start = max(0, min(start, upper_bound))
    end = max(0, min(end, upper_bound))
    if end <= start and start < upper_bound:
        end = start + 1
    return start, end


def build_intervention_steps(args):
    active_end = max(1, min(args.max_new_tokens, args.effective_max_step))
    early_start, early_end = normalize_window(args.early_range, active_end)
    mid_start, mid_end = normalize_window(args.mid_range, active_end)
    late_start, late_end = normalize_window(args.late_range, active_end)

    if args.schedule == "global":
        return sample_unique_steps(0, active_end, args.k)
    if args.schedule == "early":
        return sample_unique_steps(early_start, early_end, args.k)
    if args.schedule == "mid":
        return sample_unique_steps(mid_start, mid_end, args.k)
    if args.schedule == "late":
        return sample_unique_steps(late_start, late_end, args.k)
    if args.schedule == "stratified":
        return sample_stratified_steps(active_end, args.k)
    raise ValueError(f"Unsupported schedule: {args.schedule}")


def normalize_decoded_token(text):
    text = str(text).strip().lower()
    text = text.replace("<|im_end|>", "")
    text = text.replace("<s>", "").replace("</s>", "")
    text = text.replace("▁", " ").strip()
    text = "".join(ch for ch in text if ch.isalnum() or ch in {" ", "-"})
    return " ".join(text.split())


def decoded_token_is_object_like(tokenizer, token_id):
    text = normalize_decoded_token(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    if not text:
        return False, text
    if text in OBJECT_TRIGGER_VOCAB:
        return True, text
    # A token such as "tooth" may start a multi-token object phrase like toothbrush.
    for obj in OBJECT_TRIGGER_VOCAB:
        if len(text) >= 3 and obj.startswith(text):
            return True, text
    return False, text


def object_gate_from_visual_topk(tokenizer, vis_logits, args):
    top_k = max(1, min(int(getattr(args, "object_top_k", 10)), vis_logits.shape[-1]))
    top_ids = torch.topk(vis_logits, k=top_k, dim=-1).indices[0].detach().cpu().tolist()
    matches = []
    decoded = []
    for token_id in top_ids:
        is_object, text = decoded_token_is_object_like(tokenizer, token_id)
        if text:
            decoded.append(text)
        if is_object:
            matches.append({"token_id": int(token_id), "text": text})
    return {
        "trigger": bool(matches),
        "matches": matches,
        "top_decoded": decoded[:5],
    }


def rerank_threebranch(vis_logits, black_logits, text_logits, probs_m, visual_top1_id, args):
    topk = min(args.top_k, probs_m.shape[-1])
    top_k_vals, top_k_indices = torch.topk(vis_logits, k=topk, dim=-1)
    vis_topk_ids = top_k_indices[0]

    log_p_v_k = torch.log_softmax(top_k_vals, dim=-1)[0]
    contrast_branch = getattr(args, "contrast_branch", "black")
    if contrast_branch == "text":
        if text_logits is None:
            raise ValueError("contrast_branch='text' requires text_logits.")
        negative_logits = text_logits
        alpha = getattr(args, "text_alpha", args.black_alpha)
    elif contrast_branch == "black":
        if black_logits is None:
            raise ValueError("contrast_branch='black' requires black_logits.")
        negative_logits = black_logits
        alpha = args.black_alpha
    else:
        raise ValueError(f"Unsupported contrast_branch: {contrast_branch}")

    selected_negative_logits = negative_logits[0, vis_topk_ids]
    log_p_neg_k = torch.log_softmax(selected_negative_logits, dim=-1)
    rerank_scores = (1.0 + alpha) * log_p_v_k - alpha * log_p_neg_k

    selected_probs_v = probs_m[0, vis_topk_ids]
    max_prob = float(selected_probs_v[0].item())
    apc_mask = selected_probs_v < (max_prob * args.apc_threshold)
    rerank_scores[apc_mask] = -float("inf")

    if args.guard_mode == "none":
        for idx in torch.argsort(rerank_scores, descending=True):
            idx_int = int(idx.item())
            if bool(torch.isneginf(rerank_scores[idx_int]).item()):
                continue
            return {
                "token_id": int(vis_topk_ids[idx_int].item()),
                "guardrail_blocks": 0,
                "fallback": False,
            }
        return {
            "token_id": visual_top1_id,
            "guardrail_blocks": 0,
            "fallback": True,
        }

    guard_probs = torch.softmax(text_logits, dim=-1)
    visual_top1_syntax_support = float(guard_probs[0, visual_top1_id].item())
    syntax_accept_threshold = max(
        args.syntax_threshold,
        visual_top1_syntax_support + args.syntax_margin,
    )

    blocked = 0
    for idx in torch.argsort(rerank_scores, descending=True):
        idx_int = int(idx.item())
        if bool(torch.isneginf(rerank_scores[idx_int]).item()):
            continue
        candidate_id = int(vis_topk_ids[idx_int].item())
        syntax_support = float(guard_probs[0, candidate_id].item())
        if syntax_support >= syntax_accept_threshold:
            return {
                "token_id": candidate_id,
                "guardrail_blocks": blocked,
                "fallback": False,
            }
        blocked += 1

    return {
        "token_id": visual_top1_id,
        "guardrail_blocks": blocked,
        "fallback": True,
    }


@torch.inference_mode()
def run_baseline_answer(helper, model, tokenizer, image_path, question, args):
    pixel_values = helper.load_image(str(image_path), max_num=args.max_num_tiles).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response.strip()


@torch.inference_mode()
def run_random_sparse_threebranch(helper, model, tokenizer, image_path, question, args):
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    pixel_values = helper.load_image(str(image_path), max_num=args.max_num_tiles).to(torch.bfloat16).cuda()
    contrast_branch = getattr(args, "contrast_branch", "black")
    use_black_branch = contrast_branch == "black"
    use_text_branch = args.guard_mode == "syntax" or contrast_branch == "text"
    pixel_black = torch.zeros_like(pixel_values) if use_black_branch else None
    num_patches = pixel_values.shape[0]
    image_flags = torch.ones((num_patches, 1), dtype=torch.long, device=pixel_values.device)

    mm_text_inputs = helper.build_text_inputs_for_multimodal_chat(model, tokenizer, question, num_patches)
    current_mm_input_ids = mm_text_inputs["input_ids"].cuda()
    current_mm_attention_mask = mm_text_inputs["attention_mask"].cuda()

    current_black_input_ids = current_mm_input_ids.clone() if use_black_branch else None
    current_black_attention_mask = current_mm_attention_mask.clone() if use_black_branch else None

    if use_text_branch:
        text_inputs = helper._build_text_only_inputs(model, tokenizer, question)
        current_text_input_ids = text_inputs["input_ids"].cuda()
        current_text_attention_mask = text_inputs["attention_mask"].cuda()
    else:
        current_text_input_ids = None
        current_text_attention_mask = None

    trigger_mode = getattr(args, "trigger_mode", "random")
    if trigger_mode == "random":
        intervention_steps = set(build_intervention_steps(args))
    else:
        intervention_steps = set()
    active_end = max(1, min(args.max_new_tokens, args.effective_max_step))
    max_object_triggers = int(getattr(args, "max_object_triggers", args.k))
    object_cooldown_len = int(getattr(args, "object_trigger_cooldown", 2))
    object_triggers_used = 0
    object_cooldown = 0
    diag_stats = {
        "planned_interventions": len(intervention_steps) if trigger_mode == "random" else max_object_triggers,
        "trigger_hits": 0,
        "trigger_mode": trigger_mode,
        "object_candidate_hits": 0,
        "object_trigger_hits": 0,
        "object_trigger_budget_skips": 0,
        "object_trigger_cooldown_skips": 0,
        "object_prior_risk_skips": 0,
        "black_sync_calls": 0,
        "black_same_top1_cnt": 0,
        "pass_black_prior_cnt": 0,
        "pass_black_gap_cnt": 0,
        "pass_black_both_cnt": 0,
        "guardrail_blocks": 0,
        "guardrail_accept_cnt": 0,
        "guardrail_fallback_cnt": 0,
        "total_interventions": 0,
        "effective_interventions": 0,
        "total_tokens_generated": 0,
    }

    generated_tokens = []
    vis_past_key_values = None
    black_past_key_values = None
    black_pending_ids = []

    next_mm_input_ids = current_mm_input_ids
    next_mm_attention_mask = current_mm_attention_mask

    black_logits = None
    if use_black_branch:
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
        probs_m = torch.softmax(vis_logits, dim=-1)
        visual_top1_id = int(torch.argmax(probs_m, dim=-1).item())
        next_token_id = visual_top1_id
        diag_stats["total_tokens_generated"] += 1

        should_intervene = False
        if trigger_mode == "random":
            should_intervene = step in intervention_steps
        elif step < active_end:
            object_gate = object_gate_from_visual_topk(tokenizer, vis_logits, args)
            if object_gate["trigger"]:
                diag_stats["object_candidate_hits"] += 1
                if object_triggers_used >= max_object_triggers:
                    diag_stats["object_trigger_budget_skips"] += 1
                elif object_cooldown > 0:
                    diag_stats["object_trigger_cooldown_skips"] += 1
                else:
                    should_intervene = True
                    object_triggers_used += 1
                    object_cooldown = object_cooldown_len

        if should_intervene:
            diag_stats["trigger_hits"] += 1
            if trigger_mode == "object":
                diag_stats["object_trigger_hits"] += 1

            if use_black_branch and step > 0 and black_pending_ids:
                diag_stats["black_sync_calls"] += 1
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

            if use_black_branch:
                probs_b = torch.softmax(black_logits, dim=-1)
                black_top1_id = int(torch.argmax(black_logits, dim=-1).item())
                black_top1_prior = float(probs_b[0, black_top1_id].item())
                visual_support_for_black_top1 = float(probs_m[0, black_top1_id].item())
                black_visual_gap = black_top1_prior - visual_support_for_black_top1

                if black_top1_id == visual_top1_id:
                    diag_stats["black_same_top1_cnt"] += 1
                if black_top1_prior > args.black_prior_threshold:
                    diag_stats["pass_black_prior_cnt"] += 1
                if black_visual_gap > args.black_visual_gap_threshold:
                    diag_stats["pass_black_gap_cnt"] += 1
                if (
                    black_top1_prior > args.black_prior_threshold
                    and black_visual_gap > args.black_visual_gap_threshold
                    and black_top1_id != visual_top1_id
                ):
                    diag_stats["pass_black_both_cnt"] += 1

            skip_rerank_for_prior_risk = False
            if (
                trigger_mode == "object"
                and use_black_branch
                and getattr(args, "object_require_prior_risk", True)
            ):
                skip_rerank_for_prior_risk = not (
                    black_top1_prior > args.black_prior_threshold
                    and black_visual_gap > args.black_visual_gap_threshold
                    and black_top1_id != visual_top1_id
                )
                if skip_rerank_for_prior_risk:
                    diag_stats["object_prior_risk_skips"] += 1

            text_logits = None
            if use_text_branch and not skip_rerank_for_prior_risk:
                text_outputs = model.language_model(
                    input_ids=current_text_input_ids,
                    attention_mask=current_text_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                text_logits = text_outputs.logits[:, -1, :].float()

            if not skip_rerank_for_prior_risk:
                diag_stats["total_interventions"] += 1
                rerank_info = rerank_threebranch(
                    vis_logits=vis_logits,
                    black_logits=black_logits,
                    text_logits=text_logits,
                    probs_m=probs_m,
                    visual_top1_id=visual_top1_id,
                    args=args,
                )
                next_token_id = rerank_info["token_id"]
                diag_stats["guardrail_blocks"] += int(rerank_info["guardrail_blocks"])
                if rerank_info["fallback"]:
                    diag_stats["guardrail_fallback_cnt"] += 1
                else:
                    diag_stats["guardrail_accept_cnt"] += 1
                if next_token_id != visual_top1_id:
                    diag_stats["effective_interventions"] += 1

        if object_cooldown > 0:
            object_cooldown -= 1

        if next_token_id == eos_token_id:
            break

        generated_tokens.append(next_token_id)
        if use_black_branch:
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
        if use_black_branch:
            current_black_attention_mask = torch.cat(
                [
                    current_black_attention_mask,
                    torch.ones((1, 1), device=current_black_attention_mask.device, dtype=current_black_attention_mask.dtype),
                ],
                dim=1,
            )
        if use_text_branch:
            current_text_input_ids = torch.cat([current_text_input_ids, next_token_tensor], dim=1)
            current_text_attention_mask = torch.cat(
                [
                    current_text_attention_mask,
                    torch.ones((1, 1), device=current_text_attention_mask.device, dtype=current_text_attention_mask.dtype),
                ],
                dim=1,
            )

    caption = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {
        "caption": caption,
        "generated_len": len(generated_tokens),
        "diag_stats": diag_stats,
    }


def save_checkpoint(args, outputs, stats):
    if "baseline_results" in outputs:
        save_json(args.baseline_output, outputs["baseline_results"])
        save_json(args.baseline_stats_output, add_derived_stats(stats["baseline"]))
    if "threebranch_results" in outputs:
        save_json(args.threebranch_output, outputs["threebranch_results"])
        save_json(args.threebranch_stats_output, add_derived_stats(stats["threebranch"]))


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("InternVL MMHal inference requires CUDA. Run this script in the GPU environment.")

    helper = load_internvl_helper()
    configure_helper(helper, args)

    print(f"Loading InternVL from {args.model_path}")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("mode:", args.mode)
    print("method:", "random_sparse_threebranch")
    print("schedule:", args.schedule)
    print("k:", args.k)
    print("seed:", args.seed)
    print("max_new_tokens:", args.max_new_tokens)

    set_seed(args.seed)
    model, tokenizer = load_model(args)
    print("model loaded")

    dataset = load_dataset(args.data_json, args.max_samples)

    outputs = {}
    stats = {}
    if args.mode in ("baseline", "both"):
        outputs["baseline_results"] = []
        stats["baseline"] = init_base_stats(args, "internvl_baseline")
    if args.mode in ("threebranch", "both"):
        outputs["threebranch_results"] = []
        stats["threebranch"] = init_threebranch_stats(args)

    for idx, item in enumerate(tqdm(dataset, desc=f"MMHal InternVL {args.mode}"), start=1):
        image_path = resolve_image_path(item, args.image_dir)
        question = item.get("question", QUESTION_FALLBACK)

        if image_path is None or not image_path.exists():
            if "baseline_results" in outputs:
                out_item = dict(item)
                out_item[args.answer_key] = ""
                out_item["internvl_answer"] = ""
                out_item["error"] = f"missing image: {image_path}"
                outputs["baseline_results"].append(out_item)
                stats["baseline"]["missing_image_cnt"] += 1
            if "threebranch_results" in outputs:
                out_item = dict(item)
                out_item[args.answer_key] = ""
                out_item["internvl_threebranch_answer"] = ""
                out_item["error"] = f"missing image: {image_path}"
                outputs["threebranch_results"].append(out_item)
                stats["threebranch"]["missing_image_cnt"] += 1
            continue

        if "baseline_results" in outputs:
            out_item = dict(item)
            try:
                set_seed(args.seed + idx)
                answer = run_baseline_answer(helper, model, tokenizer, image_path, question, args)
                out_item[args.answer_key] = answer
                out_item["internvl_answer"] = answer
                out_item["internvl_generated_len_approx"] = token_len_approx(tokenizer, answer)
                stats["baseline"]["success_cnt"] += 1
                stats["baseline"]["avg_generated_len_approx_sum"] += float(out_item["internvl_generated_len_approx"])
            except Exception as exc:
                stats["baseline"]["error_cnt"] += 1
                out_item[args.answer_key] = ""
                out_item["internvl_answer"] = ""
                out_item["error"] = repr(exc)
                traceback.print_exc()
            outputs["baseline_results"].append(out_item)

        if "threebranch_results" in outputs:
            out_item = dict(item)
            try:
                set_seed(args.seed + idx)
                out = run_random_sparse_threebranch(helper, model, tokenizer, image_path, question, args)
                answer = out["caption"]
                out_item[args.answer_key] = answer
                out_item["internvl_threebranch_answer"] = answer
                out_item["internvl_threebranch_generated_len"] = out["generated_len"]
                out_item["internvl_threebranch_diag"] = out["diag_stats"]
                stats["threebranch"]["success_cnt"] += 1
                stats["threebranch"]["avg_generated_len_approx_sum"] += float(out["generated_len"])
                merge_numeric_stats(stats["threebranch"], out["diag_stats"])
            except Exception as exc:
                stats["threebranch"]["error_cnt"] += 1
                out_item[args.answer_key] = ""
                out_item["internvl_threebranch_answer"] = ""
                out_item["error"] = repr(exc)
                traceback.print_exc()
            outputs["threebranch_results"].append(out_item)

        if idx % args.save_every == 0:
            save_checkpoint(args, outputs, stats)

    save_checkpoint(args, outputs, stats)

    print("\nDone.")
    if "baseline" in stats:
        print("Baseline stats:")
        print(json.dumps(add_derived_stats(stats["baseline"]), indent=4, ensure_ascii=False))
        print(f"saved baseline responses -> {args.baseline_output}")
    if "threebranch" in stats:
        print("Threebranch stats:")
        print(json.dumps(add_derived_stats(stats["threebranch"]), indent=4, ensure_ascii=False))
        print(f"saved threebranch responses -> {args.threebranch_output}")


if __name__ == "__main__":
    main()
