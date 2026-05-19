# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
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
    LogitsProcessor,
    LogitsProcessorList,
)


ROOT = Path(".")
MODEL_PATH = "<PATH_TO_LLAVA_NEXT>"
JSONL_PATH = "<PATH_TO_COCO_500_JSONL>"
IMAGE_FOLDER = "<PATH_TO_COCO>/val2014"
QUESTION = "Please describe this image in detail."
TORCH_DTYPE = torch.float16


def resolve_torch_dtype():
    return TORCH_DTYPE if torch.cuda.is_available() else torch.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA-1.6 native baseline and three-branch sparse decoding."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["baseline", "three", "both"],
        help="Run baseline once, three-branch across seeds, or both.",
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--question", type=str, default=QUESTION)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--schedule", type=str, default="global", choices=["global", "early_fixed"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--effective-max-step", type=int, default=500)
    parser.add_argument("--early-start", type=int, default=8)
    parser.add_argument("--early-end", type=int, default=56)
    parser.add_argument("--early-interval", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260422, 20260423, 20260424])
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
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--tag", type=str, default="")
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


def get_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


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


def build_mm_inputs(processor, tokenizer, image_pil, question, device):
    messages = [
        {
            "role": "user",
            "content": f"<image>\n{question}",
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt")
    out = {}
    for key, value in inputs.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def build_text_inputs(processor, tokenizer, question, device):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt")
    out = {}
    for key, value in inputs.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def build_black_image(image_pil):
    return Image.new("RGB", image_pil.size, (0, 0, 0))


def build_blur_image(image_pil, radius=12.0):
    return image_pil.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius)))


def build_negative_image(image_pil, args):
    if args.negative_image == "black":
        return build_black_image(image_pil)
    if args.negative_image == "blur":
        return build_blur_image(image_pil, args.blur_radius)
    raise ValueError(f"Unsupported negative image: {args.negative_image}")


def sample_unique_steps(start: int, end: int, k: int):
    if end <= start or k <= 0:
        return []
    pool = list(range(start, end))
    return sorted(random.sample(pool, min(k, len(pool))))


def build_intervention_steps(schedule, k, max_steps, effective_max_step, early_start=8, early_end=56, early_interval=12):
    active_end = max(1, min(max_steps, effective_max_step))
    if schedule == "global":
        return sample_unique_steps(0, active_end, k)
    if schedule == "early_fixed":
        end = min(active_end - 1, early_end)
        if end < early_start:
            return []
        return list(range(early_start, end + 1, max(1, early_interval)))
    raise ValueError(f"Unsupported schedule: {schedule}")


def rerank_three_branch(vis_logits, black_logits, text_logits, probs_m, top_k, black_alpha, apc_threshold,
                        syntax_threshold, syntax_margin):
    topk = min(top_k, probs_m.shape[-1])
    top_k_vals, top_k_indices = torch.topk(vis_logits, k=topk, dim=-1)
    vis_topk_ids = top_k_indices[0]
    visual_top1_id = int(vis_topk_ids[0].item())

    log_p_v_k = torch.log_softmax(top_k_vals, dim=-1)[0]
    selected_black_logits = black_logits[0, vis_topk_ids]
    log_p_b_k = torch.log_softmax(selected_black_logits, dim=-1)
    rerank_scores = (1.0 + black_alpha) * log_p_v_k - black_alpha * log_p_b_k

    selected_probs_v = probs_m[0, vis_topk_ids]
    max_prob = float(selected_probs_v[0].item())
    apc_mask = selected_probs_v < (max_prob * apc_threshold)
    rerank_scores[apc_mask] = -float("inf")

    guard_probs = torch.softmax(text_logits, dim=-1)
    visual_top1_syntax_support = float(guard_probs[0, visual_top1_id].item())
    syntax_accept_threshold = max(syntax_threshold, visual_top1_syntax_support + syntax_margin)

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
                "selected_rank": idx_int,
                "selected_syntax_support": syntax_support,
                "visual_top1_syntax_support": visual_top1_syntax_support,
                "guardrail_blocks": blocked,
                "fallback": False,
            }
        blocked += 1

    return {
        "token_id": visual_top1_id,
        "selected_rank": -1,
        "selected_syntax_support": -1.0,
        "visual_top1_syntax_support": visual_top1_syntax_support,
        "guardrail_blocks": blocked,
        "fallback": True,
    }


def resolve_eos_ids(tokenizer):
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, list):
        return set(int(x) for x in eos_token_id)
    return {int(eos_token_id)}


def trim_generated_ids(token_ids, eos_ids):
    trimmed = []
    saw_eos = False
    for token_id in token_ids:
        if token_id in eos_ids:
            saw_eos = True
            break
        trimmed.append(token_id)
    return trimmed, saw_eos


class ThreeBranchProcessor(LogitsProcessor):
    def __init__(
        self,
        model,
        base_text_input_ids,
        base_text_attention_mask,
        base_black_inputs,
        intervention_steps,
        top_k,
        black_alpha,
        apc_threshold,
        black_gate_mode,
        black_prior_threshold,
        black_visual_gap_threshold,
        syntax_threshold,
        syntax_margin,
    ):
        self.model = model
        self.base_text_input_ids = base_text_input_ids
        self.base_text_attention_mask = base_text_attention_mask
        self.base_black_inputs = base_black_inputs
        self.intervention_steps = set(intervention_steps)
        self.top_k = top_k
        self.black_alpha = black_alpha
        self.apc_threshold = apc_threshold
        self.black_gate_mode = black_gate_mode
        self.black_prior_threshold = black_prior_threshold
        self.black_visual_gap_threshold = black_visual_gap_threshold
        self.syntax_threshold = syntax_threshold
        self.syntax_margin = syntax_margin

        self.step_idx = 0
        self.generated_prefix_ids = []

        self.planned_interventions = len(intervention_steps)
        self.trigger_hits = 0
        self.black_same_top1_cnt = 0
        self.pass_black_prior_cnt = 0
        self.pass_black_gap_cnt = 0
        self.pass_black_both_cnt = 0
        self.black_gate_rejects = 0
        self.guardrail_blocks = 0
        self.guardrail_accept_cnt = 0
        self.guardrail_fallback_cnt = 0
        self.total_interventions = 0
        self.effective_interventions = 0

    def _build_current_inputs(self, base_input_ids, base_attention_mask):
        if not self.generated_prefix_ids:
            return base_input_ids, base_attention_mask

        prefix_tensor = torch.tensor(
            [self.generated_prefix_ids],
            device=base_input_ids.device,
            dtype=base_input_ids.dtype,
        )
        prefix_mask = torch.ones(
            (1, len(self.generated_prefix_ids)),
            device=base_attention_mask.device,
            dtype=base_attention_mask.dtype,
        )
        input_ids = torch.cat([base_input_ids, prefix_tensor], dim=1)
        attention_mask = torch.cat([base_attention_mask, prefix_mask], dim=1)
        return input_ids, attention_mask

    def _build_current_text_inputs(self):
        return self._build_current_inputs(self.base_text_input_ids, self.base_text_attention_mask)

    def _build_current_black_inputs(self):
        input_ids, attention_mask = self._build_current_inputs(
            self.base_black_inputs["input_ids"],
            self.base_black_inputs["attention_mask"],
        )
        black_inputs = dict(self.base_black_inputs)
        black_inputs["input_ids"] = input_ids
        black_inputs["attention_mask"] = attention_mask
        return black_inputs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.step_idx > 0 and len(self.generated_prefix_ids) < self.step_idx:
            self.generated_prefix_ids.append(int(input_ids[0, -1].item()))

        current_step = self.step_idx
        self.step_idx += 1

        if current_step not in self.intervention_steps:
            return scores

        self.trigger_hits += 1
        text_input_ids, text_attention_mask = self._build_current_text_inputs()
        black_inputs = self._build_current_black_inputs()

        with torch.inference_mode():
            black_outputs = self.model(
                **black_inputs,
                use_cache=False,
            )

        black_logits = black_outputs.logits[:, -1, :].float().to(scores.device)
        vis_logits = scores.float()
        probs_m = torch.softmax(vis_logits, dim=-1)
        visual_top1_id = int(torch.argmax(probs_m, dim=-1).item())

        probs_b = torch.softmax(black_logits, dim=-1)
        black_top1_id = int(torch.argmax(black_logits, dim=-1).item())
        black_top1_prior = float(probs_b[0, black_top1_id].item())
        visual_support_for_black_top1 = float(probs_m[0, black_top1_id].item())
        black_visual_gap = black_top1_prior - visual_support_for_black_top1
        black_same_top1 = black_top1_id == visual_top1_id

        if black_same_top1:
            self.black_same_top1_cnt += 1
        pass_black_prior = black_top1_prior > self.black_prior_threshold
        pass_black_gap = black_visual_gap > self.black_visual_gap_threshold
        if pass_black_prior:
            self.pass_black_prior_cnt += 1
        if pass_black_gap:
            self.pass_black_gap_cnt += 1

        pass_black_gate = True
        if self.black_gate_mode == "hard":
            pass_black_gate = pass_black_prior and pass_black_gap and (not black_same_top1)
        if pass_black_prior and pass_black_gap and (not black_same_top1):
            self.pass_black_both_cnt += 1
        if not pass_black_gate:
            self.black_gate_rejects += 1
            return scores

        self.total_interventions += 1
        with torch.inference_mode():
            text_outputs = self.model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                use_cache=False,
            )
        txt_logits = text_outputs.logits[:, -1, :].float().to(scores.device)

        rerank_info = rerank_three_branch(
            vis_logits=vis_logits,
            black_logits=black_logits,
            text_logits=txt_logits,
            probs_m=probs_m,
            top_k=self.top_k,
            black_alpha=self.black_alpha,
            apc_threshold=self.apc_threshold,
            syntax_threshold=self.syntax_threshold,
            syntax_margin=self.syntax_margin,
        )
        safe_token_id = rerank_info["token_id"]
        self.guardrail_blocks += int(rerank_info["guardrail_blocks"])
        if rerank_info["fallback"]:
            self.guardrail_fallback_cnt += 1
        else:
            self.guardrail_accept_cnt += 1
        if safe_token_id != visual_top1_id:
            self.effective_interventions += 1

        adjusted_scores = scores.clone()
        adjusted_scores[0, safe_token_id] = torch.max(adjusted_scores) + 1.0
        return adjusted_scores


@torch.inference_mode()
def run_native_baseline(model, processor, tokenizer, image_pil, args, input_device):
    mm_inputs = build_mm_inputs(processor, tokenizer, image_pil, args.question, input_device)
    output = model.generate(
        **mm_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
    )
    eos_ids = resolve_eos_ids(tokenizer)
    raw_ids = output.sequences[0, mm_inputs["input_ids"].shape[1]:].tolist()
    trimmed_ids, saw_eos = trim_generated_ids(raw_ids, eos_ids)
    caption = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
    return {
        "caption": caption,
        "generated_len": len(trimmed_ids),
        "truncated": not saw_eos and len(raw_ids) >= args.max_new_tokens,
    }


@torch.inference_mode()
def run_native_threebranch(model, processor, tokenizer, image_pil, args, input_device):
    mm_inputs = build_mm_inputs(processor, tokenizer, image_pil, args.question, input_device)
    black_inputs = build_mm_inputs(processor, tokenizer, build_negative_image(image_pil, args), args.question, input_device)
    text_inputs = build_text_inputs(processor, tokenizer, args.question, input_device)
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
    logits_processor = LogitsProcessorList([three_processor])

    output = model.generate(
        **mm_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        logits_processor=logits_processor,
        return_dict_in_generate=True,
    )
    eos_ids = resolve_eos_ids(tokenizer)
    raw_ids = output.sequences[0, mm_inputs["input_ids"].shape[1]:].tolist()
    trimmed_ids, saw_eos = trim_generated_ids(raw_ids, eos_ids)
    caption = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    return {
        "caption": caption,
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


def init_run_stats(include_sparse: bool):
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


def merge_stats(global_stats, local_stats):
    for key in global_stats:
        global_stats[key] += local_stats.get(key, 0)


def build_output_paths(args):
    tag_suffix = f"_{args.tag}" if args.tag else ""
    suffix = f"{args.max_samples}_t{args.max_new_tokens}{tag_suffix}"
    outputs = {}
    if args.mode in ("baseline", "both"):
        base_stem = f"llava16_native_baseline_{suffix}"
        outputs["baseline_output"] = ROOT / f"{base_stem}.json"
        outputs["baseline_stats"] = ROOT / f"{base_stem}_stats.json"
    if args.mode in ("three", "both"):
        outputs["three"] = {}
        for seed in args.seeds:
            neg_suffix = "" if args.negative_image == "black" else f"_{args.negative_image}neg_r{str(args.blur_radius).replace('.', 'p')}"
            three_stem = f"llava16_native_threebranch_k{args.k}_{args.schedule}_{args.max_samples}_seed{seed}_t{args.max_new_tokens}{neg_suffix}{tag_suffix}"
            outputs["three"][seed] = {
                "output": ROOT / f"{three_stem}.json",
                "stats": ROOT / f"{three_stem}_stats.json",
            }
    return outputs


def maybe_save_baseline(paths, results, stats):
    save_json(str(paths["baseline_output"]), {"annotations": results})
    save_json(str(paths["baseline_stats"]), stats)


def maybe_save_three(seed_paths, results, stats):
    save_json(str(seed_paths["output"]), {"annotations": results})
    save_json(str(seed_paths["stats"]), stats)


def main():
    args = parse_args()

    print(f"Loading LLaVA-1.6 from {args.model_path} ...")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("mode:", args.mode)
    print("schedule:", args.schedule)
    print("k:", args.k)
    print("black_gate_mode:", args.black_gate_mode)
    print("black_alpha:", args.black_alpha)
    print("negative_image:", args.negative_image)
    print("blur_radius:", args.blur_radius)
    print("max_new_tokens:", args.max_new_tokens)
    print("max_samples:", args.max_samples)
    print("seeds:", args.seeds)

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

    print("model loaded")
    print("input device:", input_device)

    target_image_ids = load_target_image_ids(JSONL_PATH, args.max_samples)
    paths = build_output_paths(args)

    if args.mode in ("baseline", "both"):
        baseline_results = []
        baseline_stats = init_run_stats(include_sparse=False)

    if args.mode in ("three", "both"):
        three_results = {seed: [] for seed in args.seeds}
        three_stats = {seed: init_run_stats(include_sparse=True) for seed in args.seeds}

    for idx, img_id in enumerate(tqdm(target_image_ids, desc="LLaVA-1.6 native three-branch"), start=1):
        img_filename = f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        image_path = os.path.join(IMAGE_FOLDER, img_filename)

        if not os.path.exists(image_path):
            if args.mode in ("baseline", "both"):
                baseline_stats["missing_image_cnt"] += 1
            if args.mode in ("three", "both"):
                for seed in args.seeds:
                    three_stats[seed]["missing_image_cnt"] += 1
            print(f"[Skip] image not found: {image_path}")
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")

            if args.mode in ("baseline", "both"):
                baseline_out = run_native_baseline(model, processor, tokenizer, raw_image, args, input_device)
                baseline_results.append({"image_id": int(img_id), "caption": baseline_out["caption"]})
                baseline_stats["success_cnt"] += 1
                baseline_stats["truncated_cnt"] += int(baseline_out["truncated"])
                baseline_stats["avg_generated_len_sum"] += float(baseline_out["generated_len"])

            if args.mode in ("three", "both"):
                for seed in args.seeds:
                    set_seed(seed + idx)
                    three_out = run_native_threebranch(model, processor, tokenizer, raw_image, args, input_device)
                    three_results[seed].append({"image_id": int(img_id), "caption": three_out["caption"]})
                    three_stats[seed]["success_cnt"] += 1
                    three_stats[seed]["truncated_cnt"] += int(three_out["truncated"])
                    three_stats[seed]["avg_generated_len_sum"] += float(three_out["generated_len"])
                    merge_stats(three_stats[seed], three_out["diag_stats"])

            if idx % args.save_every == 0:
                if args.mode in ("baseline", "both"):
                    maybe_save_baseline(paths, baseline_results, baseline_stats)
                if args.mode in ("three", "both"):
                    for seed in args.seeds:
                        maybe_save_three(paths["three"][seed], three_results[seed], three_stats[seed])

        except torch.cuda.OutOfMemoryError as exc:
            print(f"\n[OOM] image_id={img_id}: {repr(exc)}")
            if args.mode in ("baseline", "both"):
                baseline_stats["error_cnt"] += 1
            if args.mode in ("three", "both"):
                for seed in args.seeds:
                    three_stats[seed]["error_cnt"] += 1
            torch.cuda.empty_cache()
        except Exception:
            print(f"\n[Error] image_id={img_id}")
            if args.mode in ("baseline", "both"):
                baseline_stats["error_cnt"] += 1
            if args.mode in ("three", "both"):
                for seed in args.seeds:
                    three_stats[seed]["error_cnt"] += 1
            traceback.print_exc()
            break

    if args.mode in ("baseline", "both"):
        maybe_save_baseline(paths, baseline_results, baseline_stats)
    if args.mode in ("three", "both"):
        for seed in args.seeds:
            maybe_save_three(paths["three"][seed], three_results[seed], three_stats[seed])

    print("\nDone.")
    if args.mode in ("baseline", "both"):
        avg_len = baseline_stats["avg_generated_len_sum"] / max(baseline_stats["success_cnt"], 1)
        print("\nNative baseline stats:")
        print(json.dumps(baseline_stats, indent=4, ensure_ascii=False))
        print(f"avg generated len: {avg_len:.2f}")
        print(f"saved captions -> {paths['baseline_output']}")
        print(f"saved stats -> {paths['baseline_stats']}")
    if args.mode in ("three", "both"):
        for seed in args.seeds:
            avg_len = three_stats[seed]["avg_generated_len_sum"] / max(three_stats[seed]["success_cnt"], 1)
            print(f"\nNative three-branch stats (seed={seed}):")
            print(json.dumps(three_stats[seed], indent=4, ensure_ascii=False))
            print(f"avg generated len: {avg_len:.2f}")
            print(f"saved captions -> {paths['three'][seed]['output']}")
            print(f"saved stats -> {paths['three'][seed]['stats']}")


if __name__ == "__main__":
    main()
