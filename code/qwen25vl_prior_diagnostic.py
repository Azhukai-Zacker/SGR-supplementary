# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from pathlib import Path
from statistics import median

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen2_5_VLForConditionalGeneration


ROOT = Path(".")
MODEL_PATH = "<PATH_TO_QWEN2_5_VL>"
JSONL_PATH = "<PATH_TO_COCO_500_JSONL>"
IMAGE_FOLDER = "<PATH_TO_COCO>/val2014"
QUESTION = "Please describe this image in detail."


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analysis-only Qwen2.5-VL prior diagnostic. The script does not modify "
            "generation; it only records visual/prior/text distribution relations at sparse steps."
        )
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--jsonl-path", type=str, default=JSONL_PATH)
    parser.add_argument("--image-folder", type=str, default=IMAGE_FOLDER)
    parser.add_argument("--question", type=str, default=QUESTION)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--effective-max-step", type=int, default=260)
    parser.add_argument("--schedule", choices=["global", "stratified"], default="global")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--prior-modes", nargs="+", choices=["black", "gray", "blur", "text"], default=["black", "blur", "text"])
    parser.add_argument("--blur-radius", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--overlap-ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--text-alpha", type=float, default=0.20)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--tag", type=str, default="qwen25_detail_prior_diag")
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(inputs, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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
            if len(image_ids) >= max_samples:
                break
    return image_ids


def build_mm_inputs(processor, image_pil, image_ref, question, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_pil], padding=True, return_tensors="pt")
    return to_device(inputs, device)


def build_text_inputs(processor, question, device):
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    return to_device(inputs, device)


def build_prior_image(image_pil, mode, blur_radius):
    if mode == "black":
        return Image.new("RGB", image_pil.size, (0, 0, 0))
    if mode == "gray":
        return Image.new("RGB", image_pil.size, (127, 127, 127))
    if mode == "blur":
        return image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    raise ValueError(f"Unsupported image prior mode: {mode}")


def sample_unique_steps(start, end, k):
    if end <= start or k <= 0:
        return []
    return sorted(random.sample(list(range(start, end)), min(k, end - start)))


def sample_stratified_steps(max_steps, k):
    if max_steps <= 0 or k <= 0:
        return []
    steps = []
    for idx in range(k):
        start = (idx * max_steps) // k
        end = ((idx + 1) * max_steps) // k
        if end <= start:
            end = min(start + 1, max_steps)
        if start < max_steps:
            steps.append(random.randrange(start, end))
    return sorted(set(steps))


def build_intervention_steps(schedule, k, max_steps, effective_max_step):
    active_end = max(1, min(max_steps, effective_max_step))
    if schedule == "global":
        return sample_unique_steps(0, active_end, k)
    if schedule == "stratified":
        return sample_stratified_steps(active_end, k)
    raise ValueError(f"Unsupported schedule: {schedule}")


def append_generated_to_inputs(prompt_inputs, generated_ids):
    if len(generated_ids) == 0:
        return dict(prompt_inputs)

    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs["attention_mask"]
    mm_token_type_ids = prompt_inputs.get("mm_token_type_ids")
    gen_ids = torch.tensor([generated_ids], device=input_ids.device, dtype=input_ids.dtype)
    gen_mask = torch.ones((1, len(generated_ids)), device=attention_mask.device, dtype=attention_mask.dtype)

    out = dict(prompt_inputs)
    out["input_ids"] = torch.cat([input_ids, gen_ids], dim=1)
    out["attention_mask"] = torch.cat([attention_mask, gen_mask], dim=1)
    if mm_token_type_ids is not None:
        gen_type = torch.zeros((1, len(generated_ids)), device=mm_token_type_ids.device, dtype=mm_token_type_ids.dtype)
        out["mm_token_type_ids"] = torch.cat([mm_token_type_ids, gen_type], dim=1)
    return out


@torch.inference_mode()
def forward_last_logits(model, prompt_inputs, generated_ids):
    model.model.rope_deltas = None
    inputs = append_generated_to_inputs(prompt_inputs, generated_ids)
    inputs = {key: value for key, value in inputs.items() if value is not None}
    outputs = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
    return outputs.logits[:, -1, :].float()


def token_text(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def rank_of_token(logits, token_id):
    token_score = logits[0, int(token_id)]
    return int((logits[0] > token_score).sum().item()) + 1


def top_ids(logits, k):
    return torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1).indices[0].tolist()


def overlap_rate(a, b, k):
    return len(set(a[:k]).intersection(b[:k])) / max(1, k)


def rerank_candidate(vis_logits, negative_logits, text_logits, args, alpha, guard_mode):
    probs_v = torch.softmax(vis_logits, dim=-1)
    topk = min(args.top_k, probs_v.shape[-1])
    top_vals, top_indices = torch.topk(vis_logits, k=topk, dim=-1)
    cand_ids = top_indices[0]
    visual_top1_id = int(cand_ids[0].item())

    log_p_v = torch.log_softmax(top_vals, dim=-1)[0]
    selected_neg = negative_logits[0, cand_ids]
    log_p_neg = torch.log_softmax(selected_neg, dim=-1)
    scores = (1.0 + alpha) * log_p_v - alpha * log_p_neg

    selected_probs_v = probs_v[0, cand_ids]
    max_prob = float(selected_probs_v[0].item())
    scores[selected_probs_v < (max_prob * args.apc_threshold)] = -float("inf")

    blocked = 0
    fallback = True
    chosen_id = visual_top1_id
    if guard_mode == "none":
        for idx in torch.argsort(scores, descending=True):
            idx_int = int(idx.item())
            if bool(torch.isneginf(scores[idx_int]).item()):
                continue
            chosen_id = int(cand_ids[idx_int].item())
            fallback = False
            break
    else:
        guard_probs = torch.softmax(text_logits, dim=-1)
        visual_top1_syntax = float(guard_probs[0, visual_top1_id].item())
        threshold = max(args.syntax_threshold, visual_top1_syntax + args.syntax_margin)
        for idx in torch.argsort(scores, descending=True):
            idx_int = int(idx.item())
            if bool(torch.isneginf(scores[idx_int]).item()):
                continue
            candidate_id = int(cand_ids[idx_int].item())
            if float(guard_probs[0, candidate_id].item()) >= threshold:
                chosen_id = candidate_id
                fallback = False
                break
            blocked += 1

    return {
        "chosen_id": int(chosen_id),
        "changed": bool(chosen_id != visual_top1_id),
        "fallback": bool(fallback),
        "guardrail_blocks": int(blocked),
    }


class PriorDiagnosticProcessor(LogitsProcessor):
    def __init__(self, model, tokenizer, visual_prompt_len, prior_inputs, text_inputs, intervention_steps, image_id, args):
        self.model = model
        self.tokenizer = tokenizer
        self.visual_prompt_len = int(visual_prompt_len)
        self.prior_inputs = prior_inputs
        self.text_inputs = text_inputs
        self.intervention_steps = set(intervention_steps)
        self.image_id = int(image_id)
        self.args = args
        self.records = []
        self.stats = {
            "planned_interventions": len(self.intervention_steps),
            "trigger_hits": 0,
            "stateless_aux_calls": 0,
            "stateless_aux_tokens": 0,
        }

    def __call__(self, input_ids, scores):
        step = int(input_ids.shape[1] - self.visual_prompt_len)
        if step not in self.intervention_steps:
            return scores

        generated_ids = input_ids[0, self.visual_prompt_len :].tolist()
        prefix_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self.stats["trigger_hits"] += 1
        vis_logits = scores.float()
        text_logits = None
        if self.text_inputs is not None:
            text_logits = forward_last_logits(self.model, self.text_inputs, generated_ids)
            self.stats["stateless_aux_calls"] += 1
            self.stats["stateless_aux_tokens"] += len(generated_ids)

        max_overlap_k = max(max(self.args.overlap_ks), self.args.top_k)
        visual_top = top_ids(vis_logits, max_overlap_k)
        visual_top1_id = int(visual_top[0])

        event = {
            "image_id": self.image_id,
            "step": step,
            "generated_prefix_len": len(generated_ids),
            "visual_top1_id": visual_top1_id,
            "visual_top1": token_text(self.tokenizer, visual_top1_id),
            "visual_top5": [token_text(self.tokenizer, x) for x in visual_top[:5]],
            "prefix_text_tail": prefix_text[-240:],
            "priors": {},
        }

        for mode, prompt_inputs in self.prior_inputs.items():
            if mode == "text":
                prior_logits = text_logits
                alpha = self.args.text_alpha
                guard_mode = "none"
            else:
                prior_logits = forward_last_logits(self.model, prompt_inputs, generated_ids)
                self.stats["stateless_aux_calls"] += 1
                self.stats["stateless_aux_tokens"] += len(generated_ids)
                alpha = self.args.black_alpha
                guard_mode = "syntax"

            prior_top = top_ids(prior_logits, max_overlap_k)
            prior_top1_id = int(prior_top[0])
            rerank = rerank_candidate(vis_logits, prior_logits, text_logits, self.args, alpha, guard_mode)
            event["priors"][mode] = {
                "prior_top1_id": prior_top1_id,
                "prior_top1": token_text(self.tokenizer, prior_top1_id),
                "prior_top5": [token_text(self.tokenizer, x) for x in prior_top[:5]],
                "same_top1_as_visual": bool(prior_top1_id == visual_top1_id),
                "visual_top1_prior_rank": rank_of_token(prior_logits, visual_top1_id),
                "prior_top1_visual_rank": rank_of_token(vis_logits, prior_top1_id),
                "overlap": {f"top{k}": overlap_rate(visual_top, prior_top, k) for k in self.args.overlap_ks},
                "rerank_chosen_id": rerank["chosen_id"],
                "rerank_chosen": token_text(self.tokenizer, rerank["chosen_id"]),
                "rerank_changed": rerank["changed"],
                "rerank_fallback": rerank["fallback"],
                "guardrail_blocks": rerank["guardrail_blocks"],
            }

        self.records.append(event)
        return scores


def summarize_events(events, image_count, generated_lens, processor_stats):
    summary = {
        "num_images": image_count,
        "num_events": len(events),
        "avg_generated_len": sum(generated_lens) / max(1, len(generated_lens)),
        "planned_interventions": sum(s["planned_interventions"] for s in processor_stats),
        "trigger_hits": sum(s["trigger_hits"] for s in processor_stats),
        "stateless_aux_calls": sum(s["stateless_aux_calls"] for s in processor_stats),
        "stateless_aux_tokens": sum(s["stateless_aux_tokens"] for s in processor_stats),
        "priors": {},
    }
    if not events:
        return summary

    prior_modes = sorted(events[0]["priors"].keys())
    for mode in prior_modes:
        values = [event["priors"][mode] for event in events if mode in event["priors"]]
        out = {
            "same_top1_rate": sum(v["same_top1_as_visual"] for v in values) / max(1, len(values)),
            "rerank_changed_rate": sum(v["rerank_changed"] for v in values) / max(1, len(values)),
            "rerank_fallback_rate": sum(v["rerank_fallback"] for v in values) / max(1, len(values)),
            "avg_guardrail_blocks": sum(v["guardrail_blocks"] for v in values) / max(1, len(values)),
            "avg_visual_top1_prior_rank": sum(v["visual_top1_prior_rank"] for v in values) / max(1, len(values)),
            "median_visual_top1_prior_rank": median([v["visual_top1_prior_rank"] for v in values]),
            "avg_prior_top1_visual_rank": sum(v["prior_top1_visual_rank"] for v in values) / max(1, len(values)),
            "median_prior_top1_visual_rank": median([v["prior_top1_visual_rank"] for v in values]),
            "overlap": {},
        }
        for key in values[0]["overlap"]:
            out["overlap"][key] = sum(v["overlap"][key] for v in values) / max(1, len(values))
        summary["priors"][mode] = out
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading Qwen2.5-VL from {args.model_path} ...")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("question:", args.question)
    print("prior_modes:", args.prior_modes)

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": resolve_dtype(),
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    device = get_input_device(model)

    image_ids = load_target_image_ids(args.jsonl_path, args.max_samples)
    stem = f"qwen25vl_prior_diag_{args.max_samples}_seed{args.seed}_t{args.max_new_tokens}_k{args.k}_{args.tag}"
    out_path = ROOT / f"{stem}.json"
    summary_path = ROOT / f"{stem}_summary.json"

    annotations = []
    all_events = []
    generated_lens = []
    processor_stats = []

    for idx, image_id in enumerate(tqdm(image_ids, desc="Qwen2.5 prior diagnostic"), start=1):
        image_path = os.path.join(args.image_folder, f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
        image = Image.open(image_path).convert("RGB")
        mm_inputs = build_mm_inputs(processor, image, image_path, args.question, device)
        prompt_len = mm_inputs["input_ids"].shape[1]

        text_inputs = build_text_inputs(processor, args.question, device)
        prior_inputs = {}
        for mode in args.prior_modes:
            if mode == "text":
                prior_inputs["text"] = text_inputs
            else:
                prior_img = build_prior_image(image, mode, args.blur_radius)
                prior_inputs[mode] = build_mm_inputs(processor, prior_img, image_path, args.question, device)

        set_seed(args.seed + idx)
        intervention_steps = build_intervention_steps(args.schedule, args.k, args.max_new_tokens, args.effective_max_step)
        diag_processor = PriorDiagnosticProcessor(
            model=model,
            tokenizer=tokenizer,
            visual_prompt_len=prompt_len,
            prior_inputs=prior_inputs,
            text_inputs=text_inputs,
            intervention_steps=intervention_steps,
            image_id=image_id,
            args=args,
        )
        output = model.generate(
            **mm_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([diag_processor]),
        )
        raw_ids = output.sequences[0, prompt_len:].tolist()
        caption = tokenizer.decode(raw_ids, skip_special_tokens=True).strip()
        annotations.append({"image_id": int(image_id), "caption": caption})
        generated_lens.append(len(raw_ids))
        all_events.extend(diag_processor.records)
        processor_stats.append(diag_processor.stats)

        if idx % args.save_every == 0:
            summary = summarize_events(all_events, len(annotations), generated_lens, processor_stats)
            save_json(out_path, {"annotations": annotations, "events": all_events})
            save_json(summary_path, summary)

    summary = summarize_events(all_events, len(annotations), generated_lens, processor_stats)
    save_json(out_path, {"annotations": annotations, "events": all_events})
    save_json(summary_path, summary)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved events -> {out_path}")
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
