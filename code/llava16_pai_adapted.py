# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
import subprocess
import sys
import traceback
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, LlavaNextForConditionalGeneration, LlavaNextImageProcessor, LlavaNextProcessor
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

import llava16_native_threebranch_3seeds as base


ROOT = Path(".")
MODEL_PATH = base.MODEL_PATH
JSONL_PATH = base.JSONL_PATH
IMAGE_FOLDER = base.IMAGE_FOLDER
QUESTION = base.QUESTION
COCO_ANN_PATH = "<PATH_TO_COCO>/annotations"


@dataclass
class PAIContext:
    alpha: float
    start_layer: int
    end_layer: int
    attention_active: bool = True
    cfg_branch_active: bool = False
    image_token_mask: torch.Tensor | None = None
    image_span_len: int = 0
    image_start_idx: int = -1
    image_end_idx: int = -1
    attention_layer_calls: int = 0

    def reset_sample(self):
        self.cfg_branch_active = False
        self.image_token_mask = None
        self.image_span_len = 0
        self.image_start_idx = -1
        self.image_end_idx = -1
        self.attention_layer_calls = 0

    def get_key_mask(self, kv_seq_len: int, device: torch.device):
        if self.image_token_mask is None:
            return None
        key_mask = self.image_token_mask[0].to(device=device, dtype=torch.bool)
        if key_mask.numel() < kv_seq_len:
            pad = torch.zeros(kv_seq_len - key_mask.numel(), device=device, dtype=torch.bool)
            key_mask = torch.cat([key_mask, pad], dim=0)
        elif key_mask.numel() > kv_seq_len:
            key_mask = key_mask[:kv_seq_len]
        return key_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run adapted PAI on LLaVA-NeXT with attention-only and attention+CFG variants."
    )
    parser.add_argument("--mode", choices=["attention", "attention_cfg", "both"], default="both")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--jsonl-path", type=str, default=JSONL_PATH)
    parser.add_argument("--image-folder", type=str, default=IMAGE_FOLDER)
    parser.add_argument("--question", type=str, default=QUESTION)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--alpha", type=float, default=0.2, help="PAI attention amplification strength.")
    parser.add_argument("--gamma", type=float, default=1.1, help="PAI CFG guidance scale.")
    parser.add_argument("--cfg-cutoff", type=float, default=0.1, help="PAI plausibility cutoff in probability ratio.")
    parser.add_argument("--start-layer", type=int, default=2)
    parser.add_argument("--end-layer", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--tag", type=str, default="pai")
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


def format_float_for_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def extend_decoding_state(input_ids, attention_mask, next_token):
    next_token = next_token.to(device=input_ids.device, dtype=input_ids.dtype).view(1, 1)
    next_mask = torch.ones(
        (attention_mask.shape[0], 1),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat([input_ids, next_token], dim=1), torch.cat([attention_mask, next_mask], dim=1)


def prepare_cached_step(model, input_ids, attention_mask, past_key_values, static_inputs=None):
    static_inputs = static_inputs or {}
    return model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        pixel_values=static_inputs.get("pixel_values"),
        image_sizes=static_inputs.get("image_sizes"),
        use_cache=True,
    )


def compute_merged_image_mask(model, image_features, input_ids, attention_mask):
    num_images, num_image_patches, _ = image_features.shape
    batch_size, sequence_length = input_ids.shape
    image_token_index = model.config.image_token_index
    special_image_token_mask = input_ids == image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length

    left_padding = not bool(torch.sum(input_ids[:, -1] == model.pad_token_id).item())
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_image_pad[:, None]

    batch_indices, non_image_indices = torch.where(input_ids != image_token_index)
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    occupied_by_text = torch.zeros(
        batch_size,
        int(max_embed_dim.item()) if hasattr(max_embed_dim, "item") else int(max_embed_dim),
        dtype=torch.bool,
        device=input_ids.device,
    )
    occupied_by_text[batch_indices, text_to_overwrite] = True

    image_to_overwrite = ~occupied_by_text
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(input_ids.device)

    expected = int(image_features.shape[:-1].numel())
    actual = int(image_to_overwrite.sum().item())
    if actual != expected:
        warnings.warn(
            f"PAI image mask count mismatch: expected {expected}, got {actual}. "
            "Attention intervention may be misaligned.",
            RuntimeWarning,
        )
    return image_to_overwrite


def install_image_span_capture(model, context: PAIContext):
    original_merge = model._merge_input_ids_with_image_features

    def patched_merge(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        if context.attention_active and not context.cfg_branch_active:
            image_mask = compute_merged_image_mask(self, image_features, input_ids, attention_mask)
            context.image_token_mask = image_mask.detach()
            positions = torch.nonzero(image_mask[0], as_tuple=False).flatten()
            if positions.numel() > 0:
                context.image_start_idx = int(positions.min().item())
                context.image_end_idx = int(positions.max().item()) + 1
                context.image_span_len = int(positions.numel())
        return original_merge(image_features, inputs_embeds, input_ids, attention_mask, labels)

    model._merge_input_ids_with_image_features = types.MethodType(patched_merge, model)


def pai_mistral_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    if "padding_mask" in kwargs:
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("PAI attention requires layer_idx for cached generation.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
            f"but is {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        min_value = torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device, dtype=attn_weights.dtype)
        attn_weights = torch.max(attn_weights, min_value)

    context = getattr(self, "pai_context", None)
    if (
        context is not None
        and context.attention_active
        and not context.cfg_branch_active
        and self.layer_idx is not None
        and context.start_layer <= self.layer_idx < context.end_layer
    ):
        key_mask = context.get_key_mask(kv_seq_len=kv_seq_len, device=attn_weights.device)
        if key_mask is not None and bool(key_mask.any().item()):
            selected = attn_weights[:, :, -1, key_mask]
            attn_weights[:, :, -1, key_mask] = selected + context.alpha * selected.abs()
            context.attention_layer_calls += 1

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def install_pai_attention(model, context: PAIContext):
    layers = model.language_model.model.layers
    start_layer = max(0, context.start_layer)
    end_layer = min(len(layers), context.end_layer)
    for layer_idx in range(start_layer, end_layer):
        self_attn = layers[layer_idx].self_attn
        self_attn.pai_context = context
        self_attn.forward = types.MethodType(pai_mistral_attention_forward, self_attn)
    install_image_span_capture(model, context)
    return start_layer, end_layer


def apply_cfg_scores(vis_logits, text_logits, gamma, cutoff_ratio):
    vis_log_probs = F.log_softmax(vis_logits.float(), dim=-1)
    text_log_probs = F.log_softmax(text_logits.float().to(vis_logits.device), dim=-1)
    cfg_scores = gamma * (vis_log_probs - text_log_probs) + text_log_probs

    filtered = 0
    if cutoff_ratio > 0:
        cutoff = math.log(cutoff_ratio) + vis_log_probs.max(dim=-1, keepdim=True).values
        mask = vis_log_probs < cutoff
        filtered = int(mask.sum().item())
        cfg_scores = cfg_scores.masked_fill(mask, -float("inf"))

    visual_top1 = int(torch.argmax(vis_log_probs, dim=-1).item())
    cfg_top1 = int(torch.argmax(cfg_scores, dim=-1).item())
    return cfg_scores, {
        "cfg_filtered": filtered,
        "cfg_effective": int(cfg_top1 != visual_top1),
    }


@torch.inference_mode()
def run_pai_variant(model, processor, tokenizer, image_pil, args, input_device, context: PAIContext, variant: str):
    use_cfg = variant == "attention_cfg"
    context.reset_sample()
    context.attention_active = True
    context.cfg_branch_active = False

    mm_inputs = base.build_mm_inputs(processor, tokenizer, image_pil, args.question, input_device)
    vis_outputs = model(**mm_inputs, use_cache=True, return_dict=True)

    vis_input_ids = mm_inputs["input_ids"]
    vis_attention_mask = mm_inputs["attention_mask"]
    vis_past = vis_outputs.past_key_values
    vis_logits = vis_outputs.logits[:, -1, :]

    if use_cfg:
        text_inputs = base.build_text_inputs(processor, tokenizer, args.question, input_device)
        context.cfg_branch_active = True
        txt_outputs = model(**text_inputs, use_cache=True, return_dict=True)
        context.cfg_branch_active = False
        txt_input_ids = text_inputs["input_ids"]
        txt_attention_mask = text_inputs["attention_mask"]
        txt_past = txt_outputs.past_key_values
        txt_logits = txt_outputs.logits[:, -1, :]
    else:
        text_inputs = None
        txt_input_ids = None
        txt_attention_mask = None
        txt_past = None
        txt_logits = None

    eos_ids = base.resolve_eos_ids(tokenizer)
    raw_ids = []
    diag_stats = {
        "cfg_calls": 0,
        "cfg_effective_interventions": 0,
        "cfg_filtered_sum": 0,
        "image_span_len_sum": context.image_span_len,
        "image_start_idx_sum": max(context.image_start_idx, 0),
        "image_end_idx_sum": max(context.image_end_idx, 0),
    }

    for step_idx in range(args.max_new_tokens):
        if use_cfg:
            final_scores, cfg_stats = apply_cfg_scores(
                vis_logits=vis_logits,
                text_logits=txt_logits,
                gamma=args.gamma,
                cutoff_ratio=args.cfg_cutoff,
            )
            diag_stats["cfg_calls"] += 1
            diag_stats["cfg_effective_interventions"] += cfg_stats["cfg_effective"]
            diag_stats["cfg_filtered_sum"] += cfg_stats["cfg_filtered"]
        else:
            final_scores = vis_logits.float()

        next_token = torch.argmax(final_scores, dim=-1)
        next_token_id = int(next_token.item())
        raw_ids.append(next_token_id)

        if next_token_id in eos_ids or step_idx == args.max_new_tokens - 1:
            break

        vis_input_ids, vis_attention_mask = extend_decoding_state(vis_input_ids, vis_attention_mask, next_token)
        vis_step_inputs = prepare_cached_step(
            model,
            input_ids=vis_input_ids,
            attention_mask=vis_attention_mask,
            past_key_values=vis_past,
            static_inputs=mm_inputs,
        )
        context.cfg_branch_active = False
        vis_outputs = model(**vis_step_inputs, return_dict=True)
        vis_past = vis_outputs.past_key_values
        vis_logits = vis_outputs.logits[:, -1, :]

        if use_cfg:
            txt_input_ids, txt_attention_mask = extend_decoding_state(txt_input_ids, txt_attention_mask, next_token)
            txt_step_inputs = prepare_cached_step(
                model,
                input_ids=txt_input_ids,
                attention_mask=txt_attention_mask,
                past_key_values=txt_past,
                static_inputs=text_inputs,
            )
            context.cfg_branch_active = True
            txt_outputs = model(**txt_step_inputs, return_dict=True)
            context.cfg_branch_active = False
            txt_past = txt_outputs.past_key_values
            txt_logits = txt_outputs.logits[:, -1, :]

    trimmed_ids, saw_eos = base.trim_generated_ids(raw_ids, eos_ids)
    caption = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
    diag_stats["attention_layer_calls"] = context.attention_layer_calls

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
        "cfg_calls": 0,
        "cfg_effective_interventions": 0,
        "cfg_filtered_sum": 0,
        "image_span_len_sum": 0,
        "image_start_idx_sum": 0,
        "image_end_idx_sum": 0,
        "attention_layer_calls": 0,
    }


def merge_stats(global_stats, local_stats):
    for key in global_stats:
        global_stats[key] += local_stats.get(key, 0)


def build_output_paths(args, variant):
    tag_suffix = f"_{args.tag}" if args.tag else ""
    layer_tag = f"layers{args.start_layer}_{args.end_layer}"
    if variant == "attention":
        method_tag = f"pai_attention_alpha{format_float_for_name(args.alpha)}"
    else:
        method_tag = (
            f"pai_attention_cfg_alpha{format_float_for_name(args.alpha)}_"
            f"gamma{format_float_for_name(args.gamma)}_cut{format_float_for_name(args.cfg_cutoff)}"
        )
    stem = f"llava16_{method_tag}_{layer_tag}_{args.max_samples}_seed{args.seed}_t{args.max_new_tokens}{tag_suffix}"
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
    variants = ["attention", "attention_cfg"] if args.mode == "both" else [args.mode]

    print(f"Loading LLaVA-NeXT from {args.model_path} ...")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("mode:", args.mode)
    print("variants:", variants)
    print("alpha:", args.alpha)
    print("gamma:", args.gamma)
    print("cfg_cutoff:", args.cfg_cutoff)
    print("layers:", args.start_layer, args.end_layer)
    print("max_samples:", args.max_samples)
    print("max_new_tokens:", args.max_new_tokens)
    print("seed:", args.seed)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=base.resolve_torch_dtype(),
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    try:
        model.tie_weights()
    except Exception:
        pass

    context = PAIContext(alpha=args.alpha, start_layer=args.start_layer, end_layer=args.end_layer)
    patched_start, patched_end = install_pai_attention(model, context)
    print(f"patched eager attention layers: [{patched_start}, {patched_end})")

    image_processor = LlavaNextImageProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)
    tokenizer = processor.tokenizer
    input_device = base.get_input_device(model)
    print("model loaded")
    print("input device:", input_device)

    target_image_ids = base.load_target_image_ids(args.jsonl_path, args.max_samples)
    paths = {variant: build_output_paths(args, variant) for variant in variants}
    results = {variant: [] for variant in variants}
    stats = {variant: init_stats() for variant in variants}

    for idx, img_id in enumerate(tqdm(target_image_ids, desc="LLaVA-NeXT adapted PAI"), start=1):
        img_filename = f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        image_path = os.path.join(args.image_folder, img_filename)

        if not os.path.exists(image_path):
            for variant in variants:
                stats[variant]["missing_image_cnt"] += 1
            print(f"[Skip] image not found: {image_path}")
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")
            for variant in variants:
                set_seed(args.seed + idx)
                out = run_pai_variant(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    image_pil=raw_image,
                    args=args,
                    input_device=input_device,
                    context=context,
                    variant=variant,
                )
                results[variant].append({"image_id": int(img_id), "caption": out["caption"]})
                stats[variant]["success_cnt"] += 1
                stats[variant]["truncated_cnt"] += int(out["truncated"])
                stats[variant]["avg_generated_len_sum"] += float(out["generated_len"])
                merge_stats(stats[variant], out["diag_stats"])

            if idx % args.save_every == 0:
                for variant in variants:
                    maybe_save(paths[variant], results[variant], stats[variant])

        except torch.cuda.OutOfMemoryError as exc:
            print(f"\n[OOM] image_id={img_id}: {repr(exc)}")
            for variant in variants:
                stats[variant]["error_cnt"] += 1
            torch.cuda.empty_cache()
        except Exception:
            print(f"\n[Error] image_id={img_id}")
            for variant in variants:
                stats[variant]["error_cnt"] += 1
            traceback.print_exc()
            break

    for variant in variants:
        maybe_save(paths[variant], results[variant], stats[variant])

    print("\nDone.")
    for variant in variants:
        avg_len = stats[variant]["avg_generated_len_sum"] / max(stats[variant]["success_cnt"], 1)
        avg_span = stats[variant]["image_span_len_sum"] / max(stats[variant]["success_cnt"], 1)
        print(f"\nPAI stats ({variant}):")
        print(json.dumps(stats[variant], indent=4, ensure_ascii=False))
        print(f"avg generated len: {avg_len:.2f}")
        print(f"avg image span len: {avg_span:.2f}")
        print(f"saved captions -> {paths[variant]['output']}")
        print(f"saved stats -> {paths[variant]['stats']}")

    if args.run_chair:
        for variant in variants:
            if stats[variant]["success_cnt"] > 0:
                run_chair(paths[variant], args)


if __name__ == "__main__":
    main()
