# -*- coding: utf-8 -*-
import argparse
import json
import os
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
)


ROOT = Path(".")
MODEL_PATH = "<PATH_TO_LLAVA_NEXT>"
DATA_JSON_PATH = ROOT / "MMHal-Bench" / "response_template.json"
IMAGE_DIR = ROOT / "MMHal-Bench" / "images"
OUTPUT_JSON_PATH = ROOT / "mmhal_llava_next_baseline_preds.json"
STATS_JSON_PATH = ROOT / "mmhal_llava_next_baseline_stats.json"
QUESTION_FALLBACK = "Please describe this image in detail."
TORCH_DTYPE = torch.float16


def parse_args():
    parser = argparse.ArgumentParser(description="Run native LLaVA-NeXT baseline on MMHal-Bench.")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-json", type=Path, default=DATA_JSON_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON_PATH)
    parser.add_argument("--stats-output", type=Path, default=STATS_JSON_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--answer-key", type=str, default="model_answer")
    return parser.parse_args()


def resolve_torch_dtype():
    return TORCH_DTYPE if torch.cuda.is_available() else torch.float32


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def resolve_image_path(item, image_dir):
    image_src = item.get("image_src", "")
    if not image_src:
        return None
    filename = image_src.split("/")[-1]
    return image_dir / filename


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


def generate_answer(model, processor, tokenizer, image_pil, question, device, max_new_tokens):
    inputs = build_mm_inputs(processor, tokenizer, image_pil, question, device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    raw_ids = output.sequences[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(raw_ids, skip_special_tokens=True).strip()
    return answer, int(raw_ids.shape[0])


def main():
    args = parse_args()

    print(f"Loading LLaVA-NeXT from {args.model_path}")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("data:", args.data_json)
    print("image dir:", args.image_dir)
    print("output:", args.output)
    print("max_new_tokens:", args.max_new_tokens)

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

    with open(args.data_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    results = []
    stats = {
        "success_cnt": 0,
        "missing_image_cnt": 0,
        "error_cnt": 0,
        "avg_generated_len_sum": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "model_path": args.model_path,
        "data_json": str(args.data_json),
        "image_dir": str(args.image_dir),
    }

    for idx, item in enumerate(tqdm(dataset, desc="MMHal LLaVA-NeXT baseline"), start=1):
        out_item = dict(item)
        image_path = resolve_image_path(item, args.image_dir)
        if image_path is None or not image_path.exists():
            stats["missing_image_cnt"] += 1
            out_item[args.answer_key] = ""
            out_item["llava_next_answer"] = ""
            out_item["error"] = f"missing image: {image_path}"
            results.append(out_item)
            continue

        question = item.get("question", QUESTION_FALLBACK)
        try:
            image_pil = Image.open(image_path).convert("RGB")
            answer, gen_len = generate_answer(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_pil=image_pil,
                question=question,
                device=input_device,
                max_new_tokens=args.max_new_tokens,
            )
            out_item[args.answer_key] = answer
            out_item["llava_next_answer"] = answer
            out_item["llava_next_generated_len"] = gen_len
            stats["success_cnt"] += 1
            stats["avg_generated_len_sum"] += float(gen_len)
        except Exception as exc:
            stats["error_cnt"] += 1
            out_item[args.answer_key] = ""
            out_item["llava_next_answer"] = ""
            out_item["error"] = repr(exc)
            traceback.print_exc()

        results.append(out_item)
        if idx % args.save_every == 0:
            save_json(args.output, results)
            save_json(args.stats_output, stats)

    save_json(args.output, results)
    save_json(args.stats_output, stats)

    avg_len = stats["avg_generated_len_sum"] / max(stats["success_cnt"], 1)
    print("\nDone.")
    print(json.dumps(stats, indent=4, ensure_ascii=False))
    print(f"avg generated len: {avg_len:.2f}")
    print(f"saved responses -> {args.output}")
    print(f"saved stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
