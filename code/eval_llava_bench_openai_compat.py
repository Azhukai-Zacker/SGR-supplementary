# -*- coding: utf-8 -*-
import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI


def load_api_config():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return api_key, base_url


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        parts = score_pair.split()
        if len(parts) == 2:
            return [float(parts[0]), float(parts[1])]
    except Exception:
        pass
    print("score parse error:", review[:300])
    return [-1, -1]


def get_eval(client, model, content, max_tokens, temperature, retry_sleep, max_retries):
    attempt = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and precise assistant for checking the quality of the answer.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as exc:
            attempt += 1
            msg = str(exc).lower()
            print(f"API error: {exc}")
            if "quota" in msg or "insufficient" in msg or "pre_consume_token_quota_failed" in msg:
                raise RuntimeError("API quota is not enough; stop evaluation to avoid repeated requests.") from exc
            if max_retries >= 0 and attempt > max_retries:
                raise RuntimeError(f"API failed after {max_retries} retries.") from exc
            print(f"retrying in {retry_sleep} seconds...")
            time.sleep(retry_sleep)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible LLaVA-Bench judge.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--answer-list", nargs=2, required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default=os.environ.get("LLAVA_BENCH_JUDGE_MODEL", "gpt-4.1"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key, config_base_url = load_api_config()
    if not api_key:
        raise RuntimeError("No API key found. Set OPENAI_API_KEY.")
    base_url = args.base_url or config_base_url
    client = OpenAI(api_key=api_key, base_url=base_url)

    questions = load_jsonl(args.question)
    ans1_rows = load_jsonl(args.answer_list[0])
    ans2_rows = load_jsonl(args.answer_list[1])
    contexts = load_jsonl(args.context)
    rule_dict = json.load(open(args.rule, "r", encoding="utf-8"))
    image_to_context = {ctx["image"]: ctx for ctx in contexts}

    output_path = Path(args.output)
    if output_path.exists():
        current_reviews = load_jsonl(output_path)
    else:
        current_reviews = []

    total = min(len(questions), len(ans1_rows), len(ans2_rows))
    if args.limit is not None:
        total = min(total, args.limit)

    with open(output_path, "a", encoding="utf-8") as review_file:
        for idx in range(total):
            ques = questions[idx]
            ans1 = ans1_rows[idx]
            ans2 = ans2_rows[idx]
            if idx < len(current_reviews):
                print(f"[{idx + 1}/{total}] skip existing")
                continue

            inst = image_to_context[ques["image"]]
            cap = inst["caption"]
            cap_str = "\n".join(cap) if isinstance(cap, list) else cap

            category = "llava_bench_" + ques["category"]
            if category not in rule_dict:
                raise KeyError(f"Visual QA category not found in rule file: {category}")

            rule = rule_dict[category]
            role = rule["role"]
            content = (
                f"[Context]\n{cap_str}\n\n"
                f"[Question]\n{ques['text']}\n\n"
                f"[{role} 1]\n{ans1['text']}\n\n[End of {role} 1]\n\n"
                f"[{role} 2]\n{ans2['text']}\n\n[End of {role} 2]\n\n"
                f"[System]\n{rule['prompt']}\n\n"
            )
            review = get_eval(
                client=client,
                model=args.judge_model,
                content=content,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                retry_sleep=args.retry_sleep,
                max_retries=args.max_retries,
            )
            row = {
                "id": idx + 1,
                "question_id": ques["question_id"],
                "answer1_id": ans1.get("answer_id", ans1.get("question_id")),
                "answer2_id": ans2.get("answer_id", ans2.get("question_id")),
                "category": category,
                "judge_model": args.judge_model,
                "content": review,
                "tuple": parse_score(review),
            }
            review_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            review_file.flush()
            print(f"[{idx + 1}/{total}] {category} score={row['tuple']}", flush=True)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
