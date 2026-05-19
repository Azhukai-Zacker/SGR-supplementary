# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


ROOT = Path(".")
BASELINE_RESPONSE = ROOT / "mmhal_internvl_baseline_preds.json"
OURS_RESPONSE = ROOT / "mmhal_internvl_threebranch_sparse_k8_global_seed20260423_preds.json"
DEFAULT_BASE_URL = "<OPENAI_COMPATIBLE_BASE_URL>"
DEFAULT_API_KEY = "<YOUR_API_KEY>"
DEFAULT_MODEL = "gpt-5.4-mini"


MMHAL_TEMPLATE = """Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evaluation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM response, first provide a short explanation. Then rate the response by choosing exactly one option:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

End your response with a final line in exactly this format:
Rating: <score>

### Image Contents
{image_content}

### Question
{question}

### Standard Human-Generated Answer
{gt_answer}

### LMM Response to Evaluate
{answer}
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def image_content_to_text(value):
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value)


def get_answer(record, method):
    if method == "baseline":
        return (
            record.get("internvl_answer")
            or record.get("llava_next_answer")
            or record.get("qwen3vl_answer")
            or record.get("baseline_answer")
            or record.get("answer")
            or record.get("response")
            or record.get("model_answer")
            or ""
        )
    if method == "ours":
        return (
            record.get("internvl_threebranch_answer")
            or record.get("llava_next_threebranch_answer")
            or record.get("qwen3vl_threebranch_answer")
            or record.get("ours_answer")
            or record.get("answer")
            or record.get("response")
            or record.get("model_answer")
            or ""
        )
    raise ValueError(f"Unknown method: {method}")


def parse_rating(text):
    patterns = [
        r"rating\s*[:：]\s*\**\s*([0-6])\b",
        r"\brating\s+([0-6])\b",
        r"\b([0-6])\s*,\s*(?:very|somewhat|not)\s+informative",
    ]
    lowered = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, lowered, flags=re.I)
        if matches:
            return int(matches[-1])
    print("Warning: failed to parse rating from judge response:")
    print(text[:500])
    return -1


def build_prompt(record, answer):
    return MMHAL_TEMPLATE.format(
        image_content=image_content_to_text(record.get("image_content", [])),
        question=record["question"],
        gt_answer=record["gt_answer"],
        answer=answer,
    )


def call_judge(client, args, prompt):
    attempt = 0
    while True:
        try:
            request = {
                "model": args.judge_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            # Some OpenAI-compatible gateways for newer models reject optional
            # Chat Completions parameters. Keep the default request identical
            # to the minimal connectivity test unless the user opts in.
            if args.temperature is not None:
                request["temperature"] = args.temperature
            if args.max_tokens is not None:
                request["max_tokens"] = args.max_tokens
            completion = client.chat.completions.create(**request)
            return completion.choices[0].message.content
        except Exception as exc:
            attempt += 1
            msg = str(exc).lower()
            print(f"API error: {exc}")
            if "quota" in msg or "insufficient" in msg or "balance" in msg or "pre_consume_token_quota_failed" in msg:
                raise RuntimeError("API quota appears insufficient; stopping evaluation.") from exc
            if args.max_retries >= 0 and attempt > args.max_retries:
                raise RuntimeError(f"API failed after {args.max_retries} retries.") from exc
            print(f"retrying in {args.retry_sleep} seconds...")
            time.sleep(args.retry_sleep)


def summarize(rows, limit):
    summary = {
        "limit": limit,
        "methods": {},
        "paired_delta_ours_minus_baseline": {},
    }
    by_method = {"baseline": [], "ours": []}
    for row in rows:
        if row.get("score", -1) >= 0 and row.get("method") in by_method:
            by_method[row["method"]].append(row)

    for method, method_rows in by_method.items():
        scores = [row["score"] for row in method_rows]
        hallucinations = [1 if score <= 2 else 0 for score in scores]
        by_type = {}
        for row in method_rows:
            by_type.setdefault(row.get("question_type", "unknown"), []).append(row["score"])
        summary["methods"][method] = {
            "num_records": len(method_rows),
            "average_score": sum(scores) / len(scores) if scores else None,
            "hallucination_rate": sum(hallucinations) / len(hallucinations) if hallucinations else None,
            "average_score_each_question_type": {
                key: sum(vals) / len(vals) for key, vals in sorted(by_type.items())
            },
        }

    paired = {}
    for row in rows:
        key = row.get("source_index")
        paired.setdefault(key, {})[row.get("method")] = row
    score_deltas = []
    hallucination_deltas = []
    for pair in paired.values():
        if "baseline" not in pair or "ours" not in pair:
            continue
        b_score = pair["baseline"].get("score", -1)
        o_score = pair["ours"].get("score", -1)
        if b_score < 0 or o_score < 0:
            continue
        score_deltas.append(o_score - b_score)
        hallucination_deltas.append((1 if o_score <= 2 else 0) - (1 if b_score <= 2 else 0))
    summary["paired_delta_ours_minus_baseline"] = {
        "num_pairs": len(score_deltas),
        "average_score_delta": sum(score_deltas) / len(score_deltas) if score_deltas else None,
        "hallucination_rate_delta": sum(hallucination_deltas) / len(hallucination_deltas)
        if hallucination_deltas
        else None,
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible MMHal judge for InternVL baseline and ours, defaulting to the first 30 records."
    )
    parser.add_argument("--baseline-response", type=Path, default=BASELINE_RESPONSE)
    parser.add_argument("--ours-response", type=Path, default=OURS_RESPONSE)
    parser.add_argument("--output", type=Path, default=ROOT / "mmhal_internvl_openai_gpt54mini_n30.review.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "mmhal_internvl_openai_gpt54mini_n30.summary.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-reuse-identical-answers",
        action="store_true",
        help="By default, if baseline and ours have identical answers for the same item, reuse the baseline judge score.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key or args.api_key.strip() == "<YOUR_API_KEY>":
        raise RuntimeError(
            "Please set --api-key, OPENAI_API_KEY, or replace DEFAULT_API_KEY in this script."
        )

    baseline = load_json(args.baseline_response)
    ours = load_json(args.ours_response)
    total = min(args.limit, len(baseline) - args.start_index, len(ours) - args.start_index)
    if total <= 0:
        raise ValueError("No records to evaluate. Check --limit and --start-index.")

    if args.overwrite:
        if args.output.exists():
            args.output.unlink()
        if args.summary.exists():
            args.summary.unlink()

    existing_rows = load_jsonl(args.output)
    done = {(row.get("method"), row.get("source_index")) for row in existing_rows}
    row_cache = {(row.get("method"), row.get("source_index")): row for row in existing_rows}

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as f:
        for offset in range(total):
            source_index = args.start_index + offset
            for method, records in (("baseline", baseline), ("ours", ours)):
                if (method, source_index) in done:
                    print(f"[{method} {offset + 1}/{total}] skip existing")
                    continue
                record = records[source_index]
                answer = get_answer(record, method)

                if method == "ours" and not args.no_reuse_identical_answers:
                    baseline_answer = get_answer(baseline[source_index], "baseline")
                    baseline_row = row_cache.get(("baseline", source_index))
                    if baseline_row is not None and answer.strip() == baseline_answer.strip():
                        row = {
                            "id": len(done) + 1,
                            "method": method,
                            "source_index": source_index,
                            "question_type": record.get("question_type"),
                            "question_topic": record.get("question_topic"),
                            "image_id": record.get("image_id"),
                            "question": record.get("question"),
                            "gt_answer": record.get("gt_answer"),
                            "answer": answer,
                            "judge_model": args.judge_model,
                            "base_url": args.base_url,
                            "score": baseline_row.get("score", -1),
                            "hallucination": baseline_row.get("hallucination"),
                            "content": baseline_row.get("content", ""),
                            "copied_from_identical_baseline": True,
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        done.add((method, source_index))
                        row_cache[(method, source_index)] = row
                        print(
                            f"[{method} {offset + 1}/{total}] copied identical baseline score={row['score']}",
                            flush=True,
                        )
                        continue

                prompt = build_prompt(record, answer)
                content = call_judge(client, args, prompt)
                score = parse_rating(content)
                row = {
                    "id": len(done) + 1,
                    "method": method,
                    "source_index": source_index,
                    "question_type": record.get("question_type"),
                    "question_topic": record.get("question_topic"),
                    "image_id": record.get("image_id"),
                    "question": record.get("question"),
                    "gt_answer": record.get("gt_answer"),
                    "answer": answer,
                    "judge_model": args.judge_model,
                    "base_url": args.base_url,
                    "score": score,
                    "hallucination": None if score < 0 else int(score <= 2),
                    "content": content,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done.add((method, source_index))
                row_cache[(method, source_index)] = row
                print(
                    f"[{method} {offset + 1}/{total}] score={score} hallucination={row['hallucination']}",
                    flush=True,
                )
                time.sleep(0.2)

    rows = load_jsonl(args.output)
    selected_rows = [
        row
        for row in rows
        if args.start_index <= row.get("source_index", -1) < args.start_index + total
        and row.get("method") in {"baseline", "ours"}
    ]
    summary = summarize(selected_rows, total)
    summary.update(
        {
            "judge_model": args.judge_model,
            "base_url": args.base_url,
            "baseline_response": str(args.baseline_response),
            "ours_response": str(args.ours_response),
            "output": str(args.output),
        }
    )
    save_json(args.summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
