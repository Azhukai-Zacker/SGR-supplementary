# Anonymous Supplementary Package

This repository contains anonymized code, generated outputs, and evaluation summaries for reproducing and inspecting the main experimental results.
The full manuscript is submitted through the conference review system; this repository intentionally keeps only supporting materials to avoid version mismatches.

## Contents

- `figures/method_overview.png`: compact overview of the SGR workflow.
- `code/`: generation and evaluation scripts used for COCO-CHAIR, MMHal-Bench, and LLaVA-Bench experiments.
- `requirements.txt` and `ENVIRONMENT.md`: dependency and checkpoint setup notes.
- `results/chair/`: COCO-CHAIR generation/evaluation JSON files for LLaVA-NeXT and InternVL.
- `results/ablations/`: branch ablations, sparse-budget ablations, schedule diagnostics, and trace diagnostics.
- `results/trigger_selection/`: InternVL trigger-policy diagnostics, including object-candidate, online-entropy, and hindsight entropy top-8 runs.
- `results/mmhal/`: MMHal-Bench responses and judge summary files.
- `results/llava_bench/`: LLaVA-Bench judge outputs.
- `results/qwen_boundary/`: Qwen-family boundary diagnostic summaries, including the Qwen2.5-VL prior diagnostic used in the appendix.

Public benchmark images and annotations are not included. Reviewers should obtain COCO, MMHal-Bench, and LLaVA-Bench from their official sources.
Generated outputs are preserved verbatim, including occasional multilingual tokens produced by the evaluated models.

## Reproduction Scope

The uploaded JSON files support table-level verification without rerunning all LVLM generations. They include generated responses, CHAIR evaluation outputs, MMHal-Bench response files, and judge summaries used by the paper.

To rerun generation or judge scripts, replace placeholder values such as `<PATH_TO_COCO>`, `<PATH_TO_LLAVA_NEXT>`, `<PATH_TO_INTERNVL3_5_8B>`, `<OPENAI_COMPATIBLE_BASE_URL>`, and `<YOUR_API_KEY>` with local paths or credentials. See `ENVIRONMENT.md` for runtime notes.

## Evaluation Notes

### COCO-CHAIR

Saved CHAIR outputs under `results/chair/` already contain per-example annotations and aggregate metrics. To recompute CHAIR from a saved output file, provide local MSCOCO annotations and run:

```bash
python code/chair.py \
  --cap_file results/chair/internvl/internvl_sparse_k8_seed20260423_chair.json \
  --image_id_key image_id \
  --caption_key caption \
  --coco_path <PATH_TO_COCO>/annotations \
  --cache chair.pkl \
  --save_path /tmp/recomputed_chair.json
```

The evaluator accepts either raw generation files with `image_id` and `caption` fields, COCO-style files with an `annotations` field, or previously saved CHAIR outputs with a `sentences` field.

### Open-Ended Judges

The main paper uses GLM-4.5-Air as the primary open-ended judge. OpenAI-compatible judge results are included only as judge-sensitivity diagnostics. Because automatic judge scores can shift across models and prompts, judge-specific results should be compared only within the same judge setting.

COCO-CHAIR results are reported from the saved JSON files under `results/chair/`.
Trigger-selection appendix results are reported from `results/trigger_selection/`.
MMHal-Bench and LLaVA-Bench results are reported from the saved response and judge summary files under `results/mmhal/` and `results/llava_bench/`.

## Anonymization

The package does not include private API keys, private base URLs, local absolute paths, raw benchmark images, or private model mirrors.
