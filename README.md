# Anonymous Supplementary Package

This repository contains anonymized code, generated outputs, and evaluation summaries for reproducing and inspecting the main experimental results.

## Contents

- `paper/`: LaTeX source and bibliography snapshot.
- `code/`: generation and evaluation scripts used for COCO-CHAIR, MMHal-Bench, and LLaVA-Bench experiments.
- `requirements.txt` and `ENVIRONMENT.md`: dependency and checkpoint setup notes.
- `results/chair/`: COCO-CHAIR generation/evaluation JSON files for LLaVA-NeXT and InternVL.
- `results/ablations/`: branch ablations, sparse-budget ablations, schedule diagnostics, and trace diagnostics.
- `results/trigger_selection/`: InternVL trigger-policy diagnostics, including object-candidate, online-entropy, and hindsight entropy top-8 runs.
- `results/mmhal/`: MMHal-Bench responses and judge summary files.
- `results/llava_bench/`: LLaVA-Bench judge outputs.
- `results/qwen_boundary/`: Qwen-family boundary diagnostic summaries, including the Qwen2.5-VL prior diagnostic used in the appendix.

Public benchmark images and annotations are not included. Reviewers should obtain COCO, MMHal-Bench, and LLaVA-Bench from their official sources.

## Reproduction Scope

The uploaded JSON files support table-level verification without rerunning all LVLM generations. They include generated responses, CHAIR evaluation outputs, MMHal-Bench response files, and judge summaries used by the paper.

To rerun generation or judge scripts, replace placeholder values such as `<PATH_TO_COCO>`, `<PATH_TO_LLAVA_NEXT>`, `<PATH_TO_INTERNVL3_5_8B>`, `<OPENAI_COMPATIBLE_BASE_URL>`, and `<YOUR_API_KEY>` with local paths or credentials. See `ENVIRONMENT.md` for runtime notes.

## Evaluation Notes

The main paper uses GLM-4.5-Air as the primary open-ended judge. OpenAI-compatible judge results are included only as judge-sensitivity diagnostics. Because automatic judge scores can shift across models and prompts, judge-specific results should be compared only within the same judge setting.

COCO-CHAIR results are reported from the saved JSON files under `results/chair/`.
Trigger-selection appendix results are reported from `results/trigger_selection/`.
MMHal-Bench and LLaVA-Bench results are reported from the saved response and judge summary files under `results/mmhal/` and `results/llava_bench/`.

## Anonymization

The package does not include private API keys, private base URLs, local absolute paths, raw benchmark images, or private model mirrors.
