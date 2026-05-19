# Anonymous Supplementary Package

This directory contains the anonymized code and result artifacts needed to inspect and reproduce the main reported numbers.

## Contents

- `paper/`: LaTeX source and bibliography for reference.
- `code/`: generation and evaluation scripts used for COCO-CHAIR, MMHal-Bench, and LLaVA-Bench experiments.
- `requirements.txt` and `ENVIRONMENT.md`: dependency and checkpoint setup notes.
- `results/chair/`: COCO-CHAIR generation/evaluation JSON files for LLaVA-NeXT and InternVL.
- `results/ablations/`: branch ablations, sparse-budget ablations, schedule diagnostics, and trace diagnostics.
- `results/mmhal/`: MMHal-Bench responses and judge summary files.
- `results/llava_bench/`: LLaVA-Bench judge outputs.
- `results/qwen_boundary/`: Qwen-family boundary diagnostic summaries.

Public benchmark images and annotations are not included. Reviewers should obtain COCO, MMHal-Bench, and LLaVA-Bench from their official sources.

## Privacy and Anonymization

The files in this package have been scrubbed of private API keys, private base URLs, and local absolute paths. Placeholder values such as `<PATH_TO_COCO>`, `<PATH_TO_LLAVA_NEXT>`, `<PATH_TO_INTERNVL3_5_8B>`, `<OPENAI_COMPATIBLE_BASE_URL>`, and `<YOUR_API_KEY>` must be replaced locally before rerunning generation or judge scripts.

Do not submit private API keys, local user paths, private model mirrors, or raw benchmark image folders.

## What to Submit

For a double-blind paper submission, the safest supplementary material is either:

1. A zipped copy of this `upload/` directory, or
2. A clean anonymous repository containing the same contents.

The JSON response files are useful because they allow reviewers to inspect the actual model outputs and recompute metrics without rerunning expensive LVLM generation. The code is useful for reproducing the generation and evaluation pipeline.

## Anonymous Repository Link

One common workflow is:

1. Create a fresh repository containing only the contents of this `upload/` directory.
2. Do not include git history from the development repository.
3. Push the fresh repository to GitHub.
4. Use Anonymous GitHub at `https://anonymous.4open.science/` to generate a double-blind review link.
5. Put only the anonymous `anonymous.4open.science` link in the paper or supplementary material.

Before sharing the link, open it in a private browser window and check that no author names, usernames, institution names, API keys, or local filesystem paths are visible.

## Reproduction Notes

The uploaded results are intended to support table-level verification. The main paper uses GLM-4.5-Air as the primary open-ended judge and reports OpenAI-compatible judges only as judge-sensitivity diagnostics. Because automatic judge scores can shift across models and prompts, judge-specific results should be compared only within the same judge setting.
