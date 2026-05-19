# Environment Notes

This package is intended for reproducibility inspection rather than as a fully pinned Docker image. The exact model checkpoints are large public or public-facing LVLM checkpoints and are not included in the supplementary package.

## Recommended Runtime

- Python: 3.10 or 3.11
- GPU: one NVIDIA GPU with at least 24 GB memory for the 7B/8B experiments
- CUDA/PyTorch: use a PyTorch build compatible with the local CUDA driver
- Install common dependencies with:

```bash
pip install -r requirements.txt
```

Depending on the local model implementation, additional optional packages may be required by the checkpoint provider.

## Backbones

### LLaVA-NeXT

Main LLaVA-NeXT experiments use a LLaVA-1.6 / LLaVA-NeXT Mistral-7B style Hugging Face checkpoint.

Set the model path in scripts where `<PATH_TO_LLAVA_NEXT>` appears, or replace it with an environment variable in your local copy.

Relevant scripts:

- `code/llava16_native_threebranch_3seeds.py`
- `code/llava16_threebranch_fixed_global_k10.py`
- `code/llava16_threebranch_stratified_k10_3seeds.py`
- `code/llava16_vcd_adapted.py`
- `code/llava16_pai_adapted.py`
- `code/run_mmhal_llava_next_baseline.py`
- `code/run_mmhal_llava_next_threebranch.py`
- `code/run_llava_bench_llava_next_threebranch.py`

### InternVL

Main InternVL experiments use an InternVL3.5-8B style checkpoint.

Set `<PATH_TO_INTERNVL3_5_8B>` locally before running InternVL scripts.

Relevant scripts:

- `code/internvl_threebranch_aggressive_coco.py`
- `code/run_mmhal_internvl.py`
- `code/run_llava_bench_internvl_threebranch.py`
- `code/alc_github_release/experiments/internvl_coco_native_loop.py`
- `code/alc_github_release/experiments/internvl_mmhal_native_loop.py`

### Qwen-Family Diagnostics

Qwen-family experiments are included as boundary diagnostics in the result summaries. The uploaded package includes summary artifacts under `results/qwen_boundary/`, but not a full Qwen reproduction script set. Reproducing those runs requires the corresponding Qwen-VL checkpoint and any model-specific processor utilities required by that checkpoint release.

## Datasets

Datasets are not redistributed in this package.

- COCO-CHAIR experiments require MSCOCO images and annotations. Replace `<PATH_TO_COCO>` and `<PATH_TO_COCO_500_JSONL>` in the scripts.
- MMHal-Bench experiments require the public MMHal-Bench images and response template.
- LLaVA-Bench in-the-wild experiments require the public LLaVA-Bench questions/images.

## API Judges

Open-ended judge scripts use API-compatible judge models. Set credentials through environment variables:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export OPENAI_BASE_URL=<OPENAI_COMPATIBLE_BASE_URL>
export ZHIPUAI_API_KEY=<YOUR_ZHIPU_API_KEY>
```

Do not commit API keys or private base URLs.

## Notes on Exact Reproduction

The uploaded JSON files allow table-level verification without rerunning model generation. Re-running generation may still produce small differences if checkpoint revisions, CUDA kernels, or judge models differ. For this reason, the paper reports fixed prompts, fixed seeds for sparse schedules, and judge-specific summaries.

