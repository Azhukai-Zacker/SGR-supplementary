# Manifest

## Figure

- `figures/figure1.png`: SGR workflow overview used in the manuscript.

## Code

- `requirements.txt`: lightweight Python dependency list.
- `ENVIRONMENT.md`: runtime, checkpoint, dataset, and API judge setup notes.
- `code/chair.py`: COCO-CHAIR evaluator.
- `code/llava16_native_threebranch_3seeds.py`: LLaVA-NeXT SGR generation across intervention seeds.
- `code/llava16_threebranch_fixed_global_k10.py`: LLaVA-NeXT fixed global sparse run.
- `code/llava16_threebranch_stratified_k10_3seeds.py`: LLaVA-NeXT stratified sparse schedule runs.
- `code/llava16_vcd_adapted.py`: LLaVA-NeXT adapted VCD-style baseline.
- `code/llava16_pai_adapted.py`: LLaVA-NeXT adapted PAI-style baseline.
- `code/internvl_threebranch_aggressive_coco.py`: InternVL sparse guarded generation.
- `code/alc_github_release/experiments/internvl_coco_native_loop.py`: InternVL native-loop COCO generation.
- `code/alc_github_release/experiments/internvl_mmhal_native_loop.py`: InternVL MMHal generation.
- `code/run_mmhal_llava_next_baseline.py`: LLaVA-NeXT MMHal baseline generation.
- `code/run_mmhal_llava_next_threebranch.py`: LLaVA-NeXT MMHal SGR generation.
- `code/run_mmhal_internvl.py`: InternVL MMHal generation.
- `code/run_llava_bench_llava_next_threebranch.py`: LLaVA-Bench LLaVA-NeXT generation.
- `code/run_llava_bench_internvl_threebranch.py`: LLaVA-Bench InternVL generation.
- `code/eval_llava_bench_openai_compat.py`: OpenAI-compatible LLaVA-Bench judge script.
- `code/eval_mmhal_pairwise_openai_compat.py`: OpenAI-compatible MMHal pairwise judge script.
- `code/eval_mmhal_internvl_openai_compat30.py`: OpenAI-compatible MMHal judge script.
- `code/internvl_entropy_trigger_coco.py`: shared InternVL entropy-trigger diagnostic implementation.
- `code/run_internvl_entropy_online_k8.py`: online entropy-trigger diagnostic launcher.
- `code/run_internvl_entropy_oracle_topk.py`: hindsight entropy top-8 diagnostic launcher.
- `code/qwen25vl_prior_diagnostic.py`: Qwen2.5-VL prior-branch diagnostic script.

## Results

- `results/chair/llava_next/`: LLaVA-NeXT COCO-CHAIR baseline, SGR, adapted VCD, and adapted PAI outputs.
- `results/chair/internvl/`: InternVL COCO-CHAIR baseline, SGR, adapted VCD, and adapted PAI outputs.
- `results/ablations/`: branch ablations, sparse-budget ablations, schedule ablations, and trace diagnostics.
- `results/trigger_selection/`: InternVL object-candidate and entropy-trigger diagnostic outputs.
- `results/mmhal/`: MMHal-Bench response JSONs and judge summaries.
- `results/llava_bench/`: LLaVA-Bench judge outputs.
- `results/qwen_boundary/`: Qwen-family boundary diagnostic summaries and Qwen2.5-VL prior diagnostic artifacts.
