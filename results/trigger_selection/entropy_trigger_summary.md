# InternVL Entropy Trigger Diagnostic

Date: 2026-05-20

This note records the fair entropy-trigger diagnostic for the SGR paper.
Both entropy variants use the same InternVL backbone, COCO 500-image subset,
SGR reranker, and CHAIR evaluation protocol as the main InternVL random sparse
setting. Only the trigger policy changes.

## Files

- Online entropy trigger output:
  - `intern/internvl_sgr_entropy_online_k8_seed20260423_warm5_lam1p0.json`
  - `intern/internvl_sgr_entropy_online_k8_seed20260423_warm5_lam1p0_stats.json`
  - `intern/internvl_sgr_entropy_online_k8_seed20260423_warm5_lam1p0_chair.json`
- Hindsight entropy top-8 diagnostic output:
  - `intern/internvl_sgr_entropy_oracle_k8_seed20260423.json`
  - `intern/internvl_sgr_entropy_oracle_k8_seed20260423_stats.json`
  - `intern/internvl_sgr_entropy_oracle_k8_seed20260423_chair.json`

## Trigger Policies

- Online entropy: trigger at step `t` when current entropy exceeds the running
  mean plus one running standard deviation, after a 5-token warmup, with budget
  `K=8`.
- Entropy top-8 oracle: first record a baseline entropy trace, then rerun SGR at
  the eight highest-entropy positions. This uses future information and should
  be described as a hindsight diagnostic, not a deployable trigger.

## Results

| Trigger | Seeds | CHAIRs | CHAIRi | F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Vanilla | -- | 33.4 | 8.71 | 77.1 | deterministic greedy |
| Random global | 3 | 32.8 +/- 0.5 | 8.23 +/- 0.10 | 76.8 +/- 0.1 | main InternVL SGR |
| Stratified | 3 | 33.8 +/- 0.2 | 8.30 +/- 0.07 | 76.8 +/- 0.3 | fixed budget |
| Object-candidate | 1 | 33.6 | 8.66 | 76.8 | simple object-like trigger |
| Online entropy | 1 | 35.8 | 9.94 | 75.3 | running threshold, online |
| Entropy top-8 oracle | 1 | 33.6 | 8.83 | 76.3 | hindsight diagnostic |

## Diagnostic Counts

| Trigger | Trigger hits | Effective interventions | Guard blocks | Interventions/img |
| --- | ---: | ---: | ---: | ---: |
| Online entropy | 3998 | 1988 | 11403 | 8.00 |
| Entropy top-8 oracle | 3891 | 1060 | 8312 | 7.78 |

## Interpretation

The online entropy trigger fires almost exactly at the full budget but degrades
CHAIR and F1. The hindsight entropy top-8 diagnostic, despite using future
entropy information, also does not outperform the random global sparse schedule.
This supports the paper's claim that high next-token uncertainty is not a
reliable proxy for visually consequential correction points, and that uniform
random sparse triggering is a strong low-assumption baseline.
