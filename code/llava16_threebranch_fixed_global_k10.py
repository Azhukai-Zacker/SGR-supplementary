# -*- coding: utf-8 -*-
import argparse

import llava16_native_threebranch_3seeds as base


DEFAULT_FIXED_STEPS = [25, 75, 125, 175, 225, 275, 325, 375, 425, 475]


def build_intervention_steps(
    schedule,
    k,
    max_steps,
    effective_max_step,
    early_start=8,
    early_end=56,
    early_interval=12,
):
    active_end = max(1, min(max_steps, effective_max_step))
    if schedule == "fixed_global":
        steps = [step for step in build_intervention_steps.fixed_steps if 0 <= step < active_end]
        return sorted(steps[:k])
    return base._original_build_intervention_steps(
        schedule,
        k,
        max_steps,
        effective_max_step,
        early_start=early_start,
        early_end=early_end,
        early_interval=early_interval,
    )


build_intervention_steps.fixed_steps = DEFAULT_FIXED_STEPS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA-NeXT threebranch soft fixed global probes k10 on COCO."
    )
    parser.add_argument("--model-path", type=str, default=base.MODEL_PATH)
    parser.add_argument("--question", type=str, default=base.QUESTION)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--effective-max-step", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--fixed-steps",
        type=int,
        nargs="+",
        default=DEFAULT_FIXED_STEPS,
        help="Fixed global intervention steps. Default: centers of ten 50-token buckets.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260423],
        help="One seed is enough for fixed probes; pass three seeds if you want a reproducibility sanity check.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--black-alpha", type=float, default=0.25)
    parser.add_argument("--apc-threshold", type=float, default=0.10)
    parser.add_argument("--black-prior-threshold", type=float, default=0.50)
    parser.add_argument("--black-visual-gap-threshold", type=float, default=0.30)
    parser.add_argument("--syntax-threshold", type=float, default=0.05)
    parser.add_argument("--syntax-margin", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--tag", type=str, default="threebranch_soft_fixed_global")
    args = parser.parse_args()

    args.mode = "three"
    args.schedule = "fixed_global"
    args.black_gate_mode = "soft"
    args.early_start = 8
    args.early_end = 56
    args.early_interval = 12
    build_intervention_steps.fixed_steps = args.fixed_steps
    return args


def main():
    if not hasattr(base, "_original_build_intervention_steps"):
        base._original_build_intervention_steps = base.build_intervention_steps
    base.build_intervention_steps = build_intervention_steps
    base.parse_args = parse_args
    base.main()


if __name__ == "__main__":
    main()
