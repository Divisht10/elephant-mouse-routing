"""
Discount-factor and hyperparameter sweeps (proposal,
Section 5) for four simultaneous Q-learners under the
selfish, team and tolled rewards.

gamma sweep:   gamma in GAMMAS, other settings default.
hyper sweep:   one setting changed at a time from the
               defaults (learning rate 0.1, epsilon
               decay 0.997, gamma 0.9).

The equilibrium gap is measured with the same gamma the
agents learned with. Runs are independent, so they are
spread over all CPU cores.

    python scripts/sweep_experiments.py          # full
    python scripts/sweep_experiments.py --quick  # smoke test
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cooperation import (
    SEEDS,
    EPISODES,
    STEPS_PER_EPISODE,
    build_toll_cache,
    train_cooperation,
    evaluate_cooperation
)
from multi_agent import build_reference_games
from rewards import REWARD_MODES
from environment import GAMMA

DATA = ROOT / "results" / "data"

GAMMAS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)

HYPER_SETTINGS = (
    ("learning_rate", 0.05),
    ("learning_rate", 0.1),   # default
    ("learning_rate", 0.2),
    ("epsilon_decay", 0.995),
    ("epsilon_decay", 0.999),
)

_CACHE = {}


def run_one(job):
    """Train and evaluate one configuration."""

    if not _CACHE:
        _CACHE["reference"] = build_reference_games()
        _CACHE["tolls"] = build_toll_cache()

    settings = job["settings"]

    agents, _ = train_cooperation(
        mode=job["mode"],
        seed=job["seed"],
        episodes=job["episodes"],
        steps_per_episode=STEPS_PER_EPISODE,
        toll_cache=_CACHE["tolls"],
        agent_kwargs=settings
    )

    metrics = evaluate_cooperation(
        agents,
        seed=job["seed"] + 2000,
        episodes=job["eval_episodes"],
        reference=_CACHE["reference"],
        toll_cache=_CACHE["tolls"],
        gamma=settings.get("gamma", GAMMA)
    )

    return {
        "sweep": job["sweep"],
        "parameter": job["parameter"],
        "value": job["value"],
        "mode": job["mode"],
        "seed": job["seed"],
        **metrics
    }


def make_jobs(quick):
    seeds = SEEDS[:2] if quick else SEEDS
    episodes = 200 if quick else EPISODES
    eval_episodes = 2 if quick else 20
    gammas = (0.5, 0.99) if quick else GAMMAS
    hyper = HYPER_SETTINGS[:1] if quick else HYPER_SETTINGS

    configs = [
        ("gamma", "gamma", g, {"gamma": g}) for g in gammas
    ] + [
        ("hyper", name, value, {name: value})
        for name, value in hyper
    ]

    return [
        {
            "sweep": sweep,
            "parameter": parameter,
            "value": value,
            "settings": settings,
            "mode": mode,
            "seed": seed,
            "episodes": episodes,
            "eval_episodes": eval_episodes
        }
        for sweep, parameter, value, settings in configs
        for mode in REWARD_MODES
        for seed in seeds
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    jobs = make_jobs(args.quick)
    print(f"{len(jobs)} runs on {args.workers} workers")

    rows = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, row in enumerate(pool.map(run_one, jobs), 1):
            rows.append(row)
            print(
                f"[{k}/{len(jobs)}] {row['parameter']}={row['value']} "
                f"{row['mode']} seed {row['seed']}: "
                f"efficiency {row['mean_efficiency_ratio']:.4f}, "
                f"gap {row['equilibrium_gap']:.4f}"
            )

    DATA.mkdir(parents=True, exist_ok=True)
    name = "sweep_quick.csv" if args.quick else "sweep_results.csv"
    pd.DataFrame(rows).to_csv(DATA / name, index=False)
    print("Saved:", DATA / name)


if __name__ == "__main__":
    main()
