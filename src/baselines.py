"""
Non-learning baselines from the proposal (Section 5).

random:        every agent picks a route uniformly at
               random at every step.
shortest_path: every agent always takes R1, the route
               with the smallest propagation delay,
               ignoring congestion.

Both are evaluated with exactly the same code and
evaluation seeds as the learned agents. The
equilibrium gap is computed for shortest-path only:
the gap code needs a deterministic policy table, and
uniform random routing is not one.
"""

import csv
import random
from pathlib import Path

from cooperation import (
    SEEDS,
    build_toll_cache,
    evaluate_cooperation
)
from multi_agent import build_reference_games

RESULTS_DIR = Path("results/data")


class RandomAgent:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def greedy_action(self, state):
        return self.rng.randrange(3)


class ShortestPathAgent:
    def greedy_action(self, state):
        return 0  # R1: 2 ms propagation delay


def make_baseline(name, seed):
    if name == "random":
        return [RandomAgent(seed + i) for i in range(4)]

    if name == "shortest_path":
        return [ShortestPathAgent() for _ in range(4)]

    raise ValueError(f"Unknown baseline: {name}")


def main():
    reference = build_reference_games()
    toll_cache = build_toll_cache()

    rows = []

    for name in ("random", "shortest_path"):
        for seed in SEEDS:
            metrics = evaluate_cooperation(
                make_baseline(name, seed),
                seed=seed + 2000,
                reference=reference,
                toll_cache=toll_cache,
                compute_gap=(name == "shortest_path")
            )

            rows.append({
                "baseline": name,
                "seed": seed,
                **metrics
            })

            print(name, seed, metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "baselines_evaluation.csv"

    # Shortest-path rows (last) carry every column;
    # random rows leave the gap columns empty.
    fields = list(rows[-1].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


if __name__ == "__main__":
    main()
