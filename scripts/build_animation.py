"""
Build the interactive elephant/mouse routing replay.

Retrains the Q-learning runs of cooperation.py (seed 42,
selfish / team / tolled); the callback only reads the
agents, so training is unchanged. Every CHECKPOINT
episodes the greedy policies are frozen and replayed for
REPLAY_STEPS steps from the same start state and the same
background-load sequence, so differences between
snapshots come from learning alone.

The page does no modelling of its own: every cost, load,
delay and throughput it shows is looked up in a table
computed here with network.simulate_network, and the
social optimum and Nash equilibria come from
game_theory.py.

Writes results/routing_animation.html (self-contained,
opens offline) from scripts/animation_template.html.

    python scripts/build_animation.py
"""

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cooperation import build_toll_cache, train_cooperation
from environment import BETA_VALUES, SWITCHING_COST, RoutingEnvironment
from equilibrium_gap import equilibrium_gap, policy_table
from game_theory import calculate_game, find_nash_equilibria, find_social_optimum
from network import (
    AGENT_ALPHA,
    AGENT_DEMAND,
    ROUTE_CAPACITY,
    ROUTE_DELAY,
    simulate_network
)
from policy_iteration import STATES
from rewards import REWARD_MODES

SEED = 42
CHECKPOINT = 100
REPLAY_SEED = 3
REPLAY_STEPS = 60

TEMPLATE = ROOT / "scripts" / "animation_template.html"
OUTPUT = ROOT / "results" / "routing_animation.html"

PROFILES = list(product(range(3), repeat=4))   # index = a1*27+a2*9+a3*3+a4
PROFILE_INDEX = {p: i for i, p in enumerate(PROFILES)}
STATE_INDEX = {s: i for i, s in enumerate(STATES)}


def code(beta_state, actions):
    """One integer per step: background level and joint route."""
    return beta_state * 81 + PROFILE_INDEX[tuple(actions)]


def network_table():
    """[beta][profile] -> [social cost, 4 costs, 3 loads, 3 delays, 3 throughputs]."""
    table = []

    for beta in BETA_VALUES:
        rows = []

        for profile in PROFILES:
            r = simulate_network(profile, beta)
            rows.append([round(v, 5) for v in (
                r.social_cost, *r.agent_costs,
                *r.loads, *r.delays, *r.throughput
            )])

        table.append(rows)

    return table


def equilibria():
    out = {"opt": [], "so": [], "ne": []}

    for beta in BETA_VALUES:
        game = calculate_game(beta)
        optimal, cost = find_social_optimum(game)
        out["opt"].append(round(cost, 5))
        out["so"].append([PROFILE_INDEX[p] for p in optimal])
        out["ne"].append([PROFILE_INDEX[p] for p in find_nash_equilibria(game)])

    return out


def replay(tables):
    """Greedy joint routes from the fixed start state."""

    env = RoutingEnvironment(seed=REPLAY_SEED)
    state = env.reset()
    start = PROFILE_INDEX[state[1]]
    steps = []

    for _ in range(REPLAY_STEPS):
        actions = tuple(t[STATE_INDEX[state]] for t in tables)
        steps.append(code(state[0], actions))
        state, _, _ = env.step(actions)

    return start, steps


def record(mode, toll_cache):
    snapshots = []
    start = None

    def callback(episode, agents):
        nonlocal start

        if episode % CHECKPOINT:
            return

        tables = [policy_table(agent) for agent in agents]
        gap = max(g["mean_gap"] for g in equilibrium_gap(tables))
        start, steps = replay(tables)

        snapshots.append({
            "ep": episode,
            "gap": round(gap, 4),
            "steps": steps
        })

    train_cooperation(
        mode=mode, seed=SEED, toll_cache=toll_cache, callback=callback
    )

    print(f"{mode}: {len(snapshots)} snapshots, final gap {snapshots[-1]['gap']}")

    return start, snapshots


def main():
    toll_cache = build_toll_cache()
    runs = {mode: record(mode, toll_cache) for mode in REWARD_MODES}

    starts = {start for start, _ in runs.values()}
    assert len(starts) == 1, "replays must share one start state"

    loads = [s // 81 for s in runs["selfish"][1][0]["steps"]]
    print(f"Replay background: {loads.count(0)} low-load steps, "
          f"{loads.count(1)} high-load steps")

    data = {
        "seed": SEED,
        "replaySeed": REPLAY_SEED,
        "routes": [
            {"cap": c, "tau": t} for c, t in zip(ROUTE_CAPACITY, ROUTE_DELAY)
        ],
        "agents": [
            {"demand": d, "alpha": a} for d, a in zip(AGENT_DEMAND, AGENT_ALPHA)
        ],
        "beta": list(BETA_VALUES),
        "kappa": SWITCHING_COST,
        "table": network_table(),
        **equilibria(),
        "start": starts.pop(),
        "modes": {mode: snaps for mode, (_, snaps) in runs.items()}
    }

    page = TEMPLATE.read_text().replace(
        "/*DATA*/null", json.dumps(data, separators=(",", ":"))
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page)
    print("Saved:", OUTPUT, f"({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
