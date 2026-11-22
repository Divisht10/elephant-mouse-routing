
import csv
from itertools import product
from pathlib import Path

import numpy as np
import pygambit as gbt

from game_theory import calculate_game

ACTIONS = list(product(range(3), repeat=4))


def create_gambit_game(beta):
    """Convert the routing cost game into Gambit payoffs."""

    game_data = calculate_game(beta)

    # Gambit maximizes payoffs, so payoff = -cost.
    payoffs = [
        np.zeros((3, 3, 3, 3))
        for _ in range(4)
    ]

    for actions, result in game_data.items():
        for agent in range(4):
            payoffs[agent][actions] = (
                -result["agent_costs"][agent]
            )

    game = gbt.Game.from_arrays(
        *payoffs,
        title=f"Routing Game beta={beta}"
    )

    return game, game_data



def extract_probabilities(profile):
    """Extract route probabilities for all four agents."""

    probabilities = []

    for player, mixed in profile.mixed_strategies():
        strategy_probs = np.array(
            [float(mixed[s]) for s in player.strategies],
            dtype=float
        )

        probabilities.append(strategy_probs)

    return probabilities



def expected_cost(agent, route, probabilities, game_data):
    """Cost of choosing one route against mixed opponents."""

    total = 0.0

    for actions in ACTIONS:
        if actions[agent] != route:
            continue

        probability = 1.0

        for other in range(4):
            if other != agent:
                probability *= probabilities[other][actions[other]]

        total += (
            probability
            * game_data[actions]["agent_costs"][agent]
        )

    return total


def maximum_regret(probabilities, game_data):
    """Verify whether a mixed profile is approximately NE."""

    regrets = []

    for agent in range(4):
        costs = np.array([
            expected_cost(agent, route, probabilities, game_data)
            for route in range(3)
        ])

        actual = np.dot(probabilities[agent], costs)
        best = np.min(costs)

        regrets.append(max(0.0, actual - best))

    return max(regrets)


def expected_social_cost(probabilities, game_data):
    """Social cost averaged over the mixed profile."""

    total = 0.0

    for actions in ACTIONS:
        probability = 1.0

        for agent in range(4):
            probability *= probabilities[agent][actions[agent]]

        total += probability * game_data[actions]["social_cost"]

    return total


def unique_equilibria(equilibria, decimals=6):
    """Drop numerically identical solutions."""

    seen = set()
    unique = []

    for probabilities in equilibria:
        key = tuple(
            round(float(p), decimals) + 0.0
            for strategy in probabilities
            for p in strategy
        )

        if key not in seen:
            seen.add(key)
            unique.append(probabilities)

    return unique


def analyze_mixed_game(beta, max_solutions=None, max_regret=1e-6):
    """
    Enumerate the Nash equilibria of the stage game.

    max_solutions=None runs the support enumeration to
    completion. Each solution is checked independently
    (maximum regret); duplicates are removed. enumpoly
    finds isolated equilibria; a continuum of equilibria,
    if one existed, would not be listed in full.
    """

    gambit_game, game_data = create_gambit_game(beta)

    result = gbt.nash.enumpoly_solve(
        gambit_game,
        use_strategic=True,
        stop_after=max_solutions
    )

    solutions = unique_equilibria([
        extract_probabilities(eq) for eq in result.equilibria
    ])

    print(f"\nBackground load: {beta:.0%}")
    print(f"Solutions returned: {len(result.equilibria)}")
    print(f"Distinct solutions: {len(solutions)}")

    rows = []

    for index, probabilities in enumerate(solutions, 1):
        regret = maximum_regret(probabilities, game_data)

        is_genuinely_mixed = any(
            sum(p > 1e-6 for p in strategy) > 1
            for strategy in probabilities
        )

        if regret > max_regret:
            print(
                f" WARNING: solution {index} failed the "
                f"regret check ({regret:.2e}); dropped."
            )
            continue

        rows.append({
            "beta": beta,
            "equilibrium": len(rows) + 1,
            "genuinely_mixed": is_genuinely_mixed,
            "expected_social_cost": expected_social_cost(
                probabilities, game_data
            ),
            "max_regret": regret,
            **{
                f"agent_{agent + 1}_R{route + 1}": round(
                    float(probabilities[agent][route]), 8
                )
                for agent in range(4)
                for route in range(3)
            }
        })

    mixed = [r for r in rows if r["genuinely_mixed"]]

    print(f"Verified equilibria: {len(rows)}")
    print(f"  pure: {len(rows) - len(mixed)}")
    print(f"  genuinely mixed: {len(mixed)}")

    if max_solutions is not None and (
        len(result.equilibria) >= max_solutions
    ):
        print("Search stopped at the requested solution limit.")

    return rows


def summarize(rows, beta):
    """Price of anarchy/stability over all equilibria."""

    game_data = calculate_game(beta)
    optimum = min(r["social_cost"] for r in game_data.values())
    costs = [r["expected_social_cost"] for r in rows]
    mixed = [r for r in rows if r["genuinely_mixed"]]

    return {
        "background_load": beta,
        "equilibria": len(rows),
        "pure": len(rows) - len(mixed),
        "genuinely_mixed": len(mixed),
        "social_optimum": optimum,
        "best_equilibrium_cost": min(costs),
        "worst_equilibrium_cost": max(costs),
        "poa_all_equilibria": max(costs) / optimum,
        "pos_all_equilibria": min(costs) / optimum
    }


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


if __name__ == "__main__":
    all_rows = []
    summary = []

    for beta in (0.10, 0.25):
        rows = analyze_mixed_game(beta)
        all_rows.extend(rows)
        summary.append(summarize(rows, beta))

    save_csv(Path("results/data/mixed_equilibria.csv"), all_rows)
    save_csv(
        Path("results/tables/mixed_equilibrium_summary.csv"),
        summary
    )
