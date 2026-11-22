"""
Trigger strategies in the full Markov game.

repeated_game.py and minmax_punishment.py compute the
threshold of Eq. (7) for each background level as if
that level never changed, and without the switching
cost. Here both are included.

Opponents of agent i play the social optimum while
nobody has deviated; once agent i deviates they punish
it forever with a punishment profile that may depend
on the current background level. Agent i's problem is
then an MDP over

    (background level, phase in {coop, punish},
     agent i's previous route)                 12 states

with reward -c_i(a, beta) - kappa 1{a != previous}.
Policy iteration gives agent i's best response.
Cooperation is self-enforcing for i at discount gamma
iff always cooperating attains that optimum from every
cooperative state, i.e. the deviation gain is zero.

Punishments
    minmax: opponents play agent i's pure minmax profile
            at the current level (a benchmark: not
            necessarily credible).
    nash:   everyone reverts to the pure stage-game NE
            that is worst for agent i at the current
            level.
"""

import csv
from pathlib import Path

import numpy as np

from environment import (
    BETA_VALUES,
    TRANSITION_MATRIX,
    SWITCHING_COST
)
from game_theory import (
    calculate_game,
    find_nash_equilibria,
    find_social_optimum
)
from minmax_punishment import calculate_pure_minmax, make_profile

NUM_AGENTS = 4
NUM_ROUTES = 3
TOL = 1e-9

GAMMA_GRID = np.round(np.arange(0.0, 0.9995, 0.001), 3)

# Model switching cost, and zero as in Eq. (7).
KAPPAS = (SWITCHING_COST, 0.0)
RESULTS_DIR = Path("results/data")
TABLES_DIR = Path("results/tables")


def stage_profiles(kind, agent):
    """Cooperation and punishment profile at each level."""

    cooperation = {}
    punishment = {}

    for b, beta in enumerate(BETA_VALUES):
        game = calculate_game(beta)
        cooperation[b] = find_social_optimum(game)[0][0]

        if kind == "minmax":
            worst = calculate_pure_minmax(game, agent)
            opponents = (
                worst["punishment_profiles"][0]["opponent_actions"]
            )
            punishment[b] = make_profile(agent, 0, opponents)

        elif kind == "nash":
            punishment[b] = max(
                find_nash_equilibria(game),
                key=lambda a: game[a]["agent_costs"][agent]
            )

        else:
            raise ValueError(f"Unknown punishment: {kind}")

    return cooperation, punishment


def state_index(b, phase, previous):
    return (b * 2 + phase) * NUM_ROUTES + previous


def build_mdp(agent, cooperation, punishment, transition, kappa):
    """Rewards R[s, a] and transitions P[s, a, s']."""

    n = 2 * 2 * NUM_ROUTES
    R = np.zeros((n, NUM_ROUTES))
    P = np.zeros((n, NUM_ROUTES, n))
    games = {b: calculate_game(beta) for b, beta in enumerate(BETA_VALUES)}

    for b in range(2):
        for phase in range(2):
            others = cooperation[b] if phase == 0 else punishment[b]

            for previous in range(NUM_ROUTES):
                s = state_index(b, phase, previous)

                for a in range(NUM_ROUTES):
                    profile = list(others)
                    profile[agent] = a
                    cost = games[b][tuple(profile)]["agent_costs"][agent]

                    R[s, a] = -cost - kappa * (a != previous)

                    deviated = phase == 1 or a != cooperation[b][agent]
                    next_phase = 1 if deviated else 0

                    for b_next in range(2):
                        P[s, a, state_index(b_next, next_phase, a)] += (
                            transition[b][b_next]
                        )

    return R, P


def evaluate(R, P, policy, gamma):
    n = len(policy)
    rows = np.arange(n)

    return np.linalg.solve(
        np.eye(n) - gamma * P[rows, policy],
        R[rows, policy]
    )


def optimal_values(R, P, gamma):
    """Exact policy iteration on the small MDP."""

    n = R.shape[0]
    policy = np.zeros(n, dtype=int)

    while True:
        V = evaluate(R, P, policy, gamma)
        Q = R + gamma * P @ V
        best = Q.argmax(axis=1)

        improve = Q[np.arange(n), best] > (
            Q[np.arange(n), policy] + 1e-12
        )

        if not improve.any():
            return V

        policy = np.where(improve, best, policy)


def deviation_gains(
    agent,
    gammas,
    kind,
    transition=TRANSITION_MATRIX,
    kappa=SWITCHING_COST
):
    """
    For each gamma, the largest value agent i gains by
    deviating, over the cooperative states in which it
    is already on its cooperative route. Zero means
    cooperation is a best response.
    """

    cooperation, punishment = stage_profiles(kind, agent)
    R, P = build_mdp(agent, cooperation, punishment, transition, kappa)

    # Always cooperate: play the cooperative route in every
    # state (punish states are never reached from coop).
    cooperate = np.array([
        cooperation[b][agent]
        for b in range(2)
        for _ in range(2)
        for _ in range(NUM_ROUTES)
    ])

    starts = [
        state_index(b, 0, cooperation[b][agent]) for b in range(2)
    ]

    gains = []

    for gamma in gammas:
        V_coop = evaluate(R, P, cooperate, gamma)
        V_best = optimal_values(R, P, gamma)
        gains.append(max(V_best[s] - V_coop[s] for s in starts))

    return gains


def threshold(gains):
    """
    Smallest grid gamma from which cooperation holds for
    every larger grid gamma (None if it never does), and
    whether the sustainable set is an upper interval.
    """

    sustainable = [g <= TOL for g in gains]

    if not sustainable[-1]:
        return None, all(not s for s in sustainable)

    k = len(sustainable) - 1

    while k > 0 and sustainable[k - 1]:
        k -= 1

    monotone = not any(sustainable[:k])

    return float(GAMMA_GRID[k]), monotone


def main():
    grid_rows = []
    summary_rows = []

    for kappa in KAPPAS:
        for kind in ("minmax", "nash"):
            agent_thresholds = []

            for agent in range(NUM_AGENTS):
                gains = deviation_gains(
                    agent, GAMMA_GRID, kind, kappa=kappa
                )

                for gamma, gain in zip(GAMMA_GRID, gains):
                    grid_rows.append({
                        "kappa": kappa,
                        "punishment": kind,
                        "agent": agent + 1,
                        "gamma": float(gamma),
                        "deviation_gain": gain,
                        "sustainable": gain <= TOL
                    })

                value, monotone = threshold(gains)
                agent_thresholds.append(value)

                summary_rows.append({
                    "kappa": kappa,
                    "punishment": kind,
                    "agent": agent + 1,
                    "gamma_threshold": value,
                    "monotone_in_gamma": monotone,
                    "sustainable_at_0_9": (
                        gains[list(GAMMA_GRID).index(0.9)] <= TOL
                    )
                })

            overall = (
                None if None in agent_thresholds
                else max(agent_thresholds)
            )

            summary_rows.append({
                "kappa": kappa,
                "punishment": kind,
                "agent": "all",
                "gamma_threshold": overall,
                "monotone_in_gamma": None,
                "sustainable_at_0_9": (
                    overall is not None and overall <= 0.9
                )
            })

            print(
                f"kappa={kappa} {kind:6s}: per agent "
                f"{agent_thresholds} -> all {overall}"
            )

    for path, rows in (
        (RESULTS_DIR / "markov_trigger_grid.csv", grid_rows),
        (TABLES_DIR / "markov_trigger_summary.csv", summary_rows)
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print("Saved:", path)

if __name__ == "__main__":
    main()
