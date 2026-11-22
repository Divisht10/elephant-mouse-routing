"""
Equilibrium gap of a joint policy in the Markov game.

The four greedy policies are frozen. For each agent i,
the other three policies define an MDP that policy
iteration solves exactly; agent i's gap is how much
value it gives up by playing its own policy instead
of the best response:

    gap_i(s) = V_i^BR(s) - V_i^pi(s) >= 0.

A joint policy is a Markov-perfect equilibrium iff
every gap_i(s) is zero. Gaps are always measured with
the selfish reward (negative network cost minus the
switching cost), whatever reward the agents trained on.
"""

from statistics import mean

from environment import GAMMA
from policy_iteration import PolicyIterationSolver, STATES

NUM_AGENTS = 4
ZERO_Q = (0.0, 0.0, 0.0)


def policy_table(agent):
    """
    Greedy route of an agent in every state.

    Reads Q-tables without inserting unseen states, using
    the same first-maximum tie-breaking as greedy_action.
    """

    if not hasattr(agent, "Q"):
        return [agent.greedy_action(s) for s in STATES]

    table = []

    for state in STATES:
        values = list(agent.Q.get(state, ZERO_Q))
        table.append(values.index(max(values)))

    return table


def equilibrium_gap(policy_tables, gamma=GAMMA):
    """
    Best-response gap of each agent against the other
    three frozen policies.

    Returns one dict per agent:
        mean_gap: gap averaged over the uniform
                  initial-state distribution
        max_gap:  largest gap over all 162 states
    """

    results = []

    for i in range(NUM_AGENTS):
        solver = PolicyIterationSolver(
            i,
            opponent_policies=policy_tables,
            gamma=gamma
        )

        best = solver.solve()["values"]
        own, _ = solver.evaluate_policy(policy_tables[i])

        gaps = [b - o for b, o in zip(best, own)]

        results.append({
            "mean_gap": mean(gaps),
            "max_gap": max(gaps)
        })

    return results


def gap_metrics(agents, gamma=GAMMA):
    """Flat metric dictionary used by the evaluations."""

    tables = [policy_table(agent) for agent in agents]
    gaps = equilibrium_gap(tables, gamma)

    worst = max(g["mean_gap"] for g in gaps)

    metrics = {
        "equilibrium_gap": worst,
        # Values scale like 1/(1 - gamma); this version is
        # comparable across discount factors.
        "equilibrium_gap_per_step": worst * (1 - gamma),
        "equilibrium_gap_max_state": max(
            g["max_gap"] for g in gaps
        )
    }

    for i, g in enumerate(gaps):
        metrics[f"equilibrium_gap_agent_{i + 1}"] = g["mean_gap"]

    return metrics
