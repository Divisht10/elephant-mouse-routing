import random
import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from policy_iteration import PolicyIterationSolver, STATES, FIXED_ROUTES
from equilibrium_gap import equilibrium_gap, policy_table, gap_metrics
from q_learning import QLearningAgent


def random_tables(seed):
    rng = random.Random(seed)
    return [
        [rng.randrange(3) for _ in STATES] for _ in range(4)
    ]


class TestPolicyOpponents(unittest.TestCase):

    def test_constant_tables_match_fixed_routes(self):
        tables = [[r] * len(STATES) for r in FIXED_ROUTES]

        for agent in range(4):
            old = PolicyIterationSolver(agent).solve()
            new = PolicyIterationSolver(
                agent, opponent_policies=tables
            ).solve()

            for a, b in zip(old["values"], new["values"]):
                self.assertAlmostEqual(a, b, places=9)

    def test_invalid_table(self):
        tables = random_tables(0)
        tables[1] = tables[1][:-1]

        with self.assertRaises(ValueError):
            PolicyIterationSolver(0, opponent_policies=tables)

    def test_gamma_validation(self):
        with self.assertRaises(ValueError):
            PolicyIterationSolver(0, gamma=1.0)


class TestEquilibriumGap(unittest.TestCase):

    def test_nonnegative(self):
        for g in equilibrium_gap(random_tables(1)):
            self.assertGreaterEqual(g["mean_gap"], -1e-9)
            self.assertGreaterEqual(g["max_gap"], g["mean_gap"] - 1e-9)

    def test_best_response_has_zero_gap(self):
        tables = random_tables(2)

        for agent in range(4):
            best = PolicyIterationSolver(
                agent, opponent_policies=tables
            ).solve()["policy"]

            trial = list(tables)
            trial[agent] = best

            gap = equilibrium_gap(trial)[agent]
            self.assertAlmostEqual(gap["max_gap"], 0.0, places=8)

    def test_policy_table_does_not_insert_states(self):
        agent = QLearningAgent(agent_id=0)
        agent.Q[STATES[5]] = [0.0, 2.0, 1.0]

        table = policy_table(agent)

        self.assertEqual(len(agent.Q), 1)
        self.assertEqual(table[5], 1)
        self.assertEqual(table[0], 0)

    def test_gap_metrics_keys(self):
        agents = [QLearningAgent(agent_id=i) for i in range(4)]
        metrics = gap_metrics(agents, gamma=0.5)

        self.assertAlmostEqual(
            metrics["equilibrium_gap_per_step"],
            0.5 * metrics["equilibrium_gap"]
        )

        for i in range(1, 5):
            self.assertIn(f"equilibrium_gap_agent_{i}", metrics)


if __name__ == "__main__":
    unittest.main()
