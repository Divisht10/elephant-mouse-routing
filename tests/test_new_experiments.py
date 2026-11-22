"""Tests for baselines, Monte Carlo fixes, cycling metric,
mixed-equilibrium post-processing and the sequential table."""

import sys
from pathlib import Path
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from baselines import make_baseline
from cooperation import (
    build_toll_cache,
    evaluate_cooperation,
    policy_change_rate,
    train_cooperation
)
from game_theory import calculate_game
from mixed_equilibrium import expected_social_cost, unique_equilibria
from monte_carlo import MonteCarloAgent, update_steps
from multi_agent import build_reference_games
from sequential_game import analyze_sequential_game, summary_rows


class TestBaselines(unittest.TestCase):

    def test_shortest_path_uses_r1(self):
        agents = make_baseline("shortest_path", 0)
        self.assertTrue(all(a.greedy_action(None) == 0 for a in agents))

    def test_random_routes_valid(self):
        agent = make_baseline("random", 0)[0]
        actions = {agent.greedy_action(None) for _ in range(200)}
        self.assertEqual(actions, {0, 1, 2})

    def test_shortest_path_cost(self):
        metrics = evaluate_cooperation(
            make_baseline("shortest_path", 0),
            seed=1, episodes=2, steps_per_episode=50,
            reference=build_reference_games(),
            toll_cache=build_toll_cache()
        )

        expected = calculate_game(0.10)[(0, 0, 0, 0)]["social_cost"]
        self.assertAlmostEqual(metrics["low_load_cost"], expected)
        self.assertIn("equilibrium_gap", metrics)


class TestMonteCarloFixes(unittest.TestCase):

    def test_update_steps(self):
        # gamma = 0.9: 0.9^88 < 1e-4 <= 0.9^87
        self.assertEqual(update_steps(200), 112)
        self.assertEqual(update_steps(20), 20)

    def test_tail_steps_not_updated(self):
        agent = MonteCarloAgent(agent_id=0)
        s1 = (0, (0, 0, 0, 0))
        s2 = (0, (1, 0, 0, 0))

        agent.update_episode([(s1, 0, -1.0), (s2, 1, -2.0)], n_update=1)

        self.assertEqual(agent.N[s1][0], 1)
        self.assertNotIn(s2, agent.N)


class TestMultiAgentMonteCarlo(unittest.TestCase):

    def test_training_and_snapshots(self):
        agents, history = train_cooperation(
            mode="team", seed=1, episodes=200,
            steps_per_episode=5, algorithm="monte_carlo"
        )

        self.assertTrue(all(isinstance(a, MonteCarloAgent) for a in agents))
        self.assertIsNone(history[99]["policy_change_rate"])
        self.assertIsNotNone(history[199]["policy_change_rate"])
        self.assertEqual(history[0]["algorithm"], "monte_carlo")

    def test_policy_change_rate(self):
        old = [[0, 1], [2, 2]]
        new = [[0, 2], [2, 2]]
        self.assertAlmostEqual(policy_change_rate(old, new), 0.25)

    def test_agent_kwargs(self):
        agents, _ = train_cooperation(
            mode="selfish", seed=1, episodes=1, steps_per_episode=2,
            agent_kwargs={"gamma": 0.5, "learning_rate": 0.2}
        )
        self.assertEqual(agents[0].gamma, 0.5)
        self.assertEqual(agents[0].learning_rate, 0.2)


class TestMixedPostProcessing(unittest.TestCase):

    def test_unique(self):
        a = [[1.0, 0.0, 0.0]] * 4
        b = [[1.0 + 1e-9, 0.0, -0.0]] * 4
        c = [[0.0, 1.0, 0.0]] * 4
        self.assertEqual(len(unique_equilibria([a, b, c])), 2)

    def test_expected_cost_of_pure_profile(self):
        game = calculate_game(0.10)
        profile = (1, 2, 0, 0)
        probabilities = [
            [1.0 if r == a else 0.0 for r in range(3)] for a in profile
        ]
        self.assertAlmostEqual(
            expected_social_cost(probabilities, game),
            game[profile]["social_cost"]
        )


class TestSequentialSummary(unittest.TestCase):

    def test_rows(self):
        rows = summary_rows(analyze_sequential_game(0.10, (0, 1)))
        self.assertTrue(rows)
        self.assertTrue(all(r["first_movers"] == "elephants" for r in rows))
        self.assertTrue(all(r["ratio_to_optimum"] >= 1.0 for r in rows))


class TestSnapshotCallback(unittest.TestCase):

    def test_callback_does_not_change_training(self):
        from equilibrium_gap import policy_table

        seen = []

        def callback(episode, agents):
            seen.append(episode)
            [policy_table(a) for a in agents]

        plain, h1 = train_cooperation(
            mode="tolled", seed=3, episodes=6, steps_per_episode=10
        )
        hooked, h2 = train_cooperation(
            mode="tolled", seed=3, episodes=6, steps_per_episode=10,
            callback=callback
        )

        self.assertEqual(h1, h2)
        self.assertEqual(seen, list(range(7)))

        for a, b in zip(plain, hooked):
            self.assertEqual(dict(a.Q), dict(b.Q))


if __name__ == "__main__":
    unittest.main()
