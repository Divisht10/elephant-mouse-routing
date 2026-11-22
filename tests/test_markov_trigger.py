import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from markov_trigger import (
    GAMMA_GRID,
    build_mdp,
    deviation_gains,
    stage_profiles,
    threshold,
    TOL
)
from minmax_punishment import analyze_minmax
from environment import TRANSITION_MATRIX, SWITCHING_COST

FIXED_LOAD = ((1.0, 0.0), (0.0, 1.0))


class TestMarkovTrigger(unittest.TestCase):

    def test_transitions_are_distributions(self):
        cooperation, punishment = stage_profiles("minmax", 0)
        _, P = build_mdp(
            0, cooperation, punishment,
            TRANSITION_MATRIX, SWITCHING_COST
        )

        np.testing.assert_allclose(P.sum(axis=2), 1.0)

    def test_fixed_load_matches_closed_form(self):
        # With the load frozen and no switching cost, the
        # MDP threshold must equal Eq. (7) with minmax
        # punishment, taking the worse of the two levels.
        analytic = {
            i: max(
                result["agents"][i]["gamma_threshold"]
                for beta in (0.10, 0.25)
                for result in analyze_minmax(beta)[:1]
            )
            for i in range(4)
        }

        for agent in range(4):
            gains = deviation_gains(
                agent, GAMMA_GRID, "minmax",
                transition=FIXED_LOAD, kappa=0.0
            )
            value, monotone = threshold(gains)

            self.assertTrue(monotone)
            self.assertGreaterEqual(value, analytic[agent] - 1e-9)
            self.assertLessEqual(value, analytic[agent] + 0.001 + 1e-9)

    def test_switching_cost_deters_one_shot_deviation(self):
        # The largest one-shot deviation gain (0.0132) is
        # below kappa = 0.02, so even a myopic agent stays.
        for agent in range(4):
            gain = deviation_gains(agent, [0.0], "nash")[0]
            self.assertLessEqual(gain, TOL)

    def test_threshold_helper(self):
        self.assertEqual(threshold([1, 1, 0])[0], float(GAMMA_GRID[2]))
        self.assertIsNone(threshold([0, 1])[0])
        self.assertFalse(threshold([0, 1, 0])[1])


if __name__ == "__main__":
    unittest.main()
