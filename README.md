
# Learning to Route Under Conflicting QoS Objectives

## Elephant and Mouse Agents in a Shared Congested Network

A multi-agent reinforcement learning and game-theoretic
study of congestion-aware routing under conflicting
Quality of Service (QoS) objectives.

## Project Overview

We study a shared network containing four autonomous agents:

- Two elephant agents prioritizing throughput.
- Two mouse agents prioritizing low latency.

Each agent selects one of three parallel network routes.
Agents share limited network capacity, creating congestion
externalities and conflicts between individual incentives
and socially efficient routing.

We analyze strategic routing using game theory, exact
dynamic programming, model-free reinforcement learning,
and cooperative incentive mechanisms.

## Network Configuration

| Parameter | Value |
|---|---|
| Number of agents | 4 |
| Available routes | 3 |
| Route capacities | 50, 100, 40 Mb/s |
| Base delays | 2, 10, 5 ms |
| Agent demands | 20, 20, 6, 6 Mb/s |
| Background loads | 10%, 25% |
| Discount factor | 0.9 |
| Switching penalty | 0.02 |
| Markov states | 162 |

The background traffic follows a two-state Markov process.

## Implemented Methods

### Game Theory
- Pure Nash equilibrium enumeration
- Social optimum
- Price of Anarchy and Price of Stability
- Sequential move-order analysis and pure SPNE
- Complete enumeration of mixed equilibria using PyGambit
  (duplicates removed, every solution regret-checked), and
  price of anarchy/stability over all equilibria

### Reinforcement Learning
- Exact policy iteration, against fixed routes or against
  any fixed state-dependent opponent policies
- Monte Carlo control (only steps whose return is not
  affected by the episode cut-off are used as targets)
- Q-learning
- Simultaneous independent multi-agent Q-learning and
  Monte Carlo control

### Evaluation
- Uniform random and shortest-path (always R1) baselines
- Equilibrium gap: each agent's best-response value gain
  against the other three frozen policies (exact policy
  iteration); zero iff the joint policy is an equilibrium
- Cycling check: fraction of greedy actions that change
  every 100 training episodes
- Discount-factor and learning-rate / exploration sweeps

### Cooperation
- Selfish rewards
- Shared team rewards
- Externality-based congestion tolls

### Repeated Games
- Pure Nash-reversion punishment analysis
- Pure-action minmax punishment benchmarks
- Conditional discount-factor incentive thresholds
- Trigger strategies in the full Markov game (background
  load switching, switching cost included), with minmax
  and Nash-reversion punishments

## Installation

Clone the repository:

```bash
git clone https://github.com/HellyBera/elephant-mouse-routing.git
cd elephant-mouse-routing
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running Tests

```bash
python -m unittest discover -s tests -v
```

## Running Experiments

Stage-game analysis:

```bash
python src/game_theory.py
```

Sequential-game analysis:

```bash
python src/sequential_game.py
```

Mixed equilibrium:

```bash
python src/mixed_equilibrium.py
```

Exact policy iteration:

```bash
python src/policy_iteration.py
```

Monte Carlo control:

```bash
python src/monte_carlo.py
```

Q-learning:

```bash
python src/q_learning.py
```

Simultaneous multi-agent learning:

```bash
python src/multi_agent.py
```

Cooperation experiments:

```bash
python src/cooperation.py
```

Nash-reversion punishment:

```bash
python src/repeated_game.py
```

Pure minmax analysis:

```bash
python src/minmax_punishment.py
```

Trigger strategies in the Markov game:

```bash
python src/markov_trigger.py
```

Baselines:

```bash
python src/baselines.py
```

Discount-factor and hyperparameter sweeps (parallel, about
15 minutes on 2 cores; `--quick` for a smoke test):

```bash
python scripts/sweep_experiments.py
```

RL algorithm comparison:

```bash
python scripts/compare_rl_algorithms.py
```

Generate report figures and tables (run after all
experiments above):

```bash
python scripts/generate_all_results.py
```

Run every command from the repository root.

Interactive replay of the learned routes (writes the
self-contained page `results/routing_animation.html`,
about 40 seconds):

```bash
python scripts/build_animation.py
```

## Selected Results

All learning results are means over five seeds
(42, 123, 456, 789, 2026).

### Stage game

| Load | Pure NE | All NE (mixed incl.) | PoA (pure / all) | PoS (pure / all) |
|---|---:|---:|---:|---:|
| 10% | 6 | 35 | 1.654 / 3.461 | 1.355 / 1.086 |
| 25% | 8 | 18 | 1.811 / 2.167 | 1.409 / 1.409 |

At 10% load, mice moving first reaches a cheaper SPNE
outcome (ratio to optimum 1.355) than elephants moving
first (1.654); at 25% load move order does not matter.

### Learning against the baselines

| Method | Efficiency ratio | Equilibrium gap |
|---|---:|---:|
| Random baseline | 4.768 | n/a |
| Shortest-path baseline | 20.738 | 8.502 |
| Q-learning, selfish | 1.464 | 0.103 |
| Q-learning, team | 1.135 | 0.251 |
| Q-learning, tolled | 1.250 | 0.232 |
| Monte Carlo, selfish | 1.591 | 0.246 |
| Monte Carlo, team | 1.828 | 0.595 |
| Monte Carlo, tolled | 1.268 | 0.296 |

The equilibrium gap (largest per-agent best-response
value gain, gamma = 0.9) is non-zero for every learned
joint policy, so none of them is an equilibrium. Team
rewards give the lowest social cost with Q-learning.
Q-learning greedy policies keep changing at a low rate
late in training (fig10); Monte Carlo policies freeze
early.

### Repeated game

Pure Nash reversion with a common punishment profile
never sustains the social optimum in the fixed-load
analysis; pure minmax gives conditional thresholds of
0.831 (10% load) and 0.817 (25% load).

In the full Markov game (load switching), without the
switching cost the thresholds are 0.824 (minmax) and
0.951 (agent-specific Nash reversion, so not at
gamma = 0.9). With the model's switching cost
kappa = 0.02, cooperation holds at every discount factor:
the largest one-shot deviation gain (0.0132) is below
kappa.

## Results and Figures

Experimental data is stored in `results/data/`.

Publication figures are stored in `results/figures/`.

Summary tables are stored in `results/tables/`.

## Reproducibility

Experiments use specified random seeds and fixed
configurations. The numerical results can be regenerated
using the scripts above.

## Contributors

- Helly Bera
- Divisht
- Anshul

See the final report for individual contributions.

## Notes

The mixed-equilibrium enumeration runs to completion. It
finds isolated equilibria; a continuum of equilibria, if
one existed, would not be listed in full.

`repeated_game.py` and `minmax_punishment.py` use
fixed-load stage games without the switching cost.
`markov_trigger.py` includes both. It checks the
deviating agent's incentive only; it does not check that
the minmax punishers are willing to punish, so the minmax
results are benchmarks, not subgame-perfect equilibria.

Equilibrium gaps are measured with the selfish reward,
whatever reward the agents trained with. The gap is not
computed for the random baseline, whose policy is not
deterministic.
