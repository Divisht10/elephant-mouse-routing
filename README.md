# Learning to Route Under Conflicting QoS Objectives

## Elephant and mouse agents in a shared, congested network

Four agents share three parallel network routes. Two
**elephants** carry bulk traffic and care mostly about
throughput; two **mice** carry small, latency-sensitive
flows and care mostly about delay. Every step, each agent
picks a route, and the routes' capacity is shared with
each other and with stochastic background traffic.

The project studies this conflict two ways and compares
them directly:

- **Game theory:** equilibria, price of anarchy,
  sequential play, mixed strategies, repeated-game
  punishments and congestion tolls.
- **Reinforcement learning:** exact policy iteration,
  Monte Carlo control and Q-learning, for one agent and
  for all four agents learning at the same time, under
  selfish, team and tolled rewards.

### Key findings

1. **Selfish routing is inefficient.** The social optimum
   is not a Nash equilibrium of the one-shot game at
   either background load. The price of anarchy is 1.65
   (10% load) and 1.81 (25% load) over pure equilibria,
   rising to 3.46 and 2.17 once mixed equilibria are
   counted.
2. **No learner reaches an equilibrium.** Every learned
   joint policy has a non-zero equilibrium gap (some agent
   could still gain by changing its own policy).
   Q-learning with a shared team reward gets closest to
   the optimum (1.14×); independent selfish learners reach
   1.46×.
3. **The optimum can be sustained, but is not found.**
   With the model's switching cost (κ = 0.02), the largest
   one-shot gain from leaving the optimum (0.0132) is
   smaller than κ, so trigger strategies sustain the
   optimum at every discount factor. Without the switching
   cost, minmax punishment needs γ ≥ 0.824 and Nash
   reversion γ ≥ 0.951. The gap between what is
   sustainable and what independent learners reach is a
   learning problem, not an incentive problem.

---

## Contents

- [Model](#model)
- [Research questions and where they are answered](#research-questions-and-where-they-are-answered)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Reproducing every result](#reproducing-every-result)
- [Interactive replay (UI)](#interactive-replay-ui)
- [Results](#results)
- [Figures](#figures)
- [Tests](#tests)
- [Limitations](#limitations)
- [Contributors](#contributors)

---

## Model

### Network

| Route | Character | Capacity $C_r$ | Base delay $\tau_r$ |
|---|---|---:|---:|
| R1 | fast, narrow | 50 Mb/s | 2 ms |
| R2 | wide, slow | 100 Mb/s | 10 ms |
| R3 | thin backup | 40 Mb/s | 5 ms |

### Agents

| Agent | Type | Demand $d_i$ | Weight on throughput $\alpha_i$ |
|---|---|---:|---:|
| 1, 2 | elephant | 20 Mb/s | 0.8 |
| 3, 4 | mouse | 6 Mb/s | 0.1 |

### Costs (`src/network.py`)

With background level $\beta$, the load on route $r$ is
the background share of its capacity plus the demand of
every agent on it:

$$x_r = \beta C_r + \sum_{i:\,a_i = r} d_i$$

Delay grows with utilisation and is capped at
$D_{\max} = 100$ ms (the cap also applies once a route is
at or over capacity):

$$D_r = \min\!\left(D_{\max},\ \frac{\tau_r}{1 - x_r/C_r}\right)$$

Normalised throughput and latency:

$$T_r = \min\!\left(1,\ \frac{C_r}{x_r}\right), \qquad L_r = \min\!\left(1,\ \frac{D_r}{L_{\max}}\right),\ L_{\max} = 100 \text{ ms}$$

Agent $i$ on route $r$ pays

$$c_i = \alpha_i\,(1 - T_r) + (1 - \alpha_i)\,L_r$$

and the social cost is $\sum_i c_i$.

### Markov game (`src/environment.py`)

- Background load switches between 10% and 25% as a
  Markov chain: low to high with probability 0.02, high
  to low with probability 0.04.
- State: background level and the previous joint route
  choice, $2 \times 3^4 = 162$ states.
- Switching cost $\kappa = 0.02$ each time an agent
  changes route; discount factor $\gamma = 0.9$.

### Reward schemes (`src/rewards.py`)

| Scheme | Reward of agent $i$ |
|---|---|
| Selfish | $r_i = -c_i - \kappa\,\mathbb{1}[a_i \ne a_i^{\text{prev}}]$ |
| Team | every agent receives $\sum_j r_j^{\text{selfish}}$ |
| Tolled | $r_i^{\text{selfish}} - \phi_i$, with externality toll $\phi_i = \sum_{j \ne i} \left[c_j(a) - c_j(a \text{ without } i)\right]$ |

The toll charges each agent the extra cost it imposes on
the others. With it, each agent's tolled cost differs
from the social cost only by a term it cannot influence,
so the stage game is an exact potential game whose
potential is the social cost (checked in
`tests/test_rewards.py`).

---

## Research questions and where they are answered

| Question | Method | Code | Main output |
|---|---|---|---|
| RQ1: How inefficient is selfish routing, and does move order matter? | Pure and mixed equilibria, price of anarchy/stability, sequential-game SPNE | `game_theory.py`, `mixed_equilibrium.py`, `sequential_game.py` | `stage_game_summary.csv`, `mixed_equilibrium_summary.csv`, `sequential_summary.csv`, fig 1 |
| RQ2: How close do model-free learners get to the exact best response? | Policy iteration vs Monte Carlo and Q-learning against fixed opponents | `policy_iteration.py`, `monte_carlo.py`, `q_learning.py`, `scripts/compare_rl_algorithms.py` | `rl_comparison_summary.csv`, figs 7, 8 |
| RQ3: When all four learn together, do they reach an equilibrium or cycle? | Simultaneous learning, equilibrium gap, policy-change rate, baselines | `multi_agent.py`, `cooperation.py`, `equilibrium_gap.py`, `baselines.py` | `main_comparison.csv`, figs 2–5, 9, 10 |
| RQ4: Can cooperation be sustained or induced? | Team rewards, tolls, Nash reversion, minmax punishment, trigger strategies in the Markov game, γ sweep | `cooperation.py`, `repeated_game.py`, `minmax_punishment.py`, `markov_trigger.py`, `scripts/sweep_experiments.py` | `markov_trigger_summary.csv`, `sweep_summary.csv`, figs 6, 11–13 |

---

## Repository layout

```
src/
  network.py             link model: loads, delays, throughput, costs
  environment.py         Markov game: background chain, states, switching cost
  rewards.py             selfish, team and tolled rewards; externality tolls
  game_theory.py         pure Nash equilibria, social optimum, PoA, PoS
  mixed_equilibrium.py   all equilibria via PyGambit (pure and mixed)
  sequential_game.py     elephants-first / mice-first SPNE
  policy_iteration.py    exact best response (fixed or state-dependent opponents)
  monte_carlo.py         first-visit Monte Carlo control
  q_learning.py          tabular Q-learning
  multi_agent.py         four simultaneous Q-learners (selfish)
  cooperation.py         four simultaneous learners under all reward schemes
  equilibrium_gap.py     best-response gap of a joint policy
  baselines.py           random and shortest-path routing
  repeated_game.py       Nash-reversion thresholds (fixed load)
  minmax_punishment.py   pure minmax punishment thresholds (fixed load)
  markov_trigger.py      trigger strategies in the full Markov game
  check_transitions.py   sanity check of the background chain
scripts/
  compare_rl_algorithms.py   RQ2 benchmark against policy iteration
  sweep_experiments.py       discount-factor and hyperparameter sweeps
  generate_all_results.py    all figures and summary tables
  build_animation.py         builds the interactive replay page
  animation_template.html    page template used by build_animation.py
tests/                       102 unit tests
results/
  data/       raw per-seed outputs (CSV)
  tables/     summary tables (CSV)
  figures/    figures (PNG and PDF)
  routing_animation.html     interactive replay (UI)
```

---

## Setup

Python 3.10 or newer. Linux or macOS:

```bash
git clone https://github.com/USERNAME/elephant-mouse-routing.git
cd elephant-mouse-routing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell): replace the activation line with
`.venv\Scripts\Activate.ps1`.

Dependencies: numpy, pandas, matplotlib, scipy and
pygambit (used only by `mixed_equilibrium.py`). Inside an
activated environment `python` works; outside it, some
Linux systems only provide `python3`.

---

## Reproducing every result

Every experiment uses fixed seeds, so a clean run
reproduces the committed results exactly (up to the last
floating-point digits across platforms). Run everything
from the repository root, in this order; the figure script
reads the files the others write.

```bash
rm -rf results                         # optional: start clean

# Game theory (seconds each)
python src/game_theory.py              # prints pure NE, optimum, PoA, PoS
python src/sequential_game.py
python src/mixed_equilibrium.py
python src/repeated_game.py
python src/minmax_punishment.py
python src/markov_trigger.py

# Learning
python scripts/compare_rl_algorithms.py
python src/multi_agent.py
python src/cooperation.py
python src/baselines.py
python scripts/sweep_experiments.py

# Figures, tables and the interactive replay
python scripts/generate_all_results.py
python scripts/build_animation.py
```

| Step | What it does | Writes | Time* |
|---|---|---|---|
| `sequential_game.py` | SPNE for both move orders, both loads | `tables/sequential_summary.csv` | seconds |
| `mixed_equilibrium.py` | complete equilibrium enumeration, regret-checked | `data/mixed_equilibria.csv`, `tables/mixed_equilibrium_summary.csv` | seconds |
| `repeated_game.py` | Nash-reversion thresholds | `data/repeated_game_thresholds.csv` | seconds |
| `minmax_punishment.py` | minmax thresholds | `data/minmax_punishment_results.csv` | seconds |
| `markov_trigger.py` | trigger strategies, γ ∈ [0, 0.999], κ ∈ {0.02, 0} | `data/markov_trigger_grid.csv`, `tables/markov_trigger_summary.csv` | seconds |
| `compare_rl_algorithms.py` | MC and Q-learning vs policy iteration, 4 agents × 5 seeds | `data/rl_comparison.csv`, `data/rl_training_history.csv`, `tables/rl_comparison_summary.csv`, figs 7–8 | ~3 min |
| `multi_agent.py` | four selfish Q-learners, 5 seeds | `data/multi_agent_*.csv` | ~1 min |
| `cooperation.py` | Q-learning and MC × 3 reward schemes × 5 seeds | `data/cooperation_*.csv` | ~5 min |
| `baselines.py` | random and shortest-path routing, 5 seeds | `data/baselines_evaluation.csv` | seconds |
| `sweep_experiments.py` | 165 runs: γ sweep and hyperparameter sweep, all cores | `data/sweep_results.csv` | ~15 min on 2 cores |
| `generate_all_results.py` | all remaining figures and summary tables | `figures/`, `tables/` | seconds |
| `build_animation.py` | retrains seed 42 with snapshots, builds the UI | `results/routing_animation.html` | ~40 s |

\*Measured on a 2-core machine; faster with more cores.

**Seeds.** Learning experiments use seeds 42, 123, 456,
789 and 2026; evaluation episodes use seed + 2000. The
sweep runs each configuration on all five seeds.

**Training settings.** 3000 episodes of 200 steps;
learning rate 0.1; ε decays from 1.0 by ×0.997 per
episode to 0.05; γ = 0.9. Evaluation uses the greedy
policies (no exploration).

**Smoke test.** `python scripts/sweep_experiments.py
--quick` runs a reduced sweep in about 20 seconds and
writes `data/sweep_quick.csv`. That file is only a check
that the parallel sweep works; no figure or table uses it,
and `.gitignore` excludes it.

---

## Interactive replay (UI)

`results/routing_animation.html` is a self-contained page
(about 55 KB, opens offline in any browser) showing the
learned routes:

- the three routes drawn between source and sink, with
  width proportional to capacity and a meter showing
  current load;
- elephants and mice drawn on the route they chose,
  moving when they switch;
- a side-by-side comparison of the two species: how each
  weighs throughput against delay, and each agent's route,
  delivered throughput, delay and cost;
- whether the current joint route is the social optimum, a
  Nash equilibrium of the one-shot game, or neither;
- a timeline of every agent's route over the replay, and
  a chart of replay cost relative to the optimum across
  training for all three reward schemes.

**Controls.** Choose the reward scheme (Selfish, Team,
Tolled). *Through training* + Play steps through training
snapshots from episode 0 (untrained: every agent takes R1)
to 3000. *Through replay* + Play steps through the 60
replay steps. *Show social optimum* overlays faded icons
where each agent would be at the optimum.

**What it shows.** Q-learning, seed 42. Every 100
episodes the greedy policies are frozen and replayed for
60 steps from the same start state on the same
background-load sequence (replay seed 3: 45 low-load and
15 high-load steps), so differences between snapshots come
from learning alone.

**Consistency with the model.** The page does no
modelling of its own. Every cost, load, delay and
throughput it shows is looked up in a table computed by
`network.simulate_network`; the optimum and Nash
equilibria come from `game_theory.py`. Recording the
snapshots does not change training (checked in
`tests/test_new_experiments.py`), and the final
equilibrium gaps match the seed-42 rows of
`results/data/cooperation_evaluation.csv`.

**Rebuild** after any change to the code:

```bash
python scripts/build_animation.py
```

---

## Results

### RQ1: Stage game

`results/tables/stage_game_summary.csv`,
`results/tables/mixed_equilibrium_summary.csv`

| Background load | Social optimum | Pure NE | All NE (pure + mixed) | PoA pure / all | PoS pure / all |
|---|---:|---:|---:|---:|---:|
| 10% | 0.1081 | 6 | 35 (6 + 29) | 1.654 / 3.461 | 1.355 / 1.086 |
| 25% | 0.1470 | 8 | 18 (8 + 10) | 1.811 / 2.167 | 1.409 / 1.409 |

The social optimum is (R2, R3, R1, R1): the elephants
split the wide and backup routes, and both mice take the
fast route. It is not an equilibrium at either load. At
10% load the cheapest mixed equilibrium (1.086 × optimum)
beats every pure one.

**Move order** (`results/tables/sequential_summary.csv`).
At 10% load, mice moving first reaches SPNE outcomes at
1.355 × optimum, and elephants moving first at 1.654 ×.
At 25% load both orders give 1.409 ×.

### RQ2: Learners against the exact best response

`results/tables/rl_comparison_summary.csv`. Each agent
learns alone while the other three keep fixed routes
(R1, R2, R1, R3); value gap = optimal value minus learned
value, averaged over states; 5 seeds.

| Agent | MC value gap | Q-learning value gap | MC policy agreement | Q-learning policy agreement |
|---|---:|---:|---:|---:|
| Elephant 1 | 0.032 | 0.029 | 44% | 52% |
| Elephant 2 | 0.043 | 0.057 | 52% | 51% |
| Mouse 3 | 0.029 | 0.048 | 46% | 40% |
| Mouse 4 | 0.027 | 0.048 | 67% | 44% |

Both learners come within about 0.03–0.06 of the optimal
value, while matching the optimal action in only 40–67%
of states: many different policies come close to the
optimal value.

### RQ3: Four agents learning together

`results/tables/main_comparison.csv`, mean ± std over
5 seeds. Efficiency ratio = social cost ÷ optimum
(1 is optimal). Equilibrium gap = the largest gain any
one agent could get by switching to its exact best
response against the other three frozen policies (value
units, γ = 0.9; 0 means an equilibrium).

| Method | Social cost | Efficiency ratio | Equilibrium gap | Switch rate |
|---|---:|---:|---:|---:|
| Random baseline | 0.585 | 4.77 ± 0.05 | n/a | 0.665 |
| Shortest-path baseline (all on R1) | 2.486 | 20.74 ± 0.28 | 8.50 | 0.003 |
| Q-learning, selfish | 0.178 | 1.46 ± 0.12 | 0.103 ± 0.050 | 0.015 |
| Q-learning, team | 0.139 | **1.14 ± 0.08** | 0.251 ± 0.034 | 0.115 |
| Q-learning, tolled | 0.152 | 1.25 ± 0.00 | 0.232 ± 0.003 | 0.004 |
| Monte Carlo, selfish | 0.193 | 1.59 ± 0.18 | 0.246 ± 0.089 | 0.184 |
| Monte Carlo, team | 0.226 | 1.83 ± 0.32 | 0.595 ± 0.107 | 0.326 |
| Monte Carlo, tolled | 0.155 | 1.27 ± 0.02 | 0.296 ± 0.024 | 0.006 |

- Every learner beats both baselines by a wide margin.
- No learned joint policy is an equilibrium (all gaps are
  above zero).
- Team rewards help Q-learning most but hurt Monte Carlo,
  whose team learners switch routes often and settle
  poorly.
- Tolled learners are very consistent across seeds
  (efficiency std 0.001) and all settle at about 1.25 ×:
  the toll makes the optimum an equilibrium, but not the
  only one, and the learners settle short of it.
- Q-learning greedy policies are still changing slightly
  at the end of training; Monte Carlo policies stop
  changing early (fig 10).

### RQ4: Sustaining cooperation

**Fixed load, no switching cost**
(`results/data/repeated_game_thresholds.csv`,
`results/data/minmax_punishment_results.csv`). Nash
reversion with a common punishment profile never
sustains the optimum (no feasible plan among the 12 and
16 combinations tested). Pure minmax punishment sustains
it for γ ≥ 0.831 (10% load) and γ ≥ 0.817 (25% load).

**Full Markov game** (`results/tables/markov_trigger_summary.csv`).
Background load switches; each agent's best deviation is
solved exactly as an MDP.

| Switching cost κ | Punishment | Threshold γ* (all agents) | Holds at γ = 0.9 |
|---|---|---:|---|
| 0.02 (model) | minmax | 0 (any γ) | yes |
| 0.02 (model) | Nash reversion | 0 (any γ) | yes |
| 0 | minmax | 0.824 | yes |
| 0 | Nash reversion | 0.951 | no |

With the model's switching cost, deviating from the
optimum never pays: a one-step deviation gains at most
0.0132 < κ, and a lasting one is punished.

**Discount-factor sweep** (`results/tables/sweep_summary.csv`,
Q-learning, efficiency ratio):

| γ | Selfish | Team | Tolled |
|---:|---:|---:|---:|
| 0.50 | 1.57 | 1.25 | 1.24 |
| 0.70 | 1.58 | 1.24 | 1.24 |
| 0.80 | 1.54 | 1.23 | 1.24 |
| 0.90 | 1.46 | 1.14 | 1.25 |
| 0.95 | 1.47 | 1.21 | 1.22 |
| 0.99 | 1.53 | 1.16 | 1.11 |

**Hyperparameters** (one setting changed at a time from
the defaults; efficiency ratio):

| Setting | Selfish | Team | Tolled |
|---|---:|---:|---:|
| learning rate 0.05 | 1.39 | 1.16 | 1.11 |
| learning rate 0.1 (default) | 1.46 | 1.14 | 1.25 |
| learning rate 0.2 | 1.46 | 1.34 | 1.22 |
| ε decay 0.995 | 1.47 | 1.18 | 1.22 |
| ε decay 0.999 | 1.48 | 1.11 | 1.22 |

The ranking of reward schemes (team and tolled ahead of
selfish) holds across all settings tested.

---

## Figures

All in `results/figures/` as PNG and PDF.

| Figure | Shows | Made by |
|---|---|---|
| fig1_stage_game | social optimum vs best and worst pure Nash equilibrium, both loads | `generate_all_results.py` |
| fig2_learning_curves, fig2_learning_curves_mc | social cost during training, Q-learning and Monte Carlo | `generate_all_results.py` |
| fig3_social_cost | evaluation social cost by scheme and algorithm | `generate_all_results.py` |
| fig4_efficiency | efficiency ratio by scheme and algorithm (baselines noted) | `generate_all_results.py` |
| fig5_switching | route-switching rate | `generate_all_results.py` |
| fig6_punishment | fixed-load pure minmax thresholds vs γ = 0.9 | `generate_all_results.py` |
| fig7_rl_value_gap | value gap to policy iteration | `compare_rl_algorithms.py` |
| fig8_policy_agreement | agreement with the optimal policy | `compare_rl_algorithms.py` |
| fig9_equilibrium_gap | equilibrium gap of learned joint policies | `generate_all_results.py` |
| fig10_policy_changes | greedy actions changed per 100 episodes (cycling check) | `generate_all_results.py` |
| fig11_gamma_sweep | efficiency and per-step gap across γ | `generate_all_results.py` |
| fig12_hyperparameters | sensitivity to learning rate and ε decay | `generate_all_results.py` |
| fig13_markov_trigger | deviation gain vs γ in the Markov game, κ = 0.02 and 0 | `generate_all_results.py` |

---

## Tests

```bash
python -m unittest discover -s tests
```

102 tests covering the network model, environment,
rewards (including the potential-game property of the
tolls), equilibrium computations, all solvers and
learners, the equilibrium gap (a best response has zero
gap), the Markov trigger model (it reproduces the
fixed-load thresholds when load cannot switch), baselines,
the Monte Carlo cut-off fix, and the UI snapshot hook.

---

## Limitations

- **Mixed equilibria:** the enumeration finds isolated
  equilibria; a continuum of equilibria, if one existed,
  would not be listed in full.
- **Minmax punishment** checks only that the deviating
  agent is deterred, not that the punishers are willing
  to carry out the punishment, so minmax results are a
  benchmark, not a subgame-perfect equilibrium.
- **Fixed-load analyses** (`repeated_game.py`,
  `minmax_punishment.py`) omit load switching and the
  switching cost; `markov_trigger.py` includes both.
- **Equilibrium gaps** are measured with the selfish
  reward, whatever reward the agents trained with. They
  are not computed for the random baseline, whose policy
  is not deterministic.
- **Monte Carlo** updates only the first 112 of 200 steps
  per episode, so returns are not biased by the episode
  cut-off (γ⁸⁸ < 10⁻⁴).
- **γ = 0.99:** 3000 episodes may not be enough to
  converge; gaps there are large in value units, so
  compare the per-step gap.
- **RQ2 benchmark:** opponents use one fixed route profile.

---

## Contributors

- Helly Bera
- Divisht
- Anshul

See the final report for individual contributions.
