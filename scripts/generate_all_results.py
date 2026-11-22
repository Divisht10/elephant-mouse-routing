
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locate the project root regardless of where
# the script is executed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game_theory import (
    calculate_game,
    find_nash_equilibria,
    find_social_optimum
)

DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False
})

COLORS = {
    "selfish": "#C65D57",
    "team": "#3688B8",
    "tolled": "#38A17A"
}


def save_figure(name):
    plt.tight_layout()

    for extension in ("png", "pdf"):
        plt.savefig(
            FIGURES / f"{name}.{extension}",
            bbox_inches="tight"
        )

    plt.close()
    print(f"Saved figure: {name}")


def load_csv(name, required_columns):
    path = DATA / name

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding experiment."
        )

    df = pd.read_csv(path)

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )

    return df


# ----------------------------------------------------
# FIGURE 1: Nash equilibrium vs social optimum
# ----------------------------------------------------

def plot_stage_game():
    rows = []

    for beta in (0.10, 0.25):
        game = calculate_game(beta)
        equilibria = find_nash_equilibria(game)
        _, optimum = find_social_optimum(game)

        equilibrium_costs = [
            game[a]["social_cost"] for a in equilibria
        ]

        rows.append({
            "background_load": beta,
            "number_of_ne": len(equilibria),
            "social_optimum": optimum,
            "best_ne": min(equilibrium_costs),
            "worst_ne": max(equilibrium_costs),
            "poa": max(equilibrium_costs) / optimum,
            "pos": min(equilibrium_costs) / optimum
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "stage_game_summary.csv", index=False)

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(7, 4.5))

    for offset, column, label in [
        (-width, "social_optimum", "Social optimum"),
        (0, "best_ne", "Best Nash equilibrium"),
        (width, "worst_ne", "Worst Nash equilibrium")
    ]:
        plt.bar(
            x + offset,
            df[column],
            width,
            label=label
        )

    plt.xticks(x, ["10% load", "25% load"])
    plt.ylabel("Social cost (lower is better)")
    plt.title("Stage-game equilibrium inefficiency")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)

    save_figure("fig1_stage_game")


# ----------------------------------------------------
# FIGURE 2: Multi-agent learning curves
# ----------------------------------------------------

ALGORITHM_LABELS = {
    "q_learning": "Q-learning",
    "monte_carlo": "Monte Carlo"
}


def plot_learning_curves():
    all_df = load_csv(
        "cooperation_training.csv",
        ["algorithm", "mode", "seed", "episode", "mean_social_cost"]
    )

    for algorithm, suffix in (
        ("q_learning", ""), ("monte_carlo", "_mc")
    ):
        plot_learning_curve(
            all_df[all_df["algorithm"] == algorithm],
            ALGORITHM_LABELS[algorithm],
            f"fig2_learning_curves{suffix}"
        )


def plot_learning_curve(df, label, filename):
    plt.figure(figsize=(7, 4.5))

    for mode in ("selfish", "team", "tolled"):
        subset = df[df["mode"] == mode].copy()

        if subset.empty:
            raise ValueError(f"No training data for {mode}")

        smoothed = []

        for seed, group in subset.groupby("seed"):
            group = group.sort_values("episode").copy()

            group["smoothed_cost"] = (
                group["mean_social_cost"]
                .rolling(window=100, min_periods=1)
                .mean()
            )

            smoothed.append(
                group[["episode", "smoothed_cost"]]
            )

        combined = pd.concat(smoothed)

        curve = (
            combined.groupby("episode")["smoothed_cost"]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = curve["episode"].to_numpy()
        y = curve["mean"].to_numpy()
        sd = curve["std"].fillna(0).to_numpy()

        plt.plot(
            x, y,
            label=mode.capitalize(),
            color=COLORS[mode]
        )

        plt.fill_between(
            x, y - sd, y + sd,
            color=COLORS[mode],
            alpha=0.12
        )

    plt.xlabel("Training episode")
    plt.ylabel("Mean social cost")
    plt.title(f"Multi-agent learning curves ({label})")
    plt.legend()
    plt.grid(alpha=0.2)

    save_figure(filename)


# ----------------------------------------------------
# FIGURES 3, 4, 5: Cooperation comparison
# ----------------------------------------------------

METRICS = [
    "mean_social_cost",
    "mean_efficiency_ratio",
    "switch_rate",
    "mean_stage_regret",
    "mean_tolled_stage_regret",
    "equilibrium_gap",
    "equilibrium_gap_max_state"
]


def plot_cooperation_comparison():
    df = load_csv(
        "cooperation_evaluation.csv",
        ["algorithm", "mode", "seed"] + METRICS
    )
    baselines = load_csv(
        "baselines_evaluation.csv",
        ["baseline", "seed", "mean_social_cost"]
    )

    summary = df.groupby(["algorithm", "mode"])[METRICS].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary.to_csv(TABLES / "cooperation_summary.csv")

    # One table with baselines and learners side by side.
    base = baselines.rename(columns={"baseline": "method"})
    learned = df.assign(method=df["algorithm"] + "_" + df["mode"])
    combined = pd.concat([base, learned], ignore_index=True)
    main_table = combined.groupby("method", sort=False)[
        METRICS
    ].agg(["mean", "std"])
    main_table.columns = [
        f"{metric}_{stat}" for metric, stat in main_table.columns
    ]
    main_table.to_csv(TABLES / "main_comparison.csv")

    baseline_means = baselines.groupby("baseline").mean(numeric_only=True)

    modes = ["selfish", "team", "tolled"]
    algorithms = ["q_learning", "monte_carlo"]
    hatches = {"q_learning": "", "monte_carlo": "//"}

    figures_to_make = [
        ("mean_social_cost", "Mean network social cost",
         "fig3_social_cost", True),
        ("mean_efficiency_ratio", "Social efficiency ratio",
         "fig4_efficiency", True),
        ("switch_rate", "Route-switching rate",
         "fig5_switching", False),
        ("equilibrium_gap", "Equilibrium gap (best-response value)",
         "fig9_equilibrium_gap", False)
    ]

    x = np.arange(len(modes))
    width = 0.38

    for metric, title, filename, show_baselines in figures_to_make:
        plt.figure(figsize=(7, 4.5))
        top = 0.0

        for k, algorithm in enumerate(algorithms):
            means = [summary.loc[(algorithm, m), f"{metric}_mean"] for m in modes]
            stds = [summary.loc[(algorithm, m), f"{metric}_std"] for m in modes]
            top = max(top, max(a + b for a, b in zip(means, stds)))

            bars = plt.bar(
                x + (k - 0.5) * width, means, width,
                yerr=stds, capsize=4,
                color=[COLORS[m] for m in modes],
                hatch=hatches[algorithm], edgecolor="white",
                label=ALGORITHM_LABELS[algorithm]
            )

            for bar, value, sd in zip(bars, means, stds):
                plt.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value + sd),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7
                )

        if show_baselines:
            # Baselines are far off the bar scale, so they
            # are stated as text (full values: main_comparison).
            plt.annotate(
                "Baselines (off scale): random "
                f"{baseline_means.loc['random', metric]:.3f}, "
                "shortest-path "
                f"{baseline_means.loc['shortest_path', metric]:.3f}",
                (0.01, 0.97), xycoords="axes fraction",
                va="top", fontsize=8, color="#555555"
            )

        plt.xticks(x, [m.capitalize() for m in modes])
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(axis="y", alpha=0.2)
        plt.ylim(0, top * 1.3)

        save_figure(filename)


# ----------------------------------------------------
# FIGURE 6: Punishment incentive thresholds
# ----------------------------------------------------

def plot_punishment():
    df = load_csv(
        "minmax_punishment_results.csv",
        [
            "beta", "cooperation", "agent",
            "overall_gamma_required",
            "incentive_feasible"
        ]
    )

    rows = []

    for beta, group in df.groupby("beta"):
        # All four agents appear for every cooperation
        # profile; retain unique overall thresholds.
        thresholds = (
            group["overall_gamma_required"]
            .dropna()
            .unique()
        )

        if len(thresholds) == 0:
            threshold = np.nan
        else:
            threshold = float(np.min(thresholds))

        rows.append({
            "background_load": beta,
            "pure_minmax_gamma": threshold
        })

    summary = pd.DataFrame(rows).sort_values(
        "background_load"
    )

    summary.to_csv(
        TABLES / "punishment_summary.csv",
        index=False
    )

    plt.figure(figsize=(6.5, 4.5))

    x = np.arange(len(summary))
    thresholds = summary["pure_minmax_gamma"].to_numpy()

    plt.bar(
        x, thresholds,
        color="#3688B8",
        width=0.5,
        label="Pure minmax incentive threshold"
    )

    plt.axhline(
        0.9,
        color="#C65D57",
        linestyle="--",
        label="Model discount factor (0.9)"
    )

    plt.xticks(
        x,
        [
            f"{beta:.0%} load"
            for beta in summary["background_load"]
        ]
    )

    plt.ylim(0, 1.05)
    plt.ylabel("Discount factor")
    plt.title("Conditional punishment incentive thresholds")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)

    save_figure("fig6_punishment")


# ----------------------------------------------------
# FIGURE 10: Cycling check (greedy-policy changes)
# ----------------------------------------------------

def plot_policy_changes():
    df = load_csv(
        "cooperation_training.csv",
        ["algorithm", "mode", "episode", "policy_change_rate"]
    ).dropna(subset=["policy_change_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, algorithm in zip(axes, ("q_learning", "monte_carlo")):
        for mode in ("selfish", "team", "tolled"):
            subset = df[(df["algorithm"] == algorithm) & (df["mode"] == mode)]
            curve = subset.groupby("episode")["policy_change_rate"].agg(
                ["mean", "std"]
            )
            ax.plot(curve.index, curve["mean"], color=COLORS[mode],
                    label=mode.capitalize())
            ax.fill_between(
                curve.index, curve["mean"] - curve["std"].fillna(0),
                curve["mean"] + curve["std"].fillna(0),
                color=COLORS[mode], alpha=0.12
            )

        ax.set_title(ALGORITHM_LABELS[algorithm])
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Greedy actions changed per 100 episodes")
    axes[0].legend()
    save_figure("fig10_policy_changes")


# ----------------------------------------------------
# FIGURES 11, 12: Discount-factor and hyperparameter sweeps
# ----------------------------------------------------

def plot_sweeps():
    path = DATA / "sweep_results.csv"

    if not path.exists():
        print("Skipping sweeps: run scripts/sweep_experiments.py")
        return

    df = pd.read_csv(path)

    summary = df.groupby(["sweep", "parameter", "value", "mode"])[
        ["mean_efficiency_ratio", "equilibrium_gap_per_step",
         "switch_rate"]
    ].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(TABLES / "sweep_summary.csv")
    summary = summary.reset_index()

    gamma = summary[summary["sweep"] == "gamma"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, label in (
        (axes[0], "mean_efficiency_ratio", "Social efficiency ratio"),
        (axes[1], "equilibrium_gap_per_step",
         "Equilibrium gap per step")
    ):
        for mode in ("selfish", "team", "tolled"):
            rows = gamma[gamma["mode"] == mode].sort_values("value")
            ax.errorbar(rows["value"], rows[f"{metric}_mean"],
                        yerr=rows[f"{metric}_std"], marker="o",
                        capsize=3, color=COLORS[mode],
                        label=mode.capitalize())

        ax.set_xlabel("Discount factor")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)

    axes[0].legend()
    fig.suptitle("Q-learning across discount factors")
    save_figure("fig11_gamma_sweep")

    hyper = summary[summary["sweep"] == "hyper"]
    labels = [f"{p}={v:g}" for p, v in
              hyper[["parameter", "value"]].drop_duplicates().itertuples(index=False)]
    x = np.arange(len(labels))
    width = 0.27

    plt.figure(figsize=(9, 4.5))

    for k, mode in enumerate(("selfish", "team", "tolled")):
        rows = hyper[hyper["mode"] == mode]
        plt.bar(x + (k - 1) * width, rows["mean_efficiency_ratio_mean"],
                width, yerr=rows["mean_efficiency_ratio_std"], capsize=3,
                color=COLORS[mode], label=mode.capitalize())

    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Social efficiency ratio")
    plt.title("Sensitivity to learning rate and exploration decay")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)
    save_figure("fig12_hyperparameters")


# ----------------------------------------------------
# FIGURE 13: Trigger strategies in the Markov game
# ----------------------------------------------------

def plot_markov_trigger():
    df = load_csv(
        "markov_trigger_grid.csv",
        ["kappa", "punishment", "agent", "gamma", "deviation_gain"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, kappa in zip(axes, sorted(df["kappa"].unique(), reverse=True)):
        for (punishment, agent), rows in df[df["kappa"] == kappa].groupby(
            ["punishment", "agent"]
        ):
            if rows["deviation_gain"].max() <= 1e-9:
                continue

            ax.plot(rows["gamma"], rows["deviation_gain"],
                    linestyle="-" if punishment == "minmax" else "--",
                    label=f"Agent {agent}, {punishment}")

        ax.axvline(0.9, color="#C65D57", linewidth=1)
        ax.set_title(f"Switching cost = {kappa:g}")
        ax.set_xlabel("Discount factor")
        ax.grid(alpha=0.2)

        if not ax.lines[:-1]:
            ax.text(0.5, 0.5, "No profitable deviation\nat any discount factor",
                    transform=ax.transAxes, ha="center")

    axes[0].set_ylabel("Deviation gain (0 = cooperation holds)")
    axes[1].legend(fontsize=8)
    save_figure("fig13_markov_trigger")


def main():
    plot_stage_game()
    plot_learning_curves()
    plot_cooperation_comparison()
    plot_punishment()
    plot_policy_changes()
    plot_sweeps()
    plot_markov_trigger()

    print("\nAll figures generated.")
    print("Figures folder:", FIGURES)
    print("Tables folder:", TABLES)


if __name__ == "__main__":
    main()
