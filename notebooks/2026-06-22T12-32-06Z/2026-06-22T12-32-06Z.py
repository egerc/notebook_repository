import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from enum import Enum, auto
    from collections.abc import Callable
    import numpy as np
    import pandas as pd
    import nico2_lib as n2l
    import matplotlib.pyplot as plt
    from nico2_lib.typing import NumericArray
    import seaborn as sns

    return pd, plt, sns


@app.cell
def _(pd):
    csv_path = "./run_results.csv"
    df = pd.read_csv(csv_path)
    df
    return (df,)


@app.cell
def _(df, sns):
    sns.catplot(
        df,
        x="scoring_axis",
        y="score",
        hue="scoring_aggregation",
        row="dataset",
        col="statistical_measure",
        kind="box",
        sharey=False,
        showfliers=False,
    )
    return


@app.cell
def _(df, sns):
    sns.catplot(
        df,
        x="preprocessing",
        y="score",
        hue="score_in_raw_space",
        col="statistical_measure",
        row="dataset",
        kind="box",
        sharey=False,
        showfliers=False,
    )
    return


@app.cell
def _(df, plt, sns):
    df["setup"] = df["scoring_axis"] + "\n+ " + df["scoring_aggregation"]

    # Plotting absolute values per metric
    g = sns.catplot(
        data=df,
        hue="setup",
        x="score",
        y="dataset",
        col="statistical_measure",
        col_wrap=3,
        kind="box",
        sharex=False,
        sharey=True,  # CRITICAL: Allows absolute terms to be readable per metric
        height=4,
        aspect=1.2,
        showfliers=False,
        legend=True,
    )
    g.fig.suptitle(
        "Absolute Score Distribution Across Benchmarking Setups",
        y=1.02,
        fontsize=14,
    )
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Create a unique evaluation environment string
    df["data_space"] = (
        df["preprocessing"]
        + " (Eval Raw: "
        + df["score_in_raw_space"].astype(str)
        + ")"
    )
    df["full_configuration"] = df["setup"] + " | " + df["data_space"]

    # Pivot data to see configurations vs metrics
    # We median-aggregate over cell types to get the overall trend
    pivot_df = df.pivot_table(
        index="full_configuration",
        columns="statistical_measure",
        values="score",
        aggfunc="median",
    )

    # Normalize column-wise (Min-Max) so we can compare across metrics in one heatmap
    normalized_pivot = (pivot_df - pivot_df.min()) / (
        pivot_df.max() - pivot_df.min()
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        normalized_pivot,
        annot=pivot_df,
        fmt=".2f",
        cmap="magma",
        cbar_kws={"label": "Normalized Performance (0=Worst, 1=Best)"},
    )
    plt.title("Why One Size Doesn't Fit All: Metric Sensitivity to Space & Setup")
    plt.ylabel("Benchmarking Setup + Data Space")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, pd, plt, sns):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # --- 1. Re-running Method 1 and calculating Effect Sizes ---
    effect_sizes = {}
    for measure in df["statistical_measure"].unique():
        sub_df = df[df["statistical_measure"] == measure]
        formula = "score ~ C(scoring_axis) * C(scoring_aggregation) * C(preprocessing) * C(score_in_raw_space) * C(dataset)"

        model = ols(formula, data=sub_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Calculate Partial Eta-Squared: SS_effect / (SS_effect + SS_error)
        ss_error = anova_table.loc["Residual", "sum_sq"]
        anova_table["partial_eta_sq"] = anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_error)

        # Drop the residual row for the plot
        effect_sizes[measure] = anova_table.drop("Residual")["partial_eta_sq"]

    # Combine into a single DataFrame (Rows = Terms, Columns = Metrics)
    effect_df = pd.DataFrame(effect_sizes)

    # --- 2. Plotting Individual Heatmaps Per Statistical Measure ---
    # Loop through columns to generate one dedicated plot per metric
    for measure in effect_df.columns:
        plt.figure(figsize=(6, 7))

        # Sort terms by effect size so the most critical dependencies rise to the top
        data_to_plot = effect_df[[measure]].sort_values(by=measure, ascending=False)

        sns.heatmap(
            data_to_plot,
            annot=True,
            fmt=".1%",  # Displays as percentage of variance explained
            cmap="YlGnBu",
            cbar_kws={'label': 'Partial Eta-Squared ($\eta^2_p$)'},
            vmin=0,
            vmax=max(0.5, data_to_plot[measure].max()) # Scale nicely dynamically
        )

        plt.title(f"Dependency Driver Analysis: {measure.upper()}", fontsize=12, pad=15)
        plt.xlabel("Statistical Measure")
        plt.ylabel("Pipeline Variables & Combinations")
        plt.tight_layout()
        plt.show()
    return


if __name__ == "__main__":
    app.run()
