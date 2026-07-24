import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import polars as pl
    import seaborn as sns

    return pd, plt, sns


@app.cell
def _(pd):
    df_enrichment_scores = pd.read_csv(
        "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-05-13T07-33-19Z/output.csv"
    )
    df_enrichment_scores
    return (df_enrichment_scores,)


@app.cell
def _(df_enrichment_scores, plt, sns):
    plot_data = df_enrichment_scores.query(
        "correlation_name == 'pearson' and "
        "scoring_function_name == 'max_cosine_alignment_scoring' and "
        "preprocessing_name == 'identity'"
    )

    # 2. Initialize the figure with explicit dimensions and local styling
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # Set the background color to white explicitly for this axis
    ax.set_facecolor("white")

    # 3. Create the boxplot with publication-quality overrides
    sns.boxplot(
        data=plot_data,
        x="shuffle_probability",
        y="score",
        ax=ax,
        color="#d9d9d9",  # Neutral professional gray
        width=0.6,  # Slightly narrower boxes
        linewidth=1.2,  # Crisp, explicit line weights
        fliersize=3,  # Controlled, non-obtrusive outliers
        flierprops={"markerfacecolor": "0.4", "markeredgecolor": "none"},
    )

    # 4. Local Typography and Axis Label Styling
    font_options = {"fontsize": 10, "fontweight": "bold"}

    ax.set_xlabel("Shuffle Probability", labelpad=8, **font_options)
    ax.set_ylabel("Enrichment Score", labelpad=8, **font_options)

    # Local Tick Label Styling
    ax.tick_params(axis="both", which="major", labelsize=9, colors="black")

    # 5. Clean up borders locally
    # Explicitly turn on bottom and left lines, turn off top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Apply standard line properties to the visible spines
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)

    # Trim option alternative (offsets the axis lines for a polished look)
    sns.despine(ax=ax, trim=True)

    # 6. Save in a lossless vector format
    plt.tight_layout()
    plt.savefig("enrichment_scores_boxplot.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    return


@app.cell
def _():
    PATH = "./benchmarking_output.csv"
    return (PATH,)


@app.cell
def _(PATH, pd):
    df = pd.read_csv(PATH)
    # df = df.query("dataset_name != 'Jennifer Spatial'")
    df["metric"] = df["metric_category"] + " " + df["function_name"]
    df["model_scope_data_split"] = (
        df["model_scope"] + " " + df["dataset_split"]
    )
    df["model_name_full"] = (
        df["model_name"] + " | " + df["model_scope"] + " | " + df["transform"]
    )
    # df = df.drop_nulls()
    df
    return (df,)


@app.cell
def _(df, plt, sns):
    df_pivoted = df.pivot_table(
        index=[
            "dataset_name",
            "model_name",
            "transform",
            "model_scope",
            "dataset_split",
        ],
        columns="metric",
        values="value_transformed",
    ).reset_index()

    # 5. Generate the Seaborn Pairplot
    sns.set_theme(style="ticks")

    g = sns.pairplot(
        df_pivoted,
        hue="model_name",
        corner=False,
        diag_kind="kde",  # Draws smooth kernel density estimates on the diagonal
        plot_kws={"alpha": 0.7, "s": 40},
    )

    # --- Fix Overlapping Axis Labels ---
    # 1. Rotate the X-axis labels and adjust alignment
    for ax in g.axes.flat:
        if ax is not None:
            # Rotate x labels so they don't hit each other sideways
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            # Add explicit padding between the axis title and the tick numbers
            ax.xaxis.labelpad = 15
            ax.yaxis.labelpad = 15

    # 2. Prevent the overall title from bleeding into the top row of plots
    g.fig.suptitle(
        "Pairwise Correlations of Transformed Metric Values",
        y=1.05,  # Pushed slightly higher up
        fontsize=14,
    )

    # 3. Tell matplotlib to automatically compute a clean layout boundary
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, sns):
    sns.histplot(
        df.query(
            "metric == 'feature_prediction_performance explained_variance'"
        ).query("value > -10"),
        x="value",
    )
    return


@app.cell
def _():
    return


@app.cell
def _(df, plt, sns):
    g_metric_transform = sns.relplot(
        data=df,
        x="value",
        y="value_transformed",
        hue="model_name",
        col="metric",
        col_wrap=2,
        kind="scatter",
        s=10,
        alpha=0.6,
        facet_kws={
            "sharex": False,
            "sharey": True,
        },
    )

    # Optional: Further customize titles if you want to remove the "metric = " prefix
    g_metric_transform.set_titles(template="{col_name}")

    plt.show()
    return


@app.cell(column=1)
def _(df, plt, sns):
    custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    g = sns.catplot(
        data=df.assign(
            model_name_transform=lambda x: (
                x["model_name"] + " | " + df["transform"]
            )
        ),
        y="model_name_full",
        x="value_transformed",
        hue="dataset_split",
        col="dataset_name",
        row="metric",
        kind="box",
        showfliers=False,
        sharex=False,
        # palette=custom_palette,
        height=6,
        aspect=1,
        legend=True,
        margin_titles=False,
    )
    # Set up a grid for multiple variables

    # Map the boxplot to the grid

    g.set_axis_labels("Model Architecture", "Metric Value", fontweight="bold")
    g.set_titles(
        row_template="{row_name}", col_template="{col_name}", weight="bold"
    )

    g.add_legend(
        title="Dataset Split",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=True,
    )

    sns.despine(left=True)
    plt.tight_layout()
    g.savefig("test.svg")
    g
    return


@app.cell(column=2)
def _(pd):
    import plotly.graph_objects as go
    from plotly.graph_objs._figure import Figure

    def radar_plot_multi(
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        range_lim: tuple[int | float, int | float] | None = None,
        title: str | None = None,
    ) -> Figure:
        stats = df.groupby([x, hue])[y].mean().reset_index()

        metrics = sorted(stats[x].unique().tolist())
        metrics_closed = metrics + [metrics[0]]

        fig = go.Figure()

        for category in stats[hue].unique():
            category_data = stats[stats[hue] == category]
            values_dict = dict(zip(category_data[x], category_data[y]))
            values = [values_dict.get(m, 0) for m in metrics]
            values_closed = values + [values[0]]

            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=metrics_closed,
                    mode="lines+markers",
                    name=str(category),
                    fill="toself",
                    fillcolor="rgba(0,0,0,0)",
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=range_lim,
                    tickfont_size=10,
                    gridcolor="lightgrey",
                )
            ),
            title=title,
            showlegend=True,
        )

        return fig

    return (radar_plot_multi,)


@app.cell
def _():
    from typing import Literal

    resolve: Literal["dataset_name"] | None = None
    metrics = [
        # "embedding_autocorrelation pearsonr",
        "embedding_structure multivariate_gearys_c",
        "embedding_sparsity gini",
        # "coverage gini",
        # "feature_prediction_performance mean_squared_error",
        "feature_prediction_performance explained_variance",
        "biological_enrichment max_cosine_alignment",
    ]
    identity = lambda x: x
    z_score = lambda x: (x - x.mean()) / x.std(ddof=0)
    transform_func = z_score
    return resolve, transform_func


@app.cell
def _(
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
    transform_func,
):
    for (value_1,), sub_df_1 in (
        df.query("dataset_split == 'query'")
        if resolve is not None
        else (((None,), df),)
    ):
        fig_1 = radar_plot_multi(
            df=sub_df_1.query("dataset_split == 'query'")
            .query("metric in @metrics")
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(transform_func)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(0, 1.1),
            title=value_1,
        )
        fig_1.show()
    return


@app.cell
def _():
    comparison_columns = ["model_name", "model_scope", "transform"]
    return (comparison_columns,)


@app.cell
def _(
    comparison_columns,
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
    transform_func,
):
    comparison_2 = [
        ("NMF (3 components)", "celltype", "raw"),
        ("NMF (10 components)", "global", "raw"),
        ("scVI (10 Components)", "global", "raw"),
        ("Tangram", "global", "raw"),
    ]

    for (value_2,), sub_df_2 in (
        df.groupby([resolve]) if resolve is not None else (((None,), df),)
    ):
        fig_2 = radar_plot_multi(
            df=sub_df_2.query("dataset_split == 'query'")
            .query("metric in @metrics")
            .loc[
                lambda d: d.set_index(comparison_columns).index.isin(
                    comparison_2
                )
            ]
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(transform_func)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(-1.4, 1.4),
            title=value_2,
        )

        fig_2.show()
    return (fig_2,)


@app.cell
def _(fig_2, mo):
    mo.ui.plotly(
        fig_2,
        config={"toImageButtonOptions": {"format": "svg"}},
    )
    return


@app.cell
def _(
    comparison_columns,
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
    transform_func,
):
    comparison_3 = [
        ("NMF (3 components)", "celltype", "log"),
        ("NMF (5 components)", "celltype", "log"),
        ("NMF (10 components)", "celltype", "log"),
    ]

    for (value_3,), sub_df_3 in (
        df.groupby([resolve]) if resolve is not None else (((None,), df),)
    ):
        fig_3 = radar_plot_multi(
            df=sub_df_3.query("dataset_split == 'query'")
            .query("metric in @metrics")
            .loc[
                lambda d: d.set_index(comparison_columns).index.isin(
                    comparison_3
                )
            ]
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(transform_func)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(-1.2, 1.2),
            title=value_3,
        )
        fig_3.show()
    return (fig_3,)


@app.cell
def _(fig_3, mo):
    mo.ui.plotly(
        fig_3,
        config={"toImageButtonOptions": {"format": "svg"}},
    )
    return


@app.cell
def _(
    comparison_columns,
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
    transform_func,
):
    comparison_4 = [
        ("NMF (3 components)", "celltype", "raw"),
        ("NMF (3 components)", "celltype", "log"),
    ]

    for (value_4,), sub_df_4 in (
        df.groupby([resolve]) if resolve is not None else (((None,), df),)
    ):
        fig_4 = radar_plot_multi(
            df=sub_df_4.query("dataset_split == 'query'")
            .query("metric in @metrics")
            .loc[
                lambda d: d.set_index(comparison_columns).index.isin(
                    comparison_4
                )
            ]
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(transform_func)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(0, 0.65),
            title=value_4,
        )
        fig_4.show()
    return (fig_4,)


@app.cell
def _(fig_4, mo):
    mo.ui.plotly(
        fig_4,
        config={"toImageButtonOptions": {"format": "svg"}},
    )
    return


@app.cell
def _(
    comparison_columns,
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
    transform_func,
):
    comparison_5 = [
        ("NMF (3 components)", "celltype", "raw"),
        ("NMF (Consensus)", "celltype", "raw"),
    ]

    for (value_5,), sub_df_5 in (
        df.groupby([resolve]) if resolve is not None else (((None,), df),)
    ):
        fig_5 = radar_plot_multi(
            df=sub_df_5.query("dataset_split == 'query'")
            .query("metric in @metrics")
            .loc[
                lambda d: d.set_index(comparison_columns).index.isin(
                    comparison_5
                )
            ]
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(transform_func)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(-0.55, 0.55),
            title=value_5,
        )
        fig_5.show()
    return (fig_5,)


@app.cell
def _(fig_5, mo):
    mo.ui.plotly(
        fig_5,
        config={"toImageButtonOptions": {"format": "svg"}},
    )
    return


if __name__ == "__main__":
    app.run()
