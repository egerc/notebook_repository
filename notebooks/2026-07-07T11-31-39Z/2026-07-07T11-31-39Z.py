import marimo

__generated_with = "0.23.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo

    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import polars as pl
    import seaborn as sns

    return pd, plt, sns


@app.cell
def _():
    PATH = "./benchmarking_output.csv"
    return (PATH,)


@app.cell
def _(PATH, pd):
    df = pd.read_csv(PATH)
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
        sharex=True,
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
def _(df):
    df["metric"].unique()
    return


@app.cell
def _():
    from typing import Literal

    resolve: Literal["dataset_name"] | None = None
    metrics = [
        #"embedding_autocorrelation pearsonr",
        "embedding_structure multivariate_gearys_c",
        "embedding_sparsity gini",
        # "coverage gini",
        #"feature_prediction_performance mean_squared_error",
        "feature_prediction_performance explained_variance",
    ]
    id = lambda x: x
    z_score = lambda x: (x - x.mean()) / x.std(ddof=0)
    transform_func = id
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
        ("NMF (3 components)", "celltype", "log"),
    ]

    for (value_2,), sub_df_2 in (
        df.groupby([resolve]) if resolve is not None else (((None,), df),)
    ):
        fig_2 = radar_plot_multi(
            df=sub_df_2.query("dataset_split == 'query'")
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
            range_lim=(0, 1.1),
            title=value_2,
        )
        fig_2.show()
    return


@app.cell
def _(
    comparison_columns,
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
):
    comparison_3 = [
        ("NMF (3 components)", "celltype", "raw"),
        ("NMF (5 components)", "celltype", "log"),
        ("NMF (10 components)", "celltype", "log"),
    ]

    for (value_3,), sub_df_3 in (
        df.groupby([resolve]) if resolve is not None else (((None,), df),)
    ):
        fig_3 = radar_plot_multi(
            df=sub_df_3.query("dataset_split == 'query'")
            .loc[
                lambda d: d.set_index(comparison_columns).index.isin(
                    comparison_3
                )
            ]
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(lambda x: x)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(0, 1.1),
            title=value_3,
        )
        fig_3.show()
    return


@app.cell
def _(
    comparison_columns,
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
):
    comparison_4 = [
        ("NMF (3 components)", "celltype", "log"),
        ("scVI (3 Components)", "global", "raw"),
        ("Tangram", "celltype", "raw"),
        ("MofaFlex (3 components)", "celltype", "raw"),
        ("MofaFlex (3 components)", "global", "raw"),
    ]

    for (value_4,), sub_df_4 in (
        df.groupby([resolve]) if resolve is not None else (((None,), df),)
    ):
        fig_4 = radar_plot_multi(
            df=sub_df_4.query("dataset_split == 'query'")
            .loc[
                lambda d: d.set_index(comparison_columns).index.isin(
                    comparison_4
                )
            ]
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(lambda x: x)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(0, 1.1),
            title=value_4,
        )
        fig_4.show()
    return


@app.cell
def _(
    comparison_columns,
    df,
    radar_plot_multi,
    resolve: "Literal[\"dataset_name\"] | None",
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
            .loc[
                lambda d: d.set_index(comparison_columns).index.isin(
                    comparison_5
                )
            ]
            .assign(
                value_transformed_normalized=lambda x: x.groupby(["metric"])[
                    "value_transformed"
                ].transform(lambda x: x)
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name_full",
            range_lim=(0, 1.1),
            title=value_5,
        )
        fig_5.show()
    return


if __name__ == "__main__":
    app.run()
