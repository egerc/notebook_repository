import marimo

__generated_with = "0.23.8"
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
def _():
    PATH = "./benchmarking_output.csv"
    return (PATH,)


@app.cell
def _(PATH, pd):
    df = pd.read_csv(PATH)
    df["metric"] = df["metric_category"] + " " + df["function_name"]
    df["model_scope_data_split"] = df["model_scope"] + " " + df["dataset_split"]
    # df = df.drop_nulls()
    df
    return (df,)


@app.cell
def _():
    import altair as alt

    return


@app.cell
def _(df, mo):
    mo.ui.data_explorer(df)
    return


@app.cell(column=1)
def _(df, plt, sns):
    custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    g = sns.catplot(
        data=df,
        y="model_name",
        x="value",
        hue="model_scope_data_split",
        col="dataset_name",
        row="metric",
        kind="box",
        showfliers=False,
        sharex=False,
        palette=custom_palette,
        height=4,
        aspect=1.2,
        legend=True,
        margin_titles=True,
    )

    g.set_axis_labels("Model Architecture", "Metric Value", fontweight="bold")
    g.set_titles(row_template="{row_name}", col_template="{col_name}", weight="bold")

    # g.add_legend(
    #    title="Dataset Split",
    #    loc="upper center",
    #    bbox_to_anchor=(0.5, 1.05),
    #    ncol=3,
    #    frameon=True,
    # )

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
def _(df, normalization_function, radar_plot_multi):
    fig_1 = radar_plot_multi(
        df=df.query("model_scope_data_split == 'celltype query'").assign(
            value_transformed_normalized=lambda x: x.groupby(["metric"])[
                "value_transformed"
            ].transform(normalization_function)
        ),
        x="metric",
        y="value_transformed_normalized",
        hue="model_name",
    )
    fig_1.show()
    return


@app.cell
def _(df, plt, radar_plot_multi, sns):
    model_comparisons = [
        [
            "NMF (10 Komponenten, Log Transform)",
            "NMF (5 Komponenten, Log Transform)",
            "NMF (3 Komponenten, Log Transform)",
        ],
        [
            "NMF (3 Komponenten, Raw Counts)",
            "NMF (3 Komponenten, Log Transform)",
        ],
        [
            "NMF (3 Komponenten, Log Transform)",
            "PCA (3 Komponenten, Log Transform)",
            "SCVI (3 Komponenten, Raw Counts)",
        ],
        [
            "SCVI (3 Komponenten, Raw Counts)",
            "SCVI (5 Komponenten, Raw Counts)",
            "SCVI (10 Komponenten, Raw Counts)",
        ],
        [
            "NMF (3 Komponenten, Log Transform)",
            "NMF (Knee, Log Transform)",
            "NMF (Consensus, Log Transform)",
        ],
        [
            "NMF (3 Komponenten, Log Transform)",
            "NMF (3 Komponenten, Log Transform, Pre-Init)",
        ],
    ]

    # normalization_function = lambda x: (x - x.mean()) / x.std(ddof=0)
    normalization_function = lambda x: x

    for model_comparison in model_comparisons:
        df_filtered = df.query(
            "model_scope_data_split == 'celltype query' and model_name in @model_comparison"
        )

        fig = radar_plot_multi(
            df=(
                df_filtered.assign(
                    value_transformed_normalized=lambda x: x.groupby(["metric"])[
                        "value_transformed"
                    ].transform(normalization_function)
                )
            ),
            x="metric",
            y="value_transformed_normalized",
            hue="model_name",
            # range_lim=(-1.2, 1.2),
        )
        fig.show()

        g2 = sns.catplot(
            data=df_filtered,
            x="model_name",
            y="value",
            hue="model_name",
            col="metric",
            kind="box",
            col_wrap=2,
            sharey=False,
            height=4,
            aspect=1,
            legend=True,
            showfliers=False,
        )
        sns.move_legend(
            g2,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                -0.05,
            ),
            ncol=3,
            title="Model Name",
        )

        g2.set_xticklabels(rotation=45, ha="right")
        g2.set_titles("{col_name}")
        plt.tight_layout()
        plt.show()
    return (normalization_function,)


if __name__ == "__main__":
    app.run()
