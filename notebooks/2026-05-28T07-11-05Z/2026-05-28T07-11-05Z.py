import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import polars as pl

    return pl, sns


@app.cell
def _(pl):
    df = pl.read_csv(
        "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-05-28T07-11-05Z/results.csv"
    )
    return (df,)


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration
    """)
    return


@app.cell
def _(df, mo):
    mo.ui.dataframe(df)
    return


@app.cell
def _(df, mo):
    mo.ui.data_explorer(df)
    return


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _(df_long, mo):
    mo.ui.data_explorer(df_long)
    return


@app.cell
def _(alt, df):
    _chart = (
        alt.Chart(df)
        .mark_boxplot()
        .encode(
            # Using ordinal (:O) for the x-axis ensures a box is drawn for each unique probability
            x=alt.X("shuffle_probability:O", title="Shuffle Probability"),
            # Remove aggregate='mean' so the boxplot can calculate the distribution
            y=alt.Y("marker_recall_top50_DE:Q", title="Marker Recall"),
            column=alt.Column("dataset:N", title="Dataset"),
            tooltip=["shuffle_probability", "marker_recall_top50_DE", "dataset"],
        )
        .properties(height=150, width=350)
        .configure_axis(grid=True)
    )

    _chart
    return


@app.cell
def _(df):
    df_long = df.unpivot(
        on=[
            "mean_marker_logFC",
            "fraction_markers_significant_DE",
            "correct_top_score_fraction",
        ],
        index=[
            "dataset",
            "sample_id",
            "cluster_key",
            "marker_json",
            "shuffle_probability",
            "score_name",
            "cell_type",
        ],
        variable_name="scoring_type",
        value_name="score",
    )
    return (df_long,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting
    """)
    return


@app.cell
def _(df_long, sns):
    g = sns.relplot(
        data=df_long.to_pandas(),
        x="shuffle_probability",
        y="score",
        # hue="cell_type",
        row="dataset",
        col="scoring_type",
        kind="line",
        # sharey=False
        facet_kws={"sharey": False},
    )
    g.savefig("summary.svg")
    g
    return


if __name__ == "__main__":
    app.run()
