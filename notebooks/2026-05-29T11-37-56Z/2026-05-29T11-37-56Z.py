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
def _(df, mo):
    mo.ui.data_explorer(df)
    return


@app.cell(column=1)
def _(df, plt, sns):
    custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    g = sns.catplot(
        data=df,
        x="model_name",
        y="value",
        hue="model_scope_data_split",
        row="dataset_name",
        col="metric",
        kind="box",
        showfliers=False,
        sharey=False,
        palette=custom_palette,
        height=4,
        aspect=1.2,
        legend=False,
        margin_titles=True,
    )

    g.set_axis_labels("Model Architecture", "Metric Value", fontweight="bold")
    g.set_titles(row_template="{row_name}", col_template="{col_name}", weight="bold")

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.yaxis.set_major_formatter("{x:.2f}")

    g.add_legend(
        title="Dataset Split",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=True,
    )

    sns.despine(left=True)  # Remove left spine for a modern "open" look
    plt.tight_layout()
    g.savefig("test.svg")
    g
    return


if __name__ == "__main__":
    app.run()
