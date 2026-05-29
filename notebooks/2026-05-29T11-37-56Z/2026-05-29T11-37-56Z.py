import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo

    return


app._unparsable_cell(
    r"""
    import polars as pl
    import seaborn as
    """,
    name="_"
)


@app.cell
def _():
    PATH = "~/Projects/notebook_repository/notebooks/2026-04-24T06-54-05Z/benchmarking_output.csv"
    return (PATH,)


@app.cell
def _(PATH, pl):
    df = pl.read_csv(PATH)
    df = df.drop_nulls()
    df
    return (df,)


@app.cell(column=1)
def _(df, sns):
    sns.catplot(
        data=df,
        x="model_name",
        y="value",
        hue="dataset_split",
        row="dataset_name",
        col="function_name",
        kind="box",
        showfliers=False,
        sharey=False
    )
    return


if __name__ == "__main__":
    app.run()
