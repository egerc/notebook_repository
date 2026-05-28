import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns

    return mo, pd, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test
    """)
    return


@app.cell
def _():
    DATA_PATH = "./results.csv"
    return (DATA_PATH,)


@app.cell
def _(DATA_PATH, pd):
    df = pd.read_csv(DATA_PATH)
    df
    return (df,)


@app.cell
def _(df, sns):
    sns.catplot(
        data=df,
        x="shuffle_probability",
        y="score",
        row="dataset",
        col="score_name",
        kind="box"
    )
    return


if __name__ == "__main__":
    app.run()
