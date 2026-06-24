import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from enum import Enum, auto
    from collections.abc import Callable
    import numpy as np
    import nico2_lib as n2l
    from nico2_lib.typing import NumericArray

    return Callable, NumericArray, n2l, np


@app.cell
def _(n2l):
    data_path = "./data"
    query = n2l.dt.small_mouse_intestine_merfish(data_path)
    reference = n2l.dt.small_mouse_intestine_sc(data_path)
    print(query, "\n", reference)
    return


@app.cell
def _(Callable, NumericArray, mo, n2l, np):
    def cell(
        func: Callable[[NumericArray, NumericArray], float],
    ) -> Callable[[NumericArray], float]:
        def scoring_function(
            cells_true: NumericArray,
            cells_pred: NumericArray,
        ) -> float:
            return np.mean(
                (
                    func(
                        cell_true,
                        cell_pred,
                    )
                    for (
                        cell_true,
                        cell_pred,
                    ) in zip(
                        cells_true,
                        cells_pred,
                    )
                ),
            )

        return scoring_function


    scoring_functions = {
        "Explained Variance (Scikit)": n2l.mt.explained_variance_metric,
        "Explained Variance (Dominic)": n2l.mt.explained_variance_metric_v2,
    }
    mo_dictionary = mo.ui.dictionary(
        {
            "Scoring Function": mo.ui.dropdown(
                scoring_functions,
                value="Explained Variance (Scikit)",
                allow_select_none=False,
            ),
            "Cell or Celltype": mo.ui.dropdown(
                {
                    "cell": cell,
                    "celltype": lambda x: x,
                },
                value="celltype",
                allow_select_none=False,
            ),
        }
    )
    mo_dictionary
    return mo_dictionary, scoring_functions


@app.cell
def _(np):
    counts_size = (20, 20)
    counts = np.random.poisson(lam=1, size=counts_size)
    noisy_counts = counts + np.random.normal(0, 1, size=counts_size)
    counts, noisy_counts
    return counts, noisy_counts


@app.cell
def _(counts, mo_dictionary, noisy_counts):
    value = mo_dictionary.value
    scoring_function = value["Cell or Celltype"](value["Scoring Function"])
    print(scoring_function(counts, noisy_counts))
    return


@app.cell
def _(mo, scoring_functions):
    mo.ui.dictionary(
        {
            "Scoring Functions": mo.ui.multiselect(scoring_functions),
        }
    ).form()
    return


@app.cell
def _(my_dictionary):
    print(my_dictionary.value)
    return


@app.cell
def _():
    a = 2
    return (a,)


@app.cell
def _(a):
    import time

    time.sleep(2)
    b = a + 2
    return (b,)


@app.cell
def _(b):
    print(b)
    return


if __name__ == "__main__":
    app.run()
