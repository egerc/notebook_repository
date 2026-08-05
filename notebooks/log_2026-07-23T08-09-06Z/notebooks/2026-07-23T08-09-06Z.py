import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import nico2_lib as n2l
    import seaborn as sns
    import matplotlib.pyplot as plt
    from conf import load_dataset_configurations

    from log_2026_07_23t08_09_06z.core.api import (
        run_experiment_for_model_and_scope,
        experiment,
        create_experiment_table,
    )
    from log_2026_07_23t08_09_06z.datasets import (
        DatasetConfiguration,
        QueryPlusReference,
        SampleRemainderPanel,
        SingleCellData,
        SpatialSetup,
        PseudospatialSetup,
        NonSpatialSetup,
        HighlyVariableGenes,
    )
    from log_2026_07_23t08_09_06z.models import PredictionScope
    from log_2026_07_23t08_09_06z.types import (
        SamplingSplit,
        DatasetSplit,
        log_result,
        map_result,
        unwrap_result,
        rights,
    )
    from log_2026_07_23t08_09_06z.utils import MinRange

    return (
        PredictionScope,
        create_experiment_table,
        experiment,
        load_dataset_configurations,
        n2l,
        plt,
        rights,
        sns,
    )


@app.cell
def _(load_dataset_configurations, rights):
    dataset_configurations = list(
        rights(load_dataset_configurations("./config_cache"))
    )
    return (dataset_configurations,)


@app.cell
def _(
    PredictionScope,
    create_experiment_table,
    dataset_configurations,
    experiment,
    n2l,
):
    df = create_experiment_table(
        list(
            experiment(
                dataset_configurations,
                [
                    (
                        n2l.pd.NmfPredictor(3),
                        PredictionScope.GLOBAL,
                        "./cache/nmf",
                    ),
                ],
                {
                    "mse": lambda arr1, arr2: n2l.mt.mse_metric(
                        arr1.mean(axis=0), arr2.mean(axis=0)
                    ),
                },
            )
        )
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, plt, sns):
    sns.lmplot(
        df,
        x="n_testing_features",
        y="value",
        hue="model",
        row="setup_strategy",
        col="scoring_function",
    )
    # plt.yscale("log")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
