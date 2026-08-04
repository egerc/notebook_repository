import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import nico2_lib as n2l
    from log_2026_07_23t08_09_06z.core.api import (
        run_experiment_for_model_and_scope,
    )
    from log_2026_07_23t08_09_06z.evaluation import (
        apply_reconstruction_scoring_func,
    )
    from log_2026_07_23t08_09_06z.types import (
        bind_result,
        map_result,
        starbind_result,
        starmap_result,
    )
    from log_2026_07_23t08_09_06z.models import PredictionScope
    from log_2026_07_23t08_09_06z.utils import read_h5ad, MinRange
    from log_2026_07_23t08_09_06z.datasets import (
        DatasetConfiguration,
        QueryPlusReference,
        SingleCellData,
        SpatialSetup,
        SampleRemainderPanel,
    )

    return (
        DatasetConfiguration,
        MinRange,
        PredictionScope,
        QueryPlusReference,
        SampleRemainderPanel,
        SingleCellData,
        SpatialSetup,
        bind_result,
        n2l,
        run_experiment_for_model_and_scope,
    )


@app.cell
def _():
    adata_path = "/home/gruengroup/christian/Data/mouse_intestine/intestine_MERFISH.h5ad"
    return


@app.cell
def _(
    DatasetConfiguration,
    MinRange,
    QueryPlusReference,
    SampleRemainderPanel,
    SingleCellData,
    SpatialSetup,
):
    dataset_configuration = DatasetConfiguration.try_from_setup(
        QueryPlusReference(
            SingleCellData(
                "/home/gruengroup/christian/Data/mouse_intestine/intestine_MERFISH.h5ad",
                "C_scanvi",
            ),
            SingleCellData(
                "/home/gruengroup/christian/Data/mouse_intestine/intestine_scRNA.h5ad",
                "cluster",
            ),
        ),
        SpatialSetup(),
        SampleRemainderPanel(20),
        n_samples=2,
        seed=0,
        filtering_config=MinRange(20, 10000),
    )
    return (dataset_configuration,)


@app.cell
def _(
    PredictionScope,
    bind_result,
    dataset_configuration,
    n2l,
    run_experiment_for_model_and_scope,
):
    res = bind_result(
        dataset_configuration,
        lambda dataset_configuration: run_experiment_for_model_and_scope(
            dataset_configuration,
            model=n2l.pd.NmfPredictor(n_components=2),
            prediction_scope=PredictionScope.GLOBAL,
        ),
    )
    return (res,)


@app.cell
def _(res):
    res
    return


if __name__ == "__main__":
    app.run()
