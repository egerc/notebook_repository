import pickle

import typer
from anndata.typing import AnnData  # type: ignore
from nico2_lib.predictors import NmfPredictor, ScviPredictor
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass

from log_2026_07_23t08_09_06z.core.api import (
    DatasetConfiguration,
    run_experiment_for_model_and_scope,
    setup_datasets,
)
from log_2026_07_23t08_09_06z.datasets import (
    HighlyVariableGenes,
    NonSpatialSetup,
    PseudospatialSetup,
    QueryPlusReference,
    Random,
    SamplePanel,
    SampleRemainderPanel,
    SingleCellData,
    SpatialPseudospatialSetup,
    SpatialSetup,
)
from log_2026_07_23t08_09_06z.models import Model, PredictionScope
from log_2026_07_23t08_09_06z.types import Err, Ok, Result

app = typer.Typer()

INTESTINE_SC_PATH = (
    "/home/gruengroup/christian/Data/mouse_intestine/intestine_scRNA.h5ad"
)
INTESTINE_SC_CLUSTER_KEY = "cluster"
INTESTINE_MERFISH_PATH = (
    "/home/gruengroup/christian/Data/mouse_intestine/intestine_MERFISH.h5ad"
)
INTESTINE_MERFISH_CLUSTER_KEY = "C_scanvi"


@app.command()
def preprocess_dataset(filepath: str) -> None:
    datasets: list[Result[DatasetConfiguration, Exception]] = []
    for dataset_result in setup_datasets(
        datasets=[
            QueryPlusReference(
                SingleCellData(INTESTINE_MERFISH_PATH, INTESTINE_MERFISH_CLUSTER_KEY),  # type: ignore
                SingleCellData(INTESTINE_SC_PATH, INTESTINE_SC_CLUSTER_KEY),  # type: ignore
            ),
            QueryPlusReference(
                SingleCellData(
                    "/home/gruengroup/christian/Data/benchmark_sample_data/andy_spatial_downsampled.h5ad",  # type: ignore
                    "nico_ct",
                ),
                SingleCellData(
                    "/home/gruengroup/christian/Data/benchmark_sample_data/andy_reference_downsampled.h5ad",  # type: ignore
                    "simple_annot",
                ),
            ),
        ],
        sampling_strategies=[
            SamplePanel(20),
            SamplePanel(50),
            SamplePanel(100),
            SamplePanel(200),
            SamplePanel(200),
            SampleRemainderPanel(5),
            SampleRemainderPanel(10),
            SampleRemainderPanel(25),
            SampleRemainderPanel(50),
            SampleRemainderPanel(100),
        ],
        setup_strategies=[
            SpatialSetup(),
            SpatialPseudospatialSetup(0),
            SpatialPseudospatialSetup(1),
            PseudospatialSetup(500, HighlyVariableGenes(), 0),
            PseudospatialSetup(500, HighlyVariableGenes(), 1),
            PseudospatialSetup(250, HighlyVariableGenes(), 1),
            PseudospatialSetup(500, HighlyVariableGenes(), 0),
            PseudospatialSetup(250, Random(0), 1),
            PseudospatialSetup(500, Random(0), 0),
            NonSpatialSetup(0),
            NonSpatialSetup(1),
        ],
        n_samples=5,
        seed=0,
    ):
        match dataset_result:
            case Ok(dataset):
                print(
                    dataset.dataset, dataset.sampling_strategy, dataset.setup_strategy
                )
            case Err(e):
                print(e)
        datasets.append(dataset_result)

    with open(filepath, "wb") as f:
        pickle.dump(datasets, f)
    print("Done3")


@dataclass(frozen=True, slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class ExperimentResult:
    dataset_configuration_path: str
    dataset_configuration_id: int
    model: Model
    prediction_scope: PredictionScope
    result: dict[int, Result[AnnData, Exception]]

    @classmethod
    def from_run(
        cls,
        dataset_configuration: DatasetConfiguration,
        dataset_configuration_path: str,
        dataset_configuration_id: int,
        model: Model,
        prediction_scope: PredictionScope,
    ) -> "ExperimentResult":
        results = run_experiment_for_model_and_scope(
            dataset_configuration, model, prediction_scope
        )
        return cls(
            dataset_configuration_path=dataset_configuration_path,
            dataset_configuration_id=dataset_configuration_id,
            model=model,
            prediction_scope=prediction_scope,
            result=results,
        )

    @classmethod
    def try_from_path(cls, path: str) -> Result["ExperimentResult", Exception]:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return Ok(cls(**data))
        except Exception as e:  # noqa
            return Err(e)


@app.command()
def run_experiment(dataset_configs: str, output_dir: str) -> None:
    with open(dataset_configs, "rb") as f:
        datasets: list[Result[DatasetConfiguration, Exception]] = pickle.load(f)
    model_setups: list[tuple[Model, PredictionScope]] = [
        (NmfPredictor(n_components=3), PredictionScope.GLOBAL),
        (NmfPredictor(n_components=3), PredictionScope.CELLTYPE),
        (ScviPredictor(n_factors=10), PredictionScope.GLOBAL),
    ]
    experiment_results: list[Result[ExperimentResult, Exception]] = []
    for dataset_configuration_id, dataset_configuration_result in enumerate(datasets):
        match dataset_configuration_result:
            case Ok(dataset_configuration):
                assert isinstance(dataset_configuration_result, DatasetConfiguration)
                for model, prediction_scope in model_setups:
                    experiment_result = ExperimentResult.from_run(
                        dataset_configuration,
                        dataset_configs,
                        dataset_configuration_id,
                        model,
                        prediction_scope,
                    )
                    experiment_results.append(Ok(experiment_result))
            case Err():
                experiment_results.append(dataset_configuration_result)

    with open(output_dir, "wb") as f:
        pickle.dump(experiment_results, f)


if __name__ == "__main__":
    app()
