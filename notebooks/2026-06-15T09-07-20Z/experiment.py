from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Generator, Sequence
from functools import partial
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Callable, Literal

import mlflow
import nico2_lib as n2l
import numpy as np
import pydantic
import sqlmodel
import yaml
from anndata import read_h5ad
from anndata.typing import AnnData
from nico2_lib.typing import IndexArray, NumericArray
from pandas.core.generic import pickle
from pydantic.dataclasses import dataclass
from pydantic.functional_validators import model_validator
from pydantic.types import FilePath, NonNegativeInt, PositiveInt
from pydantic_core import PydanticCustomError


@dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ExperimentConfig:
    """Configuration for the experiment."""

    predictors: Annotated[list[PredictorConfig], pydantic.Field(min_length=1)]
    datasets: Annotated[list[DatasetConfig], pydantic.Field(min_length=1)]
    seed: NonNegativeInt | None
    sampling_config: SamplingConfig | None
    mlflow_config: MlFlowConfig | None
    log_level: int = 20


@dataclass(frozen=True, slots=True)
class NmfConfig:
    type: Literal["nmf"]
    n_components: PositiveInt | Literal["consensus", "knee"] | None
    init: Literal["random", "nndsvd", "nndsvda", "nndsvdar", "custom"] | None = None
    random_state: int | None = None
    max_iter: int = 200
    alpha_W: float = 0
    alpha_H: float | Literal["same"] = "same"
    l1_ratio: float = 0
    preprocessing_steps: Sequence[Callable[[NumericArray], NumericArray]] | None = None
    pre_init: bool = False
    solver: Literal["cd", "mu"] = "cd"
    beta_loss: str = "frobenius"
    n_shared_features: int | None = None
    embedding_size: int | None = None

    def instantiate_model(self) -> n2l.pd.NmfPredictor:
        n_components: PositiveInt | Callable[[NumericArray], int] | None
        match self.n_components:
            case int():
                n_components = self.n_components
            case "consensus":
                n_components = partial(  # type: ignore
                    n2l.pd.consensus_nmf,
                    k_range=range(2, 10),
                    n_runs=10,
                    max_iter=200,
                )
            case "knee":
                n_components = partial(  # type: ignore
                    n2l.pd.find_k_by_inflection,
                    k_range=range(2, 10),
                    max_iter=200,
                )
            case _:
                n_components = self.n_components

        return n2l.pd.NmfPredictor(
            n_components=n_components,
            random_state=self.random_state,
            init=self.init,
            max_iter=self.max_iter,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio,
            preprocessing_steps=self.preprocessing_steps,
            pre_init=self.pre_init,
            solver=self.solver,
            beta_loss=self.beta_loss,
            n_shared_features=self.n_shared_features,
        )


@dataclass(frozen=True, slots=True)
class PcaConfig:
    type: Literal["pca"]
    n_components: PositiveInt | None

    def instantiate_model(self) -> n2l.pd.PcaPredictor:
        return n2l.pd.PcaPredictor(
            n_components=self.n_components,
        )


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    name: str
    scope: Literal["global", "celltype", "both"]
    model: Annotated[
        NmfConfig | PcaConfig,
        pydantic.Field(discriminator="type"),
    ]


@dataclass(frozen=True)
class MlFlowConfig:
    run_id: str | None
    experiment_id: str | None
    run_name: str | None
    nested: bool
    parent_run_id: str | None
    tags: dict[str, Any] | None
    description: str | None
    log_system_metrics: bool | None


@dataclass(frozen=True)
class SamplingConfig:
    n_samples: PositiveInt
    sample_length: PositiveInt


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    query_path: FilePath
    query_cluster_key: str
    reference_path: FilePath
    reference_cluster_key: str
    min_n_cells_celltype: PositiveInt
    n_cells: PositiveInt | None
    max_n_pcs: PositiveInt
    max_n_neighbours: PositiveInt

    @model_validator(mode="after")
    def check_cluster_keys(self) -> DatasetConfig:
        for dataset_split, dataset_path, cluster_key in [
            ("query", self.query_path, self.query_cluster_key),
            ("reference", self.reference_path, self.reference_cluster_key),
        ]:
            match read_h5ad(dataset_path, backed="r"):
                case AnnData(obs=obs) if cluster_key not in obs.columns:
                    raise PydanticCustomError(
                        "value_error",
                        "Cluster key '{cluster_key}' not found in dataset split "
                        + "'{dataset_split}', valid columns are {valid_columns}",
                        {
                            "dataset_split": dataset_split,
                            "cluster_key": cluster_key,
                            "valid_columns": list(obs.columns),
                        },
                    )
                case _:
                    pass

        return self

    def load_query(self) -> AnnData:
        return read_h5ad(self.query_path)

    def load_reference(self) -> AnnData:
        return read_h5ad(self.reference_path)


class PklType(sqlmodel.TypeDecorator):
    impl = sqlmodel.LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return pickle.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        # if value is not None:
        #    return pickle.loads(value)
        return pickle.loads(value)  # type: ignore


class Dataset(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    name: str
    shared_cell_types: list[str] = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))
    shared_features: list[str] = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))

    celltypes: list[Celltype] = sqlmodel.Relationship(back_populates="dataset")
    samples: list[Sample] = sqlmodel.Relationship(back_populates="dataset")


class Sample(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    id_of_sample: int
    train_idx: IndexArray = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))
    test_idx: IndexArray = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))

    dataset_id: int | None = sqlmodel.Field(default=None, foreign_key="dataset.id")
    dataset: Dataset = sqlmodel.Relationship(back_populates="samples")


class Celltype(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    name: str
    reference_counts_matrix: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    reference_pca_embedding: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    reference_umap_embedding: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    reference_adjacency_matrix: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    query_counts_matrix: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    query_pca_embedding: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    query_umap_embedding: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    query_adjacency_matrix: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )

    dataset_id: int | None = sqlmodel.Field(default=None, foreign_key="dataset.id")
    dataset: Dataset = sqlmodel.Relationship(back_populates="celltypes")


class Model(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    name: str
    scope: Literal["global", "celltype", "both"]


class Result(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)


def generate_datasets(
    dataset_configs: Sequence[DatasetConfig],
) -> Generator[tuple[Dataset, tuple[AnnData, AnnData], DatasetConfig], None, None]:
    for dataset_config in dataset_configs:
        (
            reference,
            query,
        ) = (
            dataset_config.load_reference(),
            dataset_config.load_query(),
        )
        shared_celltypes: list[str] = list(
            set(reference.obs[dataset_config.reference_cluster_key]).intersection(
                set(query.obs[dataset_config.query_cluster_key])
            )
        )
        shared_features: list[str] = list(
            set(reference.var_names).intersection(set(query.var_names))
        )
        yield (
            Dataset(
                name=dataset_config.name,
                shared_cell_types=shared_celltypes,
                shared_features=shared_features,
            ),
            (query, reference),
            dataset_config,
        )


def generate_samples(
    sampling_config: SamplingConfig,
    rng: np.random.Generator,
    dataset: Dataset,
) -> Generator[Sample, None, None]:
    indices = np.vstack(
        [
            rng.permutation(len(dataset.shared_features))
            for _ in range(sampling_config.n_samples)
        ]
    )
    test_idxs, train_idxs = np.split(indices, [sampling_config.sample_length], axis=1)
    for id_of_sample, (train_idx, test_idx) in enumerate(zip(train_idxs, test_idxs)):
        yield Sample(
            id_of_sample=id_of_sample,
            train_idx=train_idx,
            test_idx=test_idx,
            dataset=dataset,
        )


def generate_celltypes(
    dataset: Dataset,
    reference: AnnData,
    query: AnnData,
    dataset_config: DatasetConfig,
) -> Generator[Celltype, None, None]:
    for celltype_name in dataset.shared_cell_types:
        yield Celltype(
            name=celltype_name,
            reference_counts_matrix=reference[
                reference.obs[dataset_config.reference_cluster_key] == celltype_name
            ].X,
            query_counts_matrix=query[
                query.obs[dataset_config.query_cluster_key] == celltype_name
            ].X,
        )


def generate_models(
    predictor_configs: Sequence[PredictorConfig],
) -> Generator[Model, None, None]: ...


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        experiment_config = ExperimentConfig(**yaml.safe_load(f))
    logging.basicConfig(
        level=experiment_config.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(
        f"Starting MLflow run using the following configuration:\n {pydantic.RootModel(experiment_config).model_dump_json(indent=2)}"
    )
    rng = np.random.default_rng(experiment_config.seed)
    mlflow_config = experiment_config.mlflow_config or MlFlowConfig(
        run_id=None,
        experiment_id=None,
        run_name=None,
        nested=False,
        parent_run_id=None,
        tags=None,
        description=None,
        log_system_metrics=None,
    )
    with mlflow.start_run(
        run_id=mlflow_config.run_id,
        experiment_id=mlflow_config.experiment_id,
        run_name=mlflow_config.run_name,
        nested=mlflow_config.nested,
        parent_run_id=mlflow_config.parent_run_id,
        tags=mlflow_config.tags,
        description=mlflow_config.description,
        log_system_metrics=mlflow_config.log_system_metrics,
    ):
        mlflow.log_params(experiment_config.__dict__)
        with TemporaryDirectory() as _:
            for (
                dataset,
                (reference, query),
                dataset_config,
            ) in generate_datasets(
                experiment_config.datasets,
            ):
                for sample in generate_samples(
                    sampling_config=(
                        experiment_config.sampling_config
                        or SamplingConfig(
                            n_samples=5,
                            sample_length=20,
                        )
                    ),
                    rng=rng,
                    dataset=dataset,
                ):
                    for celltype in generate_celltypes(
                        dataset=dataset,
                        reference=reference,
                        query=query,
                        dataset_config=dataset_config,
                    ):
                        for model in generate_models(
                            predictor_configs=experiment_config.predictors,
                        ):



if __name__ == "__main__":
    main()
