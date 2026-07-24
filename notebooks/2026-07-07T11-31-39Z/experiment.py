import logging
import os
import pprint
from argparse import ArgumentParser, Namespace
from collections.abc import Generator, Sequence
from contextlib import nullcontext
from enum import Enum
from functools import partial, reduce
from itertools import product
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Callable, Literal, assert_never

import mlflow
import mofaflex
import nico2_lib as n2l
import numpy as np
import pydantic
import scanpy as sc
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
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from umap import UMAP


@dataclass(frozen=True, slots=True)
class MofaFlexConfig:
    type: Literal["mofaflex"]
    n_components: int

    def instantiate_model(self) -> n2l.pd.MofaFlexClassicPredictor:
        return n2l.pd.MofaFlexClassicPredictor(
            n_components=self.n_components,
            max_epochs=200,
        )


@dataclass(frozen=True, slots=True)
class TangramConfig:
    type: Literal["tangram"]

    def instantiate_model(self) -> n2l.pd.TangramPredictor:
        return n2l.pd.TangramPredictor()


@dataclass(frozen=True, slots=True)
class ScviConfig:
    type: Literal["scvi"]
    n_factors: int | None = None
    max_epochs: int = 200

    def instantiate_model(self) -> n2l.pd.ScviPredictor:
        return n2l.pd.ScviPredictor(
            n_factors=self.n_factors,
            max_epochs=self.max_epochs,
        )


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
    scope: Literal["global", "celltype"]
    transform: Literal["log"] | None
    model: Annotated[
        NmfConfig | PcaConfig | ScviConfig | TangramConfig | MofaFlexConfig,
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
    panel_size: PositiveInt


@dataclass(frozen=True)
class DatasetConfigSC:
    name: str
    path: FilePath
    cluster_key: str
    min_n_cells_per_celltype: PositiveInt
    max_n_pcs: PositiveInt
    max_n_neighbours: PositiveInt
    remove_constant_features: bool
    n_cells: PositiveInt | None = None


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
    def check_cluster_keys(self) -> "DatasetConfig":
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


@dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ExperimentConfig:
    """Configuration for the experiment."""

    predictors: Annotated[list[PredictorConfig], pydantic.Field(min_length=1)]
    datasets: Annotated[list[DatasetConfigSC], pydantic.Field(min_length=1)]
    seed: NonNegativeInt | None
    sampling_config: SamplingConfig | None
    mlflow_config: MlFlowConfig | None
    log_level: int = 20


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


class ModelScope(str, Enum):
    GLOBAL = "global"
    CELLTYPE = "celltype"


class Sample(sqlmodel.SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    id_of_sample: int
    train_idx: IndexArray = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))
    test_idx: IndexArray = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))

    results: list["Result"] = sqlmodel.Relationship(back_populates="sample")

    dataset_id: int | None = sqlmodel.Field(default=None, foreign_key="dataset.id")
    dataset: "Dataset" = sqlmodel.Relationship(back_populates="samples")


class Celltype(sqlmodel.SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}
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

    results: list["Result"] = sqlmodel.Relationship(back_populates="celltype")

    dataset_id: int | None = sqlmodel.Field(default=None, foreign_key="dataset.id")
    dataset: "Dataset" = sqlmodel.Relationship(back_populates="celltypes")


class Dataset(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    name: str
    celltype_names: list[str] = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))
    feature_names: list[str] = sqlmodel.Field(sa_column=sqlmodel.Column(PklType))

    celltypes: list["Celltype"] = sqlmodel.Relationship(back_populates="dataset")
    samples: list["Sample"] = sqlmodel.Relationship(back_populates="dataset")


class Model(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    name: str
    scope: ModelScope
    transform: str

    results: list["Result"] = sqlmodel.Relationship(back_populates="model")


class Result(sqlmodel.SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}
    id: int | None = sqlmodel.Field(default=None, primary_key=True)

    model_feature_embedding: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    model_embedding_reference: NumericArray | None = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    model_counts_reference: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    model_embedding_query: NumericArray | None = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )
    model_counts_query: NumericArray = sqlmodel.Field(
        sa_column=sqlmodel.Column(PklType)
    )

    celltype_id: int | None = sqlmodel.Field(
        default=None, foreign_key="celltype.id", nullable=False
    )
    celltype: Celltype = sqlmodel.Relationship(back_populates="results")
    model_id: int | None = sqlmodel.Field(
        default=None, foreign_key="model.id", nullable=False
    )
    model: Model = sqlmodel.Relationship(back_populates="results")
    sample_id: int | None = sqlmodel.Field(
        default=None, foreign_key="sample.id", nullable=False
    )
    sample: Sample = sqlmodel.Relationship(back_populates="results")


def _adata_dense_mut(adata: AnnData) -> AnnData:
    assert adata.X is not None, "adata.X is None"
    adata = adata.copy()
    if issparse(adata.X):
        adata.X = adata.X.toarray()  # type: ignore
    adata.X = adata.X.astype(np.float32)  # type: ignore
    return adata


def _remove_constant_genes(
    adata: AnnData,
    eps: float = 1e-8,
) -> AnnData:
    if (X := adata.X) is None:
        raise ValueError("adata.X is None")
    variance = np.nanvar(X, axis=0)  # type: ignore
    non_constant_mask = variance > eps
    return adata[:, non_constant_mask].copy()


def generate_datasets(
    dataset_configs: Sequence[DatasetConfigSC],
) -> Generator[tuple[Dataset, tuple[AnnData, AnnData], DatasetConfigSC], None, None]:
    for dataset_config in dataset_configs:
        adata = read_h5ad(dataset_config.path)
        train_test_split = np.random.choice(  # type: ignore
            ["train", "test"],
            size=adata.n_obs,
            p=[0.8, 0.2],
        )
        query = adata[(train_test_split == "test")]
        reference = adata[(train_test_split == "train")]
        query, reference = _adata_dense_mut(query), _adata_dense_mut(reference)
        if dataset_config.remove_constant_features:
            query = _remove_constant_genes(query)
            reference = _remove_constant_genes(reference)

        shared_features = list(
            set(query.var_names).intersection(set(reference.var_names))
        )
        query, reference = query[:, shared_features], reference[:, shared_features]

        celltypes = list(
            set(reference.obs[dataset_config.cluster_key]).intersection(
                set(query.obs[dataset_config.cluster_key])
            )
        )
        yield (
            Dataset(
                name=dataset_config.name,
                celltype_names=celltypes,
                feature_names=shared_features,
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
            rng.permutation(len(dataset.feature_names))
            for _ in range(sampling_config.n_samples)
        ]
    )
    train_idxs, test_idxs = np.split(indices, [sampling_config.panel_size], axis=1)
    for id_of_sample, (train_idx, test_idx) in enumerate(zip(train_idxs, test_idxs)):
        yield Sample(
            id_of_sample=id_of_sample,
            train_idx=train_idx,
            test_idx=test_idx,
            dataset=dataset,
        )


def generate_celltypes(
    dataset: Dataset,
    query: AnnData,
    reference: AnnData,
    dataset_config: DatasetConfigSC,
) -> Generator[Celltype, None, None]:
    for celltype_name in dataset.celltype_names:
        query_counts_matrix = query[
            query.obs[dataset_config.cluster_key] == celltype_name
        ].X
        reference_counts_matrix = reference[
            reference.obs[dataset_config.cluster_key] == celltype_name
        ].X
        if issparse(query_counts_matrix):
            query_counts_matrix = query_counts_matrix.toarray()  # type: ignore
        if issparse(reference_counts_matrix):
            reference_counts_matrix = reference_counts_matrix.toarray()  # type: ignore

        if (query_counts_matrix.shape[0] < dataset_config.min_n_cells_per_celltype) or (  # type: ignore
            reference_counts_matrix.shape[0] < dataset_config.min_n_cells_per_celltype  # type: ignore
        ):
            continue

        n_obs_query = query_counts_matrix.shape[0]  # type: ignore
        n_obs_ref = reference_counts_matrix.shape[0]  # type: ignore
        n_vars = query_counts_matrix.shape[1]  # type: ignore
        n_pcs: int = reduce(
            min, [dataset_config.max_n_pcs, n_obs_query, n_obs_ref, n_vars]
        )
        reference_pca_embedding = np.array(
            PCA(n_components=n_pcs).fit_transform(reference_counts_matrix)
        )
        query_pca_embedding = np.array(
            PCA(n_components=n_pcs).fit_transform(query_counts_matrix)
        )
        reference_umap_embedding = UMAP(n_components=2).fit_transform(
            reference_pca_embedding
        )
        query_umap_embedding = UMAP(n_components=2).fit_transform(query_pca_embedding)
        n_neighbours = reduce(
            min, [dataset_config.max_n_neighbours, dataset_config.max_n_pcs - 1]
        )
        reference_adjacency_matrix: NumericArray = kneighbors_graph(
            reference_pca_embedding,
            n_neighbors=n_neighbours,  # type: ignore
        )  # type: ignore
        query_adjacency_matrix: NumericArray = kneighbors_graph(
            query_pca_embedding,
            n_neighbors=n_neighbours,  # type: ignore
        )  # type: ignore

        yield Celltype(
            name=celltype_name,
            reference_counts_matrix=reference_counts_matrix,  # type: ignore
            query_counts_matrix=query_counts_matrix,  # type: ignore
            reference_pca_embedding=reference_pca_embedding,
            query_pca_embedding=query_pca_embedding,
            reference_umap_embedding=reference_umap_embedding,  # type: ignore
            query_umap_embedding=query_umap_embedding,  # type: ignore
            reference_adjacency_matrix=reference_adjacency_matrix,
            query_adjacency_matrix=query_adjacency_matrix,
            dataset=dataset,
        )


def generate_models(
    predictor_configs: Sequence[PredictorConfig],
) -> Generator[tuple[Model, PredictorConfig], None, None]:
    for predictor_config in predictor_configs:
        match predictor_config.scope:
            case "global":
                scope = ModelScope.GLOBAL
            case _:
                scope = ModelScope.CELLTYPE

        transform = predictor_config.transform
        yield (
            Model(
                name=predictor_config.name,
                scope=scope,
                transform=transform if transform is not None else "raw",
            ),
            predictor_config,
        )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (skips database creation and artifact logging)",
    )
    return parser.parse_args()


def main() -> None:
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

        # Determine whether to use a real temporary directory or a dummy placeholder
        tmpdir_ctx = TemporaryDirectory() if not args.debug else nullcontext()

        with tmpdir_ctx as tmpdir:
            session_ctx = nullcontext(None)

            match tmpdir:
                case str():
                    sqlite_file_name = "database.db"
                    sqlite_folder_path = os.path.join(tmpdir, sqlite_file_name)
                    sqlite_url = f"sqlite:///{sqlite_folder_path}"
                    engine = sqlmodel.create_engine(sqlite_url, echo=True)
                    sqlmodel.SQLModel.metadata.create_all(engine)
                    session_ctx = sqlmodel.Session(engine)
                case None:
                    sqlite_folder_path = None
                    logger.info("DEBUG MODE ACTIVE: Skipping database initialization.")
                case _:
                    assert_never(tmpdir)

            with session_ctx as session:
                models = list(
                    generate_models(predictor_configs=experiment_config.predictors)
                )

                for dataset, (query, reference), dataset_config in generate_datasets(
                    experiment_config.datasets
                ):
                    logger.info(f"Processing dataset {pprint.pformat(dataset_config)}")

                    samples = list(
                        generate_samples(
                            sampling_config=experiment_config.sampling_config
                            or SamplingConfig(n_samples=5, panel_size=500),
                            rng=rng,
                            dataset=dataset,
                        )
                    )

                    celltypes = list(
                        generate_celltypes(
                            dataset=dataset,
                            query=query,
                            reference=reference,
                            dataset_config=dataset_config,
                        )
                    )

                    for model, predictor_config in models:
                        logger.info(
                            f"Fitting model {model.name} on dataset {dataset.name}"
                        )
                        predictor = predictor_config.model.instantiate_model()
                        transform = (
                            np.log1p
                            if predictor_config.transform == "log"
                            else lambda x: x
                        )

                        match predictor_config.scope:
                            case "global":
                                predictor = predictor.fit(transform(reference.X))  # type: ignore
                                my_generator = (
                                    (celltype, predictor) for celltype in celltypes
                                )
                            case "celltype":
                                my_generator = (
                                    (
                                        celltype,
                                        predictor.fit(
                                            transform(celltype.reference_counts_matrix)
                                        ),
                                    )
                                    for celltype in celltypes
                                )

                        for (celltype, predictor_protocol), sample in product(
                            my_generator, samples
                        ):
                            logger.info(
                                f"Processing celltype {pprint.pformat(celltype)}, ",
                                f"predictor {pprint.pformat(predictor_protocol)}, ",
                                f"sample {pprint.pformat((sample))}, ",
                            )
                            model_embedding_query, model_counts_query = (
                                predictor_protocol.predict(
                                    x=transform(
                                        celltype.query_counts_matrix[
                                            :, sample.train_idx
                                        ]
                                    ),
                                    indexer=sample.train_idx,
                                )
                            )
                            model_embedding_reference, model_counts_reference = (
                                predictor_protocol.predict(
                                    x=transform(
                                        celltype.reference_counts_matrix[
                                            :, sample.train_idx
                                        ]
                                    ),
                                    indexer=sample.train_idx,
                                )
                            )
                            model_feature_embedding = (
                                predictor_protocol.feature_embedding
                                if predictor_protocol.feature_embedding is not None
                                else np.array(np.nan)
                            )

                            result = Result(
                                model_embedding_reference=model_embedding_reference,
                                model_counts_reference=model_counts_reference,
                                model_embedding_query=model_embedding_query,
                                model_counts_query=model_counts_query,
                                model_feature_embedding=model_feature_embedding,
                                model=model,
                                celltype=celltype,
                                sample=sample,
                            )

                            # Only execute database writes if session exists
                            if session is not None:
                                session.add(result)
                                session.commit()
                                session.expunge(result)

                            logger.info(
                                f"Processed sample {sample.id_of_sample}, dataset {dataset.name}, "
                                f"celltype {celltype.name}, model {model.name} "
                                f"(DB tracking: {not args.debug})"
                            )

                if session is not None and sqlite_folder_path is not None:
                    logger.info("Committing final results")
                    session.commit()
                    mlflow.log_artifact(sqlite_folder_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
