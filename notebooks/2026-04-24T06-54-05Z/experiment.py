import os
import pickle
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial, reduce
from typing import Annotated, Any, Callable, Literal

import anndata as ad
import mlflow
import nico2_lib as n2l
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from anndata.io import read_h5ad
from anndata.typing import AnnData
from joblib import Memory
from numpy import intp, number
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sqlmodel import (
    Column,
    Field,
    LargeBinary,
    Relationship,
    Session,
    SQLModel,
    TypeDecorator,
    create_engine,
)
from umap import UMAP

ModelArchitecture = Literal[
    "mock",
    "nmf",
    "scvi",
    "mofaflex",
]
NumericArray = NDArray[number]
IndexArray = NDArray[intp]
LoaderFn = Callable[[], tuple[AnnData, AnnData, str, str]]


class SpatialInputDataset(BaseModel):
    type: Literal["spatial"]
    name: str
    query_path: str
    query_ct_key: str
    reference_path: str
    reference_ct_key: str


class PseudospatialInputDataset(BaseModel):
    type: Literal["pseudospatial"]
    name: str
    data_path: str
    data_ct_key: str


class Predictor(BaseModel):
    name: str
    log_transform: bool
    model_architecture: str
    kwargs: dict[str, Any]


class Config(BaseModel):
    n_samples: int = Field(ge=1)
    sample_length: int = Field(ge=1)
    min_n_cells_celltype: int
    n_cells: int | None
    max_n_pcs: int
    max_n_neighbours: int
    predictors: list[Predictor]
    datasets: list[
        Annotated[
            SpatialInputDataset | PseudospatialInputDataset, Field(discriminator="type")
        ]
    ]


class PklType(TypeDecorator):
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return pickle.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        # if value is not None:
        #    return pickle.loads(value)
        return pickle.loads(value)


class ModalityEnum(str, Enum):
    SPATIAL = "SPATIAL"
    PSEUDOSPATIAL = "PSEUDOSPATIAL"


@dataclass(frozen=True)
class PredictorWrapper:
    name: str
    log_transform: bool
    predictor: object

    def fit(self, X: NumericArray) -> "PredictorWrapper":
        # Check if X is sparse (including SparseCSRMatrixView)
        if hasattr(X, "toarray"):
            data = X.toarray()
        else:
            data = X

        if self.log_transform:
            data = np.log1p(data)

        data = np.asarray(data, dtype=np.float32)
        return replace(self, predictor=self.predictor.fit(data))

    def predict(
        self, X: NumericArray, idx: IndexArray
    ) -> tuple[NumericArray, NumericArray]:
        # Ensure data is float32 to match the fitted model
        data = np.log1p(X) if self.log_transform else X
        data = np.asarray(data, dtype=np.float32)
        return self.predictor.predict(data, idx)


@dataclass(frozen=True)
class LoaderFnWrapper:
    name: str
    loader: LoaderFn


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    name: str

    celltypes: list["Celltype"] = Relationship(back_populates="dataset")
    samples: list["Sample"] = Relationship(back_populates="dataset")


class Celltype(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}
    id: int | None = Field(default=None, primary_key=True)

    name: str
    reference_counts_matrix: NumericArray = Field(sa_column=Column(PklType))
    reference_pca_embedding: NumericArray = Field(sa_column=Column(PklType))
    reference_umap_embedding: NumericArray = Field(sa_column=Column(PklType))
    reference_adjacency_matrix: NumericArray = Field(sa_column=Column(PklType))
    query_counts_matrix: NumericArray = Field(sa_column=Column(PklType))
    query_pca_embedding: NumericArray = Field(sa_column=Column(PklType))
    query_umap_embedding: NumericArray = Field(sa_column=Column(PklType))
    query_adjacency_matrix: NumericArray = Field(sa_column=Column(PklType))
    n_pcs: int
    n_neighbours: int

    dataset_id: int | None = Field(default=None, foreign_key="dataset.id")
    dataset: Dataset = Relationship(back_populates="celltypes")

    results: list["Result"] = Relationship(back_populates="celltype")


class Model(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    name: str

    results: list["Result"] = Relationship(back_populates="model")


class Sample(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}
    id: int | None = Field(default=None, primary_key=True)

    id_of_sample: int
    train_idx: IndexArray = Field(sa_column=Column(PklType))
    test_idx: IndexArray = Field(sa_column=Column(PklType))

    dataset_id: int | None = Field(
        default=None, foreign_key="dataset.id", nullable=False
    )
    dataset: Dataset = Relationship(back_populates="samples")

    results: list["Result"] = Relationship(back_populates="sample")


class Result(SQLModel, table=True):
    model_config = {"arbitrary_types_allowed": True}
    id: int | None = Field(default=None, primary_key=True)

    global_model_embedding_reference: NumericArray = Field(sa_column=Column(PklType))
    global_model_counts_reference: NumericArray = Field(sa_column=Column(PklType))
    celltype_model_embedding_reference: NumericArray = Field(sa_column=Column(PklType))
    celltype_model_counts_reference: NumericArray = Field(sa_column=Column(PklType))
    global_model_embedding_query: NumericArray = Field(sa_column=Column(PklType))
    global_model_counts_query: NumericArray = Field(sa_column=Column(PklType))
    celltype_model_embedding_query: NumericArray = Field(sa_column=Column(PklType))
    celltype_model_counts_query: NumericArray = Field(sa_column=Column(PklType))

    celltype_id: int | None = Field(
        default=None, foreign_key="celltype.id", nullable=False
    )
    celltype: Celltype = Relationship(back_populates="results")
    model_id: int | None = Field(default=None, foreign_key="model.id", nullable=False)
    model: Model = Relationship(back_populates="results")
    sample_id: int | None = Field(default=None, foreign_key="sample.id", nullable=False)
    sample: Sample = Relationship(back_populates="results")


def mock_loader() -> tuple[AnnData, AnnData, str, str]:
    rng = np.random.default_rng(0)
    query_n_obs = 1000
    query_n_vars = 500
    ref_n_obs = 1000
    ref_n_vars = 500
    query_counts = rng.normal(loc=100, scale=20, size=(query_n_obs, query_n_vars))
    reference_counts = rng.normal(loc=100, scale=20, size=(ref_n_obs, ref_n_vars))
    label_classes = ["A", "B", "C"]
    query_annotation = rng.choice(label_classes, size=query_n_obs)
    ref_annotation = rng.choice(label_classes, size=ref_n_obs)
    query_ct_key = "cluster"
    ref_ct_key = "cluster"
    query = ad.AnnData(
        X=query_counts, obs=pd.DataFrame({query_ct_key: query_annotation})
    )
    ref = ad.AnnData(X=reference_counts, obs=pd.DataFrame({ref_ct_key: ref_annotation}))
    return query, ref, query_ct_key, ref_ct_key


@dataclass(frozen=True)
class MockPredictor:
    n_components: int

    def fit(self, X: NumericArray) -> "MockPredictor":
        return replace(self)

    def predict(
        self, X: NumericArray, idx: IndexArray
    ) -> tuple[NumericArray, NumericArray]:
        rng = np.random.default_rng(0)
        n_obs, _ = X.shape
        embeddings = rng.normal(loc=0, scale=1, size=(n_obs, self.n_components))
        reconstructed_counts = rng.normal(loc=0, scale=1, size=X.shape)
        return embeddings, reconstructed_counts


def _adata_dense_mut(adata: AnnData) -> None:
    if issparse(adata.X):
        # .toarray() works on both matrices and views
        adata.X = adata.X.toarray()


def _subsample_adata(
    adata: AnnData,
    n_cells: int | None,
    rng: np.random.Generator,
) -> AnnData:
    if n_cells is not None and adata.n_obs > n_cells:
        indices = rng.choice(adata.n_obs, size=n_cells, replace=False)
        adata = adata[indices].copy()
    return adata


def _sample_indices(
    total_features: int,
    n_samples: int,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    indices = np.vstack([rng.permutation(total_features) for _ in range(n_samples)])
    test_idx, train_idx = np.split(indices, [sample_size], axis=1)
    return train_idx, test_idx


PREDICTOR_REGISTRY: dict[ModelArchitecture, n2l.pd.PredictorProtocol] = {  # type: ignore
    "mock_predictor": MockPredictor,
    "nmf": n2l.pd.NmfPredictor,
    "scvi": n2l.pd.ScviPredictor,
    "mofaflex": n2l.pd.MofaFlexPredictor,
}


def _create_spatial_loader(
    query_path: str,
    query_ct_key: str,
    reference_path: str,
    reference_ct_key: str,
    *,
    memory: Memory,
) -> Callable[[], tuple[AnnData, AnnData, str, str]]:
    cached_transfer = memory.cache(n2l.lt.scvi_transfer)
    query_loader = partial(ad.read_h5ad, filename=query_path)
    reference_loader = partial(ad.read_h5ad, filename=reference_path)

    def loader() -> tuple[AnnData, AnnData, str, str]:
        query = query_loader()
        reference = reference_loader()

        _adata_dense_mut(query)
        _adata_dense_mut(reference)

        try:
            query.obs[query_ct_key]
        except KeyError:
            query.obs[query_ct_key] = cached_transfer(
                query,
                reference,
                reference_ct_key,
            )
        shared_features = np.intersect1d(query.var_names, reference.var_names)
        return (
            query[:, shared_features],
            reference[:, shared_features],
            query_ct_key,
            reference_ct_key,
        )

    return loader


def _create_pseudospatial_loader(data_path: str, data_ct_key: str) -> LoaderFn:
    def split_loader() -> tuple[AnnData, AnnData, str, str]:
        loader_func: Callable[[], AnnData] = partial(read_h5ad, data_path)
        adata = loader_func()
        n_cells = adata.n_obs
        shuffled_idx = np.random.permutation(n_cells)
        split_idx = n_cells // 2
        idx1, idx2 = shuffled_idx[:split_idx], shuffled_idx[split_idx:]
        query = adata[idx1].copy()
        reference = adata[idx2].copy()
        sc.pp.highly_variable_genes(
            query, n_top_genes=500, flavor="seurat_v3", inplace=True
        )
        query = query[:, query.var["highly_variable"]].copy()
        return query, reference, data_ct_key, data_ct_key

    return split_loader


def loader_config_to_loader(
    loader_config: SpatialInputDataset | PseudospatialInputDataset, memory: Memory
) -> LoaderFnWrapper:
    match loader_config:
        case SpatialInputDataset():
            return LoaderFnWrapper(
                name=loader_config.name,
                loader=_create_spatial_loader(
                    query_path=loader_config.query_path,
                    query_ct_key=loader_config.query_ct_key,
                    reference_path=loader_config.reference_path,
                    reference_ct_key=loader_config.reference_ct_key,
                    memory=memory,
                ),
            )
        case PseudospatialInputDataset():
            return LoaderFnWrapper(
                name=loader_config.name,
                loader=_create_pseudospatial_loader(
                    data_path=loader_config.data_path,
                    data_ct_key=loader_config.data_ct_key,
                ),
            )


def main():
    with open("config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    config = Config(**yaml_config)
    memory = Memory("cache")
    print(config)

    loader_fns: list[LoaderFnWrapper] = [
        loader_config_to_loader(loader_config, memory=memory)
        for loader_config in config.datasets
    ]

    predictors: list[PredictorWrapper] = [
        PredictorWrapper(
            name=predictor_config.name,
            log_transform=predictor_config.log_transform,
            predictor=PREDICTOR_REGISTRY[predictor_config.model_architecture](  # type: ignore
                **predictor_config.kwargs
            ),
        )
        for predictor_config in config.predictors
    ]

    with mlflow.start_run():  # type: ignore
        mlflow.log_params(config.__dict__)  # type: ignore
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_file_name = "database.db"
            sqlite_folder_path = os.path.join(tmpdir, sqlite_file_name)
            sqlite_url = f"sqlite:///{sqlite_folder_path}"
            engine = create_engine(sqlite_url, echo=True)
            SQLModel.metadata.create_all(engine)

            rng = np.random.default_rng(seed=0)
            with Session(engine) as session:
                model_objects = {
                    predictor.name: Model(name=predictor.name)
                    for predictor in predictors
                }
                for loader_fn in loader_fns:
                    query, reference, query_ct_key, ref_ct_key = loader_fn.loader()
                    query = _subsample_adata(
                        query,
                        config.n_cells,
                        rng,
                    )
                    reference = _subsample_adata(
                        reference,
                        config.n_cells,
                        rng,
                    )
                    dataset = Dataset(
                        name=loader_fn.name,
                    )

                    shared_celltypes: list[str] = np.intersect1d(
                        ar1=query.obs[query_ct_key]
                        .value_counts()
                        .loc[lambda x: x >= config.min_n_cells_celltype]
                        .index,  # type: ignore
                        ar2=reference.obs[ref_ct_key]
                        .value_counts()
                        .loc[lambda x: x >= config.min_n_cells_celltype]
                        .index,  # type: ignore
                    ).tolist()
                    shared_features = np.intersect1d(
                        query.var_names, reference.var_names
                    )
                    query = query[
                        query.obs[query_ct_key].isin(shared_celltypes),  # type: ignore
                        shared_features,
                    ]
                    reference = reference[
                        reference.obs[ref_ct_key].isin(shared_celltypes),  # type: ignore
                        shared_features,
                    ]
                    _adata_dense_mut(query)
                    _adata_dense_mut(reference)
                    train_idxs, test_idxs = _sample_indices(
                        len(shared_features),
                        config.n_samples,
                        config.sample_length,
                        rng,
                    )

                    sample_objects = [
                        Sample(
                            id_of_sample=id_of_sample,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            dataset=dataset,
                        )
                        for id_of_sample, (train_idx, test_idx) in enumerate(
                            zip(train_idxs, test_idxs)
                        )
                    ]
                    globally_fitted_models = {
                        predictor.name: predictor.fit(reference.X)  # type: ignore
                        for predictor in predictors
                    }

                    for celltype_name in shared_celltypes:
                        query_ct_mask = query.obs[query_ct_key] == celltype_name
                        ref_ct_mask = reference.obs[ref_ct_key] == celltype_name
                        n_obs_query = np.sum(query_ct_mask)
                        n_vars = len(shared_features)
                        n_obs_ref = np.sum(ref_ct_mask)
                        reference_ct = reference[ref_ct_mask].copy()
                        _adata_dense_mut(reference_ct)
                        reference_counts_matrix: NumericArray = reference_ct.X.copy()  # type: ignore

                        celltype_fitted_models = {
                            predictor.name: predictor.fit(reference_counts_matrix)
                            for predictor in predictors
                        }
                        query_ct = query[query_ct_mask].copy()
                        _adata_dense_mut(query_ct)
                        query_counts_matrix: NumericArray = query_ct.X.copy()  # type: ignore

                        n_pcs: int = reduce(
                            min, [config.max_n_pcs, n_obs_query, n_obs_ref, n_vars]
                        )
                        query_pca_embedding: NumericArray = np.array(
                            PCA(n_components=n_pcs).fit_transform(query_counts_matrix)
                        )
                        reference_pca_embedding = np.array(
                            PCA(n_components=n_pcs).fit_transform(
                                reference_counts_matrix
                            )
                        )
                        query_umap_embedding: NumericArray = UMAP(
                            n_components=2
                        ).fit_transform(query_pca_embedding)
                        reference_umap_embedding = UMAP(n_components=2).fit_transform(
                            reference_pca_embedding
                        )
                        n_neighbours = reduce(min, [config.max_n_neighbours, n_pcs])
                        if n_pcs == n_neighbours:
                            n_neighbours -= 1
                        query_adjacency_matrix: NumericArray = kneighbors_graph(
                            query_pca_embedding, n_neighbors=n_neighbours
                        )
                        reference_adjacency_matrix = kneighbors_graph(
                            reference_pca_embedding, n_neighbors=n_neighbours
                        )

                        celltype = Celltype(
                            name=celltype_name,
                            reference_counts_matrix=reference_counts_matrix,
                            reference_pca_embedding=reference_pca_embedding,
                            reference_umap_embedding=reference_umap_embedding,
                            reference_adjacency_matrix=reference_adjacency_matrix,
                            query_counts_matrix=query_counts_matrix,
                            query_pca_embedding=query_pca_embedding,
                            query_umap_embedding=query_umap_embedding,
                            query_adjacency_matrix=query_adjacency_matrix,
                            dataset=dataset,
                            n_pcs=n_pcs,
                            n_neighbours=n_neighbours,
                        )
                        for model_name, model in model_objects.items():
                            globally_fitted_model = globally_fitted_models[model_name]
                            celltype_fitted_model = celltype_fitted_models[model_name]
                            for sample in sample_objects:
                                (
                                    global_model_embedding_reference,
                                    global_model_counts_reference,
                                ) = globally_fitted_model.predict(
                                    reference_counts_matrix[:, sample.train_idx],
                                    sample.train_idx,
                                )
                                (
                                    global_model_embedding_query,
                                    global_model_counts_query,
                                ) = globally_fitted_model.predict(
                                    query_counts_matrix[:, sample.train_idx],
                                    sample.train_idx,
                                )
                                (
                                    celltype_model_embedding_reference,
                                    celltype_model_counts_reference,
                                ) = celltype_fitted_model.predict(
                                    reference_counts_matrix[:, sample.train_idx],
                                    sample.train_idx,
                                )
                                (
                                    celltype_model_embedding_query,
                                    celltype_model_counts_query,
                                ) = celltype_fitted_model.predict(
                                    query_counts_matrix[:, sample.train_idx],
                                    sample.train_idx,
                                )

                                result = Result(
                                    global_model_embedding_reference=global_model_embedding_reference,
                                    global_model_counts_reference=global_model_counts_reference,
                                    celltype_model_embedding_reference=celltype_model_embedding_reference,
                                    celltype_model_counts_reference=celltype_model_counts_reference,
                                    global_model_embedding_query=global_model_embedding_query,
                                    global_model_counts_query=global_model_counts_query,
                                    celltype_model_embedding_query=celltype_model_embedding_query,
                                    celltype_model_counts_query=celltype_model_counts_query,
                                    sample=sample,
                                    model=model,
                                    celltype=celltype,
                                )
                                session.add(result)
                session.commit()
                mlflow.log_artifact(sqlite_folder_path)  # type: ignore


if __name__ == "__main__":
    main()
