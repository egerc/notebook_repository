from functools import partial
import os
import pickle
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Literal

import anndata as ad
from joblib import Memory
import mlflow
import nico2_lib as n2l
import numpy as np
import pandas as pd
from pydantic import BaseModel
import yaml
from anndata.typing import AnnData
from nico2_lib.predictors._scvi._scvi_pred import ScviPredictor
from numpy import intp, number
from numpy.typing import NDArray
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

ModelArchitecture = Literal["mock", "nmf", "scvi"]
NumericArray = NDArray[number]
IndexArray = NDArray[intp]
LoaderFn = Callable[[], tuple[AnnData, AnnData, str, str]]


class InputDataset(BaseModel):
    name: str
    query_path: str
    query_ct_key: str
    reference_path: str
    reference_ct_key: str


class Predictor(BaseModel):
    name: str
    log_transform: bool
    model_architecture: str
    kwargs: dict[str, Any]


class Config(BaseModel):
    n_samples: int = Field(ge=1)
    sample_length: int = Field(ge=1)
    n_pcs: int
    n_neighbours: int
    predictors: list[Predictor]
    datasets: list[InputDataset]


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
        return pickle.loads(value)  # type: ignore


class ModalityEnum(str, Enum):
    SPATIAL = "SPATIAL"
    PSEUDOSPATIAL = "PSEUDOSPATIAL"


@dataclass(frozen=True)
class PredictorWrapper:
    name: str
    log_transform: bool
    predictor: object

    def fit(self, X: NumericArray) -> "PredictorWrapper":
        data = np.log1p(X) if self.log_transform else X
        return replace(self, predictor=self.predictor.fit(data))

    def predict(self, X: NumericArray, idx: IndexArray) -> tuple[NumericArray, NumericArray]:
        data = np.log1p(X) if self.log_transform else X
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
    counts_matrix: NumericArray = Field(sa_column=Column(PklType))
    pca_embedding: NumericArray = Field(sa_column=Column(PklType))
    umap_embedding: NumericArray = Field(sa_column=Column(PklType))
    adjacency_matrix: NumericArray = Field(sa_column=Column(PklType))

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

    global_model_embedding: NumericArray = Field(sa_column=Column(PklType))
    global_model_counts: NumericArray = Field(sa_column=Column(PklType))
    celltype_model_embedding: NumericArray = Field(sa_column=Column(PklType))
    celltype_model_counts: NumericArray = Field(sa_column=Column(PklType))

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
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()  # type: ignore


def _sample_indices(
    total_features: int,
    n_samples: int,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    indices = np.vstack([rng.permutation(total_features) for _ in range(n_samples)])
    test_idx, train_idx = np.split(indices, [sample_size], axis=1)
    return train_idx, test_idx


PREDICTOR_REGISTRY: dict[ModelArchitecture, n2l.pd.PredictorProtocol] = {
    "mock_predictor": MockPredictor,
    "nmf": n2l.pd.NmfPredictor,
    "scvi": ScviPredictor,
}


def _create_spatial_loader(
    query_path: str,
    query_ct_key: str,
    reference_path: str,
    reference_ct_key: str,
    *,
    memory: Memory,
) -> Callable[[], tuple[AnnData, AnnData, str, str]]:
    cached_transfer = memory.cache(n2l.lt.scvi_transfer)  # type: ignore
    query_loader = partial(ad.read_h5ad, filename=query_path)
    reference_loader = partial(ad.read_h5ad, filename=reference_path)

    def loader() -> tuple[AnnData, AnnData, str, str]:
        query = query_loader()
        reference = reference_loader()

        _adata_dense_mut(query)
        _adata_dense_mut(reference)

        query.obs[query_ct_key] = cached_transfer(  # type: ignore
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


def main():
    with open("config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    config = Config(**yaml_config)
    memory = Memory("cache")
    print(config)

    # loader_fns = [
    #    LoaderFnWrapper(name="mock_loader_1", loader=mock_loader),
    #    LoaderFnWrapper(name="mock_loader_2", loader=mock_loader),
    # ]
    loader_fns: list[LoaderFnWrapper] = [
        LoaderFnWrapper(
            name=loader_config.name,
            loader=_create_spatial_loader(
                query_path=loader_config.query_path,
                query_ct_key=loader_config.query_ct_key,
                reference_path=loader_config.reference_path,
                reference_ct_key=loader_config.reference_ct_key,
                memory=memory,
            ),
        )
        for loader_config in config.datasets
    ]

    predictors: list[PredictorWrapper] = [
        PredictorWrapper(
            name=predictor_config.name,
            log_transform=predictor_config.log_transform,
            predictor=PREDICTOR_REGISTRY[predictor_config.model_architecture](  # type: ignore
                **predictor_config.kwargs  # type: ignore
            ),
        )
        for predictor_config in config.predictors
    ]

    with mlflow.start_run():
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
                    dataset = Dataset(
                        name=loader_fn.name,
                    )
                    shared_celltypes = np.intersect1d(
                        ar1=np.array(query.obs[query_ct_key].unique()),
                        ar2=np.array(reference.obs[ref_ct_key].unique()),
                    )
                    shared_features = np.intersect1d(
                        query.var_names, reference.var_names
                    )
                    query = query[
                        query.obs[query_ct_key].isin(shared_celltypes.tolist()),
                        shared_features,
                    ]
                    reference = reference[
                        reference.obs[ref_ct_key].isin(shared_celltypes.tolist()),
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
                        predictor.name: predictor.fit(reference.X)
                        for predictor in predictors
                    }

                    for celltype_name in shared_celltypes:
                        query_ct_mask = query.obs[query_ct_key] == celltype_name
                        ref_ct_mask = reference.obs[ref_ct_key] == celltype_name
                        celltype_fitted_models = {
                            predictor.name: predictor.fit(reference[ref_ct_mask, :].X)
                            for predictor in predictors
                        }
                        counts_matrix = query[query_ct_mask, :].X
                        pca_embedding = PCA(n_components=config.n_pcs).fit_transform(
                            counts_matrix
                        )
                        umap_embedding = UMAP(n_components=2).fit_transform(
                            counts_matrix
                        )
                        adjacency_matrix = kneighbors_graph(
                            pca_embedding, n_neighbors=config.n_neighbours
                        )

                        celltype = Celltype(
                            name=celltype_name,
                            counts_matrix=counts_matrix,
                            pca_embedding=pca_embedding,
                            umap_embedding=umap_embedding,
                            adjacency_matrix=adjacency_matrix,
                            dataset=dataset,
                        )
                        for model_name, model in model_objects.items():
                            globally_fitted_model = globally_fitted_models[model_name]
                            celltype_fitted_model = celltype_fitted_models[model_name]
                            for sample in sample_objects:
                                global_model_embedding, global_model_counts = (
                                    globally_fitted_model.predict(
                                        celltype.counts_matrix[:, sample.train_idx],
                                        sample.train_idx,
                                    )
                                )
                                celltype_model_embedding, celltype_model_counts = (
                                    celltype_fitted_model.predict(
                                        celltype.counts_matrix[:, sample.train_idx],
                                        sample.train_idx,
                                    )
                                )

                                result = Result(
                                    global_model_embedding=global_model_embedding,
                                    global_model_counts=global_model_counts,
                                    celltype_model_embedding=celltype_model_embedding,
                                    celltype_model_counts=celltype_model_counts,
                                    sample=sample,
                                    model=model,
                                    celltype=celltype,
                                )
                                session.add(result)
                session.commit()
                mlflow.log_artifact(sqlite_folder_path)


if __name__ == "__main__":
    main()
