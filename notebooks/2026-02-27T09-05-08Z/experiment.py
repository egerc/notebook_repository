import pickle
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
from anndata.typing import AnnData
from numpy import intp, number
from numpy.typing import NDArray
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

NumericArray = NDArray[number]
IndexArray = NDArray[intp]
LoaderFn = Callable[[str], tuple[AnnData, AnnData, str, str]]


class PklType(TypeDecorator):
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return pickle.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return pickle.loads(value)
        return None


class ModalityEnum(str, Enum):
    SPATIAL = "SPATIAL"
    PSEUDOSPATIAL = "PSEUDOSPATIAL"


@dataclass(frozen=True)
class PredictorWrapper:
    name: str
    predictor: object


@dataclass(frozen=True)
class LoaderFnWrapper:
    name: str
    loader: LoaderFn


@dataclass(frozen=True)
class ModelOutputs:
    embedding: NumericArray
    predicted_counts: NumericArray


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

    global_model_outputs: ModelOutputs = Field(sa_column=Column(PklType))
    celltype_model_outputs: ModelOutputs = Field(sa_column=Column(PklType))

    celltype_id: int | None = Field(
        default=None, foreign_key="celltype.id", nullable=False
    )
    celltype: Celltype = Relationship(back_populates="results")
    model_id: int | None = Field(default=None, foreign_key="model.id", nullable=False)
    model: Model = Relationship(back_populates="results")
    sample_id: int | None = Field(default=None, foreign_key="sample.id", nullable=False)
    sample: Sample = Relationship(back_populates="results")


def mock_loader(dir: str) -> tuple[AnnData, AnnData, str, str]:
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
        n_obs, n_vars = X.shape
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


def main():
    loader_fns = [
        LoaderFnWrapper(name="mock_loader_1", loader=mock_loader),
        LoaderFnWrapper(name="mock_loader_2", loader=mock_loader),
    ]
    predictors = [
        PredictorWrapper(
            name="mock_predictor_1", predictor=MockPredictor(n_components=5)
        ),
        PredictorWrapper(
            name="mock_predictor_2", predictor=MockPredictor(n_components=5)
        ),
    ]

    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"
    engine = create_engine(sqlite_url, echo=True)
    SQLModel.metadata.create_all(engine)

    rng = np.random.default_rng(seed=0)
    with Session(engine) as session:
        model_objects = {
            predictor.name: Model(name=predictor.name) for predictor in predictors
        }
        for loader_fn in loader_fns:
            query, reference, query_ct_key, ref_ct_key = loader_fn.loader("./data")
            dataset = Dataset(
                name=loader_fn.name,
            )
            shared_celltypes = np.intersect1d(
                ar1=np.array(query.obs[query_ct_key].unique()),
                ar2=np.array(reference.obs[ref_ct_key].unique()),
            )
            shared_features = np.intersect1d(query.var_names, reference.var_names)
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
            train_idxs, test_idxs = _sample_indices(len(shared_features), 2, 10, rng)

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
                predictor.name: predictor.predictor.fit(reference.X)
                for predictor in predictors
            }

            for celltype_name in shared_celltypes:
                query_ct_mask = query.obs[query_ct_key] == celltype_name
                ref_ct_mask = reference.obs[ref_ct_key] == celltype_name
                celltype_fitted_models = {
                    predictor.name: predictor.predictor.fit(reference[ref_ct_mask, :].X)
                    for predictor in predictors
                }

                celltype = Celltype(
                    name=celltype_name,
                    counts_matrix=query[query_ct_mask, :].X,
                    dataset=dataset,
                )
                for model_name, model in model_objects.items():
                    globally_fitted_model = globally_fitted_models.get(model_name)
                    celltype_fitted_model = celltype_fitted_models.get(model_name)
                    for sample in sample_objects:
                        global_model_outputs = ModelOutputs(
                            *globally_fitted_model.predict(
                                celltype.counts_matrix[:, sample.train_idx],
                                sample.train_idx,
                            )
                        )
                        celltype_model_outputs = ModelOutputs(
                            *celltype_fitted_model.predict(
                                celltype.counts_matrix[:, sample.train_idx],
                                sample.train_idx,
                            )
                        )
                        result = Result(
                            global_model_outputs=global_model_outputs,
                            celltype_model_outputs=celltype_model_outputs,
                            sample=sample,
                            model=model,
                            celltype=celltype,
                        )
                        session.add(result)
        session.commit()


if __name__ == "__main__":
    main()
