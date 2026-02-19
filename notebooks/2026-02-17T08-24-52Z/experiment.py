from __future__ import annotations

import json
import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Union

import mlflow
import nico2_lib as n2l
import numpy as np
from anndata.typing import AnnData
from joblib import Memory
from joblib.externals.loky.backend.context import os
from nico2_lib.predictors import NmfPredictor
from numpy import intp, number
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)


NumericArray = NDArray[number]
LoaderFn = Callable[[str], tuple[AnnData, AnnData, str, str]]


@dataclass(frozen=True)
class Config:
    data_dir: str
    cache_dir: str
    n_pca_components: int
    seed: int
    n_samples: int
    sample_length: int


@dataclass(frozen=True)
class PredictorWrapper:
    name: str
    predictor: n2l.pd.PredictorProtocol


@dataclass(frozen=True)
class LoaderFnWrapper:
    name: str
    loader: LoaderFn


@dataclass(frozen=True)
class ExperimentOutput:
    dataset_name: str
    celltype_results: list[CelltypeResult]


@dataclass(frozen=True)
class CelltypeResult:
    celltype: str
    celltype_mask: NDArray[np.bool_]
    pca_embedding: NumericArray
    umap_embedding: NumericArray
    observed_counts: NumericArray
    global_model_results: list[Result]
    celltype_model_results: list[Result]


@dataclass(frozen=True)
class Result:
    model: PredictorWrapper
    sample_results: list[SampleResult]


@dataclass(frozen=True)
class SampleResult:
    sample_idx: int
    train_idx: Union[Sequence[int], NDArray[np.intp]]
    test_idx: Union[Sequence[int], NDArray[np.intp]]
    embedding: NumericArray
    predicted_counts: NumericArray


def _create_spatial_loader(
    query_loader: Callable[[str], AnnData],
    query_ct_key: str,
    reference_loader: Callable[[str], AnnData],
    reference_ct_key: str,
    *,
    memory: Memory,
) -> LoaderFn:
    cached_transfer = memory.cache(n2l.lt.scvi_transfer)

    def loader(data_dir: str) -> tuple[AnnData, AnnData, str, str]:
        logger.info("Loading spatial dataset from %s", data_dir)
        query = query_loader(data_dir)
        reference = reference_loader(data_dir)

        _adata_dense_mut(query)
        _adata_dense_mut(reference)

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
    train_idx, test_idx = np.split(indices, [sample_size], axis=1)
    return train_idx, test_idx


def _filter_celltypes(adata: AnnData, ct_key: str, min_count: int) -> list[str]:
    celltypes = adata.obs[ct_key].value_counts()
    return celltypes[celltypes >= min_count].index.tolist()  # type: ignore


def load_config(config_path: Path | None = None) -> Config:
    path = config_path or Path(__file__).with_name("config.json")
    with path.open() as f:
        raw = json.load(f)
    return Config(**raw)


def _build_dataloaders(cache_dir: str) -> list[LoaderFnWrapper]:
    return [
        LoaderFnWrapper(
            name="small_mouse_intestine_spatial",
            loader=_create_spatial_loader(
                query_loader=n2l.dt.small_mouse_intestine_merfish,
                query_ct_key="cluster",
                reference_loader=n2l.dt.small_mouse_intestine_sc,
                reference_ct_key="cluster",
                memory=Memory(cache_dir),
            ),
        ),
    ]


predictors: list[PredictorWrapper] = [
    PredictorWrapper("NMF_3", NmfPredictor(n_components=3)),
    PredictorWrapper("NMF_8", NmfPredictor(n_components=8)),
]


def run_experiment(config: Config) -> list[ExperimentOutput]:
    experiment_outputs: list[ExperimentOutput] = []
    dataloaders = _build_dataloaders(config.cache_dir)
    for dataloader in dataloaders:
        query, reference, query_ct_key, reference_ct_key = dataloader.loader(
            config.data_dir
        )
        _adata_dense_mut(query)
        _adata_dense_mut(reference)
        experiment_outputs.append(
            ExperimentOutput(
                dataset_name=dataloader.name,
                celltype_results=get_celltype_results(
                    query=query,
                    reference=reference,
                    query_ct_key=query_ct_key,
                    reference_ct_key=reference_ct_key,
                    n_pca_components=config.n_pca_components,
                    seed=config.seed,
                    n_samples=config.n_samples,
                    sample_length=config.sample_length,
                ),
            )
        )
    return experiment_outputs


def get_celltype_results(
    query: AnnData,
    reference: AnnData,
    query_ct_key: str,
    reference_ct_key: str,
    n_pca_components: int,
    seed: int,
    n_samples: int,
    sample_length: int,
) -> list[CelltypeResult]:
    query_celltypes = _filter_celltypes(query, query_ct_key, n_pca_components)
    reference_celltypes = _filter_celltypes(
        reference, reference_ct_key, n_pca_components
    )
    shared_celltypes = np.sort(np.intersect1d(query_celltypes, reference_celltypes))
    shared_features = np.intersect1d(query.var_names, reference.var_names)

    query_mask = query.obs[query_ct_key].isin(shared_celltypes.tolist())
    reference_mask = reference.obs[reference_ct_key].isin(shared_celltypes.tolist())
    query_shared = query[query_mask, shared_features]
    reference_shared = reference[reference_mask, shared_features]
    n_features = shared_features.shape[0]
    rng = np.random.default_rng(seed)
    train_indices, test_indices = _sample_indices(
        n_features, n_samples, sample_length, rng
    )
    global_models = [
        PredictorWrapper(
            name=predictor_wrapper.name,
            predictor=predictor_wrapper.predictor.fit(reference_shared.X),  # type: ignore
        )
        for predictor_wrapper in predictors
    ]
    results: list[CelltypeResult] = []
    for celltype in shared_celltypes:
        query_celltype_mask = query_shared.obs[query_ct_key] == celltype
        reference_celltype_mask = reference_shared.obs[reference_ct_key] == celltype
        pca_embeddings = PCA(n_components=n_pca_components).fit_transform(
            StandardScaler().fit_transform(query_shared[query_celltype_mask].X)
        )
        umap_embeddings = UMAP(n_components=2).fit_transform(pca_embeddings)
        celltype_models = [
            PredictorWrapper(
                name=predictor_wrapper.name,
                predictor=predictor_wrapper.predictor.fit(
                    reference_shared[reference_celltype_mask].X  # type: ignore
                ),
            )
            for predictor_wrapper in predictors
        ]
        observed_counts = query_shared[query_celltype_mask].X  # type: ignore
        results.append(
            CelltypeResult(
                celltype=celltype,
                celltype_mask=query_celltype_mask,
                pca_embedding=pca_embeddings,
                umap_embedding=umap_embeddings,  # type: ignore
                observed_counts=observed_counts,  # type: ignore
                global_model_results=get_model_results(
                    global_models,
                    counts=observed_counts,  # type: ignore
                    train_indices=train_indices,
                    test_indices=test_indices,
                ),
                celltype_model_results=get_model_results(
                    celltype_models,
                    counts=observed_counts,  # type: ignore
                    train_indices=train_indices,
                    test_indices=test_indices,
                ),
            )
        )
    return results


def get_model_results(
    predictors: Sequence[PredictorWrapper],
    counts: NumericArray,
    train_indices: Union[Sequence[NDArray[intp]], NDArray[intp]],
    test_indices: Union[Sequence[NDArray[intp]], NDArray[intp]],
) -> list[Result]:
    model_results: list[Result] = []
    for model in predictors:
        model_results.append(
            Result(
                model=model,
                sample_results=get_sample_results(
                    predictor=model.predictor,
                    counts=counts,
                    train_indices=train_indices,
                    test_indices=test_indices,
                ),
            )
        )
    return model_results


def get_sample_results(
    predictor: n2l.pd.PredictorProtocol,
    counts: NumericArray,
    train_indices: Union[Sequence[NDArray[intp]], NDArray[intp]],
    test_indices: Union[Sequence[NDArray[intp]], NDArray[intp]],
) -> list[SampleResult]:
    sample_results: list[SampleResult] = []
    for idx, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
        embedding, predicted_counts = predictor.predict(counts[:, train_idx], train_idx)
        sample_results.append(
            SampleResult(
                sample_idx=idx,
                train_idx=train_idx,
                test_idx=test_idx,
                embedding=embedding,
                predicted_counts=predicted_counts,
            )
        )
    return sample_results


def main() -> None:
    config = load_config()
    with mlflow.start_run():
        mlflow.log_params(config.__dict__)
        experiment_outputs = run_experiment(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_name = "results.pkl"
            artifact_path = os.path.join(tmpdir, artifact_name)
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            with open(artifact_path, "wb") as f:
                pickle.dump(experiment_outputs, f)
                mlflow.log_artifact(artifact_path)


if __name__ == "__main__":
    main()
