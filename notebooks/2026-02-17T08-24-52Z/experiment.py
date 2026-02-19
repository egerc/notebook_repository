from __future__ import annotations

import json
import logging
import pickle
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlflow
import nico2_lib as n2l
import numpy as np
from anndata.typing import AnnData
from joblib import Memory
from joblib.func_inspect import os
from nico2_lib.predictors import NmfPredictor
from numpy import number
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


@dataclass(frozen=True)
class PredictorWrapper:
    name: str
    predictor: n2l.pd.PredictorProtocol


@dataclass(frozen=True)
class LoaderFnWrapper:
    name: str
    loader: LoaderFn


@dataclass(frozen=True)
class Config:
    data_dir: str
    cache_dir: str
    seed: int
    n_samples: int
    sample_length: int
    n_pca_components: int
    mlflow_experiment_name: str
    mlflow_parent_run_name: str


@dataclass(frozen=True)
class Models:
    name: str
    model: n2l.pd.PredictorProtocol
    
@dataclass(frozen=True)
class DatasetResult:
    name: str
    global_models: list[Models]

@dataclass(frozen=True)
class CelltypeResults:
    pass

@dataclass(frozen=True)
class SampleResult:
    sample_idx: int
    train_idx: NDArray[np.int_]
    test_idx: NDArray[np.int_]
    observed_counts: NumericArray
    global_model_predicted_counts: NumericArray
    global_model_embeddings: NumericArray
    celltype_model_predicted_counts: NumericArray
    celltype_model_embeddings: NumericArray


@dataclass(frozen=True)
class CelltypeResult:
    celltype: str
    celltype_pca_embedding: NumericArray
    celltype_umap_embedding: NumericArray
    global_model: n2l.pd.PredictorProtocol
    global_model_embedding_size: int
    celltype_model: n2l.pd.PredictorProtocol
    celltype_model_embedding_size: int
    samples: Sequence[SampleResult]


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


def main() -> None:
    config = load_config()
    dataloaders = _build_dataloaders(config.cache_dir)
    mlflow.set_experiment(config.mlflow_experiment_name)
    with mlflow.start_run(run_name=config.mlflow_parent_run_name):
        mlflow.log_params(config.__dict__)
        mlflow.log_params(
            {
                "predictor_names": [p.name for p in predictors],
                "dataloader_names": [d.name for d in dataloaders],
            }
        )
        for dataloader in dataloaders:
            query, reference, query_ct_key, reference_ct_key = dataloader.loader(
                config.data_dir
            )
            _adata_dense_mut(query)
            _adata_dense_mut(reference)

            query_celltypes = _filter_celltypes(
                query, query_ct_key, config.n_pca_components
            )
            reference_celltypes = _filter_celltypes(
                reference, reference_ct_key, config.n_pca_components
            )
            shared_celltypes = np.sort(
                np.intersect1d(query_celltypes, reference_celltypes)
            )
            shared_features = np.intersect1d(query.var_names, reference.var_names)

            query_mask = query.obs[query_ct_key].isin(shared_celltypes.tolist())
            reference_mask = reference.obs[reference_ct_key].isin(
                shared_celltypes.tolist()
            )
            query_shared = query[query_mask, shared_features]
            reference_shared = reference[reference_mask, shared_features]

            n_features = shared_features.shape[0]
            rng = np.random.default_rng(config.seed)
            train_idxs, test_idxs = _sample_indices(
                n_features, config.n_samples, config.sample_length, rng
            )

            celltype_pca_embeddings: dict[str, NumericArray] = {}
            celltype_umap_embeddings: dict[str, NumericArray] = {}
            for celltype in shared_celltypes:
                query_celltype_mask = query_shared.obs[query_ct_key] == celltype
                celltype_pca_embeddings[celltype] = PCA(
                    n_components=config.n_pca_components
                ).fit_transform(
                    StandardScaler().fit_transform(query_shared[query_celltype_mask].X)
                )  # type: ignore
                celltype_umap_embeddings[celltype] = UMAP(n_components=2).fit_transform(
                    celltype_pca_embeddings[celltype]
                )

            for predictor in predictors:
                with mlflow.start_run(
                    nested=True, run_name=f"{predictor.name}_{dataloader.name}"
                ):
                    celltype_results: list[CelltypeResult] = []
                    global_model = predictor.predictor.fit(reference_shared.X)  # type: ignore
                    for celltype in shared_celltypes:
                        query_celltype_mask = query_shared.obs[query_ct_key] == celltype
                        reference_celltype_mask = (
                            reference_shared.obs[reference_ct_key] == celltype
                        )
                        celltype_model = predictor.predictor.fit(
                            reference_shared[reference_celltype_mask].X  # type: ignore
                        )
                        sample_results: list[SampleResult] = []
                        for idx, (train_idx, test_idx) in enumerate(
                            zip(train_idxs, test_idxs)
                        ):
                            observed_counts = query_shared[
                                query_celltype_mask, test_idx
                            ].X  # type: ignore
                            global_model_embeddings, global_model_predicted_counts = (
                                global_model.predict(
                                    query_shared[query_celltype_mask, train_idx].X,  # type: ignore
                                    train_idx,
                                )
                            )
                            (
                                celltype_model_embeddings,
                                celltype_model_predicted_counts,
                            ) = celltype_model.predict(
                                query_shared[query_celltype_mask, train_idx].X,  # type: ignore
                                train_idx,
                            )
                            sample_results.append(
                                SampleResult(
                                    sample_idx=idx,
                                    train_idx=train_idx,
                                    test_idx=test_idx,
                                    observed_counts=observed_counts,  # type: ignore
                                    global_model_predicted_counts=global_model_predicted_counts[
                                        :, test_idx
                                    ],
                                    global_model_embeddings=global_model_embeddings,
                                    celltype_model_predicted_counts=celltype_model_predicted_counts[
                                        :, test_idx
                                    ],
                                    celltype_model_embeddings=celltype_model_embeddings,
                                )
                            )

                        celltype_results.append(
                            CelltypeResult(
                                celltype=celltype,
                                celltype_pca_embedding=celltype_pca_embeddings[
                                    celltype
                                ],
                                celltype_umap_embedding=celltype_umap_embeddings[
                                    celltype
                                ],  # type: ignore
                                global_model=global_model,
                                global_model_embedding_size=global_model_embeddings.shape[
                                    1
                                ],
                                celltype_model=celltype_model,
                                celltype_model_embedding_size=celltype_model_embeddings.shape[
                                    1
                                ],
                                samples=sample_results,
                            )
                        )

                    with tempfile.TemporaryDirectory() as tmpdir:
                        results_path = os.path.join(tmpdir, "results.pkl")

                        with open(results_path, "wb") as f:
                            pickle.dump(celltype_results, f)

                        mlflow.log_artifact(results_path)
                        mlflow.log_params(
                            {
                                "dataset": dataloader.name,
                                "predictor": predictor.name,
                            }
                        )


if __name__ == "__main__":
    main()
