from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from types import CellType
from typing import Any, Callable, List, Sequence

import mlflow
import nico2_lib as n2l
import numpy as np
import polars as pl
import scanpy as sc
from anndata.typing import AnnData
from joblib import Memory
from nico2_lib.predictors._scvi._scvi_pred import ScviPredictor
from numpy import number, str_
from numpy.typing import NDArray
from pandas.core.arrays.sparse.array import SequenceIndexer
from scanpy._utils.random import SeedLike
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

DATA_DIR = "./data"
CACHE_DIR = "./cache"
DEFAULT_SEED = 0
DEFAULT_N_SAMPLES = 2
DEFAULT_SAMPLE_LENGTH = 20
DEFAULT_PCA_COMPONENTS = 25
DEFAULT_CLUSTER_RANDOM_STATE = 0
DEFAULT_MLFLOW_EXPERIMENT = "nico2_sweep"

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger(__name__)


NumericArray = NDArray[number]
StringArray = NDArray[str_]
LoaderFn = Callable[[str], tuple[AnnData, AnnData, str, str]]
PredictorFactory = Callable[..., n2l.pd.PredictorProtocol]
MetricFn = Callable[[NumericArray, NumericArray], float]


DEFAULT_CONFIG: dict[str, Any] = {
    "data_dir": DATA_DIR,
    "cache_dir": CACHE_DIR,
    "seed": DEFAULT_SEED,
    "n_samples": DEFAULT_N_SAMPLES,
    "sample_length": DEFAULT_SAMPLE_LENGTH,
    "dataset_names": (),
    "predictor_names": (),
    "mlflow_tracking_uri": None,
    "mlflow_experiment_name": DEFAULT_MLFLOW_EXPERIMENT,
    "mlflow_parent_run_name": "dataset_predictor_sweep",
    "pca_components": DEFAULT_PCA_COMPONENTS,
    "cluster_random_state": DEFAULT_CLUSTER_RANDOM_STATE,
}


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


@dataclass(frozen=True)
class SampleResult:
    train_idx: NDArray[np.int_]
    test_idx: NDArray[np.int_]
    observed_counts: NumericArray
    global_model_predicted_counts: NumericArray
    celltype_model_predicted_counts: NumericArray


@dataclass(frozen=True)
class CelltypeResult:
    celltype: str
    global_model: n2l.pd.PredictorProtocol
    celltype_model: n2l.pd.PredictorProtocol
    samples: Sequence[SampleResult]


def experiment(
    dataset: tuple[AnnData, AnnData, str, str],
    predictor: n2l.pd.PredictorProtocol,
    *,
    seed: int,
    n_samples: int,
    sample_length: int,
) -> List[CelltypeResult]:
    query, reference, query_ct_key, reference_ct_key = dataset

    _adata_dense_mut(query)
    _adata_dense_mut(reference)

    query_celltypes = query.obs[query_ct_key].to_numpy()
    reference_celltypes = reference.obs[reference_ct_key].to_numpy()
    shared_celltypes = np.intersect1d(query_celltypes, reference_celltypes)
    shared_features = np.intersect1d(query.var_names, reference.var_names)

    query_mask = query.obs[query_ct_key].isin(shared_celltypes.tolist())
    reference_mask = reference.obs[reference_ct_key].isin(shared_celltypes.tolist())
    query_shared = query[query_mask, shared_features]
    reference_shared = reference[reference_mask, shared_features]

    n_features = shared_features.shape[0]
    rng = np.random.default_rng(seed)
    train_idxs, test_idxs = _sample_indices(n_features, n_samples, sample_length, rng)

    celltype_results: List[CelltypeResult] = []
    global_model = predictor.fit(reference_shared.X)  # type: ignore
    for celltype in shared_celltypes:
        query_celltype_mask = query_shared.obs[query_ct_key] == celltype
        reference_celltype_mask = reference_shared.obs[reference_ct_key] == celltype
        celltype_model = predictor.fit(
            reference_shared[reference_celltype_mask].X  # type: ignore
        )
        sample_results = [
            SampleResult(
                train_idx=train_idx,
                test_idx=test_idx,
                observed_counts=query_shared[query_celltype_mask, test_idx].X,  # type: ignore
                global_model_predicted_counts=global_model.predict(
                    query_shared[query_celltype_mask, train_idx].X,  # type: ignore
                    train_idx,
                )[:, test_idx],
                celltype_model_predicted_counts=celltype_model.predict(
                    query_shared[query_celltype_mask, train_idx].X,  # type: ignore
                    train_idx,
                )[:, test_idx],
            )
            for train_idx, test_idx in zip(train_idxs, test_idxs)
        ]
        celltype_results.append(
            CelltypeResult(
                celltype=celltype,
                global_model=global_model,
                celltype_model=celltype_model,
                samples=sample_results,
            )
        )
    return celltype_results
