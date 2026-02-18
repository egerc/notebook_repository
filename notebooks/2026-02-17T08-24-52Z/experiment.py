from __future__ import annotations

import logging
import pickle
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
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

DATA_DIR = "../data"
CACHE_DIR = "../cache"
SEED = 0
N_SAMPLES = 2
SAMPLE_LENGTH = 20
MLFLOW_EXPERIMENT = "nico2_sweep"


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
    data_dir: str = "../data"
    cache_dir: str = "../cache"
    seed: int = 0
    n_samples: int = 10
    sample_length: int = 20
    mlflow_experiment_name: str = "explainability_benchmark"
    mlflow_parent_run_name: str = "dataset_predictor_sweep"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: tuple[AnnData, AnnData, str, str]
    predictor: n2l.pd.PredictorProtocol
    seed: int
    n_samples: int
    sample_length: int


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
    global_model: n2l.pd.PredictorProtocol
    celltype_model: n2l.pd.PredictorProtocol
    samples: Sequence[SampleResult]


def _run_experiment(
    dataset: tuple[AnnData, AnnData, str, str],
    predictor: n2l.pd.PredictorProtocol,
    *,
    seed: int,
    n_samples: int,
    sample_length: int,
) -> list[CelltypeResult]:
    """Run per-celltype masking experiments with global and celltype-specific models.

    The function restricts query and reference AnnData objects to shared
    celltypes and shared features, samples train/test feature index splits, then
    compares predictions from:
    1) a global model fit on all reference cells and
    2) a celltype-specific model fit on each reference celltype subset.

    Parameters
    ----------
    dataset
        Tuple containing `(query, reference, query_ct_key, reference_ct_key)`.
        `query_ct_key` and `reference_ct_key` are `.obs` column names with
        celltype labels.
    predictor
        Predictor factory implementing `fit(...) -> model` and
        `model.predict(masked_counts, train_idx)`.
    seed
        Random seed used to generate feature train/test splits.
    n_samples
        Number of train/test index samples to evaluate.
    sample_length
        Number of training features per sampled split.

    Returns
    -------
    list[CelltypeResult]
        One result per shared celltype, each containing sampled predictions,
        embeddings, and observed counts for both model variants.
    """
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

    celltype_results: list[CelltypeResult] = []
    global_model = predictor.fit(reference_shared.X)  # type: ignore
    for celltype in shared_celltypes:
        query_celltype_mask = query_shared.obs[query_ct_key] == celltype
        reference_celltype_mask = reference_shared.obs[reference_ct_key] == celltype
        celltype_model = predictor.fit(
            reference_shared[reference_celltype_mask].X  # type: ignore
        )
        sample_results: list[SampleResult] = []
        for idx, (train_idx, test_idx) in enumerate(zip(train_idxs, test_idxs)):
            observed_counts = query_shared[query_celltype_mask, test_idx].X  # type: ignore
            global_model_embeddings, global_model_predicted_counts = (
                global_model.predict(
                    query_shared[query_celltype_mask, train_idx].X,  # type: ignore
                    train_idx,
                )
            )
            celltype_model_embeddings, celltype_model_predicted_counts = (
                celltype_model.predict(
                    query_shared[query_celltype_mask, train_idx].X,  # type: ignore
                    train_idx,
                )
            )
            sample_result = SampleResult(
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
            sample_results.append(sample_result)

        celltype_results.append(
            CelltypeResult(
                celltype=celltype,
                global_model=global_model,
                celltype_model=celltype_model,
                samples=sample_results,
            )
        )
    return celltype_results


def run_experiment(config: ExperimentConfig) -> list[CelltypeResult]:
    return _run_experiment(
        dataset=config.dataset,
        predictor=config.predictor,
        seed=config.seed,
        n_samples=config.n_samples,
        sample_length=config.sample_length,
    )


predictors: list[PredictorWrapper] = [
    PredictorWrapper("NMF_3", NmfPredictor(n_components=3)),
    PredictorWrapper("NMF_8", NmfPredictor(n_components=8)),
]
dataloaders: list[LoaderFnWrapper] = [
    LoaderFnWrapper(
        name="small_mouse_intestine_spatial",
        loader=_create_spatial_loader(
            query_loader=n2l.dt.small_mouse_intestine_merfish,
            query_ct_key="cluster",
            reference_loader=n2l.dt.small_mouse_intestine_sc,
            reference_ct_key="cluster",
            memory=Memory(CACHE_DIR),
        ),
    ),
]


def main() -> None:
    config = Config()
    mlflow.set_experiment(config.mlflow_experiment_name)
    with mlflow.start_run(run_name=config.mlflow_parent_run_name):
        mlflow.log_params(config.__dict__)
        for predictor, dataloader in product(predictors, dataloaders):
            with mlflow.start_run(
                nested=True, run_name=f"{predictor.name}_{dataloader.name}"
            ):
                experiment_config = ExperimentConfig(
                    dataset=dataloader.loader(config.data_dir),
                    predictor=predictor.predictor,
                    seed=config.seed,
                    n_samples=config.n_samples,
                    sample_length=config.sample_length,
                )

                results = run_experiment(experiment_config)

                with tempfile.TemporaryDirectory() as tmpdir:
                    results_path = os.path.join(tmpdir, "results.pkl")

                    with open(results_path, "wb") as f:
                        pickle.dump(results, f)
                        
                    mlflow.log_artifact(results_path)
                    mlflow.log_params(
                        {
                            "dataset": dataloader.name,
                            "predictor": predictor.name,
                        }
                    )


if __name__ == "__main__":
    main()
