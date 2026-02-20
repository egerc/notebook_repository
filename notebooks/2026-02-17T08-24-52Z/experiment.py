from __future__ import annotations

import logging
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar, Union

import mlflow
import nico2_lib as n2l
import numpy as np
import scanpy as sc
import yaml
from anndata.typing import AnnData
from joblib import Memory
from joblib.externals.loky.backend.context import os
from nico2_lib.predictors import NmfPredictor
from nico2_lib.predictors._scvi._scvi_pred import ScviPredictor
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


@dataclass(frozen=True)
class Config:
    data_dir: str
    cache_dir: str
    n_pca_components: int
    seed: int
    n_samples: int
    sample_length: int
    dataloader_keys: list[str] = field(default_factory=list)
    predictor_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PredictorWrapper:
    name: str
    predictor: n2l.pd.PredictorProtocol


@dataclass(frozen=True)
class PredictorFactoryWrapper:
    name: str
    create_predictor: Callable[[], n2l.pd.PredictorProtocol]


@dataclass(frozen=True)
class LoaderFnWrapper:
    name: str
    loader: LoaderFn


@dataclass(frozen=True)
class ExperimentOutput:
    dataset_name: str
    celltype_metadata: list[CelltypeMetadata]
    split_results: list[SplitResult]


@dataclass(frozen=True)
class CelltypeMetadata:
    celltype: str
    celltype_mask: NDArray[np.bool_]
    pca_embedding: NumericArray
    umap_embedding: NumericArray
    observed_counts: NumericArray


@dataclass(frozen=True)
class SplitResult:
    sample_idx: int
    train_idx: Union[Sequence[int], NDArray[np.intp]]
    test_idx: Union[Sequence[int], NDArray[np.intp]]
    model_results: list[ModelSplitResult]


@dataclass(frozen=True)
class ModelSplitResult:
    model_name: str
    celltype_results: list[CelltypeSplitResult]


@dataclass(frozen=True)
class CelltypeSplitResult:
    celltype: str
    n_cells: int
    observed_test_counts: NumericArray
    global_model_embedding: NumericArray
    global_model_predicted_test_counts: NumericArray
    celltype_model_embedding: NumericArray
    celltype_model_predicted_test_counts: NumericArray


@dataclass(frozen=True)
class _PreparedCelltype:
    metadata: CelltypeMetadata
    celltype_models_by_name: dict[str, PredictorWrapper]


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


def _create_pseudospatial_loader(
    loader_func: Callable[[str], AnnData], ct_key: str
) -> LoaderFn:
    def split_loader(
        dir: str,
        *,
        cached_label_transfer: bool = True,
    ) -> tuple[AnnData, AnnData, str, str]:
        adata = loader_func(dir)
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

        return query, reference, ct_key, ct_key

    return split_loader


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


def _filter_celltypes(adata: AnnData, ct_key: str, min_count: int) -> list[str]:
    celltypes = adata.obs[ct_key].value_counts()
    return celltypes[celltypes >= min_count].index.tolist()  # type: ignore


def load_config(config_path: Path | None = None) -> Config:
    path = config_path or Path(__file__).with_name("config.yaml")
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _build_dataloader_registry(cache_dir: str) -> dict[str, LoaderFnWrapper]:
    registry = [
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
        LoaderFnWrapper(
            name="small_mouse_intestine_pseudospatial",
            loader=_create_pseudospatial_loader(
                n2l.dt.small_mouse_intestine_sc, ct_key="cluster"
            ),
        ),
        LoaderFnWrapper(
            name="human_liver_spatial",
            loader=_create_spatial_loader(
                query_loader=lambda dir: n2l.dt.xenium_10x_loader(
                    name="Xenium_V1_hLiver_nondiseased_section_FFPE", dir=dir
                ),
                query_ct_key="annot",
                reference_loader=n2l.dt.human_liver_cell_atlas,
                reference_ct_key="annot",
                memory=Memory(cache_dir),
            ),
        ),
        LoaderFnWrapper(
            name="human_liver_pseudospatial",
            loader=_create_pseudospatial_loader(
                loader_func=n2l.dt.human_liver_cell_atlas,
                ct_key="annot",
            ),
        ),
    ]
    return {loader.name: loader for loader in registry}


def _build_predictor_registry() -> dict[str, PredictorFactoryWrapper]:
    registry = [
        PredictorFactoryWrapper(
            "NMF_3", create_predictor=lambda: NmfPredictor(n_components=3)
        ),
        PredictorFactoryWrapper(
            "NMF_8", create_predictor=lambda: NmfPredictor(n_components=8)
        ),
        PredictorFactoryWrapper(
            "SCVI_3", create_predictor=lambda: ScviPredictor(n_factors=3)
        ),
        PredictorFactoryWrapper(
            "SCVI_8", create_predictor=lambda: ScviPredictor(n_factors=8)
        ),
    ]
    return {predictor.name: predictor for predictor in registry}


RegistryT = TypeVar("RegistryT")


def _select_registry_entries(
    registry: Mapping[str, RegistryT],
    selected_keys: Sequence[str],
    *,
    entry_type: str,
) -> list[RegistryT]:
    keys = list(registry.keys()) if len(selected_keys) == 0 else list(selected_keys)
    unknown_keys = [key for key in keys if key not in registry]
    if unknown_keys:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown {entry_type} key(s): {', '.join(unknown_keys)}. "
            f"Available keys: {available}"
        )
    return [registry[key] for key in keys]


def run_experiment(config: Config) -> list[ExperimentOutput]:
    experiment_outputs: list[ExperimentOutput] = []
    dataloader_registry = _build_dataloader_registry(config.cache_dir)
    dataloaders = _select_registry_entries(
        dataloader_registry,
        config.dataloader_keys,
        entry_type="dataloader",
    )
    predictor_registry = _build_predictor_registry()
    predictor_factories = _select_registry_entries(
        predictor_registry,
        config.predictor_keys,
        entry_type="predictor",
    )
    for dataloader in dataloaders:
        query, reference, query_ct_key, reference_ct_key = dataloader.loader(
            config.data_dir
        )
        _adata_dense_mut(query)
        _adata_dense_mut(reference)
        celltype_metadata, split_results = get_split_results(
            query=query,
            reference=reference,
            query_ct_key=query_ct_key,
            reference_ct_key=reference_ct_key,
            n_pca_components=config.n_pca_components,
            seed=config.seed,
            n_samples=config.n_samples,
            sample_length=config.sample_length,
            predictor_factories=predictor_factories,
        )
        experiment_outputs.append(
            ExperimentOutput(
                dataset_name=dataloader.name,
                celltype_metadata=celltype_metadata,
                split_results=split_results,
            )
        )
    return experiment_outputs


def get_split_results(
    query: AnnData,
    reference: AnnData,
    query_ct_key: str,
    reference_ct_key: str,
    n_pca_components: int,
    seed: int,
    n_samples: int,
    sample_length: int,
    predictor_factories: Sequence[PredictorFactoryWrapper],
) -> tuple[list[CelltypeMetadata], list[SplitResult]]:
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

    global_models_by_name = {
        predictor_factory.name: PredictorWrapper(
            name=predictor_factory.name,
            predictor=predictor_factory.create_predictor().fit(reference_shared.X),  # type: ignore[arg-type]
        )
        for predictor_factory in predictor_factories
    }

    prepared_celltypes: list[_PreparedCelltype] = []
    celltype_metadata: list[CelltypeMetadata] = []
    for celltype in shared_celltypes:
        query_celltype_mask = query_shared.obs[query_ct_key] == celltype
        reference_celltype_mask = reference_shared.obs[reference_ct_key] == celltype
        observed_counts = np.asarray(query_shared[query_celltype_mask].X)  # type: ignore[arg-type]
        pca_embedding = PCA(n_components=n_pca_components).fit_transform(
            StandardScaler().fit_transform(query_shared[query_celltype_mask].X)
        )
        umap_embedding = UMAP(n_components=2).fit_transform(pca_embedding)
        metadata = CelltypeMetadata(
            celltype=str(celltype),
            celltype_mask=query_celltype_mask,
            pca_embedding=pca_embedding,
            umap_embedding=umap_embedding,  # type: ignore[arg-type]
            observed_counts=observed_counts,
        )
        celltype_metadata.append(metadata)
        celltype_models_by_name = {
            predictor_factory.name: PredictorWrapper(
                name=predictor_factory.name,
                predictor=predictor_factory.create_predictor().fit(
                    reference_shared[reference_celltype_mask].X  # type: ignore
                ),
            )
            for predictor_factory in predictor_factories
        }
        prepared_celltypes.append(
            _PreparedCelltype(
                metadata=metadata,
                celltype_models_by_name=celltype_models_by_name,
            )
        )

    split_results: list[SplitResult] = []
    for sample_idx, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
        model_results: list[ModelSplitResult] = []
        for model_name, global_model in global_models_by_name.items():
            celltype_results: list[CelltypeSplitResult] = []
            for prepared_celltype in prepared_celltypes:
                observed_counts = np.asarray(prepared_celltype.metadata.observed_counts)
                observed_train_counts = observed_counts[:, train_idx]
                global_embedding, global_predicted_counts = global_model.predictor.predict(
                    observed_train_counts,
                    train_idx,
                )
                celltype_model = prepared_celltype.celltype_models_by_name[model_name]
                celltype_embedding, celltype_predicted_counts = (
                    celltype_model.predictor.predict(
                        observed_train_counts,
                        train_idx,
                    )
                )
                observed_test_counts = observed_counts[:, test_idx]
                celltype_results.append(
                    CelltypeSplitResult(
                        celltype=prepared_celltype.metadata.celltype,
                        n_cells=int(observed_counts.shape[0]),
                        observed_test_counts=observed_test_counts,
                        global_model_embedding=global_embedding,
                        global_model_predicted_test_counts=global_predicted_counts[
                            :, test_idx
                        ],
                        celltype_model_embedding=celltype_embedding,
                        celltype_model_predicted_test_counts=celltype_predicted_counts[
                            :, test_idx
                        ],
                    )
                )
            model_results.append(
                ModelSplitResult(
                    model_name=model_name,
                    celltype_results=celltype_results,
                )
            )
        split_results.append(
            SplitResult(
                sample_idx=sample_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                model_results=model_results,
            )
        )

    return celltype_metadata, split_results


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
