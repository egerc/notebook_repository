import argparse
import csv
import logging
import pprint
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from itertools import groupby, product
from typing import Annotated, Any, Literal, TypedDict, assert_never

import nico2_lib as n2l
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import read_h5ad
from anndata.typing import AnnData
from nico2_lib.typing import IndexArray, NumericArray
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

type RawNumericArray = Annotated[NumericArray, "raw"]

type StatisticalMeasure1D = Callable[[NumericArray, NumericArray], float]

type StatisticalMeasureKey = Literal[
    "pearson",
    "spearman",
    "cosine_similarity",
    "mse",
    "explained_variance",
    "explained_variance_v2",
]


def prepare_scoring_function(
    func: Callable[[NumericArray, NumericArray], float | NumericArray],
) -> StatisticalMeasure1D:
    def scoring_function(x_true: NumericArray, x_pred: NumericArray) -> float:
        assert x_true.shape == x_pred.shape, (
            "x_true and x_pred must have the same shape"
        )
        assert x_true.ndim == 1 and x_pred.ndim == 1, (
            "x_true and x_pred must be 1D arrays"
        )
        value = func(x_true, x_pred)
        assert isinstance(value, (float, np.floating)), (
            "scoring function must return a float"
        )
        return value

    return scoring_function


STATISTICAL_MEASURE_REGISTRY: dict[StatisticalMeasureKey, StatisticalMeasure1D] = {
    "pearson": prepare_scoring_function(
        n2l.mt.pearson_metric,
    ),
    "spearman": prepare_scoring_function(
        n2l.mt.spearman_metric,
    ),
    "cosine_similarity": prepare_scoring_function(
        n2l.mt.cosine_similarity_metric,
    ),
    "mse": prepare_scoring_function(
        n2l.mt.mse_metric,
    ),
    "explained_variance": prepare_scoring_function(
        n2l.mt.explained_variance_metric,
    ),
    "explained_variance_v2": prepare_scoring_function(
        n2l.mt.explained_variance_metric_v2
    ),
}
type StatisticalMeasure2D = StatisticalMeasure1D  # just for reference, the types are the same but at runtime 1d only takes 2 1d arrays and 2d takes 2 2d arrays
type ScoringAggregation = Callable[[StatisticalMeasure1D], StatisticalMeasure2D]
type ScoringAggregationKey = Literal[
    "mean_score_of_expression",
    "score_of_mean_expression",
]

SCORING_AGGREGATION_REGISTRY: dict[ScoringAggregationKey, ScoringAggregation] = {
    "mean_score_of_expression": lambda statistical_measure: (
        lambda x_true, x_pred: float(
            np.array(
                [
                    statistical_measure(vec_true, vec_pred)
                    for vec_true, vec_pred in zip(x_true, x_pred)
                ]
            ).mean()
        )
    ),
    "score_of_mean_expression": lambda statistical_measure: (
        lambda x_true, x_pred: float(
            statistical_measure(x_true.mean(axis=0), x_pred.mean(axis=0))
        )
    ),
}


type ScoringAxis = Callable[[StatisticalMeasure2D], StatisticalMeasure2D]
type ScoringAxisKey = Literal["cell-wise", "gene-wise"]

SCORING_AXIS_REGISTRY: dict[ScoringAxisKey, ScoringAxis] = {
    "cell-wise": lambda statistical_measure: (
        lambda x_true, x_pred: statistical_measure(x_true, x_pred)
    ),
    "gene-wise": lambda statistical_measure: (
        lambda x_true, x_pred: statistical_measure(x_true.T, x_pred.T)
    ),
}


@dataclass
class InputDataset:
    sample_id: int
    celltype: str
    counts: NumericArray
    training_cells_index: IndexArray
    testing_cells_index: IndexArray
    training_genes_index: IndexArray
    testing_genes_index: IndexArray


type DatasetGenerator = Generator[InputDataset, None, None]


def _dense_counts_from_anndata(adata: AnnData) -> NumericArray:
    return adata.X.toarray() if sp.issparse(adata.X) else adata.X  # type: ignore


def make_spatial_dataset_generator(
    query: tuple[str, str],
    reference: tuple[str, str],
    n_panel_genes: int,
    n_samples: int,
    seed: int | None = None,
) -> DatasetGenerator:
    rng = np.random.default_rng(seed)
    (
        (query_path, query_cluster_key),
        (reference_path, reference_cluster_key),
    ) = (
        query,
        reference,
    )
    query_anndata, reference_anndata = read_h5ad(query_path), read_h5ad(reference_path)
    shared_celltypes = np.intersect1d(
        query_anndata.obs[query_cluster_key],
        reference_anndata.obs[reference_cluster_key],
    ).tolist()
    assert len(shared_celltypes) > 0, "No shared cell types between query and reference"
    shared_genes = np.intersect1d(
        query_anndata.var_names,
        reference_anndata.var_names,
    ).tolist()
    assert len(shared_genes) > 0, "No shared genes between query and reference"
    samples: list[tuple[IndexArray, IndexArray]] = [  # type: ignore
        (
            train_test_split(
                np.arange(len(shared_genes)),
                train_size=len(shared_genes) - n_panel_genes,
                random_state=rng.integers(0, 1000),
            )
        )
        for _ in range(n_samples)
    ]
    for celltype in shared_celltypes:
        query_celltype_mask = query_anndata.obs[query_cluster_key] == celltype
        reference_celltype_mask = (
            reference_anndata.obs[reference_cluster_key] == celltype
        )
        query_counts = _dense_counts_from_anndata(
            query_anndata[query_celltype_mask, shared_genes]
        )
        reference_counts = _dense_counts_from_anndata(
            reference_anndata[reference_celltype_mask, shared_genes]
        )
        counts = np.vstack([reference_counts, query_counts])

        training_cells_index, testing_cells_index = np.split(
            np.arange(counts.shape[0]), [reference_counts.shape[0]]
        )
        for sample_id, (training_genes_index, testing_genes_index) in enumerate(
            samples
        ):
            yield InputDataset(
                sample_id=sample_id,
                celltype=celltype,
                counts=counts,
                training_cells_index=training_cells_index,
                testing_cells_index=testing_cells_index,
                training_genes_index=training_genes_index,
                testing_genes_index=testing_genes_index,
            )


@dataclass
class Subpanel:
    size: int
    panel_ranking: Literal["HVG"]
    panel_size: int


@dataclass
class Outofpanel:
    panel_size: int


def make_pseudospatial_dataset_generator(
    adata_path: str,
    cluster_key: str,
    training_cells_fraction: float,
    n_samples: int,
    mode: Subpanel | Outofpanel,
    seed: int | None = None,
) -> DatasetGenerator:
    rng = np.random.default_rng(seed)
    adata = read_h5ad(adata_path)
    celltypes = adata.obs[cluster_key].unique().tolist()

    match mode:
        case Subpanel(size, "HVG", panel_size):
            hvg_df = sc.pp.highly_variable_genes(
                adata,
                n_top_genes=panel_size,
                subset=True,
                flavor="seurat_v3",
                inplace=False,
            )
            assert hvg_df is not None, "Failed to compute highly variable genes"
            panel_gene_indices = np.where(
                adata.var_names.isin(hvg_df[hvg_df["highly_variable"]].index)
            )[0]

            for celltype in celltypes:
                celltype_mask = adata.obs[cluster_key] == celltype
                counts = _dense_counts_from_anndata(
                    adata[celltype_mask, panel_gene_indices]
                )
                num_cells, _ = counts.shape
                training_cells_index: IndexArray
                testing_cells_index: IndexArray
                training_cells_index, testing_cells_index = train_test_split(  # type: ignore
                    np.arange(num_cells),
                    train_size=training_cells_fraction,
                    random_state=rng.integers(0, 1000000),
                )

                for sample_id in range(n_samples):
                    training_genes_index: IndexArray
                    testing_genes_index: IndexArray
                    training_genes_index, testing_genes_index = train_test_split(  # type: ignore
                        np.arange(panel_size),
                        test_size=size,
                        random_state=rng.integers(0, 1000000),
                    )
                    yield InputDataset(
                        sample_id=sample_id,
                        celltype=celltype,
                        counts=counts,
                        training_cells_index=training_cells_index,
                        testing_cells_index=testing_cells_index,
                        training_genes_index=training_genes_index,
                        testing_genes_index=testing_genes_index,
                    )

        case Outofpanel(panel_size):
            all_gene_indices = np.arange(adata.n_vars)

            for celltype in celltypes:
                celltype_mask = adata.obs[cluster_key] == celltype
                counts = _dense_counts_from_anndata(adata[celltype_mask, :])

                num_cells, _ = counts.shape

                for sample_id in range(n_samples):
                    training_cells_index: IndexArray
                    testing_cells_index: IndexArray
                    training_cells_index, testing_cells_index = train_test_split(  # type: ignore
                        np.arange(num_cells),
                        train_size=training_cells_fraction,
                        random_state=rng.integers(0, 1000000),
                    )
                    shuffled_genes = rng.permutation(all_gene_indices)
                    training_genes_index, testing_genes_index = np.split(
                        shuffled_genes, [panel_size]
                    )

                    yield InputDataset(
                        sample_id=sample_id,
                        celltype=celltype,
                        counts=counts,
                        training_cells_index=training_cells_index,
                        testing_cells_index=testing_cells_index,
                        training_genes_index=training_genes_index,
                        testing_genes_index=testing_genes_index,
                    )

        case _:
            assert_never(mode)


type DatasetKey = Literal[
    "small_mouse_intestine_spatial",
    "small_mouse_intestine_pseudospatial_full",
    "small_mouse_intestine_pseudospatial_panel",
]

DATASET_REGISTRY: dict[DatasetKey, DatasetGenerator] = {
    "small_mouse_intestine_spatial": make_spatial_dataset_generator(
        query=(
            "/home/gruengroup/christian/Data/mouse_intestine/intestine_MERFISH.h5ad",
            "C_scanvi",
        ),
        reference=(
            "/home/gruengroup/christian/Projects/notebook_repository/notebooks/data/mouse_small_intestine_sc/mouse_small_intestine_sc.h5ad",
            "cluster",
        ),
        n_panel_genes=20,
        n_samples=5,
    ),
    "small_mouse_intestine_pseudospatial_full": make_pseudospatial_dataset_generator(
        adata_path="/home/gruengroup/christian/Data/mouse_intestine/intestine_scRNA.h5ad",
        cluster_key="cluster",
        training_cells_fraction=0.8,
        n_samples=5,
        mode=Outofpanel(
            panel_size=500,
        ),
    ),
    "small_mouse_intestine_pseudospatial_panel": make_pseudospatial_dataset_generator(
        adata_path="/home/gruengroup/christian/Data/mouse_intestine/intestine_scRNA.h5ad",
        cluster_key="cluster",
        training_cells_fraction=0.8,
        n_samples=5,
        mode=Subpanel(
            size=20,
            panel_ranking="HVG",
            panel_size=500,
        ),
    ),
}


type ProcessedNumericArray = Annotated[NumericArray, "processed"]
type PreprocessingFunctionKey = Literal["identity", "log1p"]
type PreprocessingFunction = Callable[[RawNumericArray], ProcessedNumericArray]
type InvertPreprocessingFunction = Callable[[ProcessedNumericArray], RawNumericArray]

PREPROCESSING_FUNCTION_REGISTRY: dict[
    PreprocessingFunctionKey, tuple[PreprocessingFunction, InvertPreprocessingFunction]
] = {
    "identity": (lambda x: x, lambda x: x),
    "log1p": (lambda x: np.log1p(x), lambda x: np.expm1(x)),
}


def _run_experiment_outer_product(
    predictor: n2l.pd.PredictorProtocol,
    dataset_keys: Sequence[DatasetKey],
    scoring_axis_keys: Sequence[ScoringAxisKey],
    statistical_measure_keys: Sequence[StatisticalMeasureKey],
    scoring_aggregation_keys: Sequence[ScoringAggregationKey],
    preprocessing_keys: Sequence[tuple[PreprocessingFunctionKey, bool]],
) -> Generator[dict[str, Any], None, None]:
    for dataset_key in dataset_keys:
        dataset_generator = DATASET_REGISTRY[dataset_key]
        for dataset in dataset_generator:
            for preprocessing_key, score_in_raw in preprocessing_keys:
                preprocessing_function, invert_preprocessing_function = (
                    PREPROCESSING_FUNCTION_REGISTRY[preprocessing_key]
                )
                counts = preprocessing_function(dataset.counts)
                _, counts_predicted = predictor.fit(
                    counts[dataset.training_cells_index]
                ).predict(
                    counts[dataset.testing_cells_index][
                        :, dataset.training_genes_index
                    ],
                    dataset.training_genes_index,
                )
                for (
                    scoring_axis_key,
                    statistical_measure_key,
                    scoring_aggregation_key,
                ) in product(
                    scoring_axis_keys,
                    statistical_measure_keys,
                    scoring_aggregation_keys,
                ):
                    scoring_axis = SCORING_AXIS_REGISTRY[scoring_axis_key]
                    statistical_measure = STATISTICAL_MEASURE_REGISTRY[
                        statistical_measure_key
                    ]
                    scoring_aggregation = SCORING_AGGREGATION_REGISTRY[
                        scoring_aggregation_key
                    ]
                    x_true = counts[dataset.testing_cells_index][
                        :, dataset.testing_genes_index
                    ]
                    x_pred = counts_predicted[:, dataset.testing_genes_index]
                    scoring_function = scoring_axis(
                        scoring_aggregation(
                            statistical_measure,
                        ),
                    )
                    if score_in_raw:
                        score = scoring_function(
                            invert_preprocessing_function(x_true),
                            invert_preprocessing_function(x_pred),
                        )
                    else:
                        score = scoring_function(x_true, x_pred)
                    yield {
                        "dataset": dataset_key,
                        "sample_id": dataset.sample_id,
                        "celltype": dataset.celltype,
                        "scoring_axis": scoring_axis_key,
                        "statistical_measure": statistical_measure_key,
                        "scoring_aggregation": scoring_aggregation_key,
                        "preprocessing": preprocessing_key,
                        "score_in_raw_space": score_in_raw,
                        "score": score,
                    }


def run_experiment_outer_product(
    predictor: n2l.pd.PredictorProtocol,
    dataset_keys: Sequence[DatasetKey],
    scoring_axis_keys: Sequence[ScoringAxisKey],
    scoring_function_keys: Sequence[StatisticalMeasureKey],
    scoring_aggregation_keys: Sequence[ScoringAggregationKey],
    preprocessing_keys: Sequence[tuple[PreprocessingFunctionKey, bool]],
) -> pd.DataFrame:
    return pd.DataFrame(
        (
            result
            for result in _run_experiment_outer_product(
                predictor,
                dataset_keys,
                scoring_axis_keys,
                scoring_function_keys,
                scoring_aggregation_keys,
                preprocessing_keys,
            )
        )
    )


BenchmarkingConfig = TypedDict(
    "BenchmarkingConfig",
    {
        "name": str,
        "dataset_generator": DatasetKey,
        "preprocessing": PreprocessingFunctionKey,
        "score_in_raw": bool,
        "scoring_axis": ScoringAxisKey,
        "statistical_measure": StatisticalMeasureKey,
        "scoring_aggregation": ScoringAggregationKey,
    },
)


def targeted_experiment(
    predictor: n2l.pd.PredictorProtocol,
    benchmarking_configs: Sequence[BenchmarkingConfig],
) -> Generator[dict[str, Any], None, None]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output csv"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    SCORING_AXIS_KEYS: list[ScoringAxisKey] = [
        "cell-wise",
        "gene-wise",
    ]
    SCORING_FUNCTION_KEYS: list[StatisticalMeasureKey] = [
        "pearson",
        "spearman",
        "cosine_similarity",
        "mse",
        "explained_variance",
        "explained_variance_v2",
    ]
    dataset_keys: list[DatasetKey] = [
        "small_mouse_intestine_spatial",
        "small_mouse_intestine_pseudospatial_panel",
        "small_mouse_intestine_pseudospatial_full",
    ]
    scoring_aggregation_keys: list[ScoringAggregationKey] = [
        "mean_score_of_expression",
        "score_of_mean_expression",
    ]
    preprocessing_keys: list[tuple[PreprocessingFunctionKey, bool]] = [
        ("identity", False),
        ("log1p", True),
        ("log1p", False),
    ]
    logger.info(
        f"Starting experiment with the following configuration:\n  Datasets: {dataset_keys}\n  Scoring axes: {SCORING_AXIS_KEYS}\n  Measures: {SCORING_FUNCTION_KEYS}\n  Aggregations: {scoring_aggregation_keys}\n  Preprocessing: {preprocessing_keys}"
    )
    with open(args.output, "w", newline="", buffering=1) as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "sample_id",
                "celltype",
                "scoring_axis",
                "statistical_measure",
                "scoring_aggregation",
                "preprocessing",
                "score_in_raw_space",
                "score",
            ],
        )
        csv_writer.writeheader()
        for result in tqdm(
            _run_experiment_outer_product(
                n2l.pd.NmfPredictor(n_components=3),
                dataset_keys,
                SCORING_AXIS_KEYS,
                SCORING_FUNCTION_KEYS,
                scoring_aggregation_keys,
                preprocessing_keys,
            )
        ):
            csv_writer.writerow(result)
            logger.info(f"Processed:\n{pprint.pformat(result, indent=4)}")


if __name__ == "__main__":
    main()
