import sys
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import replace
from enum import Enum, auto
from itertools import groupby, product
from typing import Callable, Literal, assert_never

import nico2_lib as n2l
import numpy as np
import scanpy as sc
from anndata import read_h5ad
from anndata.typing import AnnData
from nico2_lib.typing import IndexArray, NumericArray
from pandas.io.parsers.python_parser import csv
from pydantic.dataclasses import dataclass

type H5adData = tuple[str, str]

type Predictor = n2l.pd.NmfPredictor | n2l.pd.TangramPredictor


@dataclass(frozen=True, slots=True)
class SpatialSubpanel:
    sample_size: int
    n_samples: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class PseudospatialSubpanel:
    panel_size: int
    sample_size: int
    n_samples: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class PseudospatialFull:
    panel_size: int
    n_samples: int
    seed: int | None


type Dataset = (
    tuple[tuple[H5adData, H5adData], SpatialSubpanel]
    | tuple[H5adData, PseudospatialSubpanel]
    | tuple[H5adData, PseudospatialFull]
)


class TransformationFunction(Enum):
    LOG1P = auto()
    EXPM1 = auto()


def apply_transformation_func(
    transform: TransformationFunction | None,
    x: NumericArray,
) -> NumericArray:
    if transform is None:
        return x
    match transform:
        case TransformationFunction.LOG1P:
            return np.log1p(x)
        case TransformationFunction.EXPM1:
            return np.expm1(x)
        case _:
            assert_never(transform)


type PreprocessingTransformation = (
    tuple[
        TransformationFunction,
        TransformationFunction | None,
    ]
    | tuple[None, None]
)


def identity[T](input: T) -> T:
    return input


class ScoringAxis(Enum):
    CELL = auto()
    GENE = auto()


class AggregationType(Enum):
    MEAN_SCORE_OF_EXPRESSION = auto()
    SCORE_OF_MEAN_EXPRESSION = auto()
    SCORE_OF_RAVEL = auto()


class StatisticalMeasure(Enum):
    PEARSON = auto()
    SPEARMAN = auto()
    COSINE_SIM = auto()
    MSE = auto()
    EXPLAINED_VARIANCE_1 = auto()
    EXPLAINED_VARIANCE_2 = auto()


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    dataset: Dataset
    preprocessing: PreprocessingTransformation
    scoring_axis: ScoringAxis
    aggregation_type: AggregationType
    statistical_measure: StatisticalMeasure


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    dataset: Dataset
    sample_id: int
    celltype: str
    preprocessing_transformation: PreprocessingTransformation
    aggregation_type: AggregationType
    scoring_axis: ScoringAxis
    statistical_measure: StatisticalMeasure
    value: float


def extract_dense_counts(adata: AnnData) -> NumericArray:
    return (
        np.asarray(adata.X.toarray())  # type: ignore
        if hasattr(adata.X, "toarray")
        else np.asarray(adata.X)
    )


def combine_adatas(
    adata1: AnnData,
    adata2: AnnData,
    cluster_key: str,
) -> tuple[NumericArray, list[Literal["train", "test"]], list[str]]:
    assert cluster_key in adata1.obs.columns, (
        f"{cluster_key} not found in adata1.obs.columns"
    )
    assert cluster_key in adata2.obs.columns, (
        f"{cluster_key} not found in adata2.obs.columns"
    )
    shared_celltypes: list[str] = list(
        set(adata1.obs[cluster_key].unique()).intersection(
            set(adata2.obs[cluster_key].unique())
        )
    )
    adata1 = adata1[adata1.obs[cluster_key].isin(shared_celltypes)]
    adata2 = adata2[adata2.obs[cluster_key].isin(shared_celltypes)]

    adata_concat = sc.concat(
        [adata2, adata1],
        join="inner",
        label="split",
        keys=["train", "test"],
    )
    split_annotation: list[Literal["train", "test"]] = (
        adata_concat.obs["split"].values.astype(str).tolist()
    )
    celltype_annotation: list[str] = adata_concat.obs[cluster_key].tolist()
    return extract_dense_counts(adata_concat), split_annotation, celltype_annotation


def process_dataset(
    dataset: Dataset,
) -> tuple[
    NumericArray,
    list[Literal["train", "test"]],
    list[str],
    list[tuple[IndexArray, IndexArray]],
]:
    match dataset:
        case (
            (
                (query_h5ad_path, query_h5ad_cluster_key),
                (reference_h5ad_path, reference_h5ad_cluster_key),
            ),
            SpatialSubpanel(
                sample_size,
                n_samples,
                seed,
            ),
        ):
            query_adata, reference_adata = (
                read_h5ad(query_h5ad_path),
                read_h5ad(reference_h5ad_path),
            )
            query_adata.obs[reference_h5ad_cluster_key] = query_adata.obs[
                query_h5ad_cluster_key
            ]
            counts, split_annotation, celltype_annotation = combine_adatas(
                query_adata,
                reference_adata,
                reference_h5ad_cluster_key,
            )

        case (
            (h5ad_path, cluster_key),
            PseudospatialSubpanel(
                panel_size,
                sample_size,
                n_samples,
                seed,
            ),
        ):
            rng = np.random.default_rng(seed)
            adata = read_h5ad(h5ad_path)
            hvg_df = sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=panel_size,
                subset=True,
                inplace=False,
            )
            assert hvg_df is not None, "Failed to identify highly variable genes"
            panel_gene_indices = np.where(
                adata.var_names.isin(hvg_df[hvg_df["highly_variable"]].index)
            )[0]
            counts = extract_dense_counts(adata[:, panel_gene_indices])
            split_annotation: list[Literal["train", "test"]] = rng.choice(
                ["train", "test"], size=counts.shape[0], p=[0.8, 0.2]
            ).tolist()
            celltype_annotation: list[str] = adata.obs[cluster_key].tolist()

        case (
            (h5ad_path, cluster_key),
            PseudospatialFull(
                panel_size,
                n_samples,
                seed,
            ),
        ):
            rng = np.random.default_rng(seed)
            adata = read_h5ad(h5ad_path)
            counts = extract_dense_counts(adata)
            n_cells, n_genes = counts.shape
            sample_size = n_genes - panel_size
            split_annotation: list[Literal["train", "test"]] = rng.choice(
                ["train", "test"], size=n_cells, p=[0.8, 0.2]
            ).tolist()
            celltype_annotation: list[str] = adata.obs[cluster_key].tolist()

        case _:
            assert_never(dataset)
    rng = np.random.default_rng(seed)

    n_genes = counts.shape[1]
    gene_samples: list[tuple[IndexArray, IndexArray]] = [  # type: ignore
        tuple(np.split(rng.permutation(n_genes), [sample_size]))
        for _ in range(n_samples)
    ]
    return (
        counts,
        split_annotation,
        celltype_annotation,
        gene_samples,
    )


def mean_score_of_expression(
    statistical_measure: Callable[[NumericArray, NumericArray], float],
) -> Callable[[NumericArray, NumericArray], float]:
    return lambda x, y: np.array(
        [statistical_measure(xi, yi) for xi, yi in zip(x, y)]
    ).mean()


def score_of_mean_expression(
    statistical_measure: Callable[[NumericArray, NumericArray], float],
) -> Callable[[NumericArray, NumericArray], float]:
    return lambda x, y: statistical_measure(np.mean(x, axis=0), np.mean(y, axis=0))


def score_of_ravel(
    statistical_measure: Callable[[NumericArray, NumericArray], float],
) -> Callable[[NumericArray, NumericArray], float]:
    return lambda x, y: statistical_measure(x.ravel(), y.ravel())


def make_scorer(
    scoring_axis: ScoringAxis,
    aggregation_type: AggregationType,
    statistical_measure: StatisticalMeasure,
) -> Callable[[NumericArray, NumericArray], float]:
    transpose_func: Callable[[NumericArray], NumericArray]
    match scoring_axis:
        case ScoringAxis.CELL:
            transpose_func = identity
        case ScoringAxis.GENE:
            transpose_func = np.transpose
        case _:
            assert_never(scoring_axis)
    aggregation_function: Callable[
        [Callable[[NumericArray, NumericArray], float]],
        Callable[[NumericArray, NumericArray], float],
    ]
    match aggregation_type:
        case AggregationType.MEAN_SCORE_OF_EXPRESSION:
            aggregation_function = mean_score_of_expression
        case AggregationType.SCORE_OF_MEAN_EXPRESSION:
            aggregation_function = score_of_mean_expression
        case AggregationType.SCORE_OF_RAVEL:
            aggregation_function = score_of_ravel
        case _:
            assert_never(aggregation_type)
    statistical_measure_function: Callable[[NumericArray, NumericArray], float]
    match statistical_measure:
        case StatisticalMeasure.PEARSON:
            statistical_measure_function = n2l.mt.pearson_metric  # type: ignore
        case StatisticalMeasure.SPEARMAN:
            statistical_measure_function = n2l.mt.spearman_metric  # type: ignore
        case StatisticalMeasure.COSINE_SIM:
            statistical_measure_function = n2l.mt.cosine_similarity_metric  # type: ignore
        case StatisticalMeasure.MSE:
            statistical_measure_function = n2l.mt.mse_metric  # type: ignore
        case StatisticalMeasure.EXPLAINED_VARIANCE_1:
            statistical_measure_function = n2l.mt.explained_variance_metric  # type: ignore
        case StatisticalMeasure.EXPLAINED_VARIANCE_2:
            statistical_measure_function = n2l.mt.explained_variance_metric_v2  # type: ignore
        case _:
            assert_never(statistical_measure)
    return lambda x, y: aggregation_function(statistical_measure_function)(
        transpose_func(x), transpose_func(y)
    )


def expand_configurations(
    datasets: Sequence[Dataset],
    preprocessing_transformations: Sequence[PreprocessingTransformation],
    scoring_configurations: Sequence[
        tuple[ScoringAxis, AggregationType, StatisticalMeasure]
    ],
) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            dataset=dataset,
            preprocessing=preprocess,
            scoring_axis=scoring_axis,
            aggregation_type=aggregation_type,
            statistical_measure=statistical_measure,
        )
        for dataset, preprocess, (
            scoring_axis,
            aggregation_type,
            statistical_measure,
        ) in product(
            datasets,
            preprocessing_transformations,
            scoring_configurations,
        )
    ]


def run_experiment(
    predictor: Predictor | n2l.pd.PredictorProtocol,
    experiment_configs: Sequence[ExperimentConfig],
) -> Generator[ExperimentResult]:
    for dataset, dataset_configs in groupby(
        experiment_configs, key=lambda x: x.dataset
    ):
        (
            counts,
            split_annotation,
            celltype_annotation,
            gene_samples,
        ) = process_dataset(dataset)
        celltypes = list(set(celltype_annotation))
        breakpoint()
        for preprocessing_transformation, preprocessing_configs in groupby(
            dataset_configs, key=lambda x: x.preprocessing
        ):
            counts = apply_transformation_func(preprocessing_transformation[0], counts)

            for celltype in celltypes:
                celltype_mask = np.array(celltype_annotation) == celltype
                train_mask, test_mask = (
                    (np.array(split_annotation) == "train"),
                    (np.array(split_annotation) == "test"),
                )
                predictor = predictor.fit(counts[train_mask & celltype_mask])
                for sample_id, (testing_genes, training_genes) in enumerate(
                    gene_samples
                ):
                    _, counts_pred = predictor.predict(
                        counts[test_mask & celltype_mask][:, training_genes],
                        training_genes,
                    )
                    for (
                        scoring_axis,
                        aggregation_type,
                        statistical_measure,
                    ), _ in groupby(
                        preprocessing_configs,
                        key=lambda x: (
                            x.scoring_axis,
                            x.aggregation_type,
                            x.statistical_measure,
                        ),
                    ):
                        scorer = make_scorer(
                            scoring_axis, aggregation_type, statistical_measure
                        )
                        value = scorer(
                            apply_transformation_func(
                                preprocessing_transformation[1],
                                counts[test_mask & celltype_mask][:, testing_genes],
                            ),
                            apply_transformation_func(
                                preprocessing_transformation[1],
                                counts_pred[:, testing_genes],
                            ),
                        )
                        experiment_result = ExperimentResult(
                            dataset=dataset,
                            sample_id=sample_id,
                            celltype=celltype,
                            preprocessing_transformation=preprocessing_transformation,
                            aggregation_type=aggregation_type,
                            scoring_axis=scoring_axis,
                            statistical_measure=statistical_measure,
                            value=value,
                        )
                        yield experiment_result


def run_targeted_experiment(
    predictor: Predictor | n2l.pd.PredictorProtocol,
    grouped_experiment_configs: Mapping[str, list[ExperimentConfig]],
) -> Generator[tuple[str, ExperimentResult]]:
    for name, experiment_configs in grouped_experiment_configs.items():
        for experiment_result in run_experiment(predictor, experiment_configs):
            yield name, experiment_result


def main():
    import os

    from dotenv import load_dotenv

    load_dotenv()
    query_h5ad_data_intestine: H5adData = (  # type: ignore
        os.getenv("QUERY_H5AD"),
        os.getenv("QUERY_CLUSTER_KEY"),
    )
    reference_h5ad_data_intestine: H5adData = (  # type: ignore
        os.getenv("REFERENCE_H5AD"),
        os.getenv("REFERENCE_CLUSTER_KEY"),
    )
    spatial_subpanel_config = SpatialSubpanel(
        sample_size=20,
        n_samples=10,
        seed=0,
    )

    pseudospatial_full_config_250 = PseudospatialFull(
        panel_size=250,
        n_samples=10,
        seed=0,
    )
    pseudospatial_full_config_500 = PseudospatialFull(
        panel_size=500,
        n_samples=10,
        seed=0,
    )
    configurations = {
        "christian": expand_configurations(
            datasets=[
                (
                    (query_h5ad_data_intestine, reference_h5ad_data_intestine),
                    spatial_subpanel_config,
                ),
            ],
            preprocessing_transformations=[
                (TransformationFunction.LOG1P, None),
                (None, None),
            ],
            scoring_configurations=[
                (
                    ScoringAxis.CELL,
                    AggregationType.MEAN_SCORE_OF_EXPRESSION,
                    StatisticalMeasure.EXPLAINED_VARIANCE_1,
                ),
            ],
        ),
        "helene": expand_configurations(
            datasets=[
                (reference_h5ad_data_intestine, pseudospatial_full_config_500),
                (reference_h5ad_data_intestine, pseudospatial_full_config_250),
            ],
            preprocessing_transformations=[
                (TransformationFunction.LOG1P, None),
                (TransformationFunction.LOG1P, TransformationFunction.EXPM1),
                (None, None),
            ],
            scoring_configurations=[
                (
                    ScoringAxis.CELL,
                    AggregationType.SCORE_OF_RAVEL,
                    StatisticalMeasure.EXPLAINED_VARIANCE_2,
                ),
            ],
        ),
    }
    with open("christian_helene_comparison.csv", "w", newline="", buffering=1) as f:
        csv_writer = csv.DictWriter(f, fieldnames=)
        for result in run_targeted_experiment(
            predictor=n2l.pd.TangramPredictor(),
            grouped_experiment_configs=configurations,
        ):
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
