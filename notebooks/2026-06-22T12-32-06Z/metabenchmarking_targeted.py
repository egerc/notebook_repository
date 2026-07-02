import csv
import logging
import pprint
import sys
from collections import Counter
from collections.abc import Generator, Mapping, Sequence
from enum import Enum, auto
from itertools import groupby, product
from typing import Callable, Literal, assert_never

import nico2_lib as n2l
import numpy as np
import scanpy as sc
from anndata import read_h5ad
from anndata.typing import AnnData
from nico2_lib.typing import IndexArray, NumericArray
from pydantic.dataclasses import dataclass
from tqdm import tqdm

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
    tuple[str, tuple[H5adData, H5adData], SpatialSubpanel]
    | tuple[str, H5adData, PseudospatialSubpanel]
    | tuple[str, H5adData, PseudospatialFull]
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
    modality: str
    prediction_scope: str
    predictor_name: str
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


def filter_rare_celltypes(
    celltype_annotation: list[str],
    split_annotation: list[Literal["train", "test"]],
    min_n_cells: int,
) -> list[bool]:
    "returns a mask for entries making sure both train and test split contain at least min_n_cells cells of a celltype"
    train_counts = Counter(
        cell
        for cell, split in zip(celltype_annotation, split_annotation)
        if split == "train"
    )
    test_counts = Counter(
        cell
        for cell, split in zip(celltype_annotation, split_annotation)
        if split == "test"
    )
    valid_celltypes = {
        cell
        for cell in set(celltype_annotation)
        if train_counts[cell] >= min_n_cells and test_counts[cell] >= min_n_cells
    }
    mask = [(cell in valid_celltypes) for cell in celltype_annotation]
    breakpoint()
    return mask


def process_dataset(
    dataset: Dataset,
    celltype_min_cells: int,
) -> tuple[
    NumericArray,
    list[Literal["train", "test"]],
    list[str],
    list[tuple[IndexArray, IndexArray]],
    tuple[str, str],
]:
    match dataset:
        case (
            _,
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
            modality, prediction_scope = "spatial", "subpanel"

        case (
            _,
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
            modality, prediction_scope = "pseudospatial", "subpanel"

        case (
            _,
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
            modality, prediction_scope = "pseudospatial", "full"

        case _:
            assert_never(dataset)
    rng = np.random.default_rng(seed)
    celltype_mask = filter_rare_celltypes(
        celltype_annotation,
        split_annotation,
        celltype_min_cells,
    )
    counts = counts[celltype_mask]
    celltype_annotation = [
        ct for ct, keep in zip(celltype_annotation, celltype_mask) if keep
    ]
    split_annotation = [s for s, keep in zip(split_annotation, celltype_mask) if keep]
    assert counts.shape[0] == len(split_annotation) == len(celltype_annotation), (
        f"Counts shape mismatch: {counts.shape[0]} vs {len(split_annotation)} vs {len(celltype_annotation)}"
    )
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
        (modality, prediction_scope),
    )


def mean_score_of_expression(
    statistical_measure: Callable[[NumericArray, NumericArray], float],
) -> Callable[[NumericArray, NumericArray], float]:
    def wrapped(x, y):
        scores = np.array([statistical_measure(xi, yi) for xi, yi in zip(x, y)])
        return scores.mean()

    return wrapped


def score_of_mean_expression(
    statistical_measure: Callable[[NumericArray, NumericArray], float],
) -> Callable[[NumericArray, NumericArray], float]:

    def wrapped(x, y):
        mean_x = np.mean(x, axis=0)
        mean_y = np.mean(y, axis=0)
        return statistical_measure(mean_x, mean_y)

    return wrapped


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
    celltype_min_n_cells: int,
    predictors: Sequence[tuple[str, Predictor | n2l.pd.PredictorProtocol]],
    experiment_configs: Sequence[ExperimentConfig],
) -> Generator[ExperimentResult]:
    for dataset, dataset_iterable in groupby(
        experiment_configs, key=lambda x: x.dataset
    ):
        dataset_configs = list(dataset_iterable)

        (
            counts_raw,
            split_annotation,
            celltype_annotation,
            gene_samples,
            (modality, prediction_scope),
        ) = process_dataset(dataset, celltype_min_n_cells)
        celltypes = list(set(celltype_annotation))
        for preprocessing_transformation, preprocessing_iterable in groupby(
            dataset_configs, key=lambda x: x.preprocessing
        ):
            preprocessing_configs = list(preprocessing_iterable)
            counts = apply_transformation_func(
                preprocessing_transformation[0], counts_raw.copy()
            )

            for celltype in celltypes:
                celltype_mask = np.array(celltype_annotation) == celltype
                train_mask, test_mask = (
                    (np.array(split_annotation) == "train"),
                    (np.array(split_annotation) == "test"),
                )
                for predictor_name, predictor in predictors:
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
                                modality=modality,
                                prediction_scope=prediction_scope,
                                predictor_name=predictor_name,
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
        for experiment_result in tqdm(
            run_experiment(
                0,
                [("", predictor)],
                experiment_configs,
            ),
            total=len(experiment_configs),
        ):
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

    fieldnames = [
        "dataset_name",
        "modality",
        "prediction_scope",
        "predictor_name",
        "sample_id",
        "celltype",
        "preprocessing_transformation",
        "aggregation_type",
        "scoring_axis",
        "statistical_measure",
        "value",
    ]
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Running experiments")
    experiment_configs = expand_configurations(
        datasets=[
            (
                "intestine_spatial",
                (query_h5ad_data_intestine, reference_h5ad_data_intestine),
                spatial_subpanel_config,
            ),
            (
                "intestine_pseudospatial",
                reference_h5ad_data_intestine,
                PseudospatialSubpanel(
                    panel_size=500,
                    sample_size=20,
                    n_samples=10,
                    seed=0,
                ),
            ),
            (
                "intestine_full_250",
                reference_h5ad_data_intestine,
                pseudospatial_full_config_250,
            ),
            (
                "intestine_full_500",
                reference_h5ad_data_intestine,
                pseudospatial_full_config_500,
            ),
        ],
        preprocessing_transformations=[
            (None, None),
            (TransformationFunction.LOG1P, None),
            (TransformationFunction.LOG1P, TransformationFunction.EXPM1),
        ],
        scoring_configurations=[
            (scoring_axis, aggregation_type, statistical_measure)
            for scoring_axis, aggregation_type, statistical_measure in product(
                [
                    ScoringAxis.CELL,
                    ScoringAxis.GENE,
                ],
                [
                    AggregationType.MEAN_SCORE_OF_EXPRESSION,
                    AggregationType.SCORE_OF_MEAN_EXPRESSION,
                    AggregationType.SCORE_OF_RAVEL,
                ],
                [
                    StatisticalMeasure.PEARSON,
                    StatisticalMeasure.EXPLAINED_VARIANCE_1,
                    StatisticalMeasure.EXPLAINED_VARIANCE_2,
                    StatisticalMeasure.MSE,
                ],
            )
        ],
    )
    with open("benchmarking_full_comparison.csv", "w", newline="", buffering=1) as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        logger.info(
            f"Running full benchmark experiment with configs {pprint.pformat(experiment_configs)}"
        )
        for result in run_experiment(
            celltype_min_n_cells=10,
            predictors=[
                ("tangram", n2l.pd.TangramPredictor()),
                ("nmf_3", n2l.pd.NmfPredictor(n_components=3)),
            ],
            experiment_configs=experiment_configs,
        ):
            logger.info(pprint.pformat(result))
            csv_writer.writerow(
                {
                    "dataset_name": result.dataset[0],
                    "modality": result.modality,
                    "prediction_scope": result.prediction_scope,
                    "predictor_name": result.predictor_name,
                    "sample_id": result.sample_id,
                    "celltype": result.celltype,
                    "preprocessing_transformation": "->".join(
                        t.name if t is not None else "none"
                        for t in result.preprocessing_transformation
                    ),
                    "aggregation_type": result.aggregation_type.name,
                    "scoring_axis": result.scoring_axis.name,
                    "statistical_measure": result.statistical_measure.name,
                    "value": result.value,
                }
            )

    logger.info("Done")
    sys.exit(0)


if __name__ == "__main__":
    main()
