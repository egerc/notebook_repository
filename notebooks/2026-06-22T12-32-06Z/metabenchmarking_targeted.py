from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import replace
from enum import Enum, auto
from itertools import groupby, product
from typing import Callable, Literal, assert_never

import numpy as np
import scanpy as sc
from anndata import read_h5ad
from anndata.typing import AnnData
from nico2_lib.typing import IndexArray, NumericArray
from pydantic.dataclasses import dataclass

type H5adData = tuple[str, str]


@dataclass(frozen=True, slots=True)
class SpatialSubpanel:
    name: str
    query_h5ad: H5adData
    reference_h5ad: H5adData
    subpanel_size: int
    n_samples: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class PseudospatialSubpanel:
    name: str
    h5ad: H5adData
    panel_size: int
    subpanel_size: int
    n_samples: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class PseudospatialFull:
    name: str
    h5ad: H5adData
    panel_size: int
    n_samples: int
    seed: int | None


type Dataset = SpatialSubpanel | PseudospatialSubpanel | PseudospatialFull


@dataclass(frozen=True, slots=True)
class PreprocessingTransformation:
    transform: Callable[[NumericArray], NumericArray]
    inverse_transform: Callable[[NumericArray], NumericArray]
    invert: bool


def identity[T](input: T) -> T:
    return input


class Preprocessing(Enum):
    LOG1P = PreprocessingTransformation(
        transform=np.log1p,
        inverse_transform=np.expm1,
        invert=True,
    )
    IDENTITY = PreprocessingTransformation(
        transform=identity,
        inverse_transform=identity,
        invert=False,
    )


class ScoringAxis(Enum):
    CELL = auto()
    GENE = auto()


class AggregationType(Enum):
    MEAN_SCORE_OF_EXPRESSION = auto()
    SCORE_OF_MEAN_EXPRESSION = auto()


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


def process_dataset(dataset: Dataset):
    match dataset:
        case SpatialSubpanel(
            name,
            (query_h5ad_path, query_h5ad_cluster_key),
            (reference_h5ad_path, reference_h5ad_cluster_key),
            subpanel_size,
            n_samples,
            seed,
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
            raise Warning(
                f"make sure samples are correctly generated, eg the size of the predicted features is {subpanel_size}"
            )

        case PseudospatialSubpanel(
            name,
            (h5ad_path, cluster_key),
            n_genes_total,
            panel_size,
            n_samples,
            seed,
        ):
            rng = np.random.default_rng(seed)
            adata = read_h5ad(h5ad_path)
            hvg_df = sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=n_genes_total,
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

        case PseudospatialFull(
            name,
            (h5ad_path, cluster_key),
            panel_size,
            n_samples,
            seed,
        ):
            rng = np.random.default_rng(seed)
            adata = read_h5ad(h5ad_path)
            counts = extract_dense_counts(adata)
            n_cells, n_genes = counts.shape
            split_annotation: list[Literal["train", "test"]] = rng.choice(
                ["train", "test"], size=n_cells, p=[0.8, 0.2]
            ).tolist()
            celltype_annotation: list[str] = adata.obs[cluster_key].tolist()

        case _:
            assert_never(dataset)
    rng = np.random.default_rng(seed)
    gene_samples: list[IndexArray] = [
        rng.choice(counts.shape[1], panel_size, replace=False) for _ in range(n_samples)
    ]
    return (
        name,
        counts,
        split_annotation,
        celltype_annotation,
        gene_samples,
    )


def make_scorer(
    scoring_axis: ScoringAxis,
    aggregation_type: AggregationType,
    statistical_measure: StatisticalMeasure,
) -> Callable[[NumericArray, NumericArray], float]:
    transpose_func: Callable[[NumericArray], NumericArray]
    match scoring_axis:
        case ScoringAxis.CELL:
            transpose_func = lambda x: x
        case ScoringAxis.GENE:
            transpose_func = lambda x: x.T
        case _:
            assert_never(scoring_axis)
    raise NotImplementedError


def expand_configurations(
    datasets: Sequence[Dataset],
    preprocessing_transformations: Sequence[PreprocessingTransformation],
    scoring_axes: Sequence[ScoringAxis],
    aggregation_types: Sequence[AggregationType],
    statistical_measures: Sequence[StatisticalMeasure],
) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            dataset=dataset,
            preprocessing=preprocess,
            scoring_axis=scoring_axis,
            aggregation_type=aggregation_type,
            statistical_measure=statistical_measure,
        )
        for dataset, preprocess, scoring_axis, aggregation_type, statistical_measure in product(
            datasets,
            preprocessing_transformations,
            scoring_axes,
            aggregation_types,
            statistical_measures,
        )
    ]


def run_experiment(
    experiment_configs: Sequence[ExperimentConfig],
) -> Generator[ExperimentResult]:
    for dataset, dataset_configs in groupby(
        experiment_configs, key=lambda x: x.dataset
    ):
        (
            name,
            counts,
            split_annotation,
            celltype_annotation,
            gene_samples,
        ) = process_dataset(dataset)
        celltypes = list(set(celltype_annotation))
        for celltype, (sample_id, gene_sample), (
            preprocessing_transformation,
            preprocessing_configs,
        ) in product(
            celltypes,
            enumerate(gene_samples),
            groupby(dataset_configs, key=lambda x: x.preprocessing),
        ):
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
                yield ExperimentResult(
                    dataset=dataset,
                    sample_id=sample_id,
                    celltype=celltype,
                    preprocessing_transformation=preprocessing_transformation,
                    aggregation_type=aggregation_type,
                    scoring_axis=scoring_axis,
                    statistical_measure=statistical_measure,
                    value=0.0,
                )


def run_targeted_experiment(
    grouped_experiment_configs: Mapping[str, list[ExperimentConfig]],
) -> Generator[tuple[str, ExperimentResult]]:
    for name, experiment_configs in grouped_experiment_configs.items():
        for experiment_result in run_experiment(experiment_configs):
            yield name, experiment_result


def main():
    import os

    from dotenv import load_dotenv

    load_dotenv()
    query_h5ad: str = os.getenv("QUERY_H5AD")  # type: ignore
    query_cluster_key: str = os.getenv("QUERY_CLUSTER_KEY")  # type: ignore
    reference_h5ad: str = os.getenv("REFERENCE_H5AD")  # type: ignore
    reference_cluster_key: str = os.getenv("REFERENCE_CLUSTER_KEY")  # type: ignore
    SPATIAL_SUBPANEL_PANEL_SIZE = 20
    run_targeted_experiment(
        [
            {
                "christian": expand_configurations(
                    datasets=[
                        SpatialSubpanel(subpanel_size=SPATIAL_SUBPANEL_PANEL_SIZE)
                    ],
                    preprocessing_transformations=[
                        PreprocessingTransformation(np.log1p, np.expm1, False),
                        PreprocessingTransformation(identity, identity, False),
                    ],
                    scoring_axes=[ScoringAxis.CELL],
                    aggregation_types=[AggregationType.MEAN_SCORE_OF_EXPRESSION],
                    statistical_measures=[StatisticalMeasure.COSINE_SIM],
                ),
            },
        ]
    )


if __name__ == "__main__":
    main()
