from collections.abc import Iterable, Sequence
from dataclasses import replace
from enum import Enum, auto
from itertools import groupby
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
    dataset_setup: Literal["SpatialSubpanel"]
    name: str
    query_h5ad: H5adData
    reference_h5ad: H5adData
    subpanel_size: int
    n_samples: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class PseudospatialSubpanel:
    dataset_setup: Literal["PseudospatialSubpanel"]
    name: str
    h5ad: H5adData
    panel_size: int
    subpanel_size: int
    n_samples: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class PseudospatialFull:
    dataset_setup: Literal["PseudospatialFull"]
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
    name: str | None
    dataset_name: str
    fold: int
    celltype: str
    preprocessing: str
    score_in_raw: bool
    scoring_axis: Literal["CELL", "GENE"]


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
            dataset_setup,
            name,
            (query_h5ad_path, query_h5ad_cluster_key),
            (reference_h5ad_path, reference_h5ad_cluster_key),
            panel_size,
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

        case PseudospatialSubpanel(
            dataset_setup,
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
            dataset_setup,
            name,
            (h5ad_path, cluster_key),
            panel_size,
            n_samples,
            seed,
        ):
            rng = np.random.default_rng(seed)
            adata = read_h5ad(h5ad_path)
            counts = extract_dense_counts(adata)
            split_annotation: list[Literal["train", "test"]] = rng.choice(
                ["train", "test"], size=counts.shape[0], p=[0.8, 0.2]
            ).tolist()
            celltype_annotation: list[str] = adata.obs[cluster_key].tolist()

        case _:
            assert_never(dataset)
    rng = np.random.default_rng(seed)
    gene_samples: list[IndexArray] = [
        rng.choice(counts.shape[1], panel_size, replace=False) for _ in range(n_samples)
    ]
    return (
        dataset_setup,
        name,
        counts,
        split_annotation,
        celltype_annotation,
        gene_samples,
    )


def test(
    experiment_configs: Sequence[ExperimentConfig],
):
    for dataset, dataset_configs in groupby(
        experiment_configs, key=lambda x: x.dataset
    ):
        for preprocessing, preprocessing_configs in groupby(
            dataset_configs, key=lambda x: x.preprocessing
        ):
            for (
                scoring_axis,
                aggregation_type,
                statistical_measure,
            ), scoring_configs in groupby(
                preprocessing_configs,
                key=lambda x: (
                    x.scoring_axis,
                    x.aggregation_type,
                    x.statistical_measure,
                ),
            ):
                pass


def main():
    import os
    import pprint
    from dotenv import load_dotenv

    load_dotenv()
    query_h5ad: str = os.getenv("QUERY_H5AD")  # type: ignore
    query_cluster_key: str = os.getenv("QUERY_CLUSTER_KEY")  # type: ignore
    reference_h5ad: str = os.getenv("REFERENCE_H5AD")  # type: ignore
    reference_cluster_key: str = os.getenv("REFERENCE_CLUSTER_KEY")  # type: ignore
    for dataset in [
        SpatialSubpanel(
            "SpatialSubpanel",
            "test",
            (query_h5ad, query_cluster_key),
            (reference_h5ad, reference_cluster_key),
            20,
            2,
            42,
        ),
        PseudospatialSubpanel(
            "PseudospatialSubpanel",
            "test2",
            (reference_h5ad, reference_cluster_key),
            500,
            20,
            2,
            42,
        ),
        PseudospatialFull(
            "PseudospatialFull",
            "test3",
            (reference_h5ad, reference_cluster_key),
            500,
            2,
            42,
        ),
    ]:
        (
            dataset_setup,
            name,
            counts,
            split_annotation,
            celltype_annotation,
            gene_samples,
        ) = process_dataset(dataset)
        print(
            f"Processed {name} with setup {pprint.pformat(dataset)}: counts shape {counts.shape}, samples {len(gene_samples)} of length {gene_samples[0].size}, {len(split_annotation)=}, {len(celltype_annotation)=}"
        )
        breakpoint()


if __name__ == "__main__":
    main()
