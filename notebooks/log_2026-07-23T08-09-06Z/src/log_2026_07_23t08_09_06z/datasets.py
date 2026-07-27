from collections.abc import Sequence
from enum import StrEnum, auto
from typing import Literal, assert_never

import anndata as ad
import numpy as np
import pandas as pd
import pandera.pandas as pa
import scanpy as sc
from anndata.typing import AnnData  # type: ignore
from numpy import intp, number
from numpy.typing import NDArray
from pandera.typing.pandas import DataFrame
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath, NonNegativeInt, PositiveInt

from log_2026_07_23t08_09_06z.types import (
    Err,
    Just,
    Maybe,
    Null,
    Ok,
    Result,
    bind_maybe,
    bind_result,
    unwrap_result,
)
from log_2026_07_23t08_09_06z.utils import (
    read_h5ad,
    slice_adata_obs,
    validate_pandas_pandera,
)
    read_h5ad,
    slice_adata_obs,
    validate_pandas_pandera,
)

type NumericArray = NDArray[number]
type IndexArray = NDArray[intp]


class HighlyVariableGenes:
    pass


@dataclass(frozen=True)
class Random:
    seed: int


type GeneOrderingConfig = HighlyVariableGenes | Random


def compute_hvgs(adata: AnnData) -> Result[IndexArray, ValueError]:
    match hvg_df := sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        inplace=False,
    ):
        case pd.DataFrame():
            return Ok(
                hvg_df.sort_values("variances_norm", ascending=False).index.to_numpy()
            )
        case None:
            return Err(ValueError("hvg_df is None"))
        case _:
            assert_never(hvg_df)


def rank_genes(adata: AnnData, gene_ordering: GeneOrderingConfig) -> IndexArray:
    match gene_ordering:
        case HighlyVariableGenes():
            return unwrap_result(compute_hvgs(adata))
        case Random(seed):
            return np.random.default_rng(seed).choice(
                adata.var_names, size=len(adata.var_names), replace=False
            )
        case _:
            assert_never(gene_ordering)


@dataclass(frozen=True, slots=True)
class SamplePanel:
    value: PositiveInt


@dataclass(frozen=True, slots=True)
class SampleRemainderPanel:
    value: PositiveInt


type SamplingStrategy = SamplePanel | SampleRemainderPanel


class SamplingSchema(pa.DataFrameModel):
    sample_id: NonNegativeInt
    split: str = pa.Field(isin=["train", "test"])
    gene: str


def _sample_genes(
    genes: Sequence[str],
    sampling_strategy: SamplingStrategy,
    n_samples: PositiveInt,
    rng: np.random.Generator,
) -> Result[pd.DataFrame, ValueError]:
    n_total = len(genes)
    genes_arr = np.asarray(genes)
    match sampling_strategy:
        case SamplePanel(n_genes):
            n_train = n_genes
        case SampleRemainderPanel(n_genes):
            n_train = n_total - n_genes
        case _:
            assert_never(sampling_strategy)
    if not (0 <= n_train <= n_total):
        return Err(
            ValueError(
                f"Invalid n_genes ({sampling_strategy.value}) for total genes count ({n_total})."
            )
        )
    n_test = n_total - n_train
    tiled_genes = np.tile(genes_arr, (n_samples, 1))
    shuffled_genes = rng.permuted(tiled_genes, axis=1)
    train_genes = shuffled_genes[:, :n_train]
    test_genes = shuffled_genes[:, n_train:]
    sample_ids = np.arange(n_samples)
    train_sample_ids = np.repeat(sample_ids, n_train)
    test_sample_ids = np.repeat(sample_ids, n_test)
    df = pd.DataFrame(
        {
            "sample_id": np.concatenate([train_sample_ids, test_sample_ids]),
            "split": np.repeat(
                ["train", "test"], [n_samples * n_train, n_samples * n_test]
            ),
            "gene": np.concatenate([train_genes.ravel(), test_genes.ravel()]),
        }
    )

    return Ok(df)


def sample_genes(
    genes: Sequence[str],
    sampling_strategy: SamplingStrategy,
    n_samples: PositiveInt,
    rng: np.random.Generator,
) -> DataFrame[SamplingSchema]:
    return unwrap_result(
        bind_result(
            _sample_genes(genes, sampling_strategy, n_samples, rng),
            lambda df: validate_pandas_pandera(SamplingSchema, df),
        )
    )


@dataclass(frozen=True, slots=True)
class SpatialSetup:
    pass


@dataclass(frozen=True, slots=True)
class SpatialPseudospatialSetup:
    seed: int


@dataclass(frozen=True, slots=True)
class PseudospatialSetup:
    panel_size: PositiveInt
    panel_ordering: GeneOrderingConfig
    seed: int


@dataclass(frozen=True, slots=True)
class NonSpatialSetup:
    seed: int


@dataclass(frozen=True, slots=True)
class SingleCellData:
    path: FilePath
    cluster_key: str


@dataclass(frozen=True, slots=True)
class QueryPlusReference:
    query: SingleCellData
    reference: SingleCellData


class AnnotationSchema(pa.DataFrameModel):
    barcode: str
    split: Literal["train", "test"]
    annotation: str


def validate_single_cell_data(
    single_cell_data: SingleCellData,
) -> Result[SingleCellData, KeyError]:
    match single_cell_data:
        case SingleCellData(path, cluster_key):
            try:
                adata = ad.read_h5ad(path, backed="r")
                adata.obs[cluster_key]
                return Ok(single_cell_data)
            except KeyError as e:
                return Err(e)
        case _:
            assert_never(single_cell_data)


type DatasetSetup = (
    tuple[SpatialSetup, QueryPlusReference]
    | tuple[SpatialPseudospatialSetup, QueryPlusReference]
    | tuple[PseudospatialSetup, SingleCellData | QueryPlusReference]
    | tuple[NonSpatialSetup, SingleCellData | QueryPlusReference]
)


def _(dataset_setup: DatasetSetup) -> tuple[DataFrame[AnnotationSchema], list[str]]:
    annotation: DataFrame[AnnotationSchema]
    genes: list[str]
    match dataset_setup:
        case SpatialSetup(), QueryPlusReference(
            SingleCellData(query_path, query_cluster_key),
            SingleCellData(reference_path, reference_cluster_key),
        ):
            reference_adata = ad.read_h5ad(reference_path)
            reference_adata.obs["annotation"] = reference_adata.obs[
                reference_cluster_key
            ]
            query_adata = ad.read_h5ad(query_path)
            query_adata.obs["annotation"] = query_adata.obs[query_cluster_key]
            celltypes: set[str] = set(reference_adata.obs["annotation"].values) & set(
                query_adata.obs["annotation"].values
            )
            reference_adata = reference_adata[
                reference_adata.obs["annotation"].isin(list(celltypes))
            ]
            query_adata = query_adata[
                query_adata.obs["annotation"].isin(list(celltypes))
            ]

            adata = ad.concat(
                {
                    "train": reference_adata,
                    "test": query_adata,
                },
                label="split",
            )
        case SpatialPseudospatialSetup(seed), QueryPlusReference(
            SingleCellData(query_path, query_cluster_key),
            SingleCellData(reference_path, reference_cluster_key),
        ):
            adata = ad.read_h5ad(reference_path)
            adata.obs["annotation"] = adata.obs[reference_cluster_key]
            query_adata = ad.read_h5ad(query_path, backed="r")
            query_adata.obs["annotation"] = query_adata.obs[query_cluster_key]
            celltypes: set[str] = set(adata.obs["annotation"].values) & set(
                query_adata.obs["annotation"].values
            )
            adata = adata[adata.obs["annotation"].isin(list(celltypes))]

            adata.obs["split"] = np.random.default_rng(seed).choice(
                ["train", "test"], size=len(adata), p=[0.8, 0.2]
            )
        case PseudospatialSetup(
            panel_size, panel_ordering, seed
        ), SingleCellData(adata_path, cluster_key) | QueryPlusReference(_, SingleCellData(adata_path, cluster_key)):
            ...
        case NonSpatialSetup(seed), SingleCellData(
            adata_path, cluster_key
        ) | QueryPlusReference(_, SingleCellData(adata_path, cluster_key)):
            ...
        case _:
            assert_never(dataset_setup)

    annotation = unwrap_result(bind_result(bind_result(slice_adata_obs(adata, ["index", "annotation", "split"]), lambda df: Ok(df.rename(columns={"index": "barcode"}))), lambda df: validate_pandas_pandera(AnnotationSchema, df)),)

    genes: list[str] = adata.var_names.str.tolist(a)
    return annotation, genes


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ProcessedData:
    dataset_setup: DatasetSetup
    counts: NumericArray
    training_cells_index: IndexArray
    testing_cells_index: IndexArray
    celltypes: set[str]
    annotation: list[str]


def sample_data(
    dataset_setup: DatasetSetup,
    rng: np.random.Generator,
) -> ProcessedData:
    match dataset_setup:
        case SpatialSetup(), QueryPlusReference(
            SingleCellData(query_path, query_cluster_key),
            SingleCellData(reference_path, reference_cluster_key),
        ):
            reference_adata = ad.read_h5ad(reference_path)
            reference_adata.obs["annotation"] = reference_adata.obs[
                reference_cluster_key
            ]
            query_adata = ad.read_h5ad(query_path)
            query_adata.obs["annotation"] = query_adata.obs[query_cluster_key]
            celltypes: set[str] = set(reference_adata.obs["annotation"].values) & set(
                query_adata.obs["annotation"].values
            )
            reference_adata = reference_adata[
                reference_adata.obs["annotation"].isin(list(celltypes))
            ]
            query_adata = query_adata[
                query_adata.obs["annotation"].isin(list(celltypes))
            ]

            adata = ad.concat(
                {
                    "train": reference_adata,
                    "test": query_adata,
                },
                label="split",
            )
            training_cells_index, testing_cells_index = (
                np.where((mask := adata.obs["split"].values) == "train")[0],  # type: ignore
                np.where(mask == "test")[0],  # type: ignore
            )
            annotation = adata.obs["annotation"].values.tolist()

        case PseudospatialSetup(panel_size, _, panel_ordering), SingleCellData(
            path, cluster_key
        ) | QueryPlusReference(_, SingleCellData(path, cluster_key)):
            adata = ad.read_h5ad(path)
            adata = adata[:, rank_genes(adata, panel_ordering)[:panel_size]]
            adata.obs["annotation"] = adata.obs[cluster_key].values
            mask = rng.choice(["train", "test"], adata.n_obs, p=[0.8, 0.2])
            training_cells_index, testing_cells_index = (
                np.where(mask == "train")[0],
                np.where(mask == "test")[0],
            )
            celltypes = set(adata[training_cells_index].obs["annotation"].values) & set(
                adata[testing_cells_index].obs["annotation"].values
            )
            adata = adata[adata.obs["annotation"].isin(list(celltypes))]
            annotation: list[str] = adata.obs["annotation"].values.tolist()
        case NonSpatialSetup(_), SingleCellData(path, cluster_key) | QueryPlusReference(
            _, SingleCellData(path, cluster_key)
        ):
            adata = ad.read_h5ad(path)
            adata.obs["annotation"] = adata.obs[cluster_key].values
            mask = rng.choice(["train", "test"], adata.n_obs, p=[0.8, 0.2])
            training_cells_index, testing_cells_index = (
                np.where(mask == "train")[0],
                np.where(mask == "test")[0],
            )
            annotation: list[str] = adata.obs["annotation"].values.tolist()
            celltypes = set(annotation)

        case _:
            assert_never(dataset_setup)
    counts: NumericArray = adata.X  # type: ignore
    return ProcessedData(
        dataset_setup=dataset_setup,
        counts=counts,
        training_cells_index=training_cells_index,
        testing_cells_index=testing_cells_index,
        celltypes=celltypes,
        annotation=annotation,
    )
