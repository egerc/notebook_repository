from collections.abc import Generator, Sequence
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
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath, NonNegativeInt, PositiveInt

from log_2026_07_23t08_09_06z.types import (
    Err,
    Ok,
    Result,
    bind_result,
    ok_or,
    unwrap_maybe,
    unwrap_result,
)
from log_2026_07_23t08_09_06z.utils import (
    get_dense_counts,
    read_h5ad,
    slice_adata_obs,
    validate_pandas_pandera,
)

type NumericArray = NDArray[number]
type IndexArray = NDArray[intp]


@dataclass(frozen=True, slots=True)
class HighlyVariableGenes:
    pass


@dataclass(frozen=True, slots=True)
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


class GeneAnnotationSchema(pa.DataFrameModel):
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
) -> Result[DataFrame[GeneAnnotationSchema], Exception]:
    return bind_result(
        _sample_genes(genes, sampling_strategy, n_samples, rng),
        lambda df: validate_pandas_pandera(GeneAnnotationSchema, df),
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


class CellAnnotationSchema(pa.DataFrameModel):
    barcode: str
    split: str = pa.Field(isin=["train", "test"])
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


def split_cells(
    dataset_setup: DatasetSetup,
) -> tuple[Result[DataFrame[CellAnnotationSchema], Exception], list[str]]:
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
            adata = adata[adata.obs["annotation"].isin(list(celltypes))].copy()

            adata.obs["split"] = np.random.default_rng(seed).choice(
                ["train", "test"], size=len(adata), p=[0.8, 0.2]
            )
        case PseudospatialSetup(panel_size, panel_ordering, seed), SingleCellData(
            adata_path, cluster_key
        ) | QueryPlusReference(_, SingleCellData(adata_path, cluster_key)):
            adata = ad.read_h5ad(adata_path)
            adata = adata[:, rank_genes(adata, panel_ordering)[:panel_size]].copy()
            adata.obs["annotation"] = adata.obs[cluster_key].values
            adata.obs["split"] = np.random.default_rng(seed).choice(
                ["train", "test"], size=len(adata), p=[0.8, 0.2]
            )
            celltypes: set[str] = set(adata.obs["annotation"].values)
        case NonSpatialSetup(seed), SingleCellData(
            adata_path, cluster_key
        ) | QueryPlusReference(_, SingleCellData(adata_path, cluster_key)):
            adata = ad.read_h5ad(adata_path)
            adata.obs["annotation"] = adata.obs[cluster_key].values
            celltypes: set[str] = set(adata.obs["annotation"].values)
            adata.obs["split"] = np.random.default_rng(seed).choice(
                ["train", "test"], size=len(adata), p=[0.8, 0.2]
            )
        case _:
            assert_never(dataset_setup)

    annotation_result = bind_result(
        bind_result(
            bind_result(
                slice_adata_obs(adata, ["index", "annotation", "split"]),
                lambda df: Ok(df.rename(columns={"index": "barcode"})),
            ),
            lambda df: Ok(
                df.assign(
                    annotation=df["annotation"].astype(str),
                    split=df["split"].astype(str),
                )
            ),
        ),
        lambda df: validate_pandas_pandera(CellAnnotationSchema, df),
    )

    genes: list[str] = adata.var_names.tolist()
    return annotation_result, genes


def get_dataset_counts(
    single_cell_data: SingleCellData,
    cell_barcodes: Sequence[str],
    gene_ids: Sequence[str],
) -> Result[NumericArray, Exception | ValueError]:
    def get_counts(adata: AnnData) -> Result[NumericArray, ValueError]:
        if ~np.isin(gene_ids, adata.var_names):
            return Err(ValueError("gene_ids not found in adata"))
        elif ~np.isin(cell_barcodes, adata.obs_names):
            return Err(ValueError("cell_barcodes not found in adata"))
        else:
            return ok_or(
                get_dense_counts(adata[cell_barcodes, gene_ids]),
                ValueError("Counts empty"),
            )

    match single_cell_data:
        case SingleCellData(adata_path):
            counts_result = bind_result(
                read_h5ad(adata_path, backed="r"),
                get_counts,
            )
        case _:
            assert_never(single_cell_data)
    return counts_result


def resolve_single_cell_object(
    dataset_setup: SingleCellData | QueryPlusReference, split: Literal["train", "test"]
) -> SingleCellData:
    match dataset_setup, split:
        case SingleCellData(), _:
            return dataset_setup
        case QueryPlusReference(), "train":
            return dataset_setup.reference
        case QueryPlusReference(), "test":
            return dataset_setup.query
        case _:
            assert_never(dataset_setup)


def retrieve_counts(
    dataset_setup: DatasetSetup,
    cell_annotation: DataFrame[CellAnnotationSchema],
    gene_annotation: DataFrame[GeneAnnotationSchema],
) -> Generator[NumericArray]:
    raise NotImplementedError()
    dataset_format, dataset = dataset_setup
    match (dataset_format, dataset):
        case SpatialSetup(), QueryPlusReference(SingleCellData(), SingleCellData()):
            pass
        case SpatialSetup(), QueryPlusReference(SingleCellData(), SingleCellData()):
            pass
        case _:
            assert_never(dataset_setup)
