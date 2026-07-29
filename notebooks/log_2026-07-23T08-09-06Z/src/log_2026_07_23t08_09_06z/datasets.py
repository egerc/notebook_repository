from collections.abc import Generator, Sequence
from typing import Literal, assert_never

import anndata as ad
import numpy as np
import pandas as pd
import pandera.pandas as pa
import scanpy as sc
from anndata.typing import AnnData  # type: ignore
from pandera.typing.pandas import DataFrame
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath, NonNegativeInt, PositiveInt

from log_2026_07_23t08_09_06z.types import (
    Err,
    IndexArray,
    NumericArray,
    Ok,
    Result,
    bind_result,
    ok_or,
    unwrap_result,
)
from log_2026_07_23t08_09_06z.utils import (
    get_dense_counts,
    read_h5ad,
    slice_adata_obs,
    validate_pandas_pandera,
)


@dataclass(frozen=True, slots=True)
class HighlyVariableGenes:
    """Select genes by highly-variable ranking."""


@dataclass(frozen=True, slots=True)
class Random:
    """Select genes in a random order.

    Attributes:
        seed: Random seed used to shuffle gene names.
    """

    seed: int


type GeneOrderingConfig = HighlyVariableGenes | Random


def compute_hvgs(adata: AnnData) -> Result[IndexArray, ValueError]:
    """Compute highly variable genes using the Seurat v3 method.

    Genes are returned sorted by ``variances_norm`` in descending order.

    Args:
        adata: Input AnnData object.

    Returns:
        ``Result[IndexArray, ValueError]``.
    """
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
    """Rank genes of ``adata`` according to ``gene_ordering``.

    Args:
        adata: Input AnnData object.
        gene_ordering: Strategy used to determine gene order.

    Returns:
        Array of gene indices in the chosen order.

    Raises:
        ValueError: If ``gene_ordering`` is ``HighlyVariableGenes`` and the
            HVG computation fails.
    """
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
    """Use a fixed-size panel of genes per sample.

    Attributes:
        value: Number of genes in each panel.
    """

    value: PositiveInt


@dataclass(frozen=True, slots=True)
class SampleRemainderPanel:
    """Use a panel consisting of all genes except a fixed-size remainder.

    Attributes:
        value: Number of genes excluded from each panel.
    """

    value: PositiveInt


type SamplingStrategy = SamplePanel | SampleRemainderPanel


class GeneAnnotationSchema(pa.DataFrameModel):
    """Pandera schema for gene annotation rows."""

    sample_id: NonNegativeInt
    split: str = pa.Field(isin=["train", "test"])
    gene: str


def get_gene_ids_by_sample(
    gene_annotation_df: DataFrame[GeneAnnotationSchema],
) -> dict[int, dict[Literal["train", "test"], list[str]]]:
    """Group gene annotations by sample and split.

    Args:
        gene_annotation_df: Per-sample gene panel annotations.

    Returns:
        Mapping ``sample_id -> {split: [gene, ...]}``.
    """
    return (
        gene_annotation_df.groupby(["sample_id", "split"])["gene"]
        .apply(list)
        .unstack(fill_value=[])  # type: ignore
        .to_dict(orient="index")  # type: ignore
    )


def _sample_genes(
    genes: Sequence[str],
    sampling_strategy: SamplingStrategy,
    n_samples: PositiveInt,
    rng: np.random.Generator,
) -> Result[pd.DataFrame, ValueError]:
    """Sample train/test gene panels for each sample.

    For each sample, a permutation of ``genes`` is generated and split into a
    train portion and a test portion according to ``sampling_strategy``.

    Args:
        genes: Pool of gene names to sample from.
        sampling_strategy: Determines how the train/test split is sized.
        n_samples: Number of samples to generate.
        rng: NumPy random generator used for shuffling.

    Returns:
        ``Result[pd.DataFrame, ValueError]``.
    """
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
    """Sample gene panels and validate the result against ``GeneAnnotationSchema``.

    Args:
        genes: Pool of gene names to sample from.
        sampling_strategy: Determines how the train/test split is sized.
        n_samples: Number of samples to generate.
        rng: NumPy random generator used for shuffling.

    Returns:
        ``Result[DataFrame[GeneAnnotationSchema], Exception]``.
    """
    return bind_result(
        _sample_genes(genes, sampling_strategy, n_samples, rng),
        lambda df: validate_pandas_pandera(GeneAnnotationSchema, df),
    )


@dataclass(frozen=True, slots=True)
class SpatialSetup:
    """Setup configuration for a spatial transcriptomics dataset."""


@dataclass(frozen=True, slots=True)
class SpatialPseudospatialSetup:
    """Setup combining a spatial reference with a pseudospatial query.

    Attributes:
        seed: Random seed used to assign train/test splits.
    """

    seed: int


@dataclass(frozen=True, slots=True)
class PseudospatialSetup:
    """Setup for a pseudospatial dataset derived from a single-cell source.

    Attributes:
        panel_size: Number of top-ranked genes to retain.
        panel_ordering: Strategy used to rank genes.
        seed: Random seed used to assign train/test splits.
    """

    panel_size: PositiveInt
    panel_ordering: GeneOrderingConfig
    seed: int


@dataclass(frozen=True, slots=True)
class NonSpatialSetup:
    """Setup for a non-spatial single-cell dataset.

    Attributes:
        seed: Random seed used to assign train/test splits.
    """

    seed: int


@dataclass(frozen=True, slots=True)
class SingleCellData:
    """Reference to a single AnnData file used as a data source.

    Attributes:
        path: Path to an ``.h5ad`` file.
        cluster_key: Column in ``adata.obs`` holding the cell-type annotation.
    """

    path: FilePath
    cluster_key: str


@dataclass(frozen=True, slots=True)
class QueryPlusReference:
    """Pair of single-cell datasets used as query and reference.

    Attributes:
        query: Dataset used as the query split.
        reference: Dataset used as the reference/train split.
    """

    query: SingleCellData
    reference: SingleCellData


class CellAnnotationSchema(pa.DataFrameModel):
    """Pandera schema for per-cell annotation rows."""

    barcode: str
    split: str = pa.Field(isin=["train", "test"])
    annotation: str


def get_barcodes(
    cell_annotation_df: DataFrame[CellAnnotationSchema],
) -> dict[Literal["train", "test"], list[str]]:
    """Group cell barcodes by split.

    Args:
        cell_annotation_df: Per-cell train/test annotations.

    Returns:
        Mapping ``split -> [barcode, ...]``.
    """
    return cell_annotation_df.groupby("split")["barcode"].apply(list).to_dict()


def validate_single_cell_data(
    single_cell_data: SingleCellData,
) -> Result[SingleCellData, KeyError]:
    """Validate that ``cluster_key`` exists in the referenced AnnData file.

    Args:
        single_cell_data: Reference to the AnnData file to check.

    Returns:
        ``Result[SingleCellData, KeyError]``.
    """
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
    """Build the train/test cell annotations for ``dataset_setup``.

    Depending on the setup variant, this reads the relevant AnnData files,
    restricts cells to those with annotations shared between splits, assigns
    ``train``/``test`` labels (randomly for pseudospatial and non-spatial
    setups, by file of origin otherwise), and validates the resulting table.

    Args:
        dataset_setup: Discriminated pair describing the dataset layout and
            its data sources.

    Returns:
        ``(Result[DataFrame[CellAnnotationSchema], Exception], list[str])``.
    """
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

    genes: list[str] = adata.var_names.tolist()  # type: ignore
    return annotation_result, genes


def get_dataset_counts(
    single_cell_data: SingleCellData,
    cell_barcodes: Sequence[str],
    gene_ids: Sequence[str],
) -> Result[NumericArray, Exception | ValueError]:
    """Load a dense count matrix for ``cell_barcodes`` x ``gene_ids``.

    Args:
        single_cell_data: Reference to the AnnData file to load from.
        cell_barcodes: Cell barcodes (rows) to subset.
        gene_ids: Gene identifiers (columns) to subset.

    Returns:
        ``Result[NumericArray, Exception | ValueError]``.
    """

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
            return bind_result(
                read_h5ad(adata_path, backed="r"),
                get_counts,
            )
        case _:
            assert_never(single_cell_data)


def retrieve_counts(
    dataset_setup: DatasetSetup,
    cell_annotation_df: DataFrame[CellAnnotationSchema],
    gene_annotation_df: DataFrame[GeneAnnotationSchema],
) -> Generator[tuple[NonNegativeInt, AnnData, AnnData], None, None]:
    """Yield per-sample AnnData train/test views for ``dataset_setup``.

    Args:
        dataset_setup: Discriminated pair describing the dataset layout and
            its data sources.
        cell_annotation_df: Per-cell train/test annotations.
        gene_annotation_df: Per-sample gene panel annotations.

    Yields:
        Tuples of ``(sample_id, train_adata, test_adata)`` for each sample.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError()
    match dataset_setup:
        case SpatialSetup(), QueryPlusReference(SingleCellData(), SingleCellData()):
            pass
        case SpatialPseudospatialSetup(), QueryPlusReference(
            SingleCellData(), SingleCellData()
        ):
            pass
        case PseudospatialSetup(), SingleCellData() | QueryPlusReference(
            SingleCellData(), SingleCellData()
        ):
            pass
        case NonSpatialSetup(), SingleCellData() | QueryPlusReference(
            SingleCellData(), SingleCellData()
        ):
            pass
        case _:
            assert_never(dataset_setup)
    training_cells_mask, testing_cells_mask = (
        (split := cell_annotation_df["split"].values) == "train",
        split == "test",
    )
    _, _ = (
        cell_annotation_df["barcode"][training_cells_mask],
        cell_annotation_df["barcode"][testing_cells_mask],
    )
    for (sample_id,), sample_df in gene_annotation_df.groupby(["sample_id"]):
        pass
