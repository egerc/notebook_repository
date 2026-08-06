from collections.abc import Generator, Sequence
from functools import cached_property
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
from sklearn.model_selection import train_test_split

from log_2026_07_23t08_09_06z.types import (
    DownsamplingConfig,
    EitherOrBoth,
    Err,
    IndexArray,
    Ok,
    Result,
    SamplingSplit,
    bind_result,
    map_err,
    map_result,
    starmap_result,
    unwrap_result,
    zip_result,
)
from log_2026_07_23t08_09_06z.utils import (
    FilteringConfig,
    dataframe_to_json,
    filter_adata_label,
    pandas_pandera_from_json,
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


def rank_genes(
    adata: AnnData, gene_ordering: GeneOrderingConfig
) -> Result[IndexArray, ValueError]:
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
            return compute_hvgs(adata)
        case Random(seed):
            return Ok(
                np.random.default_rng(seed).choice(
                    adata.var_names, size=len(adata.var_names), replace=False
                )
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
type SamplingStrategyEither = EitherOrBoth[SamplePanel, SampleRemainderPanel]


class GeneAnnotationSchema(pa.DataFrameModel):
    """Pandera schema for gene annotation rows."""

    sample_id: NonNegativeInt
    split: str = pa.Field(isin=["train", "test"])
    gene: str


def get_gene_ids_by_sample(
    gene_annotation_df: DataFrame[GeneAnnotationSchema],
) -> Result[dict[int, dict[Literal["train", "test"], list[str]]], Exception]:
    """Group gene annotations by sample and split.

    Args:
        gene_annotation_df: Per-sample gene panel annotations.

    Returns:
        Mapping ``sample_id -> {split: [gene, ...]}``.
    """
    try:
        return Ok(
            gene_annotation_df.groupby(["sample_id", "split"])["gene"]
            .apply(list)
            .unstack(fill_value=[])  # type: ignore
            .to_dict(orient="index")  # type: ignore
        )
    except Exception as e:  # noqa
        return Err(e)


def _sample_genes(
    genes: Sequence[str],
    sampling_strategy: EitherOrBoth[SamplePanel, SampleRemainderPanel],
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
        case SamplePanel():
            return Err(NotImplementedError())
        case SampleRemainderPanel():
            return Err(NotImplementedError())
        case (
            SamplePanel(n_panel_genes),
            SampleRemainderPanel(n_panel_remainder_genes),
        ):
            if n_total < n_panel_genes + n_panel_remainder_genes:
                return Err(
                    ValueError(
                        f"Total genes count ({n_total}) is less than the sum of panel genes ({n_panel_genes}) and remainder genes ({n_panel_remainder_genes})."
                    )
                )
            return Err(NotImplementedError())
        case _:
            assert_never(sampling_strategy)


def sample_genes(
    genes: Sequence[str],
    sampling_strategy: EitherOrBoth[SamplePanel, SampleRemainderPanel],
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

    @pa.dataframe_check
    def check_split_annotation(cls, df: pd.DataFrame) -> pd.Series:
        train_labels = set(df.loc[df["split"] == "train", "annotation"])
        test_labels = set(df.loc[df["split"] == "test", "annotation"])
        mismatched_labels = train_labels.symmetric_difference(test_labels)
        return ~df["annotation"].isin(mismatched_labels)  # type: ignore


class CellLabels(pa.DataFrameModel):
    barcode: str
    annotation: str


def group_cells_by_split(
    df: DataFrame[CellAnnotationSchema],
) -> dict[Literal["train", "test"], Result[DataFrame[CellLabels], KeyError]]:
    return {  # type: ignore
        split: validate_pandas_pandera(CellLabels, sub_df[["barcode", "annotation"]])  # type: ignore
        for split, sub_df in df.groupby("split")
        if split in ("train", "test")
    }


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


# for now the fucntions only suport the case of pairing any of the setups with QueryPlusReference, the other valid types are for the future
type DatasetSetup = (
    tuple[SpatialSetup, QueryPlusReference]
    | tuple[SpatialPseudospatialSetup, QueryPlusReference]
    | tuple[PseudospatialSetup, SingleCellData | QueryPlusReference]
    | tuple[NonSpatialSetup, SingleCellData | QueryPlusReference]
)


def _stratified_split(
    adata: AnnData,
    annotation_col: str,
    test_size: float,
    seed: int,
) -> pd.Series:
    """Assign ``train``/``test`` to every cell, stratified by cell type.

    The resulting series is guaranteed (modulo tiny cell types — see below)
    to contain at least one cell of every annotation class in both splits,
    which is what ``CellAnnotationSchema.check_split_annotation`` requires.

    Args:
        adata: Cells to split. ``adata.obs[annotation_col]`` provides labels.
        annotation_col: Column in ``adata.obs`` holding cell-type annotations.
        test_size: Fraction of cells routed to the ``test`` split.
        seed: Random seed used by the underlying splitter.

    Returns:
        ``Result[pd.Series, ValueError]`` — ``pd.Series`` of ``"train"`` /
        ``"test"`` labels aligned to ``adata.obs.index``, or ``Err`` if any
        cell type has fewer than ``ceil(1 / test_size)`` cells (stratified
        splitting cannot place at least one of every class in each split
        otherwise).
    """
    annotations = adata.obs[annotation_col].astype(str).to_numpy()
    _, test_idx = train_test_split(
        np.arange(len(adata)),
        test_size=test_size,
        random_state=seed,
        stratify=annotations,
    )
    split = pd.Series("train", index=adata.obs_names)
    split.iloc[test_idx] = "test"
    return split


def split_cells(
    dataset_setup: DatasetSetup,
    filtering_config: FilteringConfig,
    downsampling_config: DownsamplingConfig | None,
) -> tuple[Result[DataFrame[CellAnnotationSchema], Exception], list[str]]:
    """Build the train/test cell annotations for ``dataset_setup``.

    Depending on the setup variant, this reads the relevant AnnData files,
    restricts cells to those with annotations shared between splits, assigns
    ``train``/``test`` labels (stratified by cell type for pseudospatial and
    non-spatial setups, by file of origin otherwise), and validates the
    resulting table. Stratification guarantees every cell type present in
    ``adata`` also appears in both splits, satisfying
    ``CellAnnotationSchema.check_split_annotation``.

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
            shared_genes = list(
                set(query_adata.var_names).intersection(adata.var_names)
            )
            celltypes: set[str] = set(adata.obs["annotation"].values) & set(
                query_adata.obs["annotation"].values
            )
            adata = unwrap_result(
                map_result(
                    bind_result(
                        Ok(adata[adata.obs["annotation"].isin(list(celltypes))].copy()),
                        lambda adata: filter_adata_label(
                            adata, "annotation", filtering_config
                        ),
                    ),
                    lambda adata: adata[:, shared_genes].copy(),
                )
            )
            adata.obs["split"] = _stratified_split(
                adata, "annotation", test_size=0.2, seed=seed
            )

        case PseudospatialSetup(panel_size, panel_ordering, seed), SingleCellData(
            adata_path, cluster_key
        ) | QueryPlusReference(_, SingleCellData(adata_path, cluster_key)):
            adata = ad.read_h5ad(adata_path)
            adata = adata[
                :, unwrap_result(rank_genes(adata, panel_ordering))[:panel_size]
            ].copy()
            adata.obs["annotation"] = adata.obs[cluster_key].values
            celltypes: set[str] = set(adata.obs["annotation"].values)
            adata = unwrap_result(
                bind_result(
                    Ok(adata[adata.obs["annotation"].isin(list(celltypes))].copy()),
                    lambda adata: filter_adata_label(
                        adata, "annotation", filtering_config
                    ),
                )
            )
            adata.obs["split"] = _stratified_split(
                adata, "annotation", test_size=0.2, seed=seed
            )

            celltypes: set[str] = set(adata.obs["annotation"].values)
        case NonSpatialSetup(seed), SingleCellData(
            adata_path, cluster_key
        ) | QueryPlusReference(_, SingleCellData(adata_path, cluster_key)):
            adata = ad.read_h5ad(adata_path)
            adata.obs["annotation"] = adata.obs[cluster_key].values
            celltypes = set(adata.obs["annotation"].values)
            adata = unwrap_result(
                bind_result(
                    Ok(adata[adata.obs["annotation"].isin(list(celltypes))].copy()),
                    lambda adata: filter_adata_label(
                        adata, "annotation", filtering_config
                    ),
                )
            )
            adata.obs["split"] = _stratified_split(
                adata, "annotation", test_size=0.2, seed=seed
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
        lambda df: map_err(
            validate_pandas_pandera(CellAnnotationSchema, df),
            lambda error: ValueError(
                f"Failed to verify cell annotation schema for {dataset_setup=}; reason: {error}"
            ),
        ),
    )

    genes: list[str] = adata.var_names.tolist()  # type: ignore
    return annotation_result, genes


def get_dataset_filtered(
    single_cell_data: SingleCellData,
    cell_barcodes: Sequence[str],
    gene_ids: Sequence[str],
) -> Result[AnnData, Exception | ValueError]:
    """Load a dense count matrix for ``cell_barcodes`` x ``gene_ids``.

    Args:
        single_cell_data: Reference to the AnnData file to load from.
        cell_barcodes: Cell barcodes (rows) to subset.
        gene_ids: Gene identifiers (columns) to subset.

    Returns:
        ``Result[NumericArray, Exception | ValueError]``.
    """

    def stream_adata(
        adata: AnnData,
    ) -> Result[AnnData, AttributeError | TypeError | ValueError]:
        if not set(gene_ids).issubset(adata.var_names):
            return Err(ValueError("gene_ids need to be a subset of adata.var_names"))
        elif not set(cell_barcodes).issubset(adata.obs_names):
            return Err(
                ValueError("cell_barcodes need to be a subset of adata.obs_names")
            )
        else:
            return Ok(adata[cell_barcodes, gene_ids])

    match single_cell_data:
        case SingleCellData(adata_path):
            return bind_result(
                read_h5ad(adata_path, backed="r"),
                stream_adata,
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


def join_table_adata[S: pa.DataFrameModel](
    adata: AnnData, table: DataFrame[S], key: Literal["obs", "var"]
) -> DataFrame[S]: ...


type SetupStrategy = (
    SpatialSetup | SpatialPseudospatialSetup | PseudospatialSetup | NonSpatialSetup
)


@dataclass(frozen=True)
class DatasetConfiguration:
    """Materialized dataset setup with serialized cell and gene annotations.

    Attributes:
        dataset: Query/reference pair backing the configuration.
        sampling_strategy: Strategy used to build the gene panels.
        setup_strategy: Discriminator describing the dataset layout.
        _cell_annotation_data: JSON-serialized per-cell annotations.
        _gene_annotation_data: JSON-serialized per-sample gene annotations.
    """

    dataset: QueryPlusReference
    sampling_strategy: SamplingStrategy
    setup_strategy: SetupStrategy
    _cell_annotation_data: str
    _gene_annotation_data: str

    def __str__(self) -> str:
        return f"DatasetConfiguration(dataset={self.dataset}, setup_strategy={self.setup_strategy}, sampling_strategy={self.sampling_strategy})"

    @classmethod
    def try_from_setup(
        cls,
        dataset: QueryPlusReference,
        setup_strategy: SetupStrategy,
        sampling_strategy: SamplingStrategy,
        n_samples: int,
        seed: NonNegativeInt,
        filtering_config: FilteringConfig,
        downsampling_config: DownsamplingConfig | None,
    ) -> Result["DatasetConfiguration", Exception]:
        dataset_setup: tuple[SetupStrategy, QueryPlusReference] = (
            setup_strategy,
            dataset,
        )  # type: ignore
        cells_df_result: Result[DataFrame[CellAnnotationSchema], Exception]
        cells_df_result, genes = split_cells(
            dataset_setup,  # type: ignore
            filtering_config,
            downsampling_config,
        )
        return starmap_result(
            zip_result(
                bind_result(cells_df_result, dataframe_to_json),
                bind_result(
                    sample_genes(
                        genes, sampling_strategy, n_samples, np.random.default_rng(seed)
                    ),
                    dataframe_to_json,
                ),
            ),
            lambda _cell_annotation_data, _gene_annotation_data: DatasetConfiguration(
                dataset=dataset,
                setup_strategy=setup_strategy,
                sampling_strategy=sampling_strategy,
                _cell_annotation_data=_cell_annotation_data,
                _gene_annotation_data=_gene_annotation_data,
            ),
        )

    @cached_property
    def cell_annotation_df(self) -> Result[DataFrame[CellAnnotationSchema], Exception]:
        """Parsed and schema-validated per-cell annotations.

        Returns:
            ``DataFrame[CellAnnotationSchema]``.

        Raises:
            ValueError: If JSON parsing or schema validation fails.
        """
        return pandas_pandera_from_json(
            CellAnnotationSchema, self._cell_annotation_data
        )

    @cached_property
    def gene_annotation_df(self) -> Result[DataFrame[GeneAnnotationSchema], Exception]:
        """Parsed and schema-validated per-sample gene annotations.

        Returns:
            ``DataFrame[GeneAnnotationSchema]``.

        Raises:
            ValueError: If JSON parsing or schema validation fails.
        """
        return pandas_pandera_from_json(
            GeneAnnotationSchema, self._gene_annotation_data
        )


def get_counts_per_cell_split(
    setup_strategy: SetupStrategy,
    query_plus_reference: QueryPlusReference,
    barcodes: Sequence[str],
    gene_ids: Sequence[str],
    cell_split: SamplingSplit,
) -> Result[AnnData, Exception]:
    single_cell_data: SingleCellData
    match combination := (setup_strategy, cell_split):
        case SpatialSetup(), SamplingSplit.TEST:
            single_cell_data = query_plus_reference.query
        case (
            SpatialPseudospatialSetup() | PseudospatialSetup() | NonSpatialSetup(),
            _,
        ) | (SpatialSetup(), SamplingSplit.TRAIN):
            single_cell_data = query_plus_reference.reference
        case _:
            assert_never(combination)
    return get_dataset_filtered(single_cell_data, barcodes, gene_ids)


def get_split(sampling_split: SamplingSplit) -> Literal["train", "test"]:
    if sampling_split == SamplingSplit.TRAIN:
        return "train"
    return "test"
