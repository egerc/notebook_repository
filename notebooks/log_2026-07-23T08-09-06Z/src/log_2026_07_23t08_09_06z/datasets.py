from enum import StrEnum, auto
from typing import assert_never

import anndata as ad
import numpy as np
import scanpy as sc
from anndata.typing import AnnData  # type: ignore
from numpy import intp, number
from numpy.typing import NDArray
from pandas import DataFrame
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath, PositiveInt

from log_2026_07_23t08_09_06z.types import Err, Ok, Result

type NumericArray = NDArray[number]
type IndexArray = NDArray[intp]


class GeneOrdering(StrEnum):
    HVG = auto()


def rank_genes(adata: AnnData, gene_ordering: GeneOrdering) -> IndexArray:
    match hvg_df := sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        inplace=False,
    ):
        case DataFrame():
            return hvg_df.sort_values(
                "variances_norm", ascending=False
            ).index.to_numpy()
        case None:
            raise ValueError("hvg_df is None")
        case _:
            assert_never(hvg_df)


@dataclass(frozen=True, slots=True)
class SpatialSetup:
    sample_size: PositiveInt


@dataclass(frozen=True, slots=True)
class PseudospatialSetup:
    panel_size: PositiveInt
    sample_size: PositiveInt
    panel_ordering: GeneOrdering


@dataclass(frozen=True, slots=True)
class NonSpatialSetup:
    panel_size: PositiveInt


@dataclass(frozen=True, slots=True)
class SingleCellData:
    path: FilePath
    cluster_key: str


@dataclass(frozen=True, slots=True)
class QueryPlusReference:
    query: SingleCellData
    reference: SingleCellData


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
    | tuple[PseudospatialSetup, SingleCellData | QueryPlusReference]
    | tuple[NonSpatialSetup, SingleCellData | QueryPlusReference]
)


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
