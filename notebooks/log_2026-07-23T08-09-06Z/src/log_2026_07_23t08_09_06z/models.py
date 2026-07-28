from enum import StrEnum, auto

import anndata as ad
from nico2_lib.predictors import NmfPredictor, ScviPredictor
from pandera.typing.pandas import DataFrame

from log_2026_07_23t08_09_06z.datasets import (
    CellAnnotationSchema,
    GeneAnnotationSchema,
    QueryPlusReference,
    SingleCellData,
)
from log_2026_07_23t08_09_06z.types import unwrap_result
from log_2026_07_23t08_09_06z.utils import read_h5ad


class FittingScope(StrEnum):
    GLOBAL = auto()
    CELLTYPE = auto()


type Model = NmfPredictor | ScviPredictor


def generate_results(
    model: Model,
    annotation_df: DataFrame[CellAnnotationSchema],
    dataset: SingleCellData | QueryPlusReference,
    sample_df: DataFrame[GeneAnnotationSchema],
) -> None:
    match dataset:
        case SingleCellData(adata_path, _):
            _ = unwrap_result(read_h5ad(adata_path))

        case QueryPlusReference(
            SingleCellData(query_path, _), SingleCellData(reference_path, _)
        ):
            _ = ad.concat(
                [
                    unwrap_result(read_h5ad(query_path)),
                    unwrap_result(read_h5ad(reference_path)),
                ]
            )


__all__ = ["NmfPredictor", "ScviPredictor"]
