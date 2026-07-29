from collections.abc import Callable
from enum import StrEnum, auto
from typing import assert_never

import anndata as ad
from anndata.typing import AnnData  # type: ignore
from nico2_lib.predictors import NmfPredictor, ScviPredictor
from pandera.typing.pandas import DataFrame

from log_2026_07_23t08_09_06z.datasets import (
    CellAnnotationSchema,
    GeneAnnotationSchema,
    QueryPlusReference,
    SingleCellData,
)
from log_2026_07_23t08_09_06z.types import (
    IndexArray,
    NumericArray,
    unwrap_maybe,
    unwrap_result,
)
from log_2026_07_23t08_09_06z.utils import get_dense_counts, read_h5ad


class FittingScope(StrEnum):
    """Scope at which a model is fitted.

    Values:
        GLOBAL: Fit on the full reference counts.
        CELLTYPE: Fit per cell type within the reference.
    """
    GLOBAL = auto()
    CELLTYPE = auto()


type Model = NmfPredictor | ScviPredictor


def generate_results(
    model: Model,
    annotation_df: DataFrame[CellAnnotationSchema],
    dataset: SingleCellData | QueryPlusReference,
    sample_df: DataFrame[GeneAnnotationSchema],
) -> None:
    """Load the AnnData objects backing ``dataset`` for result generation.

    Args:
        model: Model whose results will be generated.
        annotation_df: Per-cell annotations.
        dataset: Single or query+reference data sources to load.
        sample_df: Per-sample gene annotations.

    Raises:
        ValueError: If reading any backing ``.h5ad`` file fails.
    """
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


class PredictionScope(StrEnum):
    """Scope at which predictions are produced.

    Values:
        GLOBAL: Predict across all cells jointly.
        CELLTYPE: Predict per cell type.
    """
    GLOBAL = auto()
    CELLTYPE = auto()


def fit_model(
    reference: AnnData, model: Model
) -> Callable[[NumericArray, IndexArray], AnnData]:
    """Fit ``model`` on the dense counts of ``reference`` and return a predictor.

    Args:
        reference: AnnData whose counts are used for fitting.
        model: Model to fit.

    Returns:
        A callable ``(x, indexer) -> AnnData`` that runs ``model.predict``.

    Raises:
        ValueError: If reference counts cannot be densified.
        Exception: Propagated from ``model.fit`` or ``model.predict``.
    """
    model.fit(
        unwrap_maybe(
            get_dense_counts(reference),
            ValueError("Failed to extract counts from reference during model fitting"),
        )
    )
    return lambda x, indexer: AnnData(model.predict(x, indexer))
