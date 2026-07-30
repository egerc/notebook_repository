from collections.abc import Callable
from enum import StrEnum, auto
from typing import assert_never

import anndata as ad
import numpy as np
from anndata.typing import AnnData  # type: ignore
from nico2_lib.predictors import NmfPredictor, ScviPredictor

from log_2026_07_23t08_09_06z.types import (
    Err,
    IndexArray,
    NumericArray,
    Ok,
    Result,
    bind_result,
    unwrap_result,
)
from log_2026_07_23t08_09_06z.utils import get_adata_table, get_dense_counts


class FittingScope(StrEnum):
    """Scope at which a model is fitted.

    Values:
        GLOBAL: Fit on the full reference counts.
        CELLTYPE: Fit per cell type within the reference.
    """

    GLOBAL = auto()
    CELLTYPE = auto()


type Model = NmfPredictor | ScviPredictor


class PredictionScope(StrEnum):
    """Scope at which predictions are produced.

    Values:
        GLOBAL: Predict across all cells jointly.
        CELLTYPE: Predict per cell type.
    """

    GLOBAL = auto()
    CELLTYPE = auto()


def predict_counts(
    model: Model,
    reference: AnnData,
    query: AnnData,
) -> Result[AnnData, ValueError | AttributeError | TypeError]:
    return bind_result(
        bind_result(
            bind_result(
                get_dense_counts(reference),
                lambda arr: Ok(model.fit(arr)),
            ),
            lambda model: Ok(
                model.predict(
                    unwrap_result(
                        get_dense_counts(query),
                        ValueError("Failed to extract counts from query"),
                    ),
                    np.arange(query.n_vars),
                )
            ),
        ),
        lambda model_output: Ok(
            ad.AnnData(
                X=model_output[1],
                obs=unwrap_result(
                    get_adata_table(query, "obs"),
                    ValueError("Failed to extract obs from query"),
                ),
                var=unwrap_result(
                    get_adata_table(reference, "var"),
                    ValueError("Failed to extract var from reference"),
                ),
                obsm={"X_embedding": model_output[0]},  # type: ignore
            )
        ),
    )


def generate_results(
    model: Model,
    reference: AnnData,
    query: AnnData,
    prediction_scope: PredictionScope,
) -> Result[AnnData, ValueError | NotImplementedError | AttributeError | TypeError]:
    """Load the AnnData objects backing ``dataset`` for result generation.

    Args:
        model: Model whose results will be generated.
        annotation_df: Per-cell annotations.
        dataset: Single or query+reference data sources to load.
        sample_df: Per-sample gene annotations.

    Raises:
        ValueError: If reading any backing ``.h5ad`` file fails.
    """
    match prediction_scope:
        case PredictionScope.GLOBAL:
            return predict_counts(model, reference, query)
        case PredictionScope.CELLTYPE:
            return Err(NotImplementedError("CELLTYPE scope is not implemented"))
        case _:
            assert_never(prediction_scope)


def fit_model(
    reference: AnnData, model: Model
) -> Callable[
    [NumericArray, IndexArray], Result[tuple[NumericArray, NumericArray], Exception]
]:
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

    def get_prediction_function(
        model: Model,
    ) -> Callable[
        [NumericArray, IndexArray], Result[tuple[NumericArray, NumericArray], Exception]
    ]:
        def predict(
            x: NumericArray, indexer: IndexArray
        ) -> Result[tuple[NumericArray, NumericArray], Exception]:
            try:
                return Ok(model.predict(x, indexer))
            except Exception as e:  # noqa: BLE001
                return Err(e)

        return predict

    return unwrap_result(
        bind_result(
            bind_result(
                get_dense_counts(reference),
                lambda training_counts: Ok(model.fit(training_counts)),
            ),
            lambda model: Ok(get_prediction_function(model)),
        )
    )
