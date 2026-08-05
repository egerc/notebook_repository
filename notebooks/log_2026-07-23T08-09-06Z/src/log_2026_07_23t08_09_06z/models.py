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
    starbind_result,
    starmap_result,
    unwrap_result,
    zip_result,
)
from log_2026_07_23t08_09_06z.utils import (
    get_adata_table,
    get_dense_counts,
    validate_tokens,
)


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


def fit_model(model: Model, counts: NumericArray) -> Result[Model, Exception]:
    try:
        return Ok(model.fit(counts))
    except Exception as e:  # noqa
        return Err(e)


def model_predict(
    model: Model, counts: NumericArray, indexer: IndexArray
) -> Result[tuple[NumericArray, NumericArray], Exception]:
    try:
        return Ok(model.predict(counts, indexer))
    except Exception as e:  # noqa
        return Err(e)


def transform_model_output(
    model_output: tuple[NumericArray, NumericArray], query: AnnData, reference: AnnData
) -> Result[AnnData, Exception]:
    cell_embedding, feature_prediction = model_output
    return starmap_result(
        zip_result(get_adata_table(query, "obs"), get_adata_table(reference, "var")),
        lambda obs, var: ad.AnnData(
            X=feature_prediction,
            obs=obs,
            var=var,
            obsm={"X_embedding": cell_embedding},  # type: ignore
        ),
    )


def predict_counts(
    model: Model,
    reference: AnnData,
    query: AnnData,
) -> Result[AnnData, ValueError | AttributeError | TypeError | Exception]:
    query_var_names = query.var_names.tolist()
    reference_var_names = reference.var_names.tolist()
    unwrap_result(validate_tokens(query_var_names, reference_var_names))

    return starbind_result(
        starbind_result(
            zip_result(
                bind_result(
                    get_dense_counts(reference), lambda arr: fit_model(model, arr)
                ),
                get_dense_counts(query),
            ),
            lambda model, arr: model_predict(model, arr, np.arange(query.n_vars)),
        ),
        lambda embedding, prediction: transform_model_output(
            (embedding, prediction), query, reference
        ),
    )


def generate_results(
    model: Model,
    reference: AnnData,
    query: AnnData,
    prediction_scope: PredictionScope,
) -> Result[
    AnnData, ValueError | NotImplementedError | AttributeError | TypeError | Exception
]:
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
            return Err(
                NotImplementedError(f"{prediction_scope} is not yet implemented")
            )
        case _:
            assert_never(prediction_scope)
