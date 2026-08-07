from enum import StrEnum, auto
from typing import assert_never

import anndata as ad
import numpy as np
from anndata.typing import AnnData  # type: ignore
from nico2_lib.predictors import NmfPredictor, ScviPredictor

from log_2026_07_23t08_09_06z.datasets import QueryPlusReference
from log_2026_07_23t08_09_06z.types import (
    Err,
    IndexArray,
    NumericArray,
    Ok,
    Result,
    TransformedSpace,
    bind_result,
    collect_result,
    starbind_result,
    starmap_result,
    unwrap_result,
    zip_result,
)
from log_2026_07_23t08_09_06z.utils import (
    get_adata_table,
    get_dense_counts,
    safe_apply,
    transform_space,
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


def fit_model(
    model: Model, counts: NumericArray, transformed_space: TransformedSpace
) -> Result[Model, Exception]:
    try:
        return Ok(
            model.fit(transform_space(counts, TransformedSpace.RAW, transformed_space))
        )
    except Exception as e:  # noqa
        return Err(e)


def model_predict(
    model: Model,
    counts: NumericArray,
    indexer: IndexArray,
    transformed_space: TransformedSpace,
) -> Result[tuple[NumericArray, NumericArray], Exception]:
    try:
        return Ok(
            model.predict(
                transform_space(counts, TransformedSpace.RAW, transformed_space),
                indexer,
            )
        )
    except Exception as e:  # noqa
        return Err(e)


def assign_model_output_to_anndata(
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
    transformed_space: TransformedSpace,
) -> Result[AnnData, ValueError | AttributeError | TypeError | Exception]:
    query_var_names = query.var_names.tolist()
    reference_var_names = reference.var_names.tolist()
    unwrap_result(validate_tokens(query_var_names, reference_var_names))

    return starbind_result(
        starbind_result(
            zip_result(
                bind_result(
                    get_dense_counts(reference),
                    lambda arr: fit_model(model, arr, transformed_space),
                ),
                get_dense_counts(query),
            ),
            lambda model, arr: model_predict(
                model, arr, np.arange(query.n_vars), transformed_space
            ),
        ),
        lambda embedding, prediction: assign_model_output_to_anndata(
            (embedding, prediction), query, reference
        ),
    )


def predict_counts_per_celltype(
    model: Model,
    reference: AnnData,
    query: AnnData,
    query_cluster_key: str,
    reference_cluster_key: str,
    transformed_space: TransformedSpace,
) -> Result[AnnData, ValueError | AttributeError | TypeError | Exception]:
    """
    similar to `predict_counts` but instead of predicting in one fell swoop, the results get produced iteratively per celltype and finally concatenated.
    this happens by fitting the model only on the counts in the reference belonging to the celltype of that iteration in the reference, the prediction is similarily only produced on the cells that belong to that celltype in the query
    """
    query_var_names = query.var_names.tolist()
    reference_var_names = reference.var_names.tolist()
    unwrap_result(validate_tokens(query_var_names, reference_var_names))

    reference_clusters = reference.obs[reference_cluster_key]
    query_clusters = query.obs[query_cluster_key]
    common_clusters = sorted(
        set(reference_clusters.unique()).intersection(query_clusters.unique())
    )
    if not common_clusters:
        return Err(
            ValueError(
                "No common cell types between reference and query "
                f"(reference_key='{reference_cluster_key}', "
                f"query_key='{query_cluster_key}')"
            )
        )

    def _predict_cluster(cluster: object) -> Result[AnnData, Exception]:
        ref_sub = reference[reference_clusters.to_numpy() == cluster, :]
        query_sub = query[query_clusters.to_numpy() == cluster, :]
        return predict_counts(model, ref_sub, query_sub, transformed_space)

    results = collect_result(_predict_cluster(cluster) for cluster in common_clusters)
    return bind_result(
        results,
        lambda adatas: safe_apply(ad.concat, adatas, axis=0, join="inner"),
    )


def generate_results(
    model: Model,
    reference: AnnData,
    query: AnnData,
    prediction_scope: PredictionScope,
    dataset: QueryPlusReference,
    transformed_space: TransformedSpace,
) -> Result[
    tuple[AnnData, TransformedSpace],
    ValueError | NotImplementedError | AttributeError | TypeError | Exception,
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
            return zip_result(
                predict_counts(model, reference, query, transformed_space),
                Ok(transformed_space),
            )
        case PredictionScope.CELLTYPE:
            return zip_result(
                predict_counts_per_celltype(
                    model,
                    reference,
                    query,
                    dataset.query.cluster_key,
                    dataset.reference.cluster_key,
                    transformed_space,
                ),
                Ok(transformed_space),
            )
        case _:
            assert_never(prediction_scope)
