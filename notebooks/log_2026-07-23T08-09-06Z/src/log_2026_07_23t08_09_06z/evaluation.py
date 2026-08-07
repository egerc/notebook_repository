from collections.abc import Callable, Generator
from typing import assert_never

from anndata.typing import AnnData  # type: ignore
from pydantic.dataclasses import dataclass

from log_2026_07_23t08_09_06z.types import (
    Err,
    NumericArray,
    Ok,
    Result,
    TransformedSpace,
    map_result,
    starbind_result,
    zip_result,
)
from log_2026_07_23t08_09_06z.utils import get_dense_counts, safe_apply, transform_space


def _validate_adata_alignment(
    adata_true: AnnData, adata_pred: AnnData
) -> Result[tuple[AnnData, AnnData], ValueError | Exception]:
    try:
        var_names_true, var_names_pred = adata_true.var_names, adata_pred.var_names
        if len(var_names_true) != len(var_names_pred):
            return Err(
                ValueError(
                    f"var_names length mismatch: true={len(var_names_true)} genes, pred={len(var_names_pred)} genes"
                )
            )
        if (var_names_true != var_names_pred).any():
            return Err(
                ValueError("var_names mismatch between true and predicted AnnData")
            )
        obs_names_true, obs_names_pred = adata_true.obs_names, adata_pred.obs_names
        if len(obs_names_true) != len(obs_names_pred):
            return Err(
                ValueError(
                    f"obs_names length mismatch: true={len(obs_names_true)} cells, pred={len(obs_names_pred)} cells"
                )
            )

        if (obs_names_true != obs_names_pred).any():
            return Err(
                ValueError("obs_names mismatch between true and predicted AnnData")
            )
        return Ok((adata_true, adata_pred))
    except Exception as e:  # noqa
        return Err(e)


@dataclass(frozen=True, slots=True)
class CellWise:
    func: Callable[[NumericArray, NumericArray], float]  # Applies to 1D-Arrays


@dataclass(frozen=True, slots=True)
class GeneWise:
    func: Callable[[NumericArray, NumericArray], float]  # Applies to 1D-Arrays


@dataclass(frozen=True, slots=True)
class PopulationCelltype:
    func: Callable[[NumericArray, NumericArray], float]  # Applies to 2D-Arrays


@dataclass(frozen=True, slots=True)
class PopulationDataset:
    func: Callable[[NumericArray, NumericArray], float]  # Applies to 2D-Arrays


type ScoringSetup = CellWise | GeneWise | PopulationCelltype | PopulationDataset


def _aligned_dense_counts(
    adata_true: AnnData, adata_pred: AnnData
) -> Result[tuple[NumericArray, NumericArray], Exception]:
    return starbind_result(
        _validate_adata_alignment(adata_true, adata_pred),
        lambda t, p: zip_result(get_dense_counts(t), get_dense_counts(p)),
    )


def apply_reconstruction_scoring_func_new(
    adata_true: AnnData,
    adata_pred_and_transformed_space: tuple[AnnData, TransformedSpace],
    scoring_setup: ScoringSetup,
    target_scoring_space: TransformedSpace,
) -> Result[Generator[Result[tuple[str, float], Exception]], Exception]:
    """Score ``adata_pred`` against ``adata_true`` in ``target_scoring_space``.

    The true counts (assumed to live in :data:`TransformedSpace.RAW`) and the
    predicted counts (stored in ``adata_pred_transformed_space``) are first
    brought into the common ``target_scoring_space`` via
    :func:`log_2026_07_23t08_09_06z.utils.transform_space` so every scoring arm
    operates on a shared, comparable representation. That transformation is
    computed once, outside the match arm, to avoid redundancy across arms.
    """
    generator_result: Result[Generator[Result[tuple[str, float], Exception]], Exception]
    adata_pred, adata_pred_transformed_space = adata_pred_and_transformed_space
    adata_true_transformed_space = TransformedSpace.RAW
    # Bring the aligned true/predicted counts into the common
    # ``target_scoring_space`` once, so the scoring arms below remain agnostic
    # of the source spaces the counts were stored in. True counts are assumed to
    # live in ``RAW``; predicted counts carry their own ``TransformedSpace``.
    transformed_aligned_counts: Result[
        tuple[NumericArray, NumericArray], Exception
    ] = starbind_result(
        _aligned_dense_counts(adata_true, adata_pred),
        lambda x_true, x_pred: Ok(
            (
                transform_space(
                    x_true, adata_true_transformed_space, target_scoring_space
                ),
                transform_space(
                    x_pred, adata_pred_transformed_space, target_scoring_space
                ),
            )
        ),
    )
    match scoring_setup:
        case CellWise(func):
            generator_result = starbind_result(
                transformed_aligned_counts,
                lambda x_true, x_pred: Ok(
                    map_result(
                        safe_apply(func, x_true[i, :], x_pred[i, :]),
                        lambda score, b=barcode: (b, score),
                    )
                    for i, barcode in enumerate(map(str, adata_true.obs_names))
                ),
            )
        case GeneWise(func):
            generator_result = starbind_result(
                transformed_aligned_counts,
                lambda x_true, x_pred: Ok(
                    map_result(
                        safe_apply(func, x_true[:, j], x_pred[:, j]),
                        lambda score, g=gene: (g, score),
                    )
                    for j, gene in enumerate(map(str, adata_true.var_names))
                ),
            )
        case PopulationCelltype(func):
            # Not implemented by design.
            _ = func
            generator_result = Err(NotImplementedError())
        case PopulationDataset(func):
            generator_result = starbind_result(
                transformed_aligned_counts,
                lambda x_true, x_pred: Ok(
                    map_result(
                        safe_apply(func, x_true, x_pred),
                        lambda score: ("dataset", score),
                    )
                    for _ in (None,)
                ),
            )
        case _:
            assert_never(scoring_setup)
    return generator_result


def apply_reconstruction_scoring_func(
    adata_true: AnnData,
    adata_pred: AnnData,
    func: Callable[[NumericArray, NumericArray], float],
) -> Result[float, Exception]:
    return starbind_result(
        starbind_result(
            _validate_adata_alignment(adata_true, adata_pred),
            lambda adata_true, adata_pred: zip_result(
                get_dense_counts(adata_true),
                get_dense_counts(adata_pred),
            ),
        ),
        lambda x, y: safe_apply(func, x, y),
    )
