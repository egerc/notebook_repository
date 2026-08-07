from collections.abc import Callable, Generator
from typing import assert_never

from anndata.typing import AnnData  # type: ignore
from pydantic.dataclasses import dataclass

from log_2026_07_23t08_09_06z.types import (
    Err,
    NumericArray,
    Ok,
    Result,
    starbind_result,
    zip_result,
)
from log_2026_07_23t08_09_06z.utils import get_dense_counts, safe_apply


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


def apply_reconstruction_scoring_func_new(
    adata_true: AnnData,
    adata_pred: AnnData,
    scoring_setup: ScoringSetup,
) -> Result[Generator[Result[tuple[str, float], Exception]], Exception]:
    generator_result: Result[Generator[Result[tuple[str, float], Exception]], Exception]
    match scoring_setup:
        case CellWise(func):
            generator_result = Err(NotImplementedError())
        case GeneWise(func):
            generator_result = Err(NotImplementedError())
        case PopulationCelltype(func):
            generator_result = Err(NotImplementedError())
        case PopulationDataset(func):
            generator_result = Err(NotImplementedError())
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
