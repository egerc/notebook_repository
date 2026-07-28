from collections.abc import Sequence
from typing import Literal, assert_never

import anndata as ad
import numpy as np
import pandas as pd
import pandera.pandas as pa
from anndata.typing import AnnData
from pandera.typing.pandas import DataFrame
from pydantic.types import FilePath, PositiveInt
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from log_2026_07_23t08_09_06z.datasets import NumericArray
from log_2026_07_23t08_09_06z.types import Err, Just, Maybe, Nothing, Ok, Result


def read_h5ad(
    filename: FilePath,
    backed: bool | Literal["r", "r+"] | None = None,
    as_sparse: Sequence[str] = (),
    as_sparse_fmt: type[csr_matrix] | type[csc_matrix] = csr_matrix,
    chunk_size: PositiveInt = 6000,
) -> Result[AnnData, Exception]:
    try:
        return Ok(
            ad.read_h5ad(
                filename,
                backed=backed,
                as_sparse=as_sparse,
                as_sparse_fmt=as_sparse_fmt,
                chunk_size=chunk_size,
            )
        )
    except Exception as e:
        return Err(e)


def get_dense_counts(adata: AnnData) -> Maybe[NumericArray]:
    match counts := adata.X:
        case np.ndarray():
            return Just(counts)
        case csr_matrix() | csc_matrix() | csr_array() | csc_array():
            return Just(counts.toarray())
        case None:
            return Nothing()
        case _:
            return Nothing()


def validate_pandas_pandera[S: pa.DataFrameModel](
    schema: type[S],
    df: pd.DataFrame,
    lazy: bool = True,
) -> Result[DataFrame[S], Exception]:
    try:
        validated_df = schema.validate(df, lazy=lazy)
        return Ok(validated_df)
    except Exception as e:
        return Err(e)


def slice_adata_obs(
    adata: AnnData, columns: list[str | Literal["index"]]
) -> Result[pd.DataFrame, KeyError]:
    try:
        match df := adata.obs.reset_index()[columns].copy():
            case pd.DataFrame():
                return Ok(df)
            case pd.Series():
                return Ok(df.to_frame())
            case _:
                assert_never(df)

    except KeyError as e:
        return Err(e)
