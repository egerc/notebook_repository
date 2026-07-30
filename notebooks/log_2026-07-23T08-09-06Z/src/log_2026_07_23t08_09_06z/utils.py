import io
from collections.abc import Callable, Sequence
from typing import Literal, assert_never

import anndata as ad
import numpy as np
import pandas as pd
import pandera.pandas as pa
from anndata._core.xarray import Dataset2D
from anndata.typing import AnnData  # type: ignore
from pandera.typing.pandas import DataFrame
from pydantic.types import FilePath, PositiveInt
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from log_2026_07_23t08_09_06z.types import (
    Err,
    NumericArray,
    Ok,
    Result,
    bind_result,
    maybe_from_optional,
    ok_or,
)


def read_h5ad(
    filename: FilePath,
    backed: bool | Literal["r", "r+"] | None = None,
    as_sparse: Sequence[str] = (),
    as_sparse_fmt: type[csr_matrix] | type[csc_matrix] = csr_matrix,
    chunk_size: PositiveInt = 6000,
) -> Result[AnnData, Exception]:
    """Read an AnnData file from disk.

    Args:
        filename: Path to the ``.h5ad`` file.
        backed: Backed mode passed through to ``anndata.read_h5ad``.
        as_sparse: Columns to load as sparse.
        as_sparse_fmt: Sparse format class used for ``as_sparse`` columns.
        chunk_size: Chunk size for backed iteration.

    Returns:
        ``Result[AnnData, Exception]``.
    """
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
    except Exception as e:  # noqa
        return Err(e)


def get_dense_counts(
    adata: AnnData,
) -> Result[NumericArray, AttributeError | TypeError]:
    """Return ``adata.X`` as a dense NumPy array.

    Args:
        adata: AnnData object whose ``.X`` should be densified.

    Returns:
        ``Maybe[NumericArray]``: ``Just`` for ndarray or sparse input,
        ``Nothing`` for unsupported storage types.
    """
    match counts := adata.X:
        case np.ndarray():
            return Ok(counts)
        case csr_matrix() | csc_matrix() | csr_array() | csc_array():
            return Ok(counts.toarray())
        case None:
            return Err(AttributeError(f"Empty field X for {adata}"))
        case _:
            return Err(TypeError(f"Unsupported storage type {type(counts)}"))


def dataframe_to_json(df: pd.DataFrame) -> Result[str, ValueError]:
    """Serialize a DataFrame to its JSON representation.

    Args:
        df: DataFrame to serialize.

    Returns:
        ``Result[str, ValueError]``.
    """
    return ok_or(
        maybe_from_optional(df.to_json()), ValueError("Json transformation failed")
    )


def validate_pandas_pandera[S: pa.DataFrameModel](
    schema: type[S],
    df: pd.DataFrame,
    lazy: bool = True,
) -> Result[DataFrame[S], Exception]:
    """Validate ``df`` against a Pandera ``DataFrameModel`` schema.

    Args:
        schema: Pandera schema class to validate against.
        df: DataFrame to validate.
        lazy: If ``True``, collect all validation errors before failing.

    Returns:
        ``Result[DataFrame[S], Exception]``.
    """
    try:
        validated_df = schema.validate(df, lazy=lazy)
        return Ok(validated_df)
    except Exception as e:  # noqa
        return Err(e)


def pandas_pandera_from_json[S: pa.DataFrameModel](
    schema: type[S],
    json_data: str,
) -> Result[DataFrame[S], Exception]:
    """Load JSON into a DataFrame and validate it against a Pandera schema.

    Args:
        schema: Pandera schema class to validate against.
        json_data: JSON string to parse.

    Returns:
        ``Result[DataFrame[S], Exception]``.
    """

    def _load_df() -> Result[pd.DataFrame, Exception]:
        try:
            return Ok(pd.read_json(io.StringIO(json_data)))
        except Exception as e:  # noqa
            return Err(e)

    return bind_result(_load_df(), lambda df: validate_pandas_pandera(schema, df))


def transform_pandas_pandera[S1: pa.DataFrameModel, S2: pa.DataFrameModel](
    schema: type[S2],
    df: DataFrame[S1],
    transform: Callable[[DataFrame[S1]], pd.DataFrame],
) -> Result[DataFrame[S2], Exception]:
    return bind_result(
        Ok(transform(df)),
        lambda df: validate_pandas_pandera(schema, df),
    )


def get_adata_table(
    adata: AnnData, table_name: Literal["obs", "var"]
) -> Result[pd.DataFrame, ValueError]:
    """Return ``adata.obs`` as a plain pandas DataFrame.

    Args:
        adata: AnnData object to read from.

    Returns:
        ``Result[pd.DataFrame, ValueError]``.
    """
    match df := adata.obs if table_name == "obs" else adata.var:
        case pd.DataFrame():
            return Ok(df)
        case Dataset2D():
            return Err(ValueError("adata.obs is a Dataset2D"))
        case _:
            assert_never(df)


def extract_columns(
    df: pd.DataFrame, columns: list[str | Literal["index"]]
) -> Result[pd.DataFrame, ValueError | KeyError]:
    """Select ``columns`` from ``df`` after promoting the index to a column.

    Args:
        df: Source DataFrame.
        columns: Column names to keep; ``"index"`` refers to the index.

    Returns:
        ``Result[pd.DataFrame, ValueError | KeyError]``.
    """
    try:
        sliced = df.reset_index()[columns]
        if isinstance(sliced, pd.DataFrame):
            return Ok(sliced)
        else:
            return Err(ValueError("adata.obs is not a DataFrame"))
    except KeyError as e:
        return Err(ValueError(f"Column not found: {e}"))


def slice_adata_obs(
    adata: AnnData, columns: list[str | Literal["index"]]
) -> Result[pd.DataFrame, KeyError | ValueError]:
    """Return a subset of ``adata.obs`` columns as a pandas DataFrame.

    Args:
        adata: AnnData object to read from.
        columns: Column names to keep; ``"index"`` refers to the index.

    Returns:
        ``Result[pd.DataFrame, KeyError | ValueError]``.
    """
    return bind_result(
        get_adata_table(adata, "obs"), lambda df: extract_columns(df, columns)
    )
