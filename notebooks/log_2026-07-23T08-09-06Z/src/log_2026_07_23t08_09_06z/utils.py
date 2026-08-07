import io
import random
from collections.abc import Callable, Generator, Sequence
from functools import partial
from itertools import chain, groupby
from pathlib import Path
from typing import Literal, assert_never

import anndata as ad
import numpy as np
import pandas as pd
import pandera.pandas as pa
from anndata._core.xarray import Dataset2D
from anndata.typing import AnnData  # type: ignore
from pandera.typing.pandas import DataFrame
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath, NonNegativeInt, PositiveInt
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from log_2026_07_23t08_09_06z.types import (
    DownsamplingConfig,
    Err,
    NumericArray,
    Ok,
    Result,
    TransformedSpace,
    bind_result,
    collect_result,
    map_result,
    maybe_from_optional,
    ok_if,
    ok_or,
    safe_apply_single,
    starbind_result,
    unwrap_result,
    zip_result,
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
            return Err(ValueError("dataframe is not a DataFrame"))
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


@dataclass(frozen=True)
class Minimum:
    value: NonNegativeInt


type FilteringConfig = Minimum


def filter_adata_label(
    adata: AnnData, obs_key: str, filtering_config: FilteringConfig
) -> Result[AnnData, ValueError]:
    min_val = filtering_config.value

    obs_res = get_adata_table(adata, "obs")
    series_res = bind_result(
        obs_res,
        lambda df: (  # type: ignore
            Ok(df[obs_key])
            if obs_key in df.columns
            else Err(ValueError(f"Column not found: {obs_key}"))
        ),
    )
    cat_series_res = bind_result(
        series_res,
        lambda s: (  # type: ignore
            Ok(s)
            if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s)
            else Err(
                ValueError(
                    f"Column '{obs_key}' must be categorical or string, got {s.dtype}"
                )
            )
        ),
    )
    indices_res = bind_result(  # type: ignore
        cat_series_res,
        lambda s: Ok(  # type: ignore
            s[  # type: ignore
                s.isin(
                    s.value_counts()
                    .loc[lambda c: c >= min_val]  # type: ignore
                    .index
                )
            ].index  # type: ignore
        ),
    )
    return bind_result(
        indices_res,
        lambda kept_indices: Ok(adata[kept_indices, :].copy()),  # type: ignore
    )


def safe_apply[**P, R](
    func: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Result[R, Exception]:
    try:
        return Ok(func(*args, **kwargs))
    except Exception as e:  # noqa
        return Err(e)


def validate_tokens[A](
    tokens_1: Sequence[A], tokens_2: Sequence[A]
) -> Result[tuple[Sequence[A], Sequence[A]], ValueError]:
    """
    Validates a couple properties:
        - there are no duplicate values in either tokens_1 or tokens_2
        - tokens_1 is shorter than tokens_2
        - the values of tokens_2 are identical in value and sequence up to the length of tokens_1
        - the values in tokens_2 beyond the length of tokens_1 are not found in tokens_1
    """

    def _unique_values(tokens: Sequence[A]) -> bool:
        return len(set(tokens)) == len(tokens)

    return bind_result(
        bind_result(
            bind_result(
                zip_result(
                    ok_if(
                        tokens_1,
                        _unique_values,
                        ValueError("Duplicate tokens found in tokens_1"),
                    ),
                    ok_if(
                        tokens_2,
                        _unique_values,
                        ValueError("Duplicate tokens found in tokens_2"),
                    ),
                ),
                lambda token_tuple: ok_if(
                    token_tuple,
                    lambda token_tuple: len(token_tuple[0]) < len(token_tuple[1]),
                    ValueError("tokens_1 must be shorter than tokens_2"),
                ),
            ),
            lambda token_tuple: ok_if(
                token_tuple,
                lambda token_tuple: all(
                    t1 == t2 for t1, t2 in zip(token_tuple[0], token_tuple[1])
                ),
                ValueError(
                    "tokens_2 must be identical to tokens_1 up to the length of tokens_1"
                ),
            ),
        ),
        lambda token_tuple: ok_if(
            token_tuple,
            lambda token_tuple: (
                not set(token_tuple[1][len(token_tuple[0]) :]).intersection(
                    token_tuple[0]
                )
            ),
            ValueError(
                "No value in tokens_2 beyond length of tokens_1 is found in tokens_1"
            ),
        ),
    )


def group_barcodes_by_cluster(
    adata: AnnData, cluster_key: str
) -> Result[Generator[tuple[str, list[str]], None, None], Exception]:
    try:
        obs_df = unwrap_result(get_adata_table(adata, "obs"))
        groupby = obs_df.groupby(cluster_key)
    except Exception as e:  # noqa
        return Err(e)

    return Ok(
        (str(cluster), [str(barcode) for barcode in group.index.tolist()])
        for cluster, group in groupby
    )


def downsample_values[A](
    n: PositiveInt,
    n_samples: PositiveInt,
    seed: NonNegativeInt,
) -> Result[list[int], Exception]:
    return safe_apply_single(
        value=(range(n), n_samples, seed),
        func=lambda combo: random.Random(combo[2]).sample(combo[0], combo[1]),
    )


def slice_list_safe[A](
    values: list[A], indices: list[int]
) -> Result[list[A], Exception]:
    try:
        return Ok([values[index] for index in indices])
    except Exception as e:  # noqa
        return Err(e)


def downsample_by_group[A](
    values: list[A],
    n_samples_per_class: PositiveInt,
    seed: NonNegativeInt,
) -> Generator[Result[list[int], Exception]]:
    """Returns a generator that yields the downsampled values by each group in values."""
    sorted_enumerated = sorted(enumerate(values), key=lambda x: x[1])  # type: ignore

    for _, group in groupby(sorted_enumerated, key=lambda x: x[1]):  # type: ignore
        entries = list(group)
        indices = [index for index, _ in entries]
        yield starbind_result(
            zip_result(
                Ok(indices),
                downsample_values(len(entries), n_samples_per_class, seed),
            ),
            lambda values, idxs: slice_list_safe(values, idxs),  # type: ignore
        )


def slice_adata(
    adata: AnnData, barcodes: list[str] | None, gene_ids: list[str] | None
) -> Result[AnnData, Exception]:
    match indexers := (barcodes, gene_ids):
        case (None, None):
            return Ok(adata.copy())
        case (barcodes, None):
            return safe_apply_single(adata, lambda adata: adata[barcodes].copy())
        case (None, gene_ids):
            return safe_apply_single(adata, lambda adata: adata[:, gene_ids].copy())
        case (barcodes, gene_ids):
            return safe_apply_single(
                adata, lambda adata: adata[barcodes, gene_ids].copy()
            )
        case _:
            assert_never(indexers)


def downsample_clusters_by_split(
    adata: AnnData,
    cluster_key: str,
    split_key: str,
    downsampling_config: DownsamplingConfig,
) -> Result[AnnData, Exception]:
    """VIBECODED"""
    try:
        if cluster_key not in adata.obs:
            raise KeyError(f"'{cluster_key}' not found in adata.obs")
        if split_key not in adata.obs:
            raise KeyError(f"'{split_key}' not found in adata.obs")

        rng = np.random.default_rng(downsampling_config.seed)
        target_count = downsampling_config.value

        grouped_indices = adata.obs.groupby(  # type: ignore
            [cluster_key, split_key], observed=False
        ).indices.values()

        selected_indices = []
        for indices in grouped_indices:
            n_cells = len(indices)
            if n_cells == 0:
                continue
            if n_cells <= target_count:
                selected_indices.extend(indices)
            else:
                sampled = rng.choice(indices, size=target_count, replace=False)
                selected_indices.extend(sampled)

        selected_indices.sort()

        return Ok(adata[selected_indices].copy())
    except Exception as e:  # noqa
        return Err(e)


def _downsample_clusters_by_split(  # type: ignore
    adata: AnnData,
    cluster_key: str,
    split_key: str,
    downsampling_config: DownsamplingConfig,
) -> Result[AnnData, Exception]:
    raise NotImplementedError()

    def extract_annotation(barcodes: list[str]) -> Result[list[str], Exception]:
        try:
            obs_df = get_adata_table(adata, "obs")
            return map_result(obs_df, lambda df: df[split_key].values.tolist())
        except Exception as e:  # noqa
            return Err(e)

    _ = map_result(
        map_result(
            group_barcodes_by_cluster(adata, cluster_key),
            lambda groupby_cluster_generator: (
                bind_result(
                    bind_result(
                        extract_annotation(barcodes),
                        lambda annotation: map_result(
                            collect_result(
                                downsample_by_group(
                                    annotation,
                                    downsampling_config.value,
                                    downsampling_config.seed,
                                )
                            ),
                            lambda nested_barcode_indices: list(
                                chain.from_iterable(nested_barcode_indices)
                            ),
                        ),
                    ),
                    partial(slice_list_safe, values=barcodes),
                )
                for _, barcodes in groupby_cluster_generator
            ),
        ),
        lambda x: x,
    )


def load_adata() -> AnnData:
    return unwrap_result(
        read_h5ad(
            Path(
                "/home/gruengroup/christian/Data/mouse_intestine/intestine_scRNA.h5ad"
            ),
        )
    )


def transform_space(
    counts: NumericArray, source_space: TransformedSpace, target_space: TransformedSpace
) -> NumericArray:
    match source_space:
        case TransformedSpace.RAW:
            match target_space:
                case TransformedSpace.RAW:
                    return counts
                case TransformedSpace.LOG:
                    return np.log1p(counts)
                case _:
                    assert_never(target_space)
        case TransformedSpace.LOG:
            match target_space:
                case TransformedSpace.RAW:
                    return np.expm1(counts)
                case TransformedSpace.LOG:
                    return counts
                case _:
                    assert_never(target_space)
        case _:
            assert_never(source_space)
