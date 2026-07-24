from collections.abc import Sequence
from typing import Literal

import anndata as ad
import pandera.pandas as pa
from anndata.typing import AnnData  # type: ignore
from pandera.typing.pandas import DataFrame
from pydantic.types import FilePath, PositiveInt
from scipy.sparse import csc_matrix, csr_matrix

from log_2026_07_23t08_09_06z.types import Err, Ok, Result


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
