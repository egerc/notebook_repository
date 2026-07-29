import os

import pytest
from dotenv import load_dotenv

from log_2026_07_23t08_09_06z.datasets import (
    QueryPlusReference,
    SingleCellData,
    validate_single_cell_data,
)
from log_2026_07_23t08_09_06z.types import unwrap_result

load_dotenv()


@pytest.fixture
def query_plus_reference() -> QueryPlusReference:
    return QueryPlusReference(
        unwrap_result(
            validate_single_cell_data(
                SingleCellData(
                    os.getenv("INTESTINE_MERFISH_PATH"),  # type: ignore
                    os.getenv("INTESTINE_MERFISH_CLUSTER_KEY"),  # type: ignore
                )
            )
        ),
        unwrap_result(
            validate_single_cell_data(
                SingleCellData(
                    os.getenv("INTESTINE_SC_PATH"),  # type: ignore
                    os.getenv("INTESTINE_SC_CLUSTER_KEY"),  # type: ignore
                )
            )
        ),
    )


@pytest.fixture
def datasets(query_plus_reference: QueryPlusReference) -> list[QueryPlusReference]:
    return [query_plus_reference]
