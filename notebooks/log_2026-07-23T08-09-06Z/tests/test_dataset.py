import os
from collections.abc import Sequence
from typing import assert_never

import numpy as np
import pytest
from pandera.typing.pandas import DataFrame
from pydantic.types import PositiveInt

from log_2026_07_23t08_09_06z.datasets import (
    CellAnnotationSchema,
    DatasetSetup,
    GeneAnnotationSchema,
    HighlyVariableGenes,
    NonSpatialSetup,
    PseudospatialSetup,
    QueryPlusReference,
    SamplePanel,
    SampleRemainderPanel,
    SamplingStrategy,
    SingleCellData,
    SpatialPseudospatialSetup,
    SpatialSetup,
    sample_genes,
    split_cells,
    validate_single_cell_data,
)
from log_2026_07_23t08_09_06z.types import Result, unwrap_result
from log_2026_07_23t08_09_06z.utils import validate_pandas_pandera

query_plus_reference = QueryPlusReference(
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


@pytest.mark.parametrize(
    "dataset_setup",
    [
        (SpatialSetup(), query_plus_reference),
        (SpatialPseudospatialSetup(0), query_plus_reference),
        (PseudospatialSetup(500, HighlyVariableGenes(), 0), query_plus_reference),
        (NonSpatialSetup(0), query_plus_reference),
    ],
)
def test_setups(
    dataset_setup: DatasetSetup,
) -> None:
    split_cells(dataset_setup)


@pytest.mark.parametrize(
    "genes",
    [
        list(map(str, range(500))),
        list(map(str, range(30_000))),
    ],
)
@pytest.mark.parametrize(
    "sampling_strategy",
    [
        SamplePanel(125),
        SamplePanel(500),
        SampleRemainderPanel(10),
        SampleRemainderPanel(25),
    ],
)
@pytest.mark.parametrize("n_samples", [1, 5])
def test_gene_sampling(
    genes: Sequence[str],
    sampling_strategy: SamplingStrategy,
    n_samples: PositiveInt,
):
    rng = np.random.default_rng()
    n_total = len(genes)
    match sampling_strategy:
        case SamplePanel(n_genes):
            n_train = n_genes
        case SampleRemainderPanel(n_genes):
            n_train = n_total - n_genes
        case _:
            assert_never(sampling_strategy)
    n_test = n_total - n_train
    sampling_df = unwrap_result(sample_genes(genes, sampling_strategy, n_samples, rng))
    expected_total_rows = n_samples * n_total
    assert len(sampling_df) == expected_total_rows
    counts = sampling_df.groupby(["sample_id", "split"]).size()
    for sample_id in range(n_samples):
        if n_train > 0:
            assert counts.loc[(sample_id, "train")] == n_train
        if n_test > 0:
            assert counts.loc[(sample_id, "test")] == n_test
        train_genes, test_genes = (
            (
                sample_rows := unwrap_result(
                    validate_pandas_pandera(
                        GeneAnnotationSchema,
                        sampling_df[sampling_df["sample_id"] == sample_id],  # type: ignore
                    )
                )
            )[sample_rows["split"] == "train"],
            sample_rows[sample_rows["split"] == "test"],
        )
        assert train_genes.index.size == n_train
        assert test_genes.index.size == n_test
        assert set(train_genes["gene"]).isdisjoint(set(test_genes["gene"]))


@pytest.mark.skip("Not Finished")
@pytest.mark.parametrize("inputs", [(None, None, None)])
def test_dataloading(
    inputs: Sequence[
        tuple[
            DatasetSetup,
            Result[DataFrame[CellAnnotationSchema], Exception],
            Result[DataFrame[GeneAnnotationSchema], Exception],
        ]
    ],
) -> None:
    for _, _, _ in []:
        pass
