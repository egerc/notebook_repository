from collections.abc import Sequence

import pytest

from log_2026_07_23t08_09_06z.core.api import SetupStrategy, setup_datasets
from log_2026_07_23t08_09_06z.datasets import (
    QueryPlusReference,
    SamplePanel,
    SampleRemainderPanel,
    SamplingStrategy,
    SpatialSetup,
)


@pytest.mark.parametrize(
    "sampling_strategies, setup_strategies",
    [
        (
            [
                SamplePanel(100),
                SamplePanel(200),
                SampleRemainderPanel(10),
                SampleRemainderPanel(20),
                SampleRemainderPanel(50),
            ],
            [SpatialSetup()],
        )
    ],
)
def test_dataset_setup(
    datasets: Sequence[QueryPlusReference],
    sampling_strategies: Sequence[SamplingStrategy],
    setup_strategies: Sequence[SetupStrategy],
):
    setup_datasets(
        datasets=datasets,
        folder="./test",
        sampling_strategies=sampling_strategies,
        setup_strategies=setup_strategies,
        n_samples=5,
        seed=0,
    )
