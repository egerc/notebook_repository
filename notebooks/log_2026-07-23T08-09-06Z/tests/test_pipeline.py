
import pytest

from log_2026_07_23t08_09_06z.core.api import SetupStrategy, setup_datasets
from log_2026_07_23t08_09_06z.datasets import (
    QueryPlusReference,
    SamplePanel,
    SampleRemainderPanel,
    SamplingStrategy,
    SpatialSetup,
)
from log_2026_07_23t08_09_06z.utils import MinRange


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
    datasets: set[QueryPlusReference],
    sampling_strategies: set[SamplingStrategy],
    setup_strategies: set[SetupStrategy],
):
    setup_datasets(
        datasets=datasets,
        sampling_strategies=sampling_strategies,
        setup_strategies=setup_strategies,
        n_samples=5,
        seed=0,
        filtering_config=MinRange(20, 1000),
    )
