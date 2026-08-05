from collections.abc import Callable, Generator
from itertools import product
from pathlib import Path

import joblib
from pydantic.types import FilePath

from log_2026_07_23t08_09_06z.core.api import setup_datasets
from log_2026_07_23t08_09_06z.datasets import (
    DatasetConfiguration,
    HighlyVariableGenes,
    NonSpatialSetup,
    PseudospatialSetup,
    QueryPlusReference,
    Random,
    SamplePanel,
    SampleRemainderPanel,
    SamplingStrategy,
    SetupStrategy,
    SingleCellData,
    SpatialPseudospatialSetup,
    SpatialSetup,
)
from log_2026_07_23t08_09_06z.types import Result
from log_2026_07_23t08_09_06z.utils import FilteringConfig, MinRange


def load_dataset_configurations(
    cache_dir: FilePath | None,
) -> list[Result[DatasetConfiguration, Exception]]:
    return list(
        setup_datasets(
            datasets={
                QueryPlusReference(
                    SingleCellData(
                        Path(
                            "/home/gruengroup/christian/Data/mouse_intestine/intestine_MERFISH.h5ad"
                        ),
                        "C_scanvi",
                    ),  # type: ignore
                    SingleCellData(
                        Path(
                            "/home/gruengroup/christian/Data/mouse_intestine/intestine_scRNA.h5ad"
                        ),
                        "cluster",
                    ),
                ),
            },
            sampling_strategies={
                SamplePanel(20),
                SamplePanel(50),
                SamplePanel(100),
                SamplePanel(200),
                SampleRemainderPanel(5),
                SampleRemainderPanel(10),
                SampleRemainderPanel(25),
                SampleRemainderPanel(50),
                SampleRemainderPanel(100),
            },
            setup_strategies={
                SpatialSetup(),
                SpatialPseudospatialSetup(0),
                SpatialPseudospatialSetup(1),
                PseudospatialSetup(500, HighlyVariableGenes(), 0),
                PseudospatialSetup(500, HighlyVariableGenes(), 1),
                PseudospatialSetup(250, HighlyVariableGenes(), 1),
                PseudospatialSetup(250, HighlyVariableGenes(), 0),
                PseudospatialSetup(250, Random(0), 1),
                PseudospatialSetup(500, Random(0), 0),
                NonSpatialSetup(0),
                NonSpatialSetup(1),
            },
            n_samples=5,
            seed=0,
            filtering_config=MinRange(20, 1000),
            cache_dir=cache_dir,
        )
    )


if __name__ == "__main__":
    load_dataset_configurations(cache_dir="./test_cache")
