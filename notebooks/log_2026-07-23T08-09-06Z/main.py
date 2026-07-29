import os

from dotenv import load_dotenv
from nico2_lib.predictors import NmfPredictor

from log_2026_07_23t08_09_06z.core.api import DatasetConfiguration
from log_2026_07_23t08_09_06z.datasets import (
    QueryPlusReference,
    SamplePanel,
    SingleCellData,
    SpatialSetup,
    get_gene_ids_by_sample,
    validate_single_cell_data,
)
from log_2026_07_23t08_09_06z.models import PredictionScope
from log_2026_07_23t08_09_06z.types import unwrap_result

load_dotenv()

DATASET = QueryPlusReference(
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


def main() -> None:
    dataset_configuration = DatasetConfiguration.from_setup(
        DATASET,
        setup_strategy=SpatialSetup(),
        sampling_strategy=SamplePanel(20),
        n_samples=5,
        seed=0,
    )
    model = NmfPredictor
    prediction_scope = PredictionScope.GLOBAL
    test = get_gene_ids_by_sample(dataset_configuration.gene_annotation_df)
    breakpoint()


if __name__ == "__main__":
    main()
