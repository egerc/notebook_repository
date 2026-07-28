from collections.abc import Sequence

import numpy as np

from log_2026_07_23t08_09_06z.datasets import DatasetSetup


def preprocess(
    dataset_setups: Sequence[DatasetSetup],
    rng: np.random.Generator,
) -> None:
    for dataset_setup in dataset_setups:
        pass


def run_experiment(): ...
def postprocess(): ...
