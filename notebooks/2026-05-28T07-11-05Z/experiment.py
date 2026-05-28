import json
import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from typing import Annotated, Any

import numpy as np
import pandas as pd
import yaml
from anndata import read_h5ad
from anndata.typing import AnnData
from pydantic import Field, FilePath, NonNegativeInt
from pydantic.dataclasses import dataclass

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    h5ad: FilePath
    cluster_key: str
    marker_json: FilePath

    def __post_init__(self) -> None:
        adata = read_h5ad(
            filename=self.h5ad,
            backed="r",
        )
        columns: list[str] = adata.obs.columns.tolist()
        if self.cluster_key not in columns:
            raise ValueError(
                f"cluster_key {self.cluster_key} not found in columns {columns}"
            )
        clusters = adata.obs[self.cluster_key].unique().tolist()
        with open(self.marker_json) as f:
            marker_data: dict[str, str] = json.load(f)
        marker_clusters = list(marker_data.keys())
        intersection_clusters = set(clusters) & set(marker_clusters)
        if not intersection_clusters:
            high_cardinality_columns = {
                column: len(
                    set(marker_clusters) & set(adata.obs[column].unique().tolist())
                )
                for column in columns
                if len(set(marker_clusters) & set(adata.obs[column].unique().tolist()))
                > 0
            }
            raise ValueError(
                f"No intersection between clusters and marker clusters, columns with high cardinality: {high_cardinality_columns}"
            )
        else:
            print(f"Intersection clusters: {intersection_clusters}")


@dataclass(frozen=True, slots=True)
class Config:
    datasets: list[Dataset]
    shuffle_probabilities: list[Probability]
    seed: NonNegativeInt


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    return parser.parse_args()


def shuffle_array(
    annotation: np.ndarray,
    shuffle_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if shuffle_probability == 0.0:
        return annotation
    shuffled_annotation = annotation.copy()
    rng.shuffle(shuffled_annotation)
    return shuffled_annotation


def dummy_score(
    adata: AnnData,
    cluster_key: str,
    marker_dict: dict[str, list[str]],
) -> float:
    return 0.0


SCORE_REGISTRY: dict[str, Callable[[AnnData, str, dict[str, list[str]]], float]] = {
    "dummy": dummy_score,
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    with open(args.config) as f:
        config = Config(**yaml.safe_load(f))
    logging.info(f"Config: {config}")
    rng = np.random.default_rng(config.seed)
    results: list[dict[str, Any]] = []
    for dataset in config.datasets:
        logging.info(f"Dataset: {dataset.name}")
        adata = read_h5ad(dataset.h5ad)
        annotation = np.array(adata.obs[dataset.cluster_key].values)
        with open(dataset.marker_json) as f:
            marker_genes_dict: dict[str, list[str]] = json.load(f)
        for shuffle_probability in config.shuffle_probabilities:
            shuffled_annotation = shuffle_array(annotation, shuffle_probability, rng)  # type: ignore
            shuffled_annotation_key = (
                f"shuffled_{dataset.cluster_key}_{shuffle_probability}"
            )
            adata.obs[shuffled_annotation_key] = shuffled_annotation
            for score_name, score_func in SCORE_REGISTRY.items():
                score = score_func(
                    adata,
                    shuffled_annotation_key,
                    marker_genes_dict,
                )
                results.append(
                    {
                        "dataset": dataset.name,
                        "cluster_key": dataset.cluster_key,
                        "shuffle_probability": shuffle_probability,
                        "score_name": score_name,
                        "score": score,
                    }
                )
                pd.DataFrame(results).to_csv(
                    "results.csv",
                    index=False,
                )
    logging.info("Results saved to results.csv")


if __name__ == "__main__":
    main()
