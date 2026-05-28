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
AnnotationTarget = tuple[str, list[FilePath]]


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    h5ad: FilePath
    targets: list[AnnotationTarget]


def __post_init__(self) -> None:
    adata = read_h5ad(filename=self.h5ad, backed="r")
    columns: list[str] = adata.obs.columns.tolist()
    for cluster_key, marker_paths in self.targets:
        if cluster_key not in columns:
            raise ValueError(
                f"[{self.name}] cluster_key '{cluster_key}' not found in columns: {columns}"
            )

        obs_clusters = set(adata.obs[cluster_key].unique().tolist())
        for marker_json in marker_paths:
            with open(marker_json) as f:
                marker_data: dict[str, str] = json.load(f)

            marker_clusters = set(marker_data.keys())
            intersection = obs_clusters & marker_clusters
            if not intersection:
                high_cardinality_columns = {
                    col: len(marker_clusters & set(adata.obs[col].unique()))
                    for col in columns
                    if len(marker_clusters & set(adata.obs[col].unique())) > 0
                }

                raise ValueError(
                    f"Target '{cluster_key}' has no overlap with markers in '{marker_json}'.\n"
                    f"Suggestions based on other columns: {high_cardinality_columns}"
                )

            print(
                f"Verified: '{cluster_key}' vs '{marker_json}' ({len(intersection)} matches)."
            )


@dataclass(frozen=True, slots=True)
class Config:
    datasets: list[Dataset]
    shuffle_probabilities: list[Probability]
    n_samples: NonNegativeInt
    n_cells_per_sample: NonNegativeInt
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
        adata_full = read_h5ad(dataset.h5ad)
        for sample_id in range(config.n_samples):
            adata_sample = adata_full[
                rng.choice(adata_full.n_obs, config.n_cells_per_sample, replace=True)
            ]
            for target in dataset.targets:
                cluster_key, marker_json_files = target
                annotation = np.array(adata_sample.obs[cluster_key].values)
                for marker_json in marker_json_files:
                    with open(marker_json) as f:
                        marker_genes_dict: dict[str, list[str]] = json.load(f)
                    for shuffle_probability in config.shuffle_probabilities:
                        shuffled_annotation = shuffle_array(
                            annotation,
                            shuffle_probability,
                            rng,
                        )
                        shuffled_annotation_key = (
                            f"shuffled_{cluster_key}_{shuffle_probability}"
                        )
                        adata_sample.obs[shuffled_annotation_key] = shuffled_annotation
                        for score_name, score_func in SCORE_REGISTRY.items():
                            score = score_func(
                                adata_sample,
                                shuffled_annotation_key,
                                marker_genes_dict,
                            )
                            results.append(
                                {
                                    "dataset": dataset.name,
                                    "sample_id": sample_id,
                                    "cluster_key": cluster_key,
                                    "marker_json": marker_json,
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
