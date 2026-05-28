import json
import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from anndata import read_h5ad
from anndata.typing import AnnData
from nico_wrapper.qc import MarkerSets, marker_annotation_qc
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


class AnnotationScorer(Protocol):
    def score(
        self,
        adata: AnnData,
        cluster_key: str,
        marker_sets: MarkerSets,
    ) -> dict[str, dict[str, float]]: ...


@dataclass(frozen=True, slots=True)
class MarkerAnnotationQC:
    layer: str | None = None
    use_raw: bool = False
    gene_symbols_key: str | None = None
    labels: Sequence[str] | None = None
    logfc_pseudocount: float = 1e-9
    logfc_threshold: float = 0.25
    score_ctrl_size: int = 50
    score_gene_pool: Sequence[str] | None = None
    score_n_bins: int = 25
    score_random_state: int | None = 0
    de_method: Literal["wilcoxon", "t-test", "t-test_overestim_var"] = "wilcoxon"
    de_alpha: float = 0.05
    de_logfc_threshold: float = 0.25
    de_top_n: int = 50
    include_optional_score_metrics: bool = False

    def score(
        self, adata: AnnData, cluster_key: str, marker_sets: MarkerSets
    ) -> dict[str, dict[str, float]]:
        df = marker_annotation_qc(
            adata=adata,
            markers=marker_sets,
            label_key=cluster_key,
            layer=self.layer,
            use_raw=self.use_raw,
            gene_symbols_key=self.gene_symbols_key,
            labels=self.labels,
            logfc_pseudocount=self.logfc_pseudocount,
            logfc_threshold=self.logfc_threshold,
            score_ctrl_size=self.score_ctrl_size,
            score_gene_pool=self.score_gene_pool,
            score_n_bins=self.score_n_bins,
            score_random_state=self.score_random_state,
            de_method=self.de_method,
            de_alpha=self.de_alpha,
            de_logfc_threshold=self.de_logfc_threshold,
            de_top_n=self.de_top_n,
            include_optional_score_metrics=self.include_optional_score_metrics,
        )
        return df.set_index("cell_type").to_dict("index")


SCORER_REGISTRY: dict[str, AnnotationScorer] = {
    "marker_annotation_qc": MarkerAnnotationQC(),
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
        sc.pp.log1p(adata_full)
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
                        for score_name, scorer in SCORER_REGISTRY.items():
                            score = scorer.score(
                                adata=adata_sample,
                                cluster_key=shuffled_annotation_key,
                                marker_sets=marker_genes_dict,
                            )
                            for cell_type, cell_score in score.items():
                                results.append(
                                    {
                                        "dataset": dataset.name,
                                        "sample_id": sample_id,
                                        "cluster_key": cluster_key,
                                        "marker_json": marker_json,
                                        "shuffle_probability": shuffle_probability,
                                        "score_name": score_name,
                                        "cell_type": cell_type,
                                        **cell_score,
                                    }
                                )
                                pd.DataFrame(results).to_csv(
                                    "results.csv",
                                    index=False,
                                )
    logging.info("Results saved to results.csv")


if __name__ == "__main__":
    main()
