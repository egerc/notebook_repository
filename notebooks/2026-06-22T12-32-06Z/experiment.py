import argparse
import csv
import sys
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

import nico2_lib as n2l
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import read_h5ad
from anndata.typing import AnnData
from nico2_lib.typing import IndexArray, NumericArray
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

type ScoringFunction = Callable[[NumericArray, NumericArray], float]
type ScoringAxis = Callable[[ScoringFunction], ScoringFunction]
type ScoringAxisKeys = Literal["cell-wise", "gene-wise"]

SCORING_AXIS_REGISTRY: dict[ScoringAxisKeys, ScoringAxis] = {
    "cell-wise": lambda scoring_function: (
        lambda x_true, x_pred: scoring_function(x_true.T, x_pred.T)
    ),
    "gene-wise": lambda scoring_function: (
        lambda x_true, x_pred: scoring_function(x_true, x_pred)
    ),
}


def prepare_scoring_function(
    func: Callable[[NumericArray, NumericArray], float | NumericArray],
) -> ScoringFunction:
    def scoring_function(x_true: NumericArray, x_pred: NumericArray) -> float:
        assert x_true.shape == x_pred.shape, (
            "x_true and x_pred must have the same shape"
        )
        assert x_true.ndim == 1 and x_pred.ndim == 1, (
            "x_true and x_pred must be 1D arrays"
        )
        value = func(x_true, x_pred)
        assert isinstance(value, (float, np.floating)), (
            "scoring function must return a float"
        )
        return value

    return scoring_function


type ScoringFunctionKeys = Literal[
    "pearson",
    "spearman",
    "cosine_similarity",
    "mse",
    "explained_variance",
    "explained_variance_v2",
]

SCORING_FUNCTION_REGISTRY: dict[ScoringFunctionKeys, ScoringFunction] = {
    "pearson": prepare_scoring_function(
        n2l.mt.pearson_metric,
    ),
    "spearman": prepare_scoring_function(
        n2l.mt.spearman_metric,
    ),
    "cosine_similarity": prepare_scoring_function(
        n2l.mt.cosine_similarity_metric,
    ),
    "mse": prepare_scoring_function(
        n2l.mt.mse_metric,
    ),
    "explained_variance": prepare_scoring_function(
        n2l.mt.explained_variance_metric,
    ),
    "explained_variance_v2": prepare_scoring_function(
        n2l.mt.explained_variance_metric_v2
    ),
}


@dataclass
class InputDataset:
    celltype: str
    counts: NumericArray
    training_cells_index: IndexArray
    testing_cells_index: IndexArray
    training_genes_index: IndexArray
    testing_genes_index: IndexArray


type DatasetGenerator = Generator[InputDataset, None, None]


def _dense_counts_from_anndata(adata: AnnData) -> NumericArray:
    return adata.X.toarray() if sp.issparse(adata.X) else adata.X  # type: ignore


def make_spatial_dataset_generator(
    query: tuple[str, str],
    reference: tuple[str, str],
    n_panel_genes: int,
    seed: int | None = None,
) -> DatasetGenerator:
    rng = np.random.default_rng(seed)
    (
        (query_path, query_cluster_key),
        (reference_path, reference_cluster_key),
    ) = (
        query,
        reference,
    )
    query_anndata, reference_anndata = read_h5ad(query_path), read_h5ad(reference_path)
    shared_celltypes = np.intersect1d(
        query_anndata.obs[query_cluster_key],
        reference_anndata.obs[reference_cluster_key],
    ).tolist()
    assert len(shared_celltypes) > 0, "No shared cell types between query and reference"
    shared_genes = np.intersect1d(
        query_anndata.var_names,
        reference_anndata.var_names,
    ).tolist()
    assert len(shared_genes) > 0, "No shared genes between query and reference"
    for celltype in shared_celltypes:
        query_celltype_mask = query_anndata.obs[query_cluster_key] == celltype
        reference_celltype_mask = (
            reference_anndata.obs[reference_cluster_key] == celltype
        )
        query_counts = _dense_counts_from_anndata(
            query_anndata[query_celltype_mask, shared_genes]
        )
        reference_counts = _dense_counts_from_anndata(
            reference_anndata[reference_celltype_mask, shared_genes]
        )
        counts = np.vstack([reference_counts, query_counts])

        training_genes_index, testing_genes_index = train_test_split(
            np.arange(len(shared_genes)),
            train_size=len(shared_genes) - n_panel_genes,
            random_state=rng.integers(0, 1000),
        )
        training_cells_index, testing_cells_index = np.split(
            np.arange(counts.shape[0]), [reference_counts.shape[0]]
        )
        yield InputDataset(
            celltype=celltype,
            counts=counts,
            training_cells_index=training_cells_index,
            testing_cells_index=testing_cells_index,
            training_genes_index=training_genes_index,  # type: ignore
            testing_genes_index=testing_genes_index,  # type: ignore
        )


@dataclass
class Subpanel:
    size: int


@dataclass
class Outofpanel:
    pass


def make_pseudospatial_dataset_generator(
    adata_path: str,
    cluster_key: str,
    training_cells_fraction: float,
    panel_ranking: Literal["HVG"],
    panel_size: int,
    mode: Subpanel | Outofpanel,
    seed: int | None = None,
) -> DatasetGenerator:
    rng = np.random.default_rng(seed)
    adata = read_h5ad(adata_path)
    match panel_ranking:
        case "HVG":
            hvg_df = sc.pp.highly_variable_genes(
                adata,
                n_top_genes=panel_size,
                subset=True,
                flavor="seurat_v3",
                inplace=False,
            )
            assert hvg_df is not None, "Failed to compute highly variable genes"
            panel_genes = hvg_df[hvg_df["highly_variable"]]["index"].tolist()
        case _:
            assert_never(panel_ranking)

    celltypes = adata.obs[cluster_key].unique().tolist()
    for celltype in celltypes:
        celltype_mask = adata.obs[cluster_key] == celltype

        match mode:
            case Subpanel(size):
                counts = _dense_counts_from_anndata(adata[celltype_mask, panel_genes])
                training_genes_index, testing_genes_index = train_test_split(
                    np.arange(counts.shape[1]),
                    test_size=size,
                    random_state=rng.integers(0, 1000),
                )

            case Outofpanel():
                counts = _dense_counts_from_anndata(adata[celltype_mask, :])
                is_panel_gene = adata.var_names.isin(panel_genes)
                training_genes_index = np.where(is_panel_gene)[0]
                testing_genes_index = np.where(~is_panel_gene)[0]

            case _:
                assert_never(mode)

        training_cells_index, testing_cells_index = train_test_split(
            np.arange(counts.shape[0]),
            train_size=training_cells_fraction,
            random_state=rng.integers(0, 1000),
        )

        yield InputDataset(
            celltype=celltype,
            counts=counts,
            training_cells_index=training_cells_index,  # type: ignore
            testing_cells_index=testing_cells_index,  # type: ignore
            training_genes_index=training_genes_index,  # type: ignore
            testing_genes_index=testing_genes_index,  # type: ignore
        )


DATASET_REGISTRY = {
    "small_mouse_intestine_spatial": make_spatial_dataset_generator(
        query=(
            "/home/gruengroup/christian/Data/mouse_intestine/intestine_MERFISH.h5ad",
            "C_scanvi",
        ),
        reference=(
            "/home/gruengroup/christian/Projects/notebook_repository/notebooks/data/mouse_small_intestine_sc/mouse_small_intestine_sc.h5ad",
            "cluster",
        ),
        n_panel_genes=20,
    ),
    "small_mouse_intestine_pseudospatial_full": make_pseudospatial_dataset_generator(
        adata_path="/home/gruengroup/christian/Data/mouse_intestine/mouse_small_intestine_sc.h5ad",
        cluster_key="cluster",
        training_cells_fraction=0.8,
        panel_ranking="HVG",
        panel_size=20,
        mode=Outofpanel(),
    ),
    "small_mouse_intestine_pseudospatial_panel": make_pseudospatial_dataset_generator(
        adata_path="/home/gruengroup/christian/Data/mouse_intestine/mouse_small_intestine_sc.h5ad",
        cluster_key="cluster",
        training_cells_fraction=0.8,
        panel_ranking="HVG",
        panel_size=20,
        mode=Subpanel(20),
    ),
}


def _run_experiment(
    scoring_axis_keys: Sequence[ScoringAxisKeys],
    scoring_function_keys: Sequence[ScoringFunctionKeys],
) -> Generator[None, None, None]: ...


def run_experiment(
    scoring_axis_keys: Sequence[ScoringAxisKeys],
    scoring_function_keys: Sequence[ScoringFunctionKeys],
) -> pd.DataFrame:
    return pd.DataFrame(
        (result for result in _run_experiment(scoring_axis_keys, scoring_function_keys))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output csv"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    is_debugging = sys.gettrace() is not None
    SCORING_AXIS_KEYS: list[ScoringAxisKeys] = [
        "cell-wise",
        "gene-wise",
    ]
    SCORING_FUNCTION_KEYS: list[ScoringFunctionKeys] = [
        "pearson",
        "spearman",
        "cosine_similarity",
        "mse",
        "explained_variance",
        "explained_variance_v2",
    ]
    with open(args.output, "w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=Result.__annotations__.keys())
        if not is_debugging:
            csv_writer.writeheader()
        for result in _run_experiment(SCORING_AXIS_KEYS, SCORING_FUNCTION_KEYS):
            if not is_debugging:
                csv_writer.writerow(result.__dict__)
            else:
                print(result)


if __name__ == "__main__":
    pass
