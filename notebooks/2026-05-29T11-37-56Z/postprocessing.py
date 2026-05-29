# %%
from pathlib import Path

print(Path().cwd())

# %%
import socket

print(f"Current Node: {socket.gethostname()}")

# %%
import itertools
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import mlflow
import nico2_lib as n2l
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import scanpy as sc
import scipy
import seaborn as sns
from experiment import Celltype, Dataset, Model, NumericArray, Result, Sample
from numpy import number
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine, select
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# %%
exp_name = "Default"
benchmark_experiment = mlflow.get_experiment_by_name(exp_name)
if not benchmark_experiment:
    raise ValueError(f"Experiment '{exp_name}' not found.")
runs = mlflow.search_runs(experiment_ids=[benchmark_experiment.experiment_id])
if runs.empty:
    raise RuntimeError(f"No runs found in experiment '{exp_name}'.")
last_run_id = runs.sort_values("start_time", ascending=False)["run_id"].iloc[0]
logger.info(f"Using run_id: {last_run_id}")
logger.info("Downloading 'database.db'...")
database_path = mlflow.artifacts.download_artifacts(
    run_id=last_run_id, artifact_path="database.db"
)
logger.info(f"Local path: {database_path}")
sqlite_url = f"sqlite:///{database_path}"
engine = create_engine(sqlite_url, echo=False)
logger.info("Database engine initialized.")

# %%
# database_path = "/Users/egerc/Documents/Projects/notebook_repository/notebooks/2026-03-11T09-44-10Z/database.db"
# database_path = Path().cwd() / "database.db"
logger.info(f"Local path: {database_path}")
sqlite_url = f"sqlite:///{database_path}"
engine = create_engine(sqlite_url, echo=False)
logger.info("Database engine initialized.")

# %% [markdown]
# ## Embedding Feature Autocorrelation


# %%
def _create_avg_pairwise_metric_fn(
    metric_fn: Callable[[NumericArray, NumericArray], float],
) -> Callable[[NumericArray], float]:
    def pairwise_metric_fn(
        features: NumericArray,
    ) -> float:
        eps = 1e-8
        return np.array(
            [
                metric_fn(a + eps, b + eps)
                for a, b in itertools.combinations(features.T, 2)
            ]
        ).mean()

    return pairwise_metric_fn


def create_embedding_evaluator(
    correlation_func: Callable[[NumericArray, NumericArray], float],
) -> Callable[[Result], tuple[float, float, float, float]]:
    average_pairwise_correlation_func = _create_avg_pairwise_metric_fn(
        metric_fn=correlation_func
    )

    def pairwise_correlation_func(
        result: Result,
    ) -> tuple[float, float, float, float]:
        results: tuple[float, float, float, float] = tuple(
            [
                average_pairwise_correlation_func(embedding)
                for embedding in [
                    result.global_model_embedding_reference,
                    result.global_model_embedding_query,
                    result.celltype_model_embedding_reference,
                    result.celltype_model_embedding_query,
                ]
            ]
        )
        return results

    return pairwise_correlation_func


# %% [markdown]
# ## Embedding Structure


# %%
def create_embedding_structure_evaluator(
    structure_function: Callable[[NumericArray, NumericArray], float],
) -> Callable[[Result], tuple[float, float, float, float]]:
    def structure_evaluator(
        result: Result,
    ) -> tuple[float, float, float, float]:
        return (
            structure_function(
                result.celltype.reference_adjacency_matrix,
                result.global_model_embedding_reference,
            ),
            structure_function(
                result.celltype.query_adjacency_matrix,
                result.global_model_embedding_query,
            ),
            structure_function(
                result.celltype.reference_adjacency_matrix,
                result.celltype_model_embedding_reference,
            ),
            structure_function(
                result.celltype.query_adjacency_matrix,
                result.celltype_model_embedding_query,
            ),
        )

    return structure_evaluator


def compute_multivariate_geary(
    adjacency: NumericArray,
    embeddings: NumericArray,
) -> float:
    """
    A direct multivariate version of Geary's C.
    Computes Euclidean distance in embedding space weighted by graph adjacency.
    """
    n, d = embeddings.shape
    w_sum = np.sum(adjacency)
    if w_sum == 0:
        return 0.0

    # Variance normalization (denominator)
    # We sum the variance of each feature column
    variance_sum = np.sum(np.var(embeddings, axis=0))
    if variance_sum == 0:
        return 0.0

    # Calculate weighted squared Euclidean distances between all neighbors
    # Using a memory-efficient approach for large n:
    # dist(i,j)^2 = ||xi||^2 + ||xj||^2 - 2<xi, xj>
    norms = np.sum(embeddings**2, axis=1)
    dist_sq = (
        norms[:, np.newaxis]
        + norms[np.newaxis, :]
        - 2 * np.dot(embeddings, embeddings.T)
    )

    numerator = np.sum(adjacency * dist_sq)
    denominator = 2 * w_sum * variance_sum

    return float(((n - 1) * numerator) / (n * denominator))


def blabla(
    func: Callable[[NumericArray, NumericArray], float],
) -> Callable[[NumericArray, NumericArray], float]:
    def blabla2(adjacency: NumericArray, embeddings: NumericArray) -> float:
        # 1. Identify non-zero columns (features)
        # We check if the absolute sum of the column is > 0
        non_zero_mask = np.any(embeddings != 0, axis=0)
        valid_features = embeddings[:, non_zero_mask]

        # 2. Guard against the case where NO features are non-zero
        if valid_features.shape[1] == 0:
            return 0.0

        # 3. Compute only for valid latent features
        scores = [func(adjacency, feature) for feature in valid_features.T]

        return float(np.mean(scores))

    return blabla2


def manual_morans_i(
    adjacency: NumericArray,
    feature: NumericArray,
) -> float:
    """
    Standard Moran's I: I = (n/W) * (z'Wz / z'z)
    Range: [-1, 1]. > 0 is clustered, < 0 is dispersed.
    """
    n = len(feature)
    z = feature - np.mean(feature)
    sum_sq_z = np.sum(z**2)

    if sum_sq_z == 0:
        return 0.0

    w_z = adjacency @ z
    numerator = z @ w_z

    w_sum = np.sum(adjacency)
    if w_sum == 0:
        return 0.0

    return float((n / w_sum) * (numerator / sum_sq_z))


# %% [markdown]
# ## Cell wise sparsity


# %%
def gini(arr: NumericArray) -> float:
    """Compute Gini coefficient of a 1D array."""
    if np.amin(arr) < 0:
        arr -= np.amin(arr)  # Values must be non-negative
    arr = np.sort(arr)
    index = np.arange(1, arr.shape[0] + 1)
    n = arr.shape[0]
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def entropy(arr: NumericArray) -> float:
    return float(scipy.stats.entropy(arr))


def kurtosis(arr: NumericArray) -> float:
    return float(scipy.stats.kurtosis(arr))


def skew(arr: NumericArray) -> float:
    return float(scipy.stats.skew(arr))


def make_aggregate_cellwise_metric_func(
    func: Callable[[NumericArray], float],
) -> Callable[[NumericArray], float]:
    def aggregate_cellwise_metric_func(arr: NumericArray) -> float:
        return float(np.mean([func(cell) for cell in arr]))

    return aggregate_cellwise_metric_func


def create_embedding_sparsity_evaluator(
    sparsity_func: Callable[[NumericArray], float],
) -> Callable[[Result], tuple[float, float, float, float]]:
    def embedding_sparsity_evaluator(
        result: Result,
    ) -> tuple[float, float, float, float]:
        embedding_sparsity_func = make_aggregate_cellwise_metric_func(
            sparsity_func,
        )
        return (
            embedding_sparsity_func(result.global_model_embedding_reference),
            embedding_sparsity_func(result.global_model_embedding_query),
            embedding_sparsity_func(result.celltype_model_embedding_reference),
            embedding_sparsity_func(result.celltype_model_embedding_query),
        )

    return embedding_sparsity_evaluator


# %% [markdown]
# ## Coverage sparsity


# %%
def make_max_aggregate_featurewise_metric_func(
    func: Callable[[NumericArray], float],
) -> Callable[[NumericArray], float]:

    def aggregate_cellwise_metric_func(arr: NumericArray) -> float:
        return float(func(np.max(arr, axis=1)))

    return aggregate_cellwise_metric_func


def create_coverage_sparsity_evaluator(
    sparsity_func: Callable[[NumericArray], float],
) -> Callable[[Result], tuple[float, float, float, float]]:
    def coverage_sparsity_evaluator(
        result: Result,
    ) -> tuple[float, float, float, float]:
        max_agg_sparsity_func = make_max_aggregate_featurewise_metric_func(
            sparsity_func,
        )
        return (
            max_agg_sparsity_func(result.global_model_embedding_reference),
            max_agg_sparsity_func(result.global_model_embedding_query),
            max_agg_sparsity_func(result.celltype_model_embedding_reference),
            max_agg_sparsity_func(result.celltype_model_embedding_query),
        )

    return coverage_sparsity_evaluator


# %% [markdown]
# ## Post-hoc interpretability

# %%
# %%
from functools import cache, reduce

import gseapy
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

GeneProgramsDB = frozenset[tuple[str, frozenset[str]]]


@cache
def get_gene_programs_db(
    name: str,
) -> GeneProgramsDB:
    library: dict[str, list[str]] = gseapy.get_library(  # type: ignore
        name=name,
    )
    return frozenset((program, frozenset(genes)) for program, genes in library.items())


def extract_gene_list(gene_programs_db: GeneProgramsDB) -> list[str]:
    return sorted(
        list(
            reduce(
                frozenset.union,
                [gene_set for _, gene_set in gene_programs_db],
                frozenset(),
            )
        )
    )


def get_gene_program_counts(
    gene_programs_db: GeneProgramsDB,
    genes: Sequence[str],
) -> NumericArray:
    binary_matrix = [
        [1 if gene in program_genes else 0 for gene in genes]
        for _, program_genes in gene_programs_db
    ]
    return np.array(binary_matrix)


def compute_factor_gene_correlations(
    x: NumericArray,
    w: NumericArray,
    corr_func: Callable[[NumericArray, NumericArray], float],
) -> NumericArray:
    return np.array(
        [
            [
                corr_func(factor_distribution.flatten(), gene_distribution.flatten())
                for factor_distribution in w.T
            ]
            for gene_distribution in x.T
        ]
    ).T


def max_cosine_alignment_scoring(
    factor_gene_loadings: NumericArray,
    gene_program_counts: NumericArray,
) -> float:
    """Calculate the average maximum cosine similarity between factors and databases.

    Reasoning:
        Dominic scoring relies on API hits and hard boundaries (top 10 genes), which
        discards the nuances of sub-dominant gene weights. This function compares the
        entire continuous vector of factor weights against the ground-truth binary vectors
        of the gene database using cosine similarity.

        By identifying the maximum cosine similarity score for each factor against
        the database, we measure how well the continuous distribution matches the
        shape of known biological modules without losing information to arbitrary thresholds.
        It scales beautifully from 0 to 1 and requires zero network requests.
    """
    similarity_matrix = cosine_similarity(factor_gene_loadings, gene_program_counts)
    max_similarities = np.max(similarity_matrix, axis=1)
    return float(np.mean(max_similarities))


def enrichment_score_evaluator(result: Result) -> tuple[float, float, float, float]:
    enrichment_db_name = "KEGG_2026"
    gene_programs_db = get_gene_programs_db(name=enrichment_db_name)
    database_gene_names = extract_gene_list(gene_programs_db=gene_programs_db)
    dataset_gene_names = [
        gene.upper() for gene in result.sample.dataset.shared_features
    ]
    genes: list[str] = np.intersect1d(database_gene_names, dataset_gene_names).tolist()
    if not genes:
        logger.warning(
            f"No overlapping genes found for result {result.id}. Returning 0.0."
        )
        return (0.0, 0.0, 0.0, 0.0)
    gene_program_counts = get_gene_program_counts(
        gene_programs_db=gene_programs_db, genes=genes
    )

    feature_idx: NDArray[np.intp] = np.where(np.isin(dataset_gene_names, genes))[0]

    def abc(x: NumericArray, w: NumericArray) -> float:
        factor_gene_correlations = compute_factor_gene_correlations(
            x=x,
            w=w,
            corr_func=n2l.mt.pearson_metric,  # type: ignore
        )
        factor_gene_correlations = np.nan_to_num(factor_gene_correlations)
        return max_cosine_alignment_scoring(
            factor_gene_loadings=factor_gene_correlations,
            gene_program_counts=gene_program_counts,
        )

    return (
        abc(
            result.celltype.reference_counts_matrix[:, feature_idx],
            result.global_model_embedding_reference,
        ),
        abc(
            result.celltype.query_counts_matrix[:, feature_idx],
            result.global_model_embedding_query,
        ),
        abc(
            result.celltype.reference_counts_matrix[:, feature_idx],
            result.celltype_model_embedding_reference,
        ),
        abc(
            result.celltype.query_counts_matrix[:, feature_idx],
            result.celltype_model_embedding_query,
        ),
    )


# %%
def identity(x: float) -> float:
    return x


def tent(x: float) -> float:
    return -np.absolute(x)


def negative(x: float) -> float:
    return -x


# %%
METRIC_FNS: dict[
    str,
    dict[
        str,
        tuple[
            Callable[[Result], tuple[float, float, float, float]],
            Callable[[float], float],
        ],
    ],
] = {
    "embedding_autocorrelation": {
        "pearsonr": (create_embedding_evaluator(n2l.mt.pearson_metric), tent),
        "spearmanr": (
            create_embedding_evaluator(n2l.mt.spearman_metric),
            tent,
        ),
    },
    "embedding_structure": {
        "multivariate_gearys_c": (
            create_embedding_structure_evaluator(compute_multivariate_geary),
            negative,
        ),
        "morans_i": (
            create_embedding_structure_evaluator(blabla(manual_morans_i)),
            identity,
        ),
    },
    "embedding_sparsity": {
        "gini": (create_embedding_sparsity_evaluator(gini), identity),
        "entropy": (create_embedding_sparsity_evaluator(entropy), negative),
    },
    "coverage_sparsity": {
        "gini": (create_coverage_sparsity_evaluator(gini), negative),
        "entropy": (create_coverage_sparsity_evaluator(entropy), identity),
    },
    "biological_enrichment": {
        "max_cosine_alignment": (enrichment_score_evaluator, identity)
    },
}

# %%
with Session(engine) as session:
    result = session.exec(select(Result)).first()
    enr_score = enrichment_score_evaluator(result)


# %%
import logging

# Silence all SQLAlchemy engine logs
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# If you still see some noise, silence the pools too
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

# %%
from itertools import product

from joblib import Memory
from tqdm import tqdm

memory = Memory("./cache")

with Session(engine) as session:
    results = session.exec(select(Result)).all()
    rows: list[dict[str, Any]] = []
    for result in tqdm(results):
        for metric_category, function_mapping in METRIC_FNS.items():
            for function_name, (
                metric_function,
                transformation_function,
            ) in function_mapping.items():
                metrics = memory.cache(metric_function)(result)
                for value, (model_scope, dataset_split) in zip(
                    metrics, product(["global", "celltype"], ["reference", "query"])
                ):
                    rows.append(
                        {
                            "dataset_name": result.celltype.dataset.name,
                            "celltype": result.celltype.name,
                            "sample_id": result.sample.id_of_sample,
                            "model_name": result.model.name,
                            "metric_category": metric_category,
                            "function_name": function_name,
                            "model_scope": model_scope,
                            "dataset_split": dataset_split,
                            "value": value,
                            "value_transformed": transformation_function(value),
                        }
                    )
    results_df = pd.DataFrame(rows)
results_df.to_csv("benchmarking_output.csv")
