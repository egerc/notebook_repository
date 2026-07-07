import itertools
import logging
from collections.abc import Callable, Sequence
from functools import cache, reduce

import gseapy
import mlflow
import nico2_lib as n2l
import numpy as np
import scipy
from nico2_lib.typing import NumericArray
from numpy.typing import NDArray
from pandas.io.parsers.python_parser import csv
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sqlmodel import Session, create_engine, select
from tqdm import tqdm

from experiment import Result


def identity(x: float) -> float:
    return x


def tent(x: float) -> float:
    return -np.absolute(x)


def negative(x: float) -> float:
    return -x


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


def log_function[T, C](
    logger: logging.Logger,
) -> Callable[[Callable[[C], T]], Callable[[C], T]]:
    """decorator factory for logging purposes"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Running {func.__name__}...")
            result = func(*args, **kwargs)
            logger.info(f"Finished {func.__name__}.")
            return result

        return wrapper

    return decorator


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
) -> Callable[[Result], tuple[float, float]]:
    average_pairwise_correlation_func = _create_avg_pairwise_metric_fn(
        metric_fn=correlation_func
    )

    def pairwise_correlation_func(
        result: Result,
    ) -> tuple[float, float]:
        if (
            result.model_embedding_reference is None
            or result.model_embedding_query is None
        ):
            return (np.nan, np.nan)
        return (
            average_pairwise_correlation_func(
                result.model_embedding_reference,
            ),
            average_pairwise_correlation_func(
                result.model_embedding_query,
            ),
        )

    return pairwise_correlation_func


def create_embedding_structure_evaluator(
    structure_function: Callable[[NumericArray, NumericArray], float],
) -> Callable[[Result], tuple[float, float]]:
    def structure_evaluator(
        result: Result,
    ) -> tuple[float, float]:
        if (
            result.model_embedding_reference is None
            or result.model_embedding_query is None
        ):
            return (np.nan, np.nan)
        return (
            structure_function(
                result.celltype.reference_adjacency_matrix.toarray(),  # type: ignore
                result.model_embedding_reference,
            ),
            structure_function(
                result.celltype.query_adjacency_matrix.toarray(),  # type: ignore
                result.model_embedding_query,
            ),
        )

    return structure_evaluator


def make_aggregate_cellwise_metric_func(
    func: Callable[[NumericArray], float],
) -> Callable[[NumericArray], float]:
    def aggregate_cellwise_metric_func(arr: NumericArray) -> float:
        return float(np.mean([func(cell) for cell in arr]))

    return aggregate_cellwise_metric_func


def create_embedding_sparsity_evaluator(
    sparsity_func: Callable[[NumericArray], float],
) -> Callable[[Result], tuple[float, float]]:
    def embedding_sparsity_evaluator(
        result: Result,
    ) -> tuple[float, float]:
        if (
            result.model_embedding_reference is None
            or result.model_embedding_query is None
        ):
            return (np.nan, np.nan)
        embedding_sparsity_func = make_aggregate_cellwise_metric_func(
            sparsity_func,
        )
        return (
            embedding_sparsity_func(result.model_embedding_reference),
            embedding_sparsity_func(result.model_embedding_query),
        )

    return embedding_sparsity_evaluator


def make_max_aggregate_featurewise_metric_func(
    func: Callable[[NumericArray], float],
) -> Callable[[NumericArray], float]:

    def aggregate_cellwise_metric_func(arr: NumericArray) -> float:
        arr = StandardScaler().fit_transform(arr)
        return float(func(np.max(arr, axis=1)))

    return aggregate_cellwise_metric_func


def create_coverage_sparsity_evaluator(
    sparsity_func: Callable[[NumericArray], float],
) -> Callable[[Result], tuple[float, float]]:
    def coverage_sparsity_evaluator(
        result: Result,
    ) -> tuple[float, float]:
        if (
            result.model_embedding_reference is None
            or result.model_embedding_query is None
        ):
            return (np.nan, np.nan)
        max_agg_sparsity_func = make_max_aggregate_featurewise_metric_func(
            sparsity_func,
        )
        return (
            max_agg_sparsity_func(result.model_embedding_reference),
            max_agg_sparsity_func(result.model_embedding_query),
        )

    return coverage_sparsity_evaluator


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


GeneProgramsDB = frozenset[tuple[str, frozenset[str]]]


@cache
def get_gene_programs_db(
    name: str,
) -> GeneProgramsDB:
    library: dict[str, list[str]] = gseapy.get_library(  # type: ignore
        name=name,
    )
    return frozenset((program, frozenset(genes)) for program, genes in library.items())


def get_gene_program_counts(
    gene_programs_db: GeneProgramsDB,
    genes: Sequence[str],
) -> NumericArray:
    binary_matrix = [
        [1 if gene in program_genes else 0 for gene in genes]
        for _, program_genes in gene_programs_db
    ]
    return np.array(binary_matrix)


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


def enrichment_score_evaluator(result: Result) -> tuple[float, float]:
    if result.model_embedding_reference is None or result.model_embedding_query is None:
        return (np.nan, np.nan)

    enrichment_db_name = "KEGG_2026"
    gene_programs_db = get_gene_programs_db(name=enrichment_db_name)
    database_gene_names = extract_gene_list(gene_programs_db=gene_programs_db)
    dataset_gene_names = [gene.upper() for gene in result.sample.dataset.feature_names]
    genes: list[str] = np.intersect1d(database_gene_names, dataset_gene_names).tolist()
    if not genes:
        print(f"No overlapping genes found for result {result.id}. Returning 0.0.")
        return (0.0, 0.0)
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
            result.model_embedding_reference,
        ),
        abc(
            result.celltype.query_counts_matrix[:, feature_idx],
            result.model_embedding_query,
        ),
    )


def explained_variance_evaluator(
    result: Result,
) -> tuple[float, float]:
    inverse_transform = np.expm1 if result.model.transform == "log" else lambda x: x

    def explained_variance_func(arr1: NumericArray, arr2: NumericArray) -> float:
        if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
            return np.nan
        score = float(n2l.mt.explained_variance_metric(arr1, arr2).mean())  # type: ignore
        return score

    x_true_reference = result.celltype.reference_counts_matrix[
        :, result.sample.test_idx
    ].T
    x_pred_reference = result.model_counts_reference[:, result.sample.test_idx].T

    x_true_query = result.celltype.query_counts_matrix[:, result.sample.test_idx].T
    x_pred_query = result.model_counts_query[:, result.sample.test_idx].T

    scores = (
        explained_variance_func(x_true_reference, inverse_transform(x_pred_reference)),
        explained_variance_func(x_true_query, inverse_transform(x_pred_query)),
    )
    return scores


def mean_squared_error_evaluator(
    result: Result,
) -> tuple[float, float]:
    inverse_transform = np.expm1 if result.model.transform == "log" else lambda x: x

    def mse_func(arr1: NumericArray, arr2: NumericArray) -> float:
        if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
            return np.nan
            logging.warning("NaN values encountered in MSE calculation")
        return float(mean_squared_error(arr1, arr2))

    return (
        mse_func(
            result.celltype.reference_counts_matrix[:, result.sample.test_idx],
            inverse_transform(result.model_counts_reference[:, result.sample.test_idx]),
        ),
        mse_func(
            result.celltype.query_counts_matrix[:, result.sample.test_idx],
            inverse_transform(result.model_counts_query[:, result.sample.test_idx]),
        ),
    )


def compose[T, U, V](  # type: ignore
    a: Callable[[U], V],
    b: Callable[[T], U],
) -> Callable[[T], V]:
    return lambda x: a(b(x))


METRIC_FNS: dict[
    str,
    dict[
        str,
        tuple[
            Callable[[Result], tuple[float, float]],
            Callable[[float], float],
        ],
    ],
] = {
    "embedding_autocorrelation": {
        "pearsonr": (
            create_embedding_evaluator(n2l.mt.pearson_metric),  # type: ignore
            compose(lambda x: x + 1, tent),
        ),
    },
    "embedding_structure": {
        "multivariate_gearys_c": (
            create_embedding_structure_evaluator(compute_multivariate_geary),
            lambda x: float(np.exp(-x)),
        ),
    },
    "embedding_sparsity": {
        "gini": (
            create_embedding_sparsity_evaluator(gini),
            identity,
        ),
    },
    "coverage": {
        "gini": (
            create_coverage_sparsity_evaluator(gini),
            compose(lambda x: x + 1, negative),
        ),
        # "entropy": (create_coverage_sparsity_evaluator(entropy), identity),
    },
    # "biological_enrichment": {
    #    "max_cosine_alignment": (
    #        enrichment_score_evaluator,
    #        lambda x: np.clip(x, 0, 1),
    #    )
    # },
    "feature_prediction_performance": {
        "mean_squared_error": (
            mean_squared_error_evaluator,
            lambda x: float(np.exp(-x)),
        ),
        "explained_variance": (
            explained_variance_evaluator,
            # lambda x: float(max(0.0, x)),
            lambda x: 1 / np.exp(-x + 1),
        ),
    },
}


def main():
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    logging_decorator = log_function(logger)

    exp_name = "Default"
    benchmark_experiment = mlflow.get_experiment_by_name(exp_name)
    if not benchmark_experiment:
        raise ValueError(f"Experiment '{exp_name}' not found.")
    runs = mlflow.search_runs(experiment_ids=[benchmark_experiment.experiment_id])
    if runs.empty:  # type: ignore
        raise RuntimeError(f"No runs found in experiment '{exp_name}'.")
    last_run_id = runs.sort_values("start_time", ascending=False)["run_id"].iloc[0]  # type: ignore
    # last_run_id = "15c787b577f0484aa8391bcb0b06e095"
    # logger.warning("last_run_id variable overridden")
    logger.info(f"Using run_id: {last_run_id}")
    logger.info("Downloading 'database.db'...")
    database_path = mlflow.artifacts.download_artifacts(  # type: ignore
        run_id=last_run_id, artifact_path="database.db"
    )
    logger.info(f"Local path: {database_path}")
    sqlite_url = f"sqlite:///{database_path}"
    engine = create_engine(sqlite_url, echo=False)
    logger.info("Database engine initialized.")

    logger.info(f"Local path: {database_path}")
    sqlite_url = f"sqlite:///{database_path}"
    engine = create_engine(sqlite_url, echo=False)
    logger.info("Database engine initialized.")
    output_file = "benchmarking_output.csv"
    fieldnames = [
        "dataset_name",
        "celltype",
        "sample_id",
        "model_name",
        "transform",
        "metric_category",
        "function_name",
        "model_scope",
        "dataset_split",
        "value",
        "value_transformed",
    ]
    with Session(engine) as session:
        results = session.exec(select(Result)).all()
        with open(output_file, "w", newline="", buffering=1) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for result in tqdm(results):
                logger.info(f"Processing result {result.id}")
                for metric_category, function_mapping in METRIC_FNS.items():
                    for function_name, (
                        metric_function,
                        transformation_function,
                    ) in function_mapping.items():
                        metric_function = logging_decorator(metric_function)
                        metrics = metric_function(result)
                        for value, dataset_split in zip(
                            metrics, ["reference", "query"]
                        ):
                            writer.writerow(
                                {
                                    "dataset_name": result.celltype.dataset.name,
                                    "celltype": result.celltype.name,
                                    "sample_id": result.sample.id_of_sample,
                                    "model_name": result.model.name,
                                    "transform": result.model.transform,
                                    "metric_category": metric_category,
                                    "function_name": function_name,
                                    "model_scope": result.model.scope,
                                    "dataset_split": dataset_split,
                                    "value": value,
                                    "value_transformed": transformation_function(value),
                                }
                            )
                            csv_file.flush()
    logger.info("Done")


if __name__ == "__main__":
    result = main()
