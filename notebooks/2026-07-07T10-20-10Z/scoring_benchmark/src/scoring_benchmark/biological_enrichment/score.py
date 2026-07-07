from collections.abc import Callable, Sequence
from typing import Literal

import gseapy  # type: ignore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scoring_benchmark.typing import NumericArray

GeneProgramsDB = frozenset[tuple[str, frozenset[str]]]


def _get_gene_programs_db(
    species: str,
    name: str,
) -> GeneProgramsDB:
    library: dict[str, list[str]] = gseapy.get_library(  # type: ignore
        organism=species,
        name=name,
    )
    return frozenset((program, frozenset(genes)) for program, genes in library.items())


def _get_gene_program_counts(
    gene_programs_db: GeneProgramsDB,
    genes: Sequence[str],
) -> NumericArray:
    binary_matrix = [
        [1 if gene in program_genes else 0 for gene in genes]
        for _, program_genes in gene_programs_db
    ]
    return np.array(binary_matrix)


def _factor_gene_correlations(
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


def _max_cosine_alignment_scoring(
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


def score_biological_enrichment(
    counts: NumericArray,
    genes: list[str],
    embeddings: NumericArray,
    species: Literal["human"],
    db_name: str,
    corr_func: Callable[[NumericArray, NumericArray], float],
) -> float:
    gene_programs_db = _get_gene_programs_db(
        species=species,
        name=db_name,
    )
    gene_program_counts = _get_gene_program_counts(
        gene_programs_db,
        genes,
    )
    _factor_gene_correlations(
        x=counts,
        w=embeddings,
        corr_func=corr_func,
    )
    score = _max_cosine_alignment_scoring(
        embeddings,
        gene_program_counts,
    )
    return score


__all__ = [
    "score_biological_enrichment",
]
