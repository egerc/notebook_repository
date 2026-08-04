from collections.abc import Generator
from functools import partial
from itertools import product

import nico2_lib as n2l
from anndata import AnnData
from pydantic.types import NonNegativeInt, PositiveInt

from log_2026_07_23t08_09_06z.datasets import (
    DatasetConfiguration,
    QueryPlusReference,
    SamplingStrategy,
    SetupStrategy,
    get_barcodes,
    get_counts_per_cell_split,
    get_gene_ids_by_sample,
    get_split,
    retrieve_counts,
)
from log_2026_07_23t08_09_06z.evaluation import apply_reconstruction_scoring_func
from log_2026_07_23t08_09_06z.models import (
    Model,
    PredictionScope,
    generate_results,
)
from log_2026_07_23t08_09_06z.types import (
    DatasetSplit,
    Err,
    EvaluationResult,
    Ok,
    Result,
    SamplingSplit,
    bind_result,
    map_err,
    starbind_result,
    zip_result,
)
from log_2026_07_23t08_09_06z.utils import (
    FilteringConfig,
)


def _(
    dataset_configuration: DatasetConfiguration,
    model: Model,
    prediction_scope: PredictionScope,
) -> Generator[AnnData, None, None]:
    """Yield per-sample predictions across all retrieved counts.

    Args:
        dataset_configuration: Configuration describing the dataset.
        model: Fitted or fit-on-the-fly model used for prediction.
        prediction_scope: Scope passed to the model's prediction step.

    Yields:
        Per-sample predicted ``AnnData`` objects.

    Raises:
        NotImplementedError: ``retrieve_counts`` is not yet implemented.
    """
    raise NotImplementedError()
    for _, reference, query in retrieve_counts(
        dataset_setup=(  # type: ignore
            dataset_configuration.setup_strategy,
            dataset_configuration.dataset,
        ),
        cell_annotation_df=dataset_configuration.cell_annotation_df,
        gene_annotation_df=dataset_configuration.gene_annotation_df,
    ):
        pass


def setup_datasets(
    datasets: set[QueryPlusReference],
    sampling_strategies: set[SamplingStrategy],
    setup_strategies: set[SetupStrategy],
    n_samples: PositiveInt,
    seed: NonNegativeInt,
    filtering_config: FilteringConfig,
) -> Generator[Result[DatasetConfiguration, Exception]]:
    """Materialize the cartesian product of dataset/strategy combinations.

    Args:
        datasets: Query/reference pairs to configure.
        sampling_strategies: Sampling strategies to apply.
        setup_strategies: Setup strategies to apply.
        n_samples: Number of samples per configuration.
        seed: Seed forwarded to each configuration.

    Returns:
        List of ``DatasetConfiguration`` objects, one per combination.

    Raises:
        ValueError: If any configuration fails to build its annotations.
    """

    for dataset, setup_strategy, sampling_strategy in product(
        datasets, setup_strategies, sampling_strategies
    ):
        yield DatasetConfiguration.try_from_setup(
            dataset,
            setup_strategy,
            sampling_strategy,
            n_samples,
            seed,
            filtering_config,
        )


def run_experiment_for_model_and_scope(
    dataset_configuration: DatasetConfiguration,
    model: Model,
    prediction_scope: PredictionScope,
) -> dict[int, EvaluationResult]:
    """Run a single experiment and collect per-sample predictions.

    Args:
        dataset_configuration: Configuration describing the experiment.
        model: Model used for fitting and prediction.
        prediction_scope: Scope passed to the prediction step.

    Returns:
        Mapping ``sample_id -> predicted AnnData``.

    Raises:
        ValueError: If annotation loading or count retrieval fails.
    """

    predictions: dict[int, EvaluationResult] = {}
    # a = group_cells_by_split(dataset_configuration.cell_annotation_df)
    barcodes = get_barcodes(dataset_configuration.cell_annotation_df)
    for sample_id, gene_ids in get_gene_ids_by_sample(
        dataset_configuration.gene_annotation_df
    ).items():
        all_genes = gene_ids["train"] + gene_ids["test"]
        adata_prediction = starbind_result(
            zip_result(
                get_counts_per_cell_split(
                    dataset_configuration.setup_strategy,
                    dataset_configuration.dataset,
                    set(barcodes["test"]),
                    set(gene_ids["train"]),
                    SamplingSplit.TEST,
                ),
                get_counts_per_cell_split(
                    dataset_configuration.setup_strategy,
                    dataset_configuration.dataset,
                    set(barcodes["train"]),
                    set(all_genes),
                    SamplingSplit.TRAIN,
                ),
            ),
            lambda query, reference: generate_results(
                model, reference, query, prediction_scope
            ),
        )

        def _slice_adata(
            adata: AnnData, barcodes: set[str], gene_ids: set[str]
        ) -> Result[AnnData, Exception]:
            try:
                return Ok(adata[list(barcodes), list(gene_ids)])
            except Exception as e:  # noqa
                return Err(e)

        evaluation_result: EvaluationResult = {}

        def _create_split_error(
            dataset_split: DatasetSplit,
            err: Exception,
        ) -> ValueError:
            return ValueError(
                f"Failed to compute evaluation result for {dataset_split}, reason: {err}"
            )

        for cell_split, gene_split in product(SamplingSplit, repeat=2):
            dataset_split = DatasetSplit(cell_split, gene_split)
            split_barcodes = set(barcodes[get_split(cell_split)])
            split_gene_ids = set(gene_ids[get_split(gene_split)])
            evaluation_result[dataset_split] = map_err(
                starbind_result(
                    zip_result(
                        get_counts_per_cell_split(
                            dataset_configuration.setup_strategy,
                            dataset_configuration.dataset,
                            split_barcodes,
                            split_gene_ids,
                            cell_split,
                        ),
                        bind_result(
                            adata_prediction,
                            partial(
                                _slice_adata,
                                barcodes=split_barcodes,
                                gene_ids=split_gene_ids,
                            ),
                        ),
                    ),
                    lambda adata_true, adata_pred: apply_reconstruction_scoring_func(
                        adata_true,
                        adata_pred,
                        lambda arr1, arr2: 0.0,  # type: ignore
                    ),
                ),
                partial(_create_split_error, dataset_split),
            )
        predictions[sample_id] = evaluation_result

    return predictions
