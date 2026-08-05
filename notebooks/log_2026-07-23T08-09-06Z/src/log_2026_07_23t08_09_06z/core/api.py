from collections.abc import Callable, Generator, Mapping, Sequence
from functools import partial
from itertools import product
from typing import Literal

import pandera.pandas as pa
from anndata import AnnData
from pandera.typing.pandas import DataFrame
from pydantic.types import FilePath, NonNegativeInt, PositiveInt
from sklearn.utils.validation import joblib

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
    NumericArray,
    Ok,
    Result,
    SamplingSplit,
    bind_result,
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
    cache_dir: FilePath | None,
) -> Generator[tuple[int, EvaluationResult]]:
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
    generate_results_function: Callable[
        [Model, AnnData, AnnData, PredictionScope],
        Result[AnnData, Exception],
    ] = (  # type: ignore
        joblib.Memory(cache_dir).cache(generate_results)
        if cache_dir is not None
        else generate_results
    )

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
                    barcodes["test"],
                    gene_ids["train"],
                    SamplingSplit.TEST,
                ),
                get_counts_per_cell_split(
                    dataset_configuration.setup_strategy,
                    dataset_configuration.dataset,
                    barcodes["train"],
                    all_genes,
                    SamplingSplit.TRAIN,
                ),
            ),
            lambda query, reference: generate_results_function(
                model, reference, query, prediction_scope
            ),
        )

        def _slice_adata(
            adata: AnnData, barcodes: Sequence[str], gene_ids: Sequence[str]
        ) -> Result[AnnData, Exception]:
            try:
                return Ok(adata[list(barcodes), list(gene_ids)])
            except Exception as e:  # noqa
                return Err(e)

        evaluation_result: EvaluationResult = {}

        dataset_split = DatasetSplit(SamplingSplit.TEST, SamplingSplit.TEST)
        split_barcodes = barcodes[get_split(dataset_split.cell_split)]
        split_gene_ids = gene_ids[get_split(dataset_split.gene_split)]
        evaluation_result[dataset_split] = zip_result(
            get_counts_per_cell_split(
                dataset_configuration.setup_strategy,
                dataset_configuration.dataset,
                split_barcodes,
                split_gene_ids,
                dataset_split.cell_split,
            ),
            bind_result(
                adata_prediction,
                partial(
                    _slice_adata,
                    barcodes=split_barcodes,
                    gene_ids=split_gene_ids,
                ),
            ),
        )
        yield sample_id, evaluation_result


class ExperimentResultSchema(pa.DataFrameModel):
    query_path: str
    query_cluster_key: str
    reference_path: str
    reference_cluster_key: str
    sample_id: int
    setup_strategy: str = pa.Field(
        isin=["Spatial", "SpatialPseudospatial", "Pseudospatial", "NonSpatial"]
    )
    panel_sample_strategy: str = pa.Field(isin=["SampleRemainderPanel", "SamplePanel"])
    model: str = pa.Field(isin=["NMF", "scVI"])
    scoring_function: str
    cell_split: str = pa.Field(isin=["train", "test"])
    gene_split: str = pa.Field(isin=["train", "test"])
    value: float


type ExperimentParameters = tuple[DatasetConfiguration, int, Model, DatasetSplit, str]


def experiment(
    dataset_configurations: Sequence[DatasetConfiguration],
    model_setups: Sequence[
        tuple[Model, PredictionScope, FilePath | None]
    ],  # the optional string is the model specific caching directory
    evaluation_functions: Mapping[str, Callable[[NumericArray, NumericArray], float]],
) -> Generator[tuple[ExperimentParameters, Result[float, Exception]]]:
    for dataset_configuration in dataset_configurations:
        for model, prediction_scope, model_cache_dir in model_setups:
            for sample_id, evaluation_result in run_experiment_for_model_and_scope(
                dataset_configuration, model, prediction_scope, model_cache_dir
            ):
                for dataset_split, adata_tuple_result in evaluation_result.items():
                    for (
                        scoring_function_name,
                        scoring_function,
                    ) in evaluation_functions.items():
                        result_score = starbind_result(
                            adata_tuple_result,
                            partial(
                                apply_reconstruction_scoring_func, func=scoring_function
                            ),
                        )
                        experiment_parameters: ExperimentParameters = (
                            dataset_configuration,
                            sample_id,
                            model,
                            dataset_split,
                            scoring_function_name,
                        )
                        yield experiment_parameters, result_score

def _(): ...