from collections.abc import Callable, Generator, Mapping, Sequence
from functools import partial
from itertools import product
from typing import assert_never

import pandas as pd
import pandera.pandas as pa
from anndata import AnnData
from pandera.typing.pandas import DataFrame
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath, NonNegativeInt, PositiveInt
from sklearn.utils.validation import joblib

from log_2026_07_23t08_09_06z.datasets import (
    DatasetConfiguration,
    QueryPlusReference,
    SampleNonPanel,
    SamplePanel,
    SamplingStrategy,
    SamplingStrategyEither,
    SetupStrategy,
    get_barcodes,
    get_counts_per_cell_split,
    get_gene_ids_by_sample,
    get_split,
)
from log_2026_07_23t08_09_06z.evaluation import (
    ScoringSetup,
    apply_reconstruction_scoring_func,
    apply_reconstruction_scoring_func_new,
)
from log_2026_07_23t08_09_06z.models import (
    Model,
    PredictionScope,
    generate_results,
)
from log_2026_07_23t08_09_06z.types import (
    CountsTrueCountsPredMapping,
    DatasetSplit,
    DownsamplingConfig,
    Err,
    NumericArray,
    Ok,
    Result,
    SamplingSplit,
    bind_result,
    map_result,
    starbind_result,
    unwrap_result,
    zip_result,
)
from log_2026_07_23t08_09_06z.utils import (
    FilteringConfig,
)


def _cached_from_setup(
    dataset: QueryPlusReference,
    setup_strategy: SetupStrategy,
    sampling_strategy: SamplingStrategy,
    n_samples: int,
    seed: NonNegativeInt,
    filtering_config: FilteringConfig,
    downsampling_config: DownsamplingConfig | None,
) -> Result[DatasetConfiguration, Exception]:
    return DatasetConfiguration.try_from_setup(
        dataset=dataset,
        setup_strategy=setup_strategy,
        sampling_strategy=sampling_strategy,
        n_samples=n_samples,
        seed=seed,
        filtering_config=filtering_config,
        downsampling_config=downsampling_config,
    )


def setup_datasets(
    datasets: set[QueryPlusReference],
    sampling_strategies: set[SamplingStrategyEither],
    setup_strategies: set[SetupStrategy],
    n_samples: PositiveInt,
    seed: NonNegativeInt,
    filtering_config: FilteringConfig,
    celltype_downsampling_config: DownsamplingConfig,
    cache_dir: FilePath | None,
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

    setup_function: Callable[
        [
            QueryPlusReference,
            SetupStrategy,
            SamplingStrategyEither,
            int,
            NonNegativeInt,
            FilteringConfig,
            DownsamplingConfig,
        ],
        Result[DatasetConfiguration, Exception],
    ] = (  # type: ignore
        joblib.Memory(cache_dir).cache(_cached_from_setup)
        if cache_dir is not None
        else DatasetConfiguration.try_from_setup
    )
    for dataset, setup_strategy, sampling_strategy in product(
        datasets, setup_strategies, sampling_strategies
    ):
        yield setup_function(
            dataset,
            setup_strategy,
            sampling_strategy,
            n_samples,
            seed,
            filtering_config,
            celltype_downsampling_config,
        )


@dataclass(frozen=True)
class RunFeatures:
    n_total_features: int
    n_training_features: int
    n_testing_features: int


def run_experiment_for_model_and_scope(
    dataset_configuration: DatasetConfiguration,
    model: Model,
    prediction_scope: PredictionScope,
    cache_dir: FilePath | None,
) -> Generator[tuple[int, CountsTrueCountsPredMapping, RunFeatures]]:
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
        [Model, AnnData, AnnData, PredictionScope, QueryPlusReference],
        Result[AnnData, Exception],
    ] = (  # type: ignore
        joblib.Memory(cache_dir).cache(generate_results)
        if cache_dir is not None
        else generate_results
    )

    # a = group_cells_by_split(dataset_configuration.cell_annotation_df)
    barcodes = unwrap_result(
        map_result(dataset_configuration.cell_annotation_df, get_barcodes)
    )

    for sample_id, gene_ids in unwrap_result(
        bind_result(dataset_configuration.gene_annotation_df, get_gene_ids_by_sample)
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
                model,
                reference,
                query,
                prediction_scope,
                dataset_configuration.dataset,
            ),
        )

        def _slice_adata(
            adata: AnnData, barcodes: Sequence[str], gene_ids: Sequence[str]
        ) -> Result[AnnData, Exception]:
            try:
                return Ok(adata[list(barcodes), list(gene_ids)])
            except Exception as e:  # noqa
                return Err(e)

        evaluation_result: CountsTrueCountsPredMapping = {}

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
        run_features = RunFeatures(
            n_total_features=len(all_genes),
            n_training_features=len(gene_ids["train"]),
            n_testing_features=len(gene_ids["test"]),
        )
        yield sample_id, evaluation_result, run_features


class ExperimentResultSchema(pa.DataFrameModel):
    query_path: str
    query_cluster_key: str
    reference_path: str
    reference_cluster_key: str
    sample_id: int
    setup_strategy: str = pa.Field(
        isin=["Spatial", "SpatialPseudospatial", "Pseudospatial", "NonSpatial"]
    )
    panel_sample_strategy: str = pa.Field(
        isin=["SampleNonPanel", "SamplePanel", "Both"]
    )
    model: str = pa.Field(isin=["NMF", "scVI"])
    scoring_function: str
    cell_split: str = pa.Field(isin=["train", "test"])
    gene_split: str = pa.Field(isin=["train", "test"])
    prediction_scope: str = pa.Field(isin=["global", "celltype"])
    scoring_setup: str = pa.Field(
        isin=["cell_wise", "gene_wise", "population_celltype", "population_dataset"]
    )
    n_total_features: int
    n_training_features: int
    n_testing_features: int
    identifier: str
    value: float


type ExperimentParameters = tuple[
    DatasetConfiguration,
    int,
    Model,
    DatasetSplit,
    str,
    PredictionScope,
    ScoringSetup,
    RunFeatures,
]


def experiment(
    dataset_configurations: Sequence[DatasetConfiguration],
    model_setups: Sequence[
        tuple[Model, PredictionScope, FilePath | None]
    ],  # the optional string is the model specific caching directory
    named_scoring_setups: set[tuple[str, ScoringSetup]],
) -> Generator[
    tuple[
        ExperimentParameters,
        Result[Generator[Result[tuple[str, float], Exception]], Exception],
    ]
]:
    for dataset_configuration in dataset_configurations:
        for model, prediction_scope, model_cache_dir in model_setups:
            for (
                sample_id,
                count_true_counts_pred_mapping,
                run_features,
            ) in run_experiment_for_model_and_scope(
                dataset_configuration, model, prediction_scope, model_cache_dir
            ):
                for (
                    dataset_split,
                    adata_tuple_result,
                ) in count_true_counts_pred_mapping.items():
                    for scoring_function_name, scoring_setup in named_scoring_setups:
                        x = starbind_result(
                            adata_tuple_result,
                            partial(
                                apply_reconstruction_scoring_func_new,
                                scoring_setup=scoring_setup,
                            ),
                        )
                        yield (
                            (
                                dataset_configuration,
                                sample_id,
                                model,
                                dataset_split,
                                scoring_function_name,
                                prediction_scope,
                                scoring_setup,
                                run_features,
                            ),
                            x,
                        )


def experiment_deprecated(
    dataset_configurations: Sequence[DatasetConfiguration],
    model_setups: Sequence[
        tuple[Model, PredictionScope, FilePath | None]
    ],  # the optional string is the model specific caching directory
    evaluation_functions: Mapping[str, Callable[[NumericArray, NumericArray], float]],
) -> Generator[tuple[ExperimentParameters, Result[float, Exception]]]:
    for dataset_configuration in dataset_configurations:
        for model, prediction_scope, model_cache_dir in model_setups:
            for (
                sample_id,
                evaluation_result,
                run_features,
            ) in run_experiment_for_model_and_scope(
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
                            prediction_scope,
                            run_features,
                        )
                        yield experiment_parameters, result_score


def _sampling_strategy_name(
    sampling_strategy: SamplingStrategyEither,
) -> str:
    match sampling_strategy:
        case SamplePanel():
            return "SamplePanel"
        case SampleNonPanel():
            return "SampleNonPanel"
        case tuple():
            return "Both"
        case _:
            assert_never(sampling_strategy)


def create_experiment_table(
    value: Generator[
        tuple[
            ExperimentParameters,
            Result[Generator[Result[tuple[str, float], Exception]], Exception],
        ]
    ],
) -> DataFrame[ExperimentResultSchema]:
    """Materialize a typed experiment-result table from collected runs.

    Each ``Err`` result is skipped; only successful ``Ok`` runs contribute a row.
    The returned frame is validated against ``ExperimentResultSchema``.

    Args:
        value: Tuples of (parameters, scoring result) produced by ``experiment``.

    Returns:
        A ``DataFrame`` conforming to ``ExperimentResultSchema``.

    Raises:
        pandera.errors.SchemaError: If the constructed frame fails validation.
    """
    _MODEL_NAMES = {"NmfPredictor": "NMF", "ScviPredictor": "scVI"}

    records: list[dict[str, object]] = []
    for parameters, result in value:
        match result:
            case Err(reason):
                print(
                    f"Skipping result for parameters {parameters} due to error: {reason}"
                )
                continue
            case Ok(metric_generator):
                (
                    dataset_configuration,
                    sample_id,
                    model,
                    dataset_split,
                    scoring_function_name,
                    prediction_scope,
                    scoring_setup,
                    run_features,
                ) = parameters
                for metric_result in metric_generator:
                    match metric_result:
                        case Err(reason):
                            pass
                        case Ok(identifier_plus_value):
                            records.append(
                                {
                                    "query_path": str(
                                        dataset_configuration.dataset.query.path
                                    ),
                                    "query_cluster_key": dataset_configuration.dataset.query.cluster_key,
                                    "reference_path": str(
                                        dataset_configuration.dataset.reference.path
                                    ),
                                    "reference_cluster_key": dataset_configuration.dataset.reference.cluster_key,
                                    "sample_id": sample_id,
                                    "setup_strategy": type(
                                        dataset_configuration.setup_strategy
                                    ).__name__.removesuffix("Setup"),
                                    "panel_sample_strategy": _sampling_strategy_name(
                                        dataset_configuration.sampling_strategy
                                    ),
                                    "model": _MODEL_NAMES[type(model).__name__],
                                    "scoring_function": scoring_function_name,
                                    "cell_split": dataset_split.cell_split.value,
                                    "gene_split": dataset_split.gene_split.value,
                                    "prediction_scope": prediction_scope.value,
                                    "scoring_setup": # fix this empty field,
                                    "n_total_features": run_features.n_total_features,
                                    "n_training_features": run_features.n_training_features,
                                    "n_testing_features": run_features.n_testing_features,
                                    "identifier": identifier_plus_value[0],
                                    "value": identifier_plus_value[1],
                                }
                            )

    columns = list(ExperimentResultSchema.to_schema().columns.keys())
    df = pd.DataFrame(records, columns=columns)
    return ExperimentResultSchema.validate(df)
