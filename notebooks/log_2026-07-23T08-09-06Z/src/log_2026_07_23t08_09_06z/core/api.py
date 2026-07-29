from collections.abc import Generator, Sequence
from functools import cache, cached_property
from itertools import product
from pathlib import Path

import numpy as np
from anndata import AnnData
from pandas.io.pickle import pickle
from pandera.typing.pandas import DataFrame
from pydantic.dataclasses import dataclass
from pydantic.types import NonNegativeInt, PositiveInt

from log_2026_07_23t08_09_06z.datasets import (
    CellAnnotationSchema,
    DatasetSetup,
    GeneAnnotationSchema,
    NonSpatialSetup,
    PseudospatialSetup,
    QueryPlusReference,
    SamplingStrategy,
    SpatialPseudospatialSetup,
    SpatialSetup,
    get_barcodes,
    get_dataset_counts,
    get_gene_ids_by_sample,
    retrieve_counts,
    sample_genes,
    split_cells,
)
from log_2026_07_23t08_09_06z.models import Model, PredictionScope, fit_model, predict
from log_2026_07_23t08_09_06z.types import (
    bind_result,
    maybe_from_optional,
    ok_or,
    unwrap_result,
)
from log_2026_07_23t08_09_06z.utils import (
    dataframe_to_json,
    pandas_pandera_from_json,
    validate_pandas_pandera,
)

type SetupStrategy = (
    SpatialSetup | SpatialPseudospatialSetup | PseudospatialSetup | NonSpatialSetup
)


@dataclass(frozen=True)
class DatasetConfiguration:
    """Materialized dataset setup with serialized cell and gene annotations.

    Attributes:
        dataset: Query/reference pair backing the configuration.
        sampling_strategy: Strategy used to build the gene panels.
        setup_strategy: Discriminator describing the dataset layout.
        _cell_annotation_data: JSON-serialized per-cell annotations.
        _gene_annotation_data: JSON-serialized per-sample gene annotations.
    """

    dataset: QueryPlusReference
    sampling_strategy: SamplingStrategy
    setup_strategy: SetupStrategy
    _cell_annotation_data: str
    _gene_annotation_data: str

    @classmethod
    def from_setup(
        cls,
        dataset: QueryPlusReference,
        setup_strategy: SetupStrategy,
        sampling_strategy: SamplingStrategy,
        n_samples: int,
        seed: NonNegativeInt,
    ) -> "DatasetConfiguration":
        """Build a configuration from a setup triple by running the splits.

        Args:
            dataset: Query/reference pair to build the configuration from.
            setup_strategy: Discriminator describing the dataset layout.
            sampling_strategy: Strategy used to build the gene panels.
            n_samples: Number of samples to draw gene panels for.
            seed: Seed for the gene-panel random generator.

        Returns:
            A populated ``DatasetConfiguration``.

        Raises:
            ValueError: If cell or gene annotation generation/validation fails.
        """
        dataset_setup: DatasetSetup = (setup_strategy, dataset)  # type: ignore
        cell_annotation_data, genes = (
            unwrap_result(
                bind_result(
                    (results := split_cells(dataset_setup))[0], dataframe_to_json
                ),
                "cell_annotation_df failure",
            ),
            results[1],
        )
        gene_annotation_data = unwrap_result(
            bind_result(
                sample_genes(
                    genes, sampling_strategy, n_samples, np.random.default_rng(seed)
                ),
                dataframe_to_json,
            ),
            "gene_annotation_df failure",
        )
        return DatasetConfiguration(
            dataset=dataset,
            sampling_strategy=sampling_strategy,
            setup_strategy=setup_strategy,
            _cell_annotation_data=cell_annotation_data,
            _gene_annotation_data=gene_annotation_data,
        )

    @cached_property
    def cell_annotation_df(self) -> DataFrame[CellAnnotationSchema]:
        """Parsed and schema-validated per-cell annotations.

        Returns:
            ``DataFrame[CellAnnotationSchema]``.

        Raises:
            ValueError: If JSON parsing or schema validation fails.
        """
        return unwrap_result(
            pandas_pandera_from_json(CellAnnotationSchema, self._cell_annotation_data)
        )

    @cached_property
    def gene_annotation_df(self) -> DataFrame[GeneAnnotationSchema]:
        """Parsed and schema-validated per-sample gene annotations.

        Returns:
            ``DataFrame[GeneAnnotationSchema]``.

        Raises:
            ValueError: If JSON parsing or schema validation fails.
        """
        return unwrap_result(
            pandas_pandera_from_json(GeneAnnotationSchema, self._gene_annotation_data)
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
    for _, reference, query in retrieve_counts(
        dataset_setup=(  # type: ignore
            dataset_configuration.setup_strategy,
            dataset_configuration.dataset,
        ),
        cell_annotation_df=dataset_configuration.cell_annotation_df,
        gene_annotation_df=dataset_configuration.gene_annotation_df,
    ):
        yield predict(reference, query, model, prediction_scope)


def setup_datasets(
    datasets: Sequence[QueryPlusReference],
    sampling_strategies: Sequence[SamplingStrategy],
    setup_strategies: Sequence[SetupStrategy],
    n_samples: PositiveInt,
    seed: NonNegativeInt,
) -> list[DatasetConfiguration]:
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
    dataset_configurations = [
        DatasetConfiguration.from_setup(
            dataset=dataset,
            setup_strategy=setup_strategy,
            sampling_strategy=sampling_strategy,
            n_samples=n_samples,
            seed=seed,
        )
        for dataset, setup_strategy, sampling_strategy in product(
            datasets, setup_strategies, sampling_strategies
        )
    ]
    return dataset_configurations


def run_experiment(
    dataset_configuration: DatasetConfiguration,
    model: Model,
    prediction_scope: PredictionScope,
) -> dict[int, AnnData]:
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
    predictions: dict[int, AnnData] = {}
    barcodes = get_barcodes(dataset_configuration.cell_annotation_df)
    for gene_ids in get_gene_ids_by_sample(
        dataset_configuration.gene_annotation_df
    ).values():
        get_dataset_counts(
            dataset_configuration.dataset.reference,
            barcodes["train"],
            gene_ids["train"],
        )
        get_dataset_counts(
            dataset_configuration.dataset.query,
            barcodes["test"],
            gene_ids["train"],
        )
        fit_model()


def postprocess(): ...
