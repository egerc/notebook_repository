from collections.abc import Generator, Sequence
from functools import cached_property
from itertools import product

import anndata as ad
import numpy as np
import pandas as pd
import pandera.pandas as pa
from anndata import AnnData
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
    get_dataset_filtered,
    get_gene_ids_by_sample,
    retrieve_counts,
    sample_genes,
    split_cells,
)
from log_2026_07_23t08_09_06z.models import (
    Model,
    PredictionScope,
    generate_results,
)
from log_2026_07_23t08_09_06z.types import (
    Err,
    Ok,
    Result,
    bind_result,
    unwrap_result,
)
from log_2026_07_23t08_09_06z.utils import (
    dataframe_to_json,
    pandas_pandera_from_json,
    read_h5ad,
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
    def try_from_setup(
        cls,
        dataset: QueryPlusReference,
        setup_strategy: SetupStrategy,
        sampling_strategy: SamplingStrategy,
        n_samples: int,
        seed: NonNegativeInt,
    ) -> Result["DatasetConfiguration", Exception]:
        dataset_setup: DatasetSetup = (setup_strategy, dataset)  # type: ignore
        cells_df_result, genes = split_cells(dataset_setup)

        return bind_result(
            cells_df_result,
            lambda cells_df: bind_result(
                dataframe_to_json(cells_df),
                lambda cell_data: bind_result(
                    sample_genes(
                        genes, sampling_strategy, n_samples, np.random.default_rng(seed)
                    ),
                    lambda genes_df: bind_result(
                        dataframe_to_json(genes_df),
                        lambda gene_data: Ok(
                            DatasetConfiguration(
                                dataset=dataset,
                                sampling_strategy=sampling_strategy,
                                setup_strategy=setup_strategy,
                                _cell_annotation_data=cell_data,
                                _gene_annotation_data=gene_data,
                            )
                        ),
                    ),
                ),
            ),
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


class ValidatedAnnData[S1: pa.DataFrameModel, S2: pa.DataFrameModel](ad.AnnData):
    _adata: ad.AnnData

    @classmethod
    def from_dataset_configuration(
        cls, dataset_configuration: DatasetConfiguration
    ) -> "ValidatedAnnData[CellAnnotationSchema, GeneAnnotationSchema]":
        reference = unwrap_result(
            read_h5ad(dataset_configuration.dataset.reference.path)
        )
        query = unwrap_result(read_h5ad(dataset_configuration.dataset.query.path))
        adata = ad.concat([reference, query])
        adata.obs = unwrap_result(
            bind_result(
                (
                    Ok(obs_df)
                    if isinstance(obs_df := adata.obs, pd.DataFrame)
                    else Err(ValueError("adata.obs is not a DataFrame"))
                ),
                lambda df: validate_pandas_pandera(
                    CellAnnotationSchema,
                    df.join(dataset_configuration.cell_annotation_df),
                ),
            )
        )
        adata.var = unwrap_result(
            bind_result(
                (
                    Ok(var_raw)
                    if isinstance(var_raw := adata.var, pd.DataFrame)
                    else Err(ValueError("adata.var is not a DataFrame"))
                ),
                lambda df: validate_pandas_pandera(
                    GeneAnnotationSchema,
                    df.join(dataset_configuration.gene_annotation_df),
                ),
            )
        )
        return ValidatedAnnData(adata)


def setup_datasets(
    datasets: Sequence[QueryPlusReference],
    sampling_strategies: Sequence[SamplingStrategy],
    setup_strategies: Sequence[SetupStrategy],
    n_samples: PositiveInt,
    seed: NonNegativeInt,
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
            dataset, setup_strategy, sampling_strategy, n_samples, seed
        )


def run_experiment_for_model_and_scope(
    dataset_configuration: DatasetConfiguration,
    model: Model,
    prediction_scope: PredictionScope,
) -> dict[int, Result[AnnData, Exception]]:
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

    predictions: dict[int, Result[AnnData, Exception]] = {}
    # a = group_cells_by_split(dataset_configuration.cell_annotation_df)
    barcodes = get_barcodes(dataset_configuration.cell_annotation_df)
    for sample_id, gene_ids in get_gene_ids_by_sample(
        dataset_configuration.gene_annotation_df
    ).items():
        all_genes = gene_ids["train"] + gene_ids["test"]
        reference = unwrap_result(
            get_dataset_filtered(
                dataset_configuration.dataset.reference,
                barcodes["train"],
                all_genes,
            )
        )
        query = unwrap_result(
            get_dataset_filtered(
                dataset_configuration.dataset.query,
                barcodes["test"],
                gene_ids["train"],
            )
        )
        predictions[sample_id] = generate_results(
            model,
            reference,
            query,
            prediction_scope,
        )
    return predictions


def postprocess(): ...
