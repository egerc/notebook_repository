import logging
from collections.abc import Mapping, Sequence
from functools import cache, reduce
from itertools import product
from pathlib import Path
from time import sleep
from typing import Any, Callable, Literal

import gseapy
import nico2_lib as n2l
import numpy as np
import yaml
from anndata import read_h5ad
from anndata.typing import AnnData
from joblib import Memory
from nico2_lib.typing import NumericArray
from pydantic import (
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationInfo,
    field_validator,
)
from pydantic.dataclasses import dataclass
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

memory = Memory("./cache")

PreprocessingNames = Literal["identity", "log1p"]
CorrelationNames = Literal["pearson", "spearman", "cosine_similarity"]
ScoringFunctionNames = Literal["dominic_scoring"]
GeneProgramsDB = frozenset[tuple[str, frozenset[str]]]
DatabaseName = Literal[
    "ARCHS4_Cell-lines",
    "ARCHS4_IDG_Coexp",
    "ARCHS4_Kinases_Coexp",
    "ARCHS4_TFs_Coexp",
    "ARCHS4_Tissues",
    "Achilles_fitness_decrease",
    "Achilles_fitness_increase",
    "Aging_Perturbations_from_GEO_down",
    "Aging_Perturbations_from_GEO_up",
    "Allen_Brain_Atlas_10x_scRNA_2021",
    "Allen_Brain_Atlas_down",
    "Allen_Brain_Atlas_up",
    "Azimuth_2023",
    "Azimuth_Cell_Types_2021",
    "BioCarta_2013",
    "BioCarta_2015",
    "BioCarta_2016",
    "BioPlanet_2019",
    "BioPlex_2017",
    "CCLE_Proteomics_2020",
    "CM4AI_U2OS_Protein_Localization_Assemblies",
    "COMPARTMENTS_Curated_2025",
    "COMPARTMENTS_Experimental_2025",
    "CORUM",
    "COVID-19_Related_Gene_Sets",
    "COVID-19_Related_Gene_Sets_2021",
    "Cancer_Cell_Line_Encyclopedia",
    "Carcinogenome",
    "CellMarker_2024",
    "CellMarker_Augmented_2021",
    "ChEA_2013",
    "ChEA_2015",
    "ChEA_2016",
    "ChEA_2022",
    "Chromosome_Location",
    "Chromosome_Location_hg19",
    "ClinVar_2019",
    "ClinVar_2025",
    "DGIdb_Drug_Targets_2024",
    "DSigDB",
    "Data_Acquisition_Method_Most_Popular_Genes",
    "DepMap_CRISPR_GeneDependency_CellLines_2023",
    "DepMap_WG_CRISPR_Screens_Broad_CellLines_2019",
    "DepMap_WG_CRISPR_Screens_Sanger_CellLines_2019",
    "Descartes_Cell_Types_and_Tissue_2021",
    "Diabetes_Perturbations_GEO_2022",
    "DisGeNET",
    "Disease_Perturbations_from_GEO_down",
    "Disease_Perturbations_from_GEO_up",
    "Disease_Signatures_from_GEO_down_2014",
    "Disease_Signatures_from_GEO_up_2014",
    "DrugMatrix",
    "Drug_Perturbations_from_GEO_2014",
    "Drug_Perturbations_from_GEO_down",
    "Drug_Perturbations_from_GEO_up",
    "ENCODE_Histone_Modifications_2013",
    "ENCODE_Histone_Modifications_2015",
    "ENCODE_TF_ChIP-seq_2014",
    "ENCODE_TF_ChIP-seq_2015",
    "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
    "ESCAPE",
    "Elsevier_Pathway_Collection",
    "Enrichr_Libraries_Most_Popular_Genes",
    "Enrichr_Submissions_TF-Gene_Coocurrence",
    "Enrichr_Users_Contributed_Lists_2020",
    "Epigenomics_Roadmap_HM_ChIP-seq",
    "FANTOM6_lncRNA_KD_DEGs",
    "GO_Biological_Process_2021",
    "GO_Biological_Process_2023",
    "GO_Biological_Process_2025",
    "GO_Biological_Process_2026",
    "GO_Cellular_Component_2021",
    "GO_Cellular_Component_2023",
    "GO_Cellular_Component_2025",
    "GO_Cellular_Component_2026",
    "GO_Molecular_Function_2021",
    "GO_Molecular_Function_2023",
    "GO_Molecular_Function_2025",
    "GO_Molecular_Function_2026",
    "GTEx_Aging_Signatures_2021",
    "GTEx_Tissue_Expression_Down",
    "GTEx_Tissue_Expression_Up",
    "GTEx_Tissues_V8_2023",
    "GWAS_Catalog_2019",
    "GWAS_Catalog_2023",
    "GWAS_Catalog_2025",
    "GeDiPNet_2023",
    "GeneSigDB",
    "Gene_Perturbations_from_GEO_down",
    "Gene_Perturbations_from_GEO_up",
    "Genes_Associated_with_NIH_Grants",
    "Genome_Browser_PWMs",
    "GlyGen_Glycosylated_Proteins_2022",
    "HDSigDB_Human_2021",
    "HDSigDB_Mouse_2021",
    "HMDB_Metabolites",
    "HMS_LINCS_KinomeScan",
    "HomoloGene",
    "HuBMAP_ASCT_plus_B_augmented_w_RNAseq_Coexpression",
    "HuBMAP_ASCTplusB_augmented_2022",
    "HumanCyc_2015",
    "HumanCyc_2016",
    "Human_Gene_Atlas",
    "Human_Phenotype_Ontology",
    "IDG_Drug_Targets_2022",
    "InterPro_Domains_2019",
    "JASPAR_PWM_Human_2025",
    "JASPAR_PWM_Mouse_2025",
    "Jensen_COMPARTMENTS",
    "Jensen_DISEASES",
    "Jensen_DISEASES_Curated_2025",
    "Jensen_DISEASES_Experimental_2025",
    "Jensen_TISSUES",
    "KEA_2013",
    "KEA_2015",
    "KEGG_2013",
    "KEGG_2015",
    "KEGG_2016",
    "KEGG_2019_Human",
    "KEGG_2019_Mouse",
    "KEGG_2021_Human",
    "KEGG_2026",
    "KOMP2_Mouse_Phenotypes_2022",
    "Kinase_Perturbations_from_GEO_down",
    "Kinase_Perturbations_from_GEO_up",
    "L1000_Kinase_and_GPCR_Perturbations_down",
    "L1000_Kinase_and_GPCR_Perturbations_up",
    "LINCS_L1000_CRISPR_KO_Consensus_Sigs",
    "LINCS_L1000_Chem_Pert_Consensus_Sigs",
    "LINCS_L1000_Chem_Pert_down",
    "LINCS_L1000_Chem_Pert_up",
    "LINCS_L1000_Ligand_Perturbations_down",
    "LINCS_L1000_Ligand_Perturbations_up",
    "Ligand_Perturbations_from_GEO_down",
    "Ligand_Perturbations_from_GEO_up",
    "MAGMA_Drugs_and_Diseases",
    "MAGNET_2023",
    "MCF7_Perturbations_from_GEO_down",
    "MCF7_Perturbations_from_GEO_up",
    "MGI_Mammalian_Phenotype_Level_4_2021",
    "MGI_Mammalian_Phenotype_Level_4_2024",
    "MSigDB_Computational",
    "MSigDB_Hallmark_2020",
    "MSigDB_Oncogenic_Signatures",
    "Metabolomics_Workbench_Metabolites_2022",
    "Microbe_Perturbations_from_GEO_down",
    "Microbe_Perturbations_from_GEO_up",
    "MoTrPAC_2023",
    "Mouse_Gene_Atlas",
    "NCI-60_Cancer_Cell_Lines",
    "NCI-Nature_2016",
    "NIBR_DRUGseq_2025_down",
    "NIBR_DRUGseq_2025_up",
    "NURSA_Human_Endogenous_Complexome",
    "OMIM_Disease",
    "OMIM_Expanded",
    "Old_CMAP_down",
    "Old_CMAP_up",
    "Orphanet_Augmented_2021",
    "PFOCR_Pathways_2023",
    "PPI_Hub_Proteins",
    "PanglaoDB_Augmented_2021",
    "Panther_2015",
    "Panther_2016",
    "PerturbAtlas",
    "PerturbAtlas_MouseGenePerturbationSigs",
    "PerturbSeq_ReplogleK562",
    "PerturbSeq_ReplogleRPE1",
    "Pfam_Domains_2019",
    "Pfam_InterPro_Domains",
    "PheWeb_2019",
    "PhenGenI_Association_2021",
    "Phosphatase_Substrates_from_DEPOD",
    "ProteomicsDB_2020",
    "Proteomics_Drug_Atlas_2023",
    "RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO",
    "Rare_Diseases_AutoRIF_ARCHS4_Predictions",
    "Rare_Diseases_AutoRIF_Gene_Lists",
    "Rare_Diseases_GeneRIF_ARCHS4_Predictions",
    "Rare_Diseases_GeneRIF_Gene_Lists",
    "Reactome_2022",
    "Reactome_Pathways_2024",
    "RummaGEO_DrugPerturbations_2025",
    "RummaGEO_GenePerturbations_2025",
    "Rummagene_kinases",
    "Rummagene_signatures",
    "Rummagene_transcription_factors",
    "SILAC_Phosphoproteomics",
    "Sciplex_Drug_Perturbation_Signatures_2025",
    "SubCell_BarCode",
    "SynGO_2022",
    "SynGO_2024",
    "SysMyo_Muscle_Gene_Sets",
    "TF-LOF_Expression_from_GEO",
    "TF_Perturbations_Followed_by_Expression",
    "TG_GATES_2020",
    "TISSUES_Curated_2025",
    "TISSUES_Experimental_2025",
    "TRANSFAC_and_JASPAR_PWMs",
    "TRRUST_Transcription_Factors_2019",
    "Table_Mining_of_CRISPR_Studies",
    "Tabula_Muris",
    "Tabula_Sapiens",
    "TargetScan_microRNA",
    "TargetScan_microRNA_2017",
    "The_Kinase_Library_2023",
    "The_Kinase_Library_2024",
    "Tissue_Protein_Expression_from_Human_Proteome_Map",
    "Tissue_Protein_Expression_from_ProteomicsDB",
    "Transcription_Factor_PPIs",
    "UK_Biobank_GWAS_v1",
    "Virus-Host_PPI_P-HIPSTer_2020",
    "VirusMINT",
    "Virus_Perturbations_from_GEO_down",
    "Virus_Perturbations_from_GEO_up",
    "WikiPathway_2021_Human",
    "WikiPathway_2023_Human",
    "WikiPathways_2013",
    "WikiPathways_2015",
    "WikiPathways_2016",
    "WikiPathways_2019_Human",
    "WikiPathways_2019_Mouse",
    "WikiPathways_2024_Human",
    "WikiPathways_2024_Mouse",
    "dbGaP",
    "huMAP",
    "lncHUB_lncRNA_Co-Expression",
    "miRTarBase_2017",
]


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    filepath: str
    name: str
    cluster_key: str
    species: str

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, v: str) -> str:
        if not Path(v).is_file():
            raise ValueError(f"filepath {v} is not a file")
        if v.split(".")[-1] != "h5ad":
            raise ValueError(f"filepath {v} does not have .h5ad extension")
        return v

    @field_validator("cluster_key")
    @classmethod
    def validate_cluster_key(cls, v: str, info: ValidationInfo) -> str:
        filepath: str | None = info.data.get("filepath")
        if filepath is None:
            return v
        valid_columns: list[str] = read_h5ad(filepath, backed="r").obs.columns.tolist()
        if v not in valid_columns:
            raise ValueError(
                f"cluster_key {v} is not in obs columns, must be one of {valid_columns}"
            )

        return v

    def load_anndata(self) -> AnnData:
        return read_h5ad(self.filepath)


@dataclass(
    frozen=True,
    slots=True,
    config=ConfigDict(validate_assignment=True),
)
class Config:
    n_components: PositiveInt
    n_iterations: PositiveInt
    n_samples: PositiveInt
    seed: NonNegativeInt
    preprocessing: Sequence[PreprocessingNames]
    database_names: Sequence[DatabaseName]
    datasets: Sequence[DatasetInfo]
    scoring_functions: Sequence[tuple[CorrelationNames | None, ScoringFunctionNames]]
    shuffle_probabilities: Sequence[NonNegativeFloat]
    output_csv_path: str


def get_gene_programs_db(
    species: str,
    name: DatabaseName,
) -> GeneProgramsDB:
    library: dict[str, list[str]] = gseapy.get_library(  # type: ignore
        organism=species,
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


def add_shuffle_noise_dense(
    counts: NumericArray,
    probability: float,
    rng: np.random.Generator | None,
) -> NumericArray:
    return (
        np.stack(
            [
                (rng or np.random.default_rng()).permutation(counts[:, i])
                if (rng or np.random.default_rng()).random() < probability
                else counts[:, i]
                for i in range(counts.shape[1])
            ],
            axis=1,
        )
        if probability != 0
        else counts
    )


def gini(arr: NumericArray) -> float:
    """Compute Gini coefficient of a 1D array."""
    if np.amin(arr) < 0:
        arr -= np.amin(arr)  # Values must be non-negative
    arr = np.sort(arr)
    index = np.arange(1, arr.shape[0] + 1)
    n = arr.shape[0]
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def factor_gene_correlations(
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


@memory.cache
def gseapy_enrichr(
    gene_list: tuple[str, ...],
    gene_sets: str,
    max_retries: int = 5,
) -> gseapy.Enrichr:
    """Run gseapy enrichr on a gene list and gene sets, returning the result.

    Retries up to max_retries times in case of network/API errors.
    """
    attempts = 0

    while attempts < max_retries:
        try:
            result = gseapy.enrichr(
                gene_list=list(gene_list),
                gene_sets=gene_sets,
            )
            return result

        except Exception as e:
            attempts += 1
            logging.warning(
                f"Attempt {attempts}/{max_retries} failed to enrichr gene list. Error: {e}"
            )

            if attempts >= max_retries:
                logging.error("Max retries reached. Raising exception.")
                raise e

            sleep(60)


def dominic_scoring(
    gene_list: Sequence[str],
    factor_gene_loadings: NumericArray,
    database_name: DatabaseName,
    gene_program_counts: NumericArray,
) -> float:
    n_genes = min(len(gene_list), 10)
    list_of_gene_lists: list[list[str]] = [
        [gene_list[i] for i in np.argpartition(factor, -n_genes)[-10:]]
        for factor in factor_gene_loadings
    ]
    scores = [
        float(
            gseapy_enrichr(
                gene_list=tuple(gene_list),
                gene_sets=database_name,
            )
            .results.sort_values(by="Adjusted P-value")
            .iloc[0]["Adjusted P-value"]
        )
        for gene_list in list_of_gene_lists
    ]
    return float(np.mean(scores))


def cosine_and_gini(
    adata: AnnData,
    factor_gene_loadings: NumericArray,
    database_name: DatabaseName,
    gene_program_counts: NumericArray,
) -> float:
    a = cosine_similarity(factor_gene_loadings, gene_program_counts)
    return gini(a)


_PREPROCESSING_REGISTRY: Mapping[
    PreprocessingNames, Callable[[NumericArray], NumericArray]
] = {
    "identity": lambda x: x,
    "log1p": np.log1p,
}
_SCORING_REGISTRY: Mapping[
    ScoringFunctionNames,
    Callable[[Sequence[str], NumericArray, DatabaseName, NumericArray], float],
] = {
    "dominic_scoring": dominic_scoring,
}

_CORRELATION_REGISTRY: Mapping[
    CorrelationNames, Callable[[NumericArray, NumericArray], float]
] = {  # type: ignore[assignment]
    "pearson": n2l.mt.pearson_metric,
    "spearman": n2l.mt.spearman_metric,
    "cosine_similarity": n2l.mt.cosine_similarity_metric,
}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("Loading config")
    with open("./config.yaml") as f:
        config = Config(**yaml.safe_load(f))
    # return 0
    rng = np.random.default_rng(config.seed)

    logger.info("Starting main loop")
    n_total_iterations = len(config.datasets) * len(config.database_names)
    for dataset, database_name in tqdm(
        product(
            config.datasets,
            config.database_names,
        ),
        total=n_total_iterations,
        desc="Processing datasets",
    ):
        adata = dataset.load_anndata()
        gene_programs_db = get_gene_programs_db(
            species=dataset.species,
            name=database_name,
        )
        gene_program_db_genes = extract_gene_list(gene_programs_db)
        if len(gene_program_db_genes) == 0:
            logger.warning(
                f"No genes found in gene program database {database_name}, Skipping"
            )
            continue
        gene_list: list[str] = np.intersect1d(
            adata.var_names,
            gene_program_db_genes,
        ).tolist()
        gene_program_counts = get_gene_program_counts(
            gene_programs_db=gene_programs_db,
            genes=gene_list,
        )
        for cluster_name, obs in adata.obs.groupby([dataset.cluster_key]):  # type: ignore
            for shuffle_probability in config.shuffle_probabilities:
                noisy_counts = add_shuffle_noise_dense(
                    counts=adata[obs.index, gene_list].X.toarray(),
                    probability=shuffle_probability,
                    rng=rng,
                )
                for preprocessing_name in config.preprocessing:
                    preprocessing_func = _PREPROCESSING_REGISTRY[preprocessing_name]
                    w: NumericArray
                    h: NumericArray
                    w, h, _ = non_negative_factorization(  # type: ignore
                        X=preprocessing_func(noisy_counts),
                        n_components=config.n_components,  # type: ignore
                    )
                    for scoring_function in config.scoring_functions:
                        correlation_name, scoring_function_name = scoring_function
                        factor_gene_loadings = (
                            factor_gene_correlations(
                                x=noisy_counts,
                                w=w,
                                corr_func=_CORRELATION_REGISTRY[correlation_name],
                            )
                            if correlation_name is not None
                            else h
                        )
                        # return (
                        #    adata,
                        #    gene_list,
                        #    factor_gene_loadings,
                        #    database_name,
                        #    gene_program_counts,
                        #    dataset,
                        # )
                        score = _SCORING_REGISTRY[scoring_function_name](
                            sorted(gene_list),
                            factor_gene_loadings,
                            database_name,
                            gene_program_counts,
                        )


if __name__ == "__main__":
    main()
