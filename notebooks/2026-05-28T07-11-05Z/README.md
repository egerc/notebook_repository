# Marker Gene Score Experiment

This experiment evaluates how well marker gene scores can distinguish between true cell type annotations and shuffled (random) annotations across different shuffling probabilities.

## Overview

The experiment:

1. Loads single-cell datasets
2. Randomly samples cells from each dataset
3. Shuffles cell type labels at various probabilities (0.0 to 1.0)
4. Scores each shuffle using marker gene-based scoring
5. Saves results to `results.csv`



## Project Structure

```
/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-05-28T07-11-05Z/
├── config.yaml          # Experiment configuration
├── download_data.py     # Script to download data from Figshare
├── convert_marker.py    # Convert marker CSV to JSON format
├── experiment.py        # Main experiment script
├── pyproject.toml       # Python dependencies
├── uv.lock              # Locked dependency versions
├── data/                # Downloaded data files
│   ├── sce_follicular.h5ad
│   ├── sce_hgsc.h5ad
│   ├── fl_celltype.csv
│   ├── fl_celltype.json
│   ├── hgsc_celltype.csv
│   └── hgsc_celltype.json
├── results.csv          # Experiment results
```

## Setup

### 1. Navigate to the Experiment Directory

```bash
cd /home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-05-28T07-11-05Z
```

### 2. Install Dependencies

This project uses `uv` for dependency management:

```bash
uv sync
```

### 3. Download Data

Run the download script to fetch data from Figshare:

```bash
uv run python download_data.py
```

This will download:

- `data/sce_follicular.h5ad`
- `data/sce_hgsc.h5ad`
- `data/fl_celltype.csv`
- `data/hgsc_celltype.csv`

### 4. Convert Marker Data (if starting from CSV)

Convert marker data from CSV to JSON format:

```bash
uv run python convert_marker.py data/fl_celltype.csv data/fl_celltype.json
uv run python convert_marker.py data/hgsc_celltype.csv data/hgsc_celltype.json
```

## Running the Experiment

### Local Run

```bash
uv run python experiment.py config.yaml
```

### SLURM Cluster Run

```bash
sbatch run_experiment
```

## Configuration

Edit `config.yaml` to modify:

```yaml
datasets:
    - name: Follicular
      h5ad: path/to/sce_follicular.h5ad
      targets:
          - [celltype, [path/to/fl_celltype.json]]

    - name: HGSC
      h5ad: path/to/sce_hgsc.h5ad
      targets:
          - [celltype, [path/to/hgsc_celltype.json]]

n_samples: 20           # Number of random samples to draw
n_cells_per_sample: 500 # Number of cells per sample
shuffle_probabilities: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
seed: 0                  # Random seed for reproducibility
```

## Output

Results are saved to `results.csv` with columns:

- `dataset`: Dataset name
- `sample_id`: Sample index (0 to n_samples-1)
- `cluster_key`: Column used for cell type annotations
- `marker_json`: Path to marker gene JSON file
- `shuffle_probability`: Probability of shuffling labels
- `score_name`: Name of scoring function used
- `score`: Computed score

## Extending the Experiment

To add new scoring methods, register them in `SCORE_REGISTRY` in `experiment.py`:

```python
def my_custom_score(
    adata: AnnData,
    cluster_key: str,
    marker_dict: dict[str, list[str]],
) -> float:
    # Your scoring logic here
    return score

SCORE_REGISTRY: dict[str, Callable[...]] = {
    "dummy": dummy_score,
    "my_custom": my_custom_score,  # Add your new scorer
}
```