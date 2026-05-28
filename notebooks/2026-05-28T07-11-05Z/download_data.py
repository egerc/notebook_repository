import logging
import subprocess
from pathlib import Path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Creating data directory...")
    save_dir = Path().cwd() / "data"
    save_dir.mkdir(parents=True, exist_ok=True)
    sce_follicular_path = save_dir / "sce_follicular.h5ad"
    sce_hgsc_path = save_dir / "sce_hgsc.h5ad"
    fl_celltype_path = save_dir / "fl_celltype.csv"
    hgsc_celltype_path = save_dir / "hgsc_celltype.csv"
    logger.info("Downloading data...")
    logger.info("Downloading sce_follicular.h5ad...")
    subprocess.run(
        [
            "wget",
            "-q",
            "https://ndownloader.figshare.com/files/27458798",
            "-O",
            str(sce_follicular_path),
        ]
    )
    logger.info(
        "Finished downloading sce_follicular.h5ad, Downloading sce_hgsc.h5ad..."
    )
    subprocess.run(
        [
            "wget",
            "-q",
            "https://ndownloader.figshare.com/files/27458822",
            "-O",
            str(sce_hgsc_path),
        ]
    )
    logger.info("Finished downloading sce_hgsc.h5ad, Downloading fl_celltype.csv...")
    subprocess.run(
        [
            "wget",
            "-q",
            "https://ndownloader.figshare.com/files/27458828",
            "-O",
            str(hgsc_celltype_path),
        ]
    )
    logger.info(
        "Finished downloading hgsc_celltype.csv, Downloading fl_celltype.csv..."
    )
    subprocess.run(
        [
            "wget",
            "-q",
            "https://ndownloader.figshare.com/files/27458831",
            "-O",
            str(fl_celltype_path),
        ]
    )
    logger.info("Finished downloading fl_celltype.csv, Exiting...")


if __name__ == "__main__":
    main()
