import typer

import scoring_benchmark

app = typer.Typer()


@app.command("enrichment_experiment")
def biological_enrichment_experiment() -> None:
    print("running biological enrichment experiment")


@app.command("sparsity_experiment")
def sparsity_experiment() -> None:
    print("running sparsity experiment")


@app.command("embedding_structure")
def embedding_structure() -> None:
    print("running embedding structure experiment")


@app.command("feature_prediction")
def feature_prediction() -> None:
    print("running feature prediction experiment")


if __name__ == "__main__":
    app()
