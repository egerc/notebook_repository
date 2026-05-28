import json
from argparse import ArgumentParser, Namespace

import pandas as pd


def pandas_to_dict(df: pd.DataFrame) -> dict[str, list[str]]:
    df = df.set_index("Gene")
    marker_map = {}
    for cell_type in df.columns:
        marker_map[cell_type] = df.index[df[cell_type] == 1].tolist()
    return marker_map


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    marker_dict = pandas_to_dict(df)
    with open(args.output_json, "w") as f:
        json.dump(marker_dict, f, indent=4)


if __name__ == "__main__":
    main()
