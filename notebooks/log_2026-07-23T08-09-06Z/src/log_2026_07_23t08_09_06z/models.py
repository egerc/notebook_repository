import nico2_lib as n2l
from pydantic.dataclasses import dataclass
from pydantic.types import PositiveInt


@dataclass(frozen=True, slots=True)
class Nmf:
    n_components: PositiveInt


@dataclass(frozen=True, slots=True)
class ScviAutoencoder:
    pass


@dataclass(frozen=True, slots=True)
class ScviAutoencoderSubset:
    latent_size: PositiveInt


def main():
    n2l.pd.NmfPredictor()
