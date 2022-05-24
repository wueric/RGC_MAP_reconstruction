import numpy as np

from typing import Tuple
from dataclasses import dataclass


@dataclass
class HQSHyperparameters:

    rho_start: float
    rho_end: float
    prior_weight: float
    max_iter: int


def glm_hqs_hyperparameters_2018_08_07_5() -> HQSHyperparameters:
    return HQSHyperparameters(0.215, 100.0, 0.1, 25)


def lnp_hqs_hyperparameters_2018_08_07_5() -> HQSHyperparameters:
    return HQSHyperparameters(0.1, 31.6227, 0.09, 25)


def make_hqs_schedule(hyperparams: HQSHyperparameters) -> np.ndarray:
    rho_schedule = np.logspace(np.log10(hyperparams.rho_start), np.log10(hyperparams.rho_end), hyperparams.max_iter)
    return rho_schedule


@dataclass
class ExactMAPHyperparameters:

    prior_weight: float
    patch_height: int
    patch_width: int


def glm_1F_exact_MAP_hyperparameters_2018_08_07_5() -> ExactMAPHyperparameters:
    return ExactMAPHyperparameters(prior_weight=0.15811388300841894, patch_height=64, patch_width=64)
