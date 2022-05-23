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


def make_hqs_schedule(hyperparams: HQSHyperparameters) -> np.ndarray:
    rho_schedule = np.logspace(np.log10(hyperparams.rho_start), np.log10(hyperparams.rho_end), hyperparams.max_iter)
    return rho_schedule

