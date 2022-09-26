from enum import Enum

import numpy as np


class ImplantingComputation(Enum):

    ROUND = 'round'
    POISSON = 'Poisson'
    BINOMIAL = "binomial"


def get_implanting_function(implanting_computation: ImplantingComputation):
    if implanting_computation == ImplantingComputation.ROUND:
        return compute_round
    elif implanting_computation == ImplantingComputation.BINOMIAL:
        return compute_binomial
    elif implanting_computation == ImplantingComputation.POISSON:
        return compute_poisson
    else:
        raise RuntimeError(f"{ImplantingComputation.__name__}: invalid implanting computation specified: {implanting_computation}. "
                           f"Valid values are: {[el.name.lower() for el in ImplantingComputation]}")


def compute_round(repertoire_implanting_rate, number_of_sequences):
    return round(repertoire_implanting_rate * number_of_sequences)


def compute_poisson(repertoire_implanting_rate, number_of_sequences):
    return np.random.poisson(repertoire_implanting_rate * number_of_sequences)


def compute_binomial(repertoire_implanting_rate, number_of_sequences):
    return np.random.binomial(n=number_of_sequences, p=repertoire_implanting_rate)