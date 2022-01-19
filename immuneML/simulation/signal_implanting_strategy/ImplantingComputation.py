from enum import Enum

import numpy as np


class ImplantingComputation(Enum):

    ROUND = 'round'
    POISSON = 'Poisson'
    BINOMIAL = "binomial"


def get_implanting_function(implanting_computation: ImplantingComputation):
    if implanting_computation == ImplantingComputation.ROUND:
        return lambda repertoire_implanting_rate, number_of_sequences: round(repertoire_implanting_rate * number_of_sequences)
    elif implanting_computation == ImplantingComputation.POISSON:
        return lambda repertoire_implanting_rate, number_of_sequences: np.random.poisson(repertoire_implanting_rate * number_of_sequences)
    elif implanting_computation == ImplantingComputation.BINOMIAL:
        return lambda repertoire_implanting_rate, number_of_sequences: np.random.binomial(n=number_of_sequences, p=repertoire_implanting_rate)
    else:
        raise RuntimeError(f"{ImplantingComputation.__name__}: invalid implanting computation specified: {implanting_computation}. "
                           f"Valid values are: {[el.name.lower() for el in ImplantingComputation]}")