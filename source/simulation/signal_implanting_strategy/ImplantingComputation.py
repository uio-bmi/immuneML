from enum import Enum

import numpy as np


class ImplantingComputation(Enum):

    ROUND = 'round'
    POISSON = 'Poisson'


def get_implanting_function(implanting_computation: ImplantingComputation):
    if implanting_computation == ImplantingComputation.ROUND:
        return lambda product: round(product)
    elif implanting_computation == ImplantingComputation.POISSON:
        return lambda l: np.random.poisson(l)
    else:
        raise RuntimeError(f"{ImplantingComputation.__name__}: invalid implanting computation specified: {implanting_computation}. "
                           f"Valid values are: {[el.name.lower() for el in ImplantingComputation]}")