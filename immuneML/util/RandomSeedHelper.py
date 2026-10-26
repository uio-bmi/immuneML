import logging
import random
from pathlib import Path

import numpy as np
import yaml

from immuneML.util.Logger import print_log
from immuneML.util.ParameterValidator import ParameterValidator


class RandomSeedHelper:

    @staticmethod
    def set_seed(seed: int):
        """Seeds Python's random, numpy.random and, if installed, torch."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

    @staticmethod
    def set_seed_from_specification(specification_path: Path, location: str):
        """Seeds Python's random, numpy.random and, if installed, torch from the top-level `random_seed` key of the
        specification. If the key is absent, nothing is seeded."""
        with Path(specification_path).open("r") as file:
            specification = yaml.safe_load(file)

        seed = specification.get("random_seed") if isinstance(specification, dict) else None
        if seed is None:
            logging.info(f"{location}: no random_seed was set in the specification, so the analysis is not seeded.")
            return

        assert not isinstance(seed, bool), f"{location}: random_seed has to be an integer, got {seed}."
        ParameterValidator.assert_type_and_value(seed, int, location, "random_seed", min_inclusive=0,
                                                 max_inclusive=2 ** 32 - 1)

        RandomSeedHelper.set_seed(seed)

        print_log(f"{location}: random seed set to {seed}.\n", include_datetime=True)
