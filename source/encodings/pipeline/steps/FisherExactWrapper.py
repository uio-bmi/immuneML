import sys

from source.util.ReflectionHandler import ReflectionHandler


class FisherExactWrapper:

    TWO_SIDED = {"fisher_exact_scipy": "two-sided", "fisher_exact_custom": "two_tail"}
    LESS = {"fisher_exact_scipy": "less"}
    GREATER = {"fisher_exact_scipy": "greater"}

    def __init__(self):
        if sys.version_info > (3, 6):
            self.fisher_exact = self.fisher_exact_scipy
        else:
            self.fisher_exact = self.fisher_exact_custom

    def fisher_exact_scipy(self, table, alternative: dict):
        import scipy.stats as stats
        odds_ratio, p_value = stats.fisher_exact(table, alternative["fisher_exact_scipy"])
        return p_value

    def fisher_exact_custom(self, table, alternative: dict):
        fisher = ReflectionHandler.import_module("fisher")
        p = fisher.pvalue(table[1, 1], table[0, 1], table[1, 0], table[0, 0])
        return p[alternative["fisher_exact_custom"]]
