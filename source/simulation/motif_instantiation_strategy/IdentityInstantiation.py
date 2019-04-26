from source.simulation.implants.MotifInstance import MotifInstance
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy


class IdentityInstantiation(MotifInstantiationStrategy):
    """
    Motif instantiation strategy which always return the seed
    """

    def get_max_gap(self) -> int:
        return 0

    def instantiate_motif(self, base, params: dict = None) -> MotifInstance:
        assert isinstance(base, str)
        return MotifInstance(base, 0)
