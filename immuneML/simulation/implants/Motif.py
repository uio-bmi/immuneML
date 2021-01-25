# quality: gold
from dataclasses import dataclass

from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from immuneML.util.ReflectionHandler import ReflectionHandler
from scripts.specification_util import update_docs_per_mapping


@dataclass
class Motif:
    """
    Class describing motifs where each motif is defined by a seed and
    a way of creating specific instances of the motif (instantiation_strategy);

    When instantiation_strategy is set, specific motif instances will be
    produced by calling instantiate_motif(seed) method of instantiation_strategy


    Arguments:

        seed (str): An amino acid sequence that represents the basic motif seed. All implanted motifs correspond to the seed, or a modified
        version thereof, as specified in it's instantiation strategy. If this argument is set, seed_chain1 and seed_chain2 arguments are not used.

        instantiation (:py:obj:`~immuneML.simulation.motif_instantiation_strategy.MotifInstantiationStrategy.MotifInstantiationStrategy`):
        Which strategy to use for implanting the seed. It should be one of the classes inheriting MotifInstantiationStrategy.
        In the YAML specification this can either be one of these values as a string in which case the default parameters will be used.
        Alternatively, instantiation can be specified with parameters as in the example YAML specification below. For the detailed list of
        parameters, see the specific instantiation strategies below.

        seed_chain1 (str): in case when representing motifs for paired chain data, it is possible to define a motif seed per chain; if this parameter
        is set, the generated motif instances will include a motif instance for both chains; for more details on how it works see `seed` argument
        above. Used only if the seed argument is not set.

        seed_chain2 (str): used for paired chain data, for the other receptor chain; for more details on how it works see `seed` argument. This
        argument is used only if the seed argument is not set.

        name_chain1: name of the first chain if paired receptor data are simulated. The value should be an instance of
        :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`. This argument is used only if the seed argument is not set.

        name_chain2: name of the second chain 2 if paired receptor data are simulated. The value should be an instance of
        :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`. This argument is used only if the seed argument is not set.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        motifs:
            # examples for single chain receptor data
            my_simple_motif: # this will be the identifier of the motif
                seed: AAA
                instantiation: GappedKmer
            my_gapped_motif:
                seed: AA/A
                instantiation:
                    GappedKmer:
                        min_gap: 1
                        max_gap: 2
            # examples for paired chain receptor data
            my_paired_motif:
                seed_chain1: AAA # seed for chain1 or chain2 can optionally include gap, same as for single chain receptor data
                name_chain1: ALPHA # alpha chain of TCR
                seed_chain2: CCC
                name_chain2: BETA # beta chain of TCR
                instantiation: GappedKmer # same as for single chain receptor data

    """

    identifier: str
    instantiation: MotifInstantiationStrategy
    seed: str = None
    seed_chain1: str = None
    name_chain1: Chain = None
    seed_chain2: str = None
    name_chain2: Chain = None

    def instantiate_motif(self, chain_name: Chain = None):
        """
        Creates a motif instance based on the seed; if seed parameter is defined for the motif, it is assumed that single chain data are used for
        the analysis. If seed is None, then it is assumed that paired chain receptor data are required in which case this function will return a
        motif instance per chain along with the names of the chains

        Returns:
             a motif instance if single chain immune receptor data are simulated or a dict where keys are chain names and values are motif instances
             for the corresponding chains
        """
        assert self.instantiation is not None, "Motif: set instantiation strategy before instantiating a motif."
        # TODO: handle PWMs also, here it always uses seed
        if self.seed is not None:
            return self.instantiation.instantiate_motif(self.seed)
        else:
            assert self.name_chain1 is not None and self.name_chain2 is not None, \
                f"Motif: chain names have to be set when working with paired chain data, here these are: {self.name_chain1} and {self.name_chain2}."
            assert chain_name is not None, "Motif: when working with paired chain data, please specify the chain for which the motif is instantiated."
            assert chain_name in [self.name_chain1, self.name_chain2], \
                f"Motif: specified chain name {chain_name.name.lower()} is not in valid list of chain names specified for motif {self.identifier}: " \
                f"{[self.name_chain1.name.lower(), self.name_chain2.name.lower()]}."

            return self.instantiation.instantiate_motif(self.seed_chain1 if chain_name == self.name_chain1 else self.seed_chain2)

    def get_max_length(self):
        if self.seed is not None:
            return len(self.seed.replace("/", "")) + self.instantiation.get_max_gap()
        else:
            return max(len(self.seed_chain1.replace("/", "")), len(self.seed_chain2.replace("/", ""))) + self.instantiation.get_max_gap()

    def __str__(self):
        return self.identifier + " - " + \
               (self.seed if self.seed is not None else f"{self.name_chain1}_{self.seed_chain1}__{self.name_chain2}_{self.seed_chain2}")

    @staticmethod
    def get_documentation():
        doc = str(Motif.__doc__)

        valid_strategy_values = ReflectionHandler.all_nonabstract_subclass_basic_names(MotifInstantiationStrategy, "Instantiation",
                                                                                       "motif_instantiation_strategy/")
        valid_strategy_values = str(valid_strategy_values)[1:-1].replace("'", "`")
        chain_values = str([name for name in Chain])[1:-1].replace("'", "`")
        mapping = {
            "It should be one of the classes inheriting MotifInstantiationStrategy.": f"Valid values are: {valid_strategy_values}.",
            "The value should be an instance of :py:obj:`~immuneML.data_model.receptor.receptor_sequence.Chain.Chain`.":
                f"Valid values are: {chain_values}."
        }
        doc = update_docs_per_mapping(doc, mapping)
        return doc

