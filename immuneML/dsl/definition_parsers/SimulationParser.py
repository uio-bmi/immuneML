import copy

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.Implanting import Implanting
from immuneML.simulation.Simulation import Simulation
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator


class SimulationParser:
    """
    YAML specification:

    .. highlight:: yaml
    .. code-block:: yaml

    definitions:
        dataset:
            my_dataset:
                ...

        motifs:
            m1:
                seed: AAC # "/" character denotes the gap in the seed if present (e.g. AA/C)
                instantiation:
                    GappedKmer:
                        # probability that when hamming distance is allowed a letter in the seed will be replaced by
                        # other alphabet letters - alphabet_weights
                        alphabet_weights:
                            A: 0.2
                            C: 0.2
                            D: 0.4
                            E: 0.2
                        # Relative probabilities of choosing each position in the seed for hamming distance modification.
                        # The probabilities will be scaled to sum to one - position_weights
                        position_weights:
                            0: 1
                            1: 0
                            2: 0
                        hamming_distance_probabilities:
                            0: 0.5 # Hamming distance of 0 (no change) with probability 0.5
                            1: 0.5 # Hamming distance of 1 (one letter change) with probability 0.5
                        min_gap: 0
                        max_gap: 1
        signals:
            s1:
                motifs: # list of all motifs for signal which will be uniformly sampled to get a motif instance for implanting
                    - m1
                sequence_position_weights: # likelihood of implanting at IMGT position of receptor sequence
                    107: 0.5
                implanting: HealthySequence # choose only sequences with no other signals for to implant one of the motifs
        simulations:
            sim1: # one Simulation object consists of a dict of Implanting objects
                i1:
                    dataset_implanting_rate: 0.5 # percentage of repertoire where the signals will be implanted
                    repertoire_implanting_rate: 0.01 # percentage of sequences within repertoire where the signals will be implanted
                    signals:
                        - s1

    instructions:
        my_simulation_instruction:
            type: Simulation
            dataset: my_dataset
            simulation: sim1
            batch_size: 5 # number of repertoires that can be loaded at the same time
                          # (only affects the speed)
            export_formats: [AIRR, Pickle]

    """

    @staticmethod
    def parse_simulations(simulations: dict, symbol_table: SymbolTable):
        for key, simulation in simulations.items():
            symbol_table = SimulationParser._parse_simulation(key, simulation, symbol_table)

        return symbol_table, simulations

    @staticmethod
    @log
    def _parse_simulation(key: str, simulation: dict, symbol_table: SymbolTable) -> SymbolTable:

        location = "SimulationParser"
        valid_implanting_keys = ["dataset_implanting_rate", "repertoire_implanting_rate", "signals", "is_noise"]
        implantings = []

        for impl_key, implanting in simulation.items():

            ParameterValidator.assert_keys(implanting.keys(), valid_implanting_keys, location, impl_key, exclusive=False)
            ParameterValidator.assert_keys(implanting["signals"], symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, impl_key, False)

            implanting_params = copy.deepcopy(implanting)
            implanting_params["signals"] = [symbol_table.get(signal) for signal in implanting["signals"]]
            implanting_params["name"] = impl_key

            implantings.append(Implanting(**implanting_params))

        assert sum([settings["dataset_implanting_rate"] for settings in simulation.values()]) <= 1, \
            "The total dataset implanting rate can not exceed 1."

        symbol_table.add(key, SymbolType.SIMULATION, Simulation(implantings))

        return symbol_table
