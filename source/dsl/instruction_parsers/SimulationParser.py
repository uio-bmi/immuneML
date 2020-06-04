from source.IO.dataset_export.DataExporter import DataExporter
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler
from source.workflows.instructions.SimulationInstruction import SimulationInstruction


class SimulationParser:

    """

    Specification:

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
                        params:
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
                implanting: HealthySequences # choose only sequences with no other signals for to implant one of the motifs
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
            export_format: AIRR

    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable) -> SimulationInstruction:

        ParameterValidator.assert_keys(instruction.keys(), ["dataset", "batch_size", "simulation", "type", "export_format"],
                                       "SimulationParser", key)

        signals = [signal.item for signal in symbol_table.get_by_type(SymbolType.SIGNAL)]
        simulation = symbol_table.get(instruction["simulation"])
        dataset = symbol_table.get(instruction["dataset"])
        batch_size = instruction["batch_size"]

        exporter = self.parse_exporter(instruction)

        process = SimulationInstruction(signals=signals, simulation=simulation, dataset=dataset, batch_size=batch_size, name=key,
                                        exporter=exporter)
        return process

    def parse_exporter(self, instruction):
        if instruction["export_format"] is not None:
            class_path = "dataset_export/"
            classes = ReflectionHandler.get_classes_by_partial_name("Exporter", class_path)
            valid_values = [cls.__name__[:-8] for cls in ReflectionHandler.all_nonabstract_subclasses(DataExporter)]
            ParameterValidator.assert_in_valid_list(instruction["export_format"], valid_values, "SimulationParser", "export_format")
            exporter = ReflectionHandler.get_class_by_name(f"{instruction['export_format']}Exporter", class_path)
        else:
            exporter = None

        return exporter
