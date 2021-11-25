import copy

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.Implanting import Implanting
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


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
            export_formats: [AIRR, ImmuneML]

    """

    @staticmethod
    def parse_simulations(simulations: dict, symbol_table: SymbolTable):
        for key, simulation in simulations.items():
            symbol_table = SimulationParser._parse_simulation(key, simulation, symbol_table)

        return symbol_table, simulations

    @staticmethod
    @log
    def _parse_simulation(key: str, simulation: dict, symbol_table: SymbolTable) -> SymbolTable:

        location = SimulationParser.__name__

        simulation_items = []

        for sim_key, simulation_item in simulation.items():

            ParameterValidator.assert_keys_present(simulation_item.keys(), ['type'], location, 'simulation')

            if simulation_item['type'] == 'Implanting':
                item = SimulationParser._parse_implanting(simulation_item, sim_key, symbol_table)
            elif simulation_item['type'] == 'LIgOSimulationItem':
                item = SimulationParser._parse_simulation_item(simulation_item, sim_key, symbol_table)
            else:
                raise ValueError(f"{location}: in simulation {key}, for simulation item {sim_key}, the type was set to {simulation_item['type']},"
                                 f" but only 'Implanting' and 'LIgOSimulationItem' are supported.")

            simulation_items.append(item)

        assert all(isinstance(item, type(simulation_items[0])) for item in simulation_items), "All simulation items must be of the same type."

        symbol_table.add(key, SymbolType.SIMULATION, Simulation(simulation_items, identifier=key))

        return symbol_table

    @staticmethod
    def _parse_implanting(simulation_item: dict, key: str, symbol_table: SymbolTable) -> Implanting:
        location = SimulationParser.__name__
        valid_simulation_item_keys = ["dataset_implanting_rate", "repertoire_implanting_rate", "signals", "is_noise", "type"]

        ParameterValidator.assert_keys(simulation_item.keys(), valid_simulation_item_keys, location, key, exclusive=False)
        ParameterValidator.assert_keys(simulation_item["signals"], symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, key, False)

        implanting_params = copy.deepcopy(simulation_item)
        implanting_params["signals"] = [symbol_table.get(signal) for signal in simulation_item["signals"]]
        implanting_params["name"] = key

        return Implanting(**implanting_params)

    @staticmethod
    def _parse_simulation_item(simulation_item: dict, key: str, symbol_table: SymbolTable) -> LIgOSimulationItem:
        location = SimulationParser.__name__
        valid_simulation_item_keys = ["number_of_examples", "repertoire_implanting_rate", "signals", "is_noise",
                                      "number_of_receptors_in_repertoire", "generative_model", "type"]

        ParameterValidator.assert_keys(simulation_item.keys(), valid_simulation_item_keys, location, key, exclusive=True)
        ParameterValidator.assert_keys(simulation_item["signals"], symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, key, False)
        ParameterValidator.assert_type_and_value(simulation_item['is_noise'], bool, location, 'is_noise')
        ParameterValidator.assert_type_and_value(simulation_item['number_of_examples'], int, location, 'number_of_examples', min_inclusive=1)

        if simulation_item['repertoire_implanting_rate']:
            ParameterValidator.assert_type_and_value(simulation_item['repertoire_implanting_rate'], float, location, 'repertoire_implanting_rate', 0, 1)

        if simulation_item['number_of_receptors_in_repertoire']:
            ParameterValidator.assert_type_and_value(simulation_item['number_of_receptors_in_repertoire'], int, location,
                                                     'number_of_receptors_in_repertoire', 1)

        gen_model = SimulationParser._parse_generative_model(simulation_item, location)

        params = copy.deepcopy(simulation_item)
        params["signals"] = [symbol_table.get(signal) for signal in simulation_item["signals"]]
        params["name"] = key
        params['generative_model'] = gen_model

        return LIgOSimulationItem(**{key: val for key, val in params.items() if key != 'type'})

    @staticmethod
    def _parse_generative_model(simulation_item: dict, location: str):
        ParameterValidator.assert_type_and_value(simulation_item['generative_model'], dict, location, 'generative_model')
        ParameterValidator.assert_keys_present(simulation_item['generative_model'].keys(), ['type'], location, 'generative_model')

        gen_model_classes = ReflectionHandler.all_nonabstract_subclass_basic_names(GenerativeModel, "", "simulation/generative_models/")
        ParameterValidator.assert_in_valid_list(simulation_item['generative_model']['type'], gen_model_classes, location, "generative_model:type")

        gen_model_cls = ReflectionHandler.get_class_by_name(simulation_item['generative_model']['type'], "simulation/generative_models/")
        params = {key: val for key, val in simulation_item['generative_model'].items() if key != 'type'}
        gen_model = gen_model_cls.build_object(**params)

        return gen_model
