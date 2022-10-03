import copy

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.Implanting import Implanting
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationStrategy import SimulationStrategy
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
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
        location = SimulationParser.__name__
        for key, simulation in simulations.items():

            ParameterValidator.assert_keys_present(simulation.keys(), ['type'], location, 'simulation')

            if simulation['type'] == 'Implanting':
                item = SimulationParser._parse_implanting(simulation, key, symbol_table)
            elif simulation['type'] == 'LIgOSimulation':
                item = SimulationParser._parse_ligo_simulation(simulation, key, symbol_table)
            else:
                raise ValueError(f"{location}: in simulation {key}, for simulation item {key}, the type was set to {simulation['type']},"
                                 f" but only 'Implanting' and 'LIgOSimulation' are supported.")

            symbol_table.add(key, SymbolType.SIMULATION, item)

        return symbol_table, simulations

    @staticmethod
    def _parse_implanting(simulation_item: dict, key: str, symbol_table: SymbolTable) -> Simulation:
        location = SimulationParser.__name__
        valid_simulation_item_keys = ["dataset_implanting_rate", "repertoire_implanting_rate", "signals", "is_noise", "type"]

        ParameterValidator.assert_keys(list(simulation_item.keys()), valid_simulation_item_keys, location, key, exclusive=False)
        ParameterValidator.assert_keys(simulation_item["signals"], symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, key, False)

        implanting_params = copy.deepcopy(simulation_item)
        implanting_params["signals"] = [symbol_table.get(signal) for signal in simulation_item["signals"]]
        implanting_params["name"] = key
        del implanting_params['type']

        return Simulation(sim_items=[Implanting(**implanting_params)])

    @staticmethod
    def _parse_ligo_simulation(simulation: dict, key: str, symbol_table: SymbolTable) -> Simulation:
        location = SimulationParser.__name__
        valid_keys = {'is_repertoire': bool, 'paired': bool, 'sequence_type': str, 'use_generation_probabilities': bool,
                      'simulation_strategy': str, 'sim_items': dict, 'type': str}

        simulation = {**DefaultParamsLoader.load("simulation", "ligo_simulation"), **simulation}

        ParameterValidator.assert_keys(simulation.keys(), valid_keys.keys(), location, key, exclusive=True)
        for k, val_type in valid_keys.items():
            ParameterValidator.assert_type_and_value(simulation[k], val_type, location, k)

        ParameterValidator.assert_in_valid_list(simulation['sequence_type'].upper(), [st.name for st in SequenceType], location, 'sequence_type')
        ParameterValidator.assert_in_valid_list(simulation['simulation_strategy'].upper(), [item.name for item in SimulationStrategy], location,
                                                'simulation_strategy')

        sim_items = []
        for sim_key, item in simulation['sim_items'].items():
            sim_items.append(SimulationParser._parse_ligo_sim_item(item, sim_key, symbol_table))

        sim_obj = Simulation(**{**{k: v for k, v in simulation.items() if k != 'type'},
                                **{'sequence_type': SequenceType[simulation['sequence_type'].upper()], "sim_items": sim_items,
                                   "identifier": key, 'simulation_strategy': SimulationStrategy[simulation['simulation_strategy'].upper()]}})

        SimulationParser._signal_content_matches_seq_type(sim_obj)
        return sim_obj

    @staticmethod
    def _parse_ligo_sim_item(simulation_item: dict, key: str, symbol_table: SymbolTable) -> LIgOSimulationItem:
        location = SimulationParser.__name__
        valid_simulation_item_keys = ["number_of_examples", "repertoire_implanting_rate", "signals", "is_noise", "seed",
                                      "number_of_receptors_in_repertoire", "generative_model"]

        simulation_item = {**DefaultParamsLoader.load('simulation', 'ligo_simulation_item'), **simulation_item}

        ParameterValidator.assert_keys(simulation_item.keys(), valid_simulation_item_keys, location, key, exclusive=True)
        ParameterValidator.assert_keys(simulation_item["signals"], symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, key, False)
        ParameterValidator.assert_type_and_value(simulation_item['is_noise'], bool, location, 'is_noise')

        for key in ['number_of_examples', 'seed']:
            ParameterValidator.assert_type_and_value(simulation_item[key], int, location, key, min_inclusive=1)

        for key, val_type in zip(['repertoire_implanting_rate', 'number_of_receptors_in_repertoire'], [float, int]):
            if simulation_item[key]:
                ParameterValidator.assert_type_and_value(simulation_item[key], val_type, location, key)

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

    @staticmethod
    def _signal_content_matches_seq_type(simulation: Simulation):
        for sim_item in simulation.sim_items:
            for signal in sim_item.signals:
                for motif in signal.motifs:
                    ParameterValidator.assert_all_in_valid_list([letter for letter in motif.seed if letter != '/'],
                                                                EnvironmentSettings.get_sequence_alphabet(simulation.sequence_type),
                                                                SimulationParser.__name__, f"motif seed {motif.seed}")

