import copy
from typing import Tuple

from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfig import SimConfig
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class SimulationParser:

    keyword = "simulations"

    @staticmethod
    def parse(simulations: dict, symbol_table: SymbolTable):
        for key, simulation in simulations.items():

            item, sim_dict = SimulationParser._parse_ligo_simulation(simulation, key, symbol_table)

            symbol_table.add(key, SymbolType.SIMULATION, item)
            simulations[key] = sim_dict

        return symbol_table, simulations

    @staticmethod
    def _parse_ligo_simulation(simulation: dict, key: str, symbol_table: SymbolTable) -> Tuple[SimConfig, dict]:
        location = SimulationParser.__name__
        valid_keys = {'is_repertoire': bool, 'paired': bool, 'sequence_type': str, 'p_gen_bin_count': int, 'simulation_strategy': str,
                      'sim_items': dict, 'keep_p_gen_dist': bool, 'remove_seqs_with_signals': bool}

        simulation = {**DefaultParamsLoader.load("simulation", "ligo_sim_config"), **simulation}

        ParameterValidator.assert_keys(simulation.keys(), valid_keys.keys(), location, key, exclusive=True)
        for k, val_type in valid_keys.items():
            ParameterValidator.assert_type_and_value(simulation[k], val_type, location, k)

        sim_strategies = ReflectionHandler.all_nonabstract_subclass_basic_names(SimulationStrategy, drop_part='Strategy', subdirectory='simulation/simulation_strategy')

        ParameterValidator.assert_in_valid_list(simulation['sequence_type'].upper(), [st.name for st in SequenceType], location, 'sequence_type')
        ParameterValidator.assert_in_valid_list(simulation['simulation_strategy'], sim_strategies, location, 'simulation_strategy')

        sim_strategy_cls = ReflectionHandler.get_class_by_name(simulation['simulation_strategy'] + "Strategy", "simulation/simulation_strategy")

        sim_items = []
        for sim_key, item in simulation['sim_items'].items():
            sim_item, sim_item_dict = SimulationParser._parse_sim_config_item(item, sim_key, symbol_table)
            sim_items.append(sim_item)
            simulation['sim_items'][sim_key] = sim_item_dict

        sim_obj = SimConfig(**{**{k: v for k, v in simulation.items() if k != 'type'},
                               **{'sequence_type': SequenceType[simulation['sequence_type'].upper()], "sim_items": sim_items, "identifier": key,
                                  'simulation_strategy': sim_strategy_cls()}})

        SimulationParser._signal_content_matches_seq_type(sim_obj)
        return sim_obj, simulation

    @staticmethod
    def _parse_sim_config_item(simulation_item: dict, key: str, symbol_table: SymbolTable) -> Tuple[SimConfigItem, dict]:
        location = SimulationParser.__name__
        valid_simulation_item_keys = ["number_of_examples", "signals", "is_noise", "seed",
                                      "false_positive_prob_in_receptors", "false_negative_prob_in_receptors",
                                      "receptors_in_repertoire_count", "generative_model", "immune_events"]

        simulation_item = {**DefaultParamsLoader.load('simulation', 'ligo_sim_config_item'), **simulation_item}

        ParameterValidator.assert_keys(simulation_item.keys(), valid_simulation_item_keys, location, key, exclusive=True)

        ParameterValidator.assert_type_and_value(simulation_item['is_noise'], bool, location, 'is_noise')
        SimulationParser._parse_signals(simulation_item, symbol_table, location, key)

        for k in ['number_of_examples', 'seed']:
            ParameterValidator.assert_type_and_value(simulation_item[k], int, location, k, min_inclusive=1)

        for k, val_type in zip(['receptors_in_repertoire_count', 'immune_events'], [int, dict]):
            if simulation_item[k]:
                ParameterValidator.assert_type_and_value(simulation_item[k], val_type, location, k)

        ParameterValidator.assert_all_type_and_value(simulation_item.keys(), str, location, 'immune_events')
        for k, val in simulation_item['immune_events'].items():
            assert isinstance(val, int) or isinstance(val, bool) or isinstance(val, str), \
                f"The values for immune events under {k} has to be int, bool or string, got {val} ({type(val)}."

        gen_model = SimulationParser._parse_generative_model(simulation_item, location)

        params = copy.deepcopy(simulation_item)
        params["signal_proportions"] = {symbol_table.get(signal): proportion for signal, proportion in simulation_item["signals"].items()}
        params["name"] = key
        params['generative_model'] = gen_model

        return SimConfigItem(**{key: val for key, val in params.items() if key not in ['signals', 'type']}), simulation_item

    @staticmethod
    def _parse_signals(sim_item: dict, symbol_table: SymbolTable, location: str, key: str) -> dict:
        if isinstance(sim_item['signals'], dict):

            ParameterValidator.assert_keys(list(sim_item["signals"].keys()), symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, key, False)
            assert 0 <= sum(sim_item['signals'].values()) <= 1, sim_item['signals']

        elif isinstance(sim_item['signals'], list):

            assert len(sim_item['signals']) == 1 and sim_item['receptors_in_repertoire_count'] != 0, \
                f'Multiple signals are not supported for receptor-level simulation for sim_item {key}.'
            ParameterValidator.assert_keys(sim_item["signals"], symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, key, False)

            sim_item['signals'] = {sim_item['signals'][0]: 1}

        return sim_item

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
    def _signal_content_matches_seq_type(simulation: SimConfig):
        for sim_item in simulation.sim_items:
            for signal in sim_item.signals:
                for motif in signal.motifs:
                    ParameterValidator.assert_all_in_valid_list([letter for letter in motif.seed if letter != '/'],
                                                                EnvironmentSettings.get_sequence_alphabet(simulation.sequence_type),
                                                                SimulationParser.__name__, f"motif seed {motif.seed}")
