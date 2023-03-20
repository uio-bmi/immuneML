import copy
import logging
from itertools import chain
from typing import Tuple

from immuneML import Constants
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.definition_parsers.SignalParser import check_clonal_frequency
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.SimConfig import SimConfig
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.generative_models.GenerativeModel import GenerativeModel
from immuneML.simulation.implants.Signal import SignalPair
from immuneML.simulation.simulation_strategy.SimulationStrategy import SimulationStrategy
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class SimulationParser:
    keyword = "simulations"

    @staticmethod
    def parse(simulations: dict, symbol_table: SymbolTable):
        for key, simulation in simulations.items():
            item, sim_dict = _parse_ligo_simulation(simulation, key, symbol_table)

            symbol_table.add(key, SymbolType.SIMULATION, item)
            simulations[key] = sim_dict

        return symbol_table, simulations


def _parse_ligo_simulation(simulation: dict, key: str, symbol_table: SymbolTable) -> Tuple[SimConfig, dict]:
    location = SimulationParser.__name__
    valid_keys = {'is_repertoire': bool, 'paired': bool, 'sequence_type': str, 'p_gen_bin_count': int, 'simulation_strategy': str,
                  'sim_items': dict, 'keep_p_gen_dist': bool, 'remove_seqs_with_signals': bool}

    simulation = {**DefaultParamsLoader.load("simulation", "ligo_sim_config"), **simulation}

    ParameterValidator.assert_keys(list(simulation.keys()), list(valid_keys.keys()), location, key, exclusive=True)
    for k, val_type in valid_keys.items():
        ParameterValidator.assert_type_and_value(simulation[k], val_type, location, k)

    sim_strategies = ReflectionHandler.all_nonabstract_subclass_basic_names(SimulationStrategy, drop_part='Strategy',
                                                                            subdirectory='simulation/simulation_strategy')

    ParameterValidator.assert_in_valid_list(simulation['sequence_type'].upper(), [st.name for st in SequenceType], location, 'sequence_type')
    ParameterValidator.assert_in_valid_list(simulation['simulation_strategy'], sim_strategies, location, 'simulation_strategy')

    sim_strategy_cls = ReflectionHandler.get_class_by_name(simulation['simulation_strategy'] + "Strategy", "simulation/simulation_strategy")

    sim_items = []
    for sim_key, item in simulation['sim_items'].items():
        sim_item, sim_item_dict = _parse_sim_config_item(item, sim_key, symbol_table, simulation['is_repertoire'])
        _check_if_supported(sim_item, sim_strategy_cls)
        sim_items.append(sim_item)
        simulation['sim_items'][sim_key] = sim_item_dict

    sim_obj = SimConfig(**{**{k: v for k, v in simulation.items() if k != 'type'},
                           **{'sequence_type': SequenceType[simulation['sequence_type'].upper()], "sim_items": sim_items, "identifier": key,
                              'simulation_strategy': sim_strategy_cls()}})

    _signal_content_matches_seq_type(sim_obj)
    return sim_obj, simulation


def _check_if_supported(sim_item, sim_strategy_cls):
    if "Implanting" in sim_strategy_cls.__name__:
        assert all(not isinstance(sig, SignalPair) for sig in sim_item.signals), \
            "Implanting does not support having more than 1 signal per sequence. Please adjust the simulation specs."


def _parse_sim_config_item(simulation_item: dict, key: str, symbol_table: SymbolTable, is_repertoire: bool) -> Tuple[SimConfigItem, dict]:
    location = SimulationParser.__name__
    valid_simulation_item_keys = ["number_of_examples", "signals", "is_noise", "seed", "default_clonal_frequency",
                                  "false_positive_prob_in_receptors", "false_negative_prob_in_receptors", "sequence_len_limits",
                                  "receptors_in_repertoire_count", "generative_model", "immune_events"]

    simulation_item = {**DefaultParamsLoader.load('simulation', 'ligo_sim_config_item'), **simulation_item}

    ParameterValidator.assert_keys(simulation_item.keys(), valid_simulation_item_keys, location, key, exclusive=True)

    ParameterValidator.assert_type_and_value(simulation_item['is_noise'], bool, location, 'is_noise')
    _parse_signals(simulation_item, symbol_table, location, key)

    _validate_sequence_len_limits(simulation_item)

    for k in ['number_of_examples', 'seed']:
        ParameterValidator.assert_type_and_value(simulation_item[k], int, location, k, min_inclusive=1)

    for k, val_type in zip(['receptors_in_repertoire_count', 'immune_events'], [int, dict]):
        if simulation_item[k]:
            ParameterValidator.assert_type_and_value(simulation_item[k], val_type, location, k)

    ParameterValidator.assert_all_type_and_value(simulation_item.keys(), str, location, 'immune_events')
    for k, val in simulation_item['immune_events'].items():
        assert isinstance(val, int) or isinstance(val, bool) or isinstance(val, str), \
            f"The values for immune events under {k} has to be int, bool or string, got {val} ({type(val)}."

    gen_model = _parse_generative_model(simulation_item, location)

    params = copy.deepcopy(simulation_item)
    params["signal_proportions"] = _make_signal_proportions(symbol_table, simulation_item["signals"], key)
    params["name"] = key
    params['generative_model'] = gen_model

    check_clonal_frequency(simulation_item, 'default_clonal_frequency', SimulationParser.__name__)
    if not is_repertoire:
        assert simulation_item['default_clonal_frequency'] is None, "Clonal frequency can be set only for repertoire simulation."
    if not is_repertoire and any(signal.clonal_frequency is not None for signal in params['signal_proportions'].keys()):
        logging.warning(f"Clonal frequency is set for some of the signals in {key}, but the simulation is not on repertoire level, "
                        f"so clonal frequency parameters will not be used.")

    return SimConfigItem(**{key: val for key, val in params.items() if key not in ['signals', 'type']}), simulation_item


def _parse_signals(sim_item: dict, symbol_table: SymbolTable, location: str, key: str) -> dict:

    assert isinstance(sim_item['signals'], dict) or sim_item['signals'] is None, \
        f"Signals under {key} have to be either null or a dictionary, got: {sim_item['signals']}."

    if sim_item['signals'] is not None:

        signals = _extract_signals_from_potential_pairs(sim_item["signals"].keys())
        ParameterValidator.assert_keys(signals, symbol_table.get_keys_by_type(SymbolType.SIGNAL), location, key, False)
        assert 0 <= sum(sim_item['signals'].values()) <= 1, sim_item['signals']

    else:

        sim_item['signals'] = {}

    return sim_item


def _extract_signals_from_potential_pairs(signal_keys: list):
    return list(set(chain.from_iterable([sig_key.split(Constants.SIGNAL_DELIMITER) for sig_key in signal_keys])))


def _make_signal_proportions(symbol_table: SymbolTable, signals: dict, location: str) -> dict:
    signal_proportions = {}

    for signal, proportion in signals.items():
        tmp_signals = signal.split(Constants.SIGNAL_DELIMITER)
        if len(tmp_signals) == 2:
            signal_pair = SignalPair(symbol_table.get(tmp_signals[0]), symbol_table.get(tmp_signals[1]))
            if signal_pair in signal_proportions:
                raise KeyError(f"The combinations of signals {signal_pair} that co-occur was defined multiple times. "
                               f"Please check the signals under {location}.")
            signal_proportions[signal_pair] = proportion
        elif len(tmp_signals) == 1:
            signal_proportions[symbol_table.get(signal)] = proportion
        else:
            raise RuntimeError(f"Couldn't parse signals listed under {location}. The keys have to be either single signal names or two "
                               f"signal names separated by {Constants.SIGNAL_DELIMITER}. Got {signal} instead.")

    return signal_proportions


def _parse_generative_model(simulation_item: dict, location: str):
    ParameterValidator.assert_type_and_value(simulation_item['generative_model'], dict, location, 'generative_model')
    ParameterValidator.assert_keys_present(simulation_item['generative_model'].keys(), ['type'], location, 'generative_model')

    gen_model_classes = ReflectionHandler.all_nonabstract_subclass_basic_names(GenerativeModel, "", "simulation/generative_models/")
    ParameterValidator.assert_in_valid_list(simulation_item['generative_model']['type'], gen_model_classes, location, "generative_model:type")

    gen_model_cls = ReflectionHandler.get_class_by_name(simulation_item['generative_model']['type'], "simulation/generative_models/")
    params = {key: val for key, val in simulation_item['generative_model'].items() if key != 'type'}
    gen_model = gen_model_cls.build_object(**params)

    return gen_model


def _signal_content_matches_seq_type(simulation: SimConfig):
    for sim_item in simulation.sim_items:
        for signal in sim_item.signals:
            if isinstance(signal, SignalPair):
                _motif_content_matches_seq_type(signal.signal1.motifs + signal.signal2.motifs, simulation.sequence_type)
            else:
                _motif_content_matches_seq_type(signal.motifs, simulation.sequence_type)


def _motif_content_matches_seq_type(motifs: list, seq_type):
    for motif_group in motifs:
        motifs = motif_group if isinstance(motif_group, list) else [motif_group]
        for motif in motifs:
            ParameterValidator.assert_all_in_valid_list(motif.get_alphabet(),
                                                        EnvironmentSettings.get_sequence_alphabet(seq_type),
                                                        SimulationParser.__name__, motif.get_alphabet())


def _validate_sequence_len_limits(sim_item: dict):
    ParameterValidator.assert_keys(sim_item['sequence_len_limits'].keys(), ['min', 'max'], SimulationParser.__name__, 'sequence_len_limits')
    for key in ['min', 'max']:
        ParameterValidator.assert_type_and_value(sim_item['sequence_len_limits'][key], int, SimulationParser.__name__, f'sequence_len_limits:{key}', -1)

    assert sim_item['sequence_len_limits']['min'] <= sim_item['sequence_len_limits']['max'] or sim_item['sequence_len_limits']['max'] == -1, \
        f"Under sequence_len_limits, min has to be less or equal to max value, if max is not -1 (max=-1 -> max is not used): {sim_item['sequence_len_limits']}"
