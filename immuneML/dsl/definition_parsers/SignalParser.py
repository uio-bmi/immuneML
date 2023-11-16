import importlib
import sys
from inspect import getmembers, isfunction
from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator


class SignalParser:
    keyword = "signals"
    custom_func_keys = ['source_file', 'is_present_func']

    @staticmethod
    @log
    def parse(signals: dict, symbol_table: SymbolTable):
        for key, signal_spec in signals.items():

            assert "__" not in key, f"{SignalParser.__name__}: '__' is not valid part of signal names, please rename the signal."

            if "motifs" in signal_spec:
                signal = _parse_signal_with_motifs(key, signal_spec, symbol_table)
            else:
                signal = _parse_custom_func_signal(key, signal_spec)

            symbol_table.add(key, SymbolType.SIGNAL, signal)

        return symbol_table, signals


def _parse_custom_func_signal(key: str, signal_spec: dict) -> Signal:
    assert all(k in signal_spec for k in SignalParser.custom_func_keys), \
        f"Signal {key}: for signals with custom functions the following keys need to be defined: " \
        f"{SignalParser.custom_func_keys}, got: {list(signal_spec.keys())}"

    assert signal_spec['source_file'].endswith(".py") and Path(signal_spec['source_file']).is_file(), \
        f"Signal {key}: no file {signal_spec['source_file']}"

    sys.path.insert(0, str(Path(signal_spec['source_file']).parent))
    source_file = importlib.import_module(Path(signal_spec['source_file']).name[:-3])
    functions = getmembers(source_file, isfunction)
    is_present_func = [func for func in functions if signal_spec['is_present_func'] == func[0]]

    assert len(is_present_func) == 1, \
        f"Signal {key}: no function named {signal_spec['is_present_func']}."

    return Signal(id=key, is_present_custom_func=is_present_func[0][1])


def _parse_signal_with_motifs(key: str, signal_spec: dict, symbol_table: SymbolTable) -> Signal:
    assert all(k not in signal_spec for k in SignalParser.custom_func_keys) or \
           all(signal_spec[k] is None for k in SignalParser.custom_func_keys), \
        f"Signal {key}: define either motifs or the custom function."

    valid_motif_keys = symbol_table.get_keys_by_type(SymbolType.MOTIF)
    signal_motifs = []

    for motif_group in signal_spec['motifs']:
        if isinstance(motif_group, str):
            ParameterValidator.assert_in_valid_list(motif_group, valid_motif_keys, SignalParser.__name__,
                                                    f'{key}:motifs')
            signal_motifs.append(symbol_table.get(motif_group))
        elif isinstance(motif_group, list):
            assert len(
                motif_group) == 2, f"{SignalParser.__name__}: {len(motif_group)} motifs specified for signal {key}, but only 2 allowed."
            for motif in motif_group:
                ParameterValidator.assert_in_valid_list(motif, valid_motif_keys, SignalParser.__name__, f'{key}:motifs')
            signal_motifs.append([symbol_table.get(motif_id) for motif_id in motif_group])

    check_clonal_frequency(signal_spec)

    signal = Signal(key, signal_motifs, v_call=signal_spec.get('v_call'), j_call=signal_spec.get('j_call'),
                    clonal_frequency=signal_spec.get('clonal_frequency', None),
                    sequence_position_weights=signal_spec.get('sequence_position_weights', {}))
    return signal


def check_clonal_frequency(spec: dict, name: str = 'clonal_frequency', location: str = SignalParser.__name__):
    if name in spec and spec[name] is not None:
        ParameterValidator.assert_type_and_value(spec[name], dict, location, name)
        ParameterValidator.assert_all_in_valid_list(spec[name].keys(), ['a', 'loc'], location, name)
