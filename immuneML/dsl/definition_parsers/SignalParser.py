from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.implants.Signal import Signal
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator


class SignalParser:

    keyword = "signals"
    VALID_KEYS = ["motifs"]

    @staticmethod
    @log
    def parse(signals: dict, symbol_table: SymbolTable):
        for key, signal_spec in signals.items():

            assert "__" not in key, f"{SignalParser.__name__}: '__' is not valid part of signal names, please rename the signal."

            ParameterValidator.assert_keys_present(signal_spec.keys(), SignalParser.VALID_KEYS, "SignalParser", key)
            valid_motif_keys = symbol_table.get_keys_by_type(SymbolType.MOTIF)
            signal_motifs = []

            for motif_group in signal_spec['motifs']:
                if isinstance(motif_group, str):
                    ParameterValidator.assert_in_valid_list(motif_group, valid_motif_keys, SignalParser.__name__, f'{key}:motifs')
                    signal_motifs.append(symbol_table.get(motif_group))
                elif isinstance(motif_group, list):
                    assert len(motif_group) == 2, f"{SignalParser.__name__}: {len(motif_group)} motifs specified for signal {key}, but only 2 allowed."
                    for motif in motif_group:
                        ParameterValidator.assert_in_valid_list(motif, valid_motif_keys, SignalParser.__name__, f'{key}:motifs')
                    signal_motifs.append([symbol_table.get(motif_id) for motif_id in motif_group])

            signal = Signal(key, signal_motifs,
                            sequence_position_weights=signal_spec['sequence_position_weights'] if 'sequence_position_weights' in signal_spec else {})
            symbol_table.add(key, SymbolType.SIGNAL, signal)

        return symbol_table, signals
