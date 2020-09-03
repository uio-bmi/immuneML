from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.simulation.implants.Signal import Signal
from source.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from source.util.Logger import log
from source.util.ParameterValidator import ParameterValidator
from source.util.ReflectionHandler import ReflectionHandler


class SignalParser:

    VALID_KEYS = ["motifs", "implanting", "sequence_position_weights"]

    @staticmethod
    @log
    def parse_signals(signals: dict, symbol_table: SymbolTable):
        for key, signal_spec in signals.items():

            ParameterValidator.assert_keys(signal_spec.keys(), SignalParser.VALID_KEYS, "SignalParser", key)

            implanting_strategy = SignalParser._get_implanting_strategy(key, signal_spec)

            ParameterValidator.assert_keys(signal_spec["motifs"], symbol_table.get_keys_by_type(SymbolType.MOTIF), "SignalParser",
                                           f"motifs in signal {key}", False)

            signal_motifs = [symbol_table.get(motif_id) for motif_id in signal_spec["motifs"]]
            signal = Signal(key, signal_motifs, implanting_strategy)
            symbol_table.add(key, SymbolType.SIGNAL, signal)

        return symbol_table, signals

    @staticmethod
    def _get_implanting_strategy(key: str, signal: dict) -> SignalImplantingStrategy:

        valid_strategies = [cls[:-10] for cls in
                            ReflectionHandler.discover_classes_by_partial_name("Implanting", "simulation/signal_implanting_strategy/")]
        ParameterValidator.assert_in_valid_list(signal["implanting"], valid_strategies, "SignalParser", key)

        if "sequence_position_weights" not in signal:
            signal["sequence_position_weights"] = None

        implanting_strategy = ReflectionHandler.get_class_by_name(f"{signal['implanting']}Implanting")(GappedMotifImplanting(),
                                                                                                       signal["sequence_position_weights"])

        return implanting_strategy
