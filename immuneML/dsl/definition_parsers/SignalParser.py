from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from immuneML.simulation.signal_implanting_strategy.ImplantingComputation import ImplantingComputation
from immuneML.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from immuneML.util.Logger import log
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class SignalParser:

    VALID_KEYS = ["motifs", "implanting"]

    @staticmethod
    @log
    def parse_signals(signals: dict, symbol_table: SymbolTable):
        for key, signal_spec in signals.items():

            ParameterValidator.assert_keys_present(signal_spec.keys(), SignalParser.VALID_KEYS, "SignalParser", key)

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

        defaults = DefaultParamsLoader.load("signal_implanting_strategy/", f"{signal['implanting']}Implanting")
        signal = {**defaults, **signal}

        ParameterValidator.assert_keys_present(list(signal.keys()), ["motifs", "implanting", "sequence_position_weights"], SignalParser.__name__, key)

        implanting_comp = None
        if 'implanting_computation' in signal:
            implanting_comp = signal['implanting_computation'].lower()
            ParameterValidator.assert_in_valid_list(implanting_comp, [el.name.lower() for el in ImplantingComputation], SignalParser.__name__,
                                                    'implanting_computation')
            implanting_comp = ImplantingComputation[implanting_comp.upper()]

        implanting_strategy = ReflectionHandler.get_class_by_name(f"{signal['implanting']}Implanting")(GappedMotifImplanting(),
                                                                                                       signal["sequence_position_weights"],
                                                                                                       implanting_comp)

        return implanting_strategy
