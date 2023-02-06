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
    VALID_IMPLANTING_STRATEGY_KEYS = ["mutation_hamming_distance", "occurrence_limit_pgen_range", "overwrite_sequences",
                                      "nr_of_decoys", "dataset_implanting_rate_per_decoy", "repertoire_implanting_rate_per_decoy"]

    @staticmethod
    @log
    def parse_signals(signals: dict, symbol_table: SymbolTable):
        for key, signal_spec in signals.items():
            ParameterValidator.assert_keys_present(signal_spec.keys(), SignalParser.VALID_KEYS, "SignalParser", key)

            implanting_strategy = SignalParser._get_implanting_strategy(key, signal_spec)

            ParameterValidator.assert_keys(signal_spec["motifs"], symbol_table.get_keys_by_type(SymbolType.MOTIF),
                                           "SignalParser",
                                           f"motifs in signal {key}", False)

            signal_motifs = [symbol_table.get(motif_id) for motif_id in signal_spec["motifs"]]
            signal = Signal(key, signal_motifs, implanting_strategy)
            symbol_table.add(key, SymbolType.SIGNAL, signal)

        return symbol_table, signals

    @staticmethod
    def _get_implanting_strategy(key: str, signal: dict) -> SignalImplantingStrategy:

        mutation_hamming_distance = None
        occurrence_limit_pgen_range = None
        overwrite_sequences = None
        nr_of_decoys = None
        dataset_implanting_rate_per_decoy = None
        repertoire_implanting_rate_per_decoy = None

        if isinstance(signal["implanting"], dict):
            assert len(signal[
                           "implanting"]) == 1, f"Implanting can only have one signal implanting strategy strategy: {signal['implanting']}"
            implanting_strategy_name = list(signal["implanting"].keys())[0]
            ParameterValidator.assert_keys(list(signal["implanting"][implanting_strategy_name].keys()),
                                           SignalParser.VALID_IMPLANTING_STRATEGY_KEYS,
                                           location=SignalParser.__name__,
                                           parameter_name="implanting_strategy_parameters",
                                           exclusive=False)

            if "mutation_hamming_distance" in signal["implanting"][implanting_strategy_name]:
                mutation_hamming_distance = signal["implanting"][implanting_strategy_name]["mutation_hamming_distance"]
                ParameterValidator.assert_type_and_value(mutation_hamming_distance, int, SignalParser.__name__,
                                                         "mutation_hamming_distance")

            if "occurrence_limit_pgen_range" in signal["implanting"][implanting_strategy_name]:
                occurrence_limit_pgen_range = signal["implanting"][implanting_strategy_name][
                    "occurrence_limit_pgen_range"]
                ParameterValidator.assert_type_and_value(occurrence_limit_pgen_range, dict, SignalParser.__name__,
                                                         "occurrence_limit_pgen_range")
                # TODO validate dict values

            if "overwrite_sequences" in signal["implanting"][implanting_strategy_name]:
                overwrite_sequences = signal["implanting"][implanting_strategy_name]["overwrite_sequences"]
                ParameterValidator.assert_type_and_value(overwrite_sequences, bool, SignalParser.__name__,
                                                         "overwrite_sequences")

            if "nr_of_decoys" in signal["implanting"][implanting_strategy_name]:
                nr_of_decoys = signal["implanting"][implanting_strategy_name]["nr_of_decoys"]
                ParameterValidator.assert_type_and_value(nr_of_decoys, int, SignalParser.__name__,
                                                         "nr_of_decoys")

            if "dataset_implanting_rate_per_decoy" in signal["implanting"][implanting_strategy_name]:
                dataset_implanting_rate_per_decoy = signal["implanting"][implanting_strategy_name]["dataset_implanting_rate_per_decoy"]
                ParameterValidator.assert_type_and_value(dataset_implanting_rate_per_decoy, float, SignalParser.__name__,
                                                         "dataset_implanting_rate_per_decoy")

            if "repertoire_implanting_rate_per_decoy" in signal["implanting"][implanting_strategy_name]:
                repertoire_implanting_rate_per_decoy = signal["implanting"][implanting_strategy_name]["repertoire_implanting_rate_per_decoy"]
                ParameterValidator.assert_type_and_value(repertoire_implanting_rate_per_decoy, float, SignalParser.__name__,
                                                         "repertoire_implanting_rate_per_decoy")
        else:
            implanting_strategy_name = signal["implanting"]

        valid_strategies = [cls[:-10] for cls in
                            ReflectionHandler.discover_classes_by_partial_name("Implanting",
                                                                               "simulation/signal_implanting_strategy/")]
        ParameterValidator.assert_in_valid_list(implanting_strategy_name, valid_strategies, "SignalParser", key)

        defaults = DefaultParamsLoader.load("signal_implanting_strategy/", f"{implanting_strategy_name}Implanting")
        signal = {**defaults, **signal}

        ParameterValidator.assert_keys_present(list(signal.keys()),
                                               ["motifs", "implanting", "sequence_position_weights"],
                                               SignalParser.__name__, key)

        implanting_comp = None
        if 'implanting_computation' in signal:
            implanting_comp = signal['implanting_computation'].lower()
            ParameterValidator.assert_in_valid_list(implanting_comp, [el.name.lower() for el in ImplantingComputation],
                                                    SignalParser.__name__,
                                                    'implanting_computation')
            implanting_comp = ImplantingComputation[implanting_comp.upper()]

        implanting_strategy = ReflectionHandler.get_class_by_name(f"{implanting_strategy_name}Implanting")(
            GappedMotifImplanting(),
            signal["sequence_position_weights"],
            implanting_comp,
            mutation_hamming_distance,
            occurrence_limit_pgen_range,
            overwrite_sequences,
            nr_of_decoys,
            dataset_implanting_rate_per_decoy,
            repertoire_implanting_rate_per_decoy)

        return implanting_strategy
