from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from source.util.ReflectionHandler import ReflectionHandler


class SimulationParser:

    @staticmethod
    def parse_simulation(workflow_specification: dict, symbol_table: SymbolTable):
        if "simulation" in workflow_specification.keys():
            simulation = workflow_specification["simulation"]
            assert "motifs" in simulation, "Workflow specification parser: no motifs were defined for the simulation."
            assert "signals" in simulation, "Workflow specification parser: no signals were defined for the simulation."

            symbol_table = SimulationParser._extract_motifs(simulation, symbol_table)
            symbol_table = SimulationParser._extract_signals(simulation, symbol_table)
            symbol_table = SimulationParser._add_signals_to_implanting(simulation, symbol_table)

        return symbol_table, workflow_specification["simulation"] if "simulation" in workflow_specification else {}

    @staticmethod
    def _add_signals_to_implanting(simulation: dict, symbol_table: SymbolTable) -> SymbolTable:
        for key in simulation["implanting"].keys():
            item = {
                "repertoires": simulation["implanting"][key]["repertoires"],
                "sequences": simulation["implanting"][key]["sequences"],
                "signals": [signal[1]["signal"] for signal in symbol_table.get_by_type(SymbolType.SIGNAL)
                            if signal[1]["signal"].id in simulation["implanting"][key]["signals"]]
            }

            symbol_table.add(key, SymbolType.SIMULATION, {"simulation": item})

        return symbol_table

    @staticmethod
    def _extract_motifs(simulation: dict, symbol_table: SymbolTable) -> SymbolTable:
        for key in simulation["motifs"].keys():
            instantiation_strategy = SimulationParser._get_instantiation_strategy(simulation["motifs"][key])
            motif = Motif(key, instantiation_strategy, simulation["motifs"][key]["seed"])
            symbol_table.add(key, SymbolType.MOTIF, {"motif": motif})
        return symbol_table

    @staticmethod
    def _extract_signals(simulation: dict, symbol_table: SymbolTable) -> SymbolTable:
        for key in simulation["signals"].keys():
            implanting_strategy = SimulationParser._get_implanting_strategy(simulation["signals"][key])
            signal_motifs = [symbol_table.get(motif_id)["motif"] for motif_id in simulation["signals"][key]["motifs"]]
            signal = Signal(key, signal_motifs, implanting_strategy)
            symbol_table.add(key, SymbolType.SIGNAL, {"signal": signal})
        return symbol_table

    @staticmethod
    def _get_implanting_strategy(signal: dict) -> SignalImplantingStrategy:
        if "implanting" in signal and signal["implanting"] == "healthy_sequences":
            implanting_strategy = HealthySequenceImplanting(GappedMotifImplanting(),
                                                            signal["sequence_position_weights"] if
                                                            "sequence_position_weights" in signal else None)
        else:
            raise NotImplementedError
        return implanting_strategy

    @staticmethod
    def _get_instantiation_strategy(motif_item: dict) -> MotifInstantiationStrategy:
        if "instantiation" in motif_item:
            return ReflectionHandler.get_class_by_name("{}Instantiation".format(motif_item["instantiation"]))
