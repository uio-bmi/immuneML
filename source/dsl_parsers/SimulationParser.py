from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.IdentityMotifInstantiation import IdentityMotifInstantiation
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting


class SimulationParser:

    @staticmethod
    def parse_simulation(simulation: dict):
        assert "motifs" in simulation, "Workflow specification parser: no motifs were defined for the simulation."
        assert "signals" in simulation, "Workflow specification parser: no signals were defined for the simulation."

        motifs = SimulationParser._extract_motifs(simulation)
        signals = SimulationParser._extract_signals(simulation, motifs)
        implanting = SimulationParser._add_signals_to_implanting(simulation, signals)

        return implanting, signals

    @staticmethod
    def _add_signals_to_implanting(simulation: dict, signals: list) -> list:
        result = []
        for item in simulation["implanting"]:
            result.append({
                "repertoires": item["repertoires"],
                "sequences": item["sequences"],
                "signals": [signal for signal in signals if signal.id in item["signals"]]
            })
        return result

    @staticmethod
    def _extract_motifs(simulation: dict) -> list:
        motifs = []
        for item in simulation["motifs"]:
            instantiation_strategy = SimulationParser._get_instantiation_strategy(item)
            motif = Motif(item["id"], instantiation_strategy, item["seed"])
            motifs.append(motif)
        return motifs

    @staticmethod
    def _extract_signals(simulation: dict, motifs: list) -> list:
        signals = []
        for item in simulation["signals"]:
            implanting_strategy = SimulationParser._get_implanting_strategy(item)
            signal_motifs = [motif for motif in motifs if motif.id in item["motifs"]]
            signal = Signal(item["id"], signal_motifs, implanting_strategy)
            signals.append(signal)
        return signals

    @staticmethod
    def _get_implanting_strategy(signal: dict) -> SignalImplantingStrategy:
        if "implanting" in signal and signal["implanting"] == "healthy_sequences":
            implanting_strategy = HealthySequenceImplanting(GappedMotifImplanting())
        else:
            raise NotImplementedError
        return implanting_strategy

    @staticmethod
    def _get_instantiation_strategy(motif_item: dict) -> MotifInstantiationStrategy:
        if "instantiation" in motif_item and motif_item["instantiation"] == "identity":
            instantiation_strategy = IdentityMotifInstantiation()
        else:
            raise NotImplementedError
        return instantiation_strategy
