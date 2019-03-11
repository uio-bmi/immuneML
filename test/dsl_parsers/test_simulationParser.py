from unittest import TestCase

from source.dsl_parsers.SimulationParser import SimulationParser
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy


class TestSimulationParser(TestCase):
    def test_parse_simulation(self):

        simulation = {
            "motifs": [
                {
                    "id": "motif1",
                    "seed": "CAS",
                    "instantiation": "identity"
                }
            ],
            "signals": [
                {
                    "id": "signal1",
                    "motifs": ["motif1"],
                    "implanting": "healthy_sequences"
                }
            ],
            "implanting": [{
                "signals": ["signal1"],
                "repertoires": 100,
                "sequences": 10
            }]
        }

        implanting, signals = SimulationParser.parse_simulation(simulation)
        self.assertTrue(all([all(isinstance(item, Signal) for item in item2["signals"]) for item2 in implanting]))
        self.assertTrue(all([all(isinstance(item.implanting_strategy, SignalImplantingStrategy)
                                 for item in item2["signals"]) for item2 in implanting]))
        self.assertTrue(all([all(isinstance(motif, Motif)
                                 for motif in item2["signals"][0].motifs) for item2 in implanting]))
