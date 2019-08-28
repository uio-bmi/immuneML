from unittest import TestCase

from source.dsl.SimulationParser import SimulationParser
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal


class TestSimulationParser(TestCase):
    def test_parse_simulation(self):

        simulation = {
            "simulation": {
                "motifs": {
                    "motif1": {
                        "seed": "CAS",
                        "instantiation": "Identity",
                    }
                },
                "signals": {
                    "signal1": {
                        "motifs": ["motif1"],
                        "implanting": "HealthySequences"
                    }
                },
                "implanting": {
                    "var1": {
                        "signals": ["signal1"],
                        "dataset_implanting_rate": 0.5,
                        "repertoire_implanting_rate": 0.1
                    }
                }
            }
        }

        symbol_table, specs = SimulationParser.parse_simulation(simulation, SymbolTable())
        self.assertEqual(1, len(symbol_table.get_by_type(SymbolType.SIGNAL)))
        self.assertTrue(isinstance(symbol_table.get("motif1"), Motif))
        self.assertTrue(isinstance(symbol_table.get("signal1"), Signal))

        self.assertTrue("implanting" in specs)
        self.assertTrue("signal1" in specs["signals"].keys())
