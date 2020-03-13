from unittest import TestCase

from source.dsl.definition_parsers.SimulationParser import SimulationParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from source.simulation.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting


class TestSimulationParser(TestCase):
    def test_parse_simulation(self):

        simulation = {
            "sim1": {
                "var1": {
                    "signals": ["signal1"],
                    "dataset_implanting_rate": 0.5,
                    "repertoire_implanting_rate": 0.1
                }
            }
        }

        symbol_table = SymbolTable()
        symbol_table.add("motif1", SymbolType.MOTIF, Motif("motif1", GappedKmerInstantiation(position_weights={0: 1}), seed="CAS"))
        symbol_table.add("signal1", SymbolType.SIGNAL, Signal("signal1", [symbol_table.get("motif1")],
                                                              HealthySequenceImplanting(GappedMotifImplanting())))

        symbol_table, specs = SimulationParser.parse_simulations(simulation, symbol_table)

        self.assertTrue(symbol_table.contains("sim1"))
        sim1 = symbol_table.get("sim1")
        self.assertEqual(1, len(sim1.implantings))
