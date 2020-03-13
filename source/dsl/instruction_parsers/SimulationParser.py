from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.util.ParameterValidator import ParameterValidator
from source.workflows.instructions.SimulationProcess import SimulationInstruction


class SimulationParser:

    """
    instruction1:
        type: Implanting
        dataset: d1
        batch_size: 8
        simulation: sim1
    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable) -> SimulationInstruction:

        ParameterValidator.assert_keys(instruction.keys(), ["dataset", "batch_size", "simulation", "type"], "SimulationParser", key)

        signals = [signal.item for signal in symbol_table.get_by_type(SymbolType.SIGNAL)]
        simulation = symbol_table.get(instruction["simulation"])
        dataset = symbol_table.get(instruction["dataset"])
        batch_size = instruction["batch_size"]

        process = SimulationInstruction(signals=signals, simulation=simulation, dataset=dataset, batch_size=batch_size)
        return process
