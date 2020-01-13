from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.workflows.instructions.SimulationProcess import SimulationInstruction


class SimulationParser:

    def parse(self, instruction: dict, symbol_table: SymbolTable) -> SimulationInstruction:

        signals = [signal.item for signal in symbol_table.get_by_type(SymbolType.SIGNAL)]
        simulations = [simulation.item for simulation in symbol_table.get_by_type(SymbolType.SIMULATION)]
        dataset = symbol_table.get(instruction["dataset"])
        batch_size = instruction["batch_size"]

        process = SimulationInstruction(signals=signals, simulations=simulations, dataset=dataset, batch_size=batch_size)
        return process
