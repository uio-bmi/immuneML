from pathlib import Path

from immuneML.dsl.instruction_parsers.LigoSimParser import get_simulation_from_symbol_table
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.ligo_sim_feasibility.FeasibilitySummaryInstruction import FeasibilitySummaryInstruction


class FeasibilitySummaryParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> FeasibilitySummaryInstruction:
        ParameterValidator.assert_keys(instruction, ["simulation", "sequence_count", "number_of_processes", "type"],
                                       FeasibilitySummaryParser.__name__, "FeasibilitySummary")

        ParameterValidator.assert_type_and_value(instruction['sequence_count'], int, FeasibilitySummaryParser.__name__, 'sequence_count', 10)
        ParameterValidator.assert_type_and_value(instruction['number_of_processes'], int, FeasibilitySummaryParser.__name__, 'number_of_processes', 1)

        simulation = get_simulation_from_symbol_table(instruction['simulation'], symbol_table, FeasibilitySummaryParser.__name__)

        return FeasibilitySummaryInstruction(simulation=simulation, sequence_count=instruction['sequence_count'], signals=symbol_table.get_signals(),
                                             number_of_processes=instruction['number_of_processes'], name=key)
