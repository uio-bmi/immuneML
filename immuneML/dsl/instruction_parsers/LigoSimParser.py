from pathlib import Path

from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.ligo_simulation.LigoSimInstruction import LigoSimInstruction


class LigoSimParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> LigoSimInstruction:

        location = LigoSimParser.__name__
        keys = ["simulation", "type", 'sequence_batch_size', "max_iterations", "export_p_gens",
                "number_of_processes"]
        ParameterValidator.assert_keys(instruction.keys(), keys, location, key)

        for param_key in ['export_p_gens']:
            ParameterValidator.assert_type_and_value(instruction[param_key], bool, location, param_key)
        for param_key in ['max_iterations', 'sequence_batch_size', 'number_of_processes']:
            ParameterValidator.assert_type_and_value(instruction[param_key], int, location, param_key, 1)

        simulation = get_simulation_from_symbol_table(instruction['simulation'], symbol_table, location)

        params = {**{key: value for key, value in instruction.items() if key != 'type'},
                  **{'simulation': simulation, 'signals': symbol_table.get_signals(), 'name': key}}
        instruction = LigoSimInstruction(**params)
        return instruction


def get_simulation_from_symbol_table(sim_key, symbol_table, location):
    ParameterValidator.assert_type_and_value(sim_key, str, location, 'simulation')
    ParameterValidator.assert_in_valid_list(sim_key,
                                            [sim.item.identifier for sim in
                                             symbol_table.get_by_type(SymbolType.SIMULATION)],
                                            location, 'simulation')
    return symbol_table.get(sim_key)