from pathlib import Path

from immuneML.dsl import Util
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.ligo_simulation.LigoSimInstruction import LigoSimInstruction


class LigoSimParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> LigoSimInstruction:

        location = LigoSimParser.__name__
        keys = ["simulation", "type", "export_formats", "store_signal_in_receptors", 'sequence_batch_size', "max_iterations", "export_p_gens",
                "number_of_processes"]
        ParameterValidator.assert_keys(instruction.keys(), keys, location, key)

        for param_key in ["store_signal_in_receptors", 'export_p_gens']:
            ParameterValidator.assert_type_and_value(instruction[param_key], bool, location, param_key)
        for param_key in ['max_iterations', 'sequence_batch_size', 'number_of_processes']:
            ParameterValidator.assert_type_and_value(instruction[param_key], int, location, param_key, 1)

        signals = [signal.item for signal in symbol_table.get_by_type(SymbolType.SIGNAL)]

        ParameterValidator.assert_in_valid_list(instruction['simulation'],
                                                [sim.item.identifier for sim in symbol_table.get_by_type(SymbolType.SIMULATION)],
                                                location, 'simulation')
        simulation = symbol_table.get(instruction["simulation"])

        exporters = Util.parse_exporters(instruction, location)

        params = {**{key: value for key, value in instruction.items() if key not in ['type', 'export_formats']},
                  **{'simulation': simulation, 'signals': signals, 'exporters': exporters, 'name': key}}
        instruction = LigoSimInstruction(**params)
        return instruction