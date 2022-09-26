from pathlib import Path

from immuneML.dsl import Util
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.Simulation import Simulation
from immuneML.simulation.SimulationStrategy import SimulationStrategy
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.ligo_simulation.LIgOSimulationInstruction import LIgOSimulationInstruction


class LIgOSimulationParser:

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> LIgOSimulationInstruction:

        location = LIgOSimulationParser.__name__
        keys = ["simulation", "type", "is_repertoire", "paired", "use_generation_probabilities", "simulation_strategy", 'sequence_type',
                "export_formats", "store_signal_in_receptors", 'sequence_batch_size', "max_iterations", "export_p_gens", "number_of_processes"]
        ParameterValidator.assert_keys(instruction.keys(), keys, location, key)

        for param_key in ['is_repertoire', 'paired', 'use_generation_probabilities', "store_signal_in_receptors", 'export_p_gens']:
            ParameterValidator.assert_type_and_value(instruction[param_key], bool, location, param_key)
        for param_key in ['max_iterations', 'sequence_batch_size', 'number_of_processes']:
            ParameterValidator.assert_type_and_value(instruction[param_key], int, location, param_key, 1)

        ParameterValidator.assert_type_and_value(instruction['simulation_strategy'], str, location, 'simulation_strategy')
        ParameterValidator.assert_in_valid_list(instruction['simulation_strategy'].upper(), [item.name for item in SimulationStrategy], location,
                                                'simulation_strategy')

        signals = [signal.item for signal in symbol_table.get_by_type(SymbolType.SIGNAL)]

        ParameterValidator.assert_in_valid_list(instruction['simulation'],
                                                [sim.item.identifier for sim in symbol_table.get_by_type(SymbolType.SIMULATION)],
                                                location, 'simulation')
        simulation = symbol_table.get(instruction["simulation"])
        sequence_type = SequenceType[instruction['sequence_type'].upper()]

        self._signal_content_matches_seq_type(simulation, sequence_type)

        exporters = Util.parse_exporters(instruction, location)

        params = {**{key: value for key, value in instruction.items() if key not in ['type', 'export_formats']},
                  **{'simulation': simulation, 'signals': signals, 'exporters': exporters,
                     'simulation_strategy': SimulationStrategy[instruction['simulation_strategy'].upper()],
                     'sequence_type': sequence_type, 'name': key}}
        instruction = LIgOSimulationInstruction(**params)
        return instruction

    def _signal_content_matches_seq_type(self, simulation: Simulation, sequence_type: SequenceType):
        for sim_item in simulation.simulation_items:
            for signal in sim_item.signals:
                for motif in signal.motifs:
                    ParameterValidator.assert_all_in_valid_list([letter for letter in motif.seed if letter != '/'],
                                                                EnvironmentSettings.get_sequence_alphabet(sequence_type),
                                                                LIgOSimulationParser.__name__, f"motif seed {motif.seed}")
