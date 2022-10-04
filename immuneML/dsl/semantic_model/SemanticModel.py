from pathlib import Path

from immuneML.util.Logger import print_log
from immuneML.util.ReflectionHandler import ReflectionHandler
from immuneML.workflows.instructions.Instruction import Instruction


class SemanticModel:

    def __init__(self, instructions: list, result_path: Path, output=None):
        assert all(isinstance(instruction, Instruction) for instruction in instructions), \
            "SemanticModel: error occurred in parsing: check instruction definitions in the configuration file."
        self.instructions = instructions
        self.result_path = result_path
        self.output = output

    def run(self):
        instruction_states = self.run_instructions()
        if self.output is not None:
            self.build_reports(instruction_states)
        return instruction_states

    def build_reports(self, instruction_states):
        report_builder = self.make_report_builder()
        print_log(f"Generating {self.output['format']} reports...", include_datetime=True)
        result_path = report_builder.build(instruction_states, self.result_path)
        print_log(f"{self.output['format']} reports are generated.", include_datetime=True)
        return result_path

    def run_instructions(self) -> list:
        instruction_states = []
        for index, instruction in enumerate(self.instructions):
            print_log(f"Instruction {index+1}/{len(self.instructions)} has started.", include_datetime=True)
            result = instruction.run(result_path=self.result_path)
            instruction_states.append(result)
            print_log(f"Instruction {index+1}/{len(self.instructions)} has finished.", include_datetime=True)
        return instruction_states

    def make_report_builder(self):
        report_builder = ReflectionHandler.get_class_by_name(f"{self.output['format']}Builder", "presentation/")
        return report_builder
