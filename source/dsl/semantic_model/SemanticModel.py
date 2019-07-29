from source.workflows.processes.InstructionProcess import InstructionProcess


class SemanticModel:

    def __init__(self, instructions: list, result_path):
        assert all(isinstance(instruction, InstructionProcess) for instruction in instructions), \
            "SemanticModel: error occurred in parsing: check instruction definitions in the configuration file."
        self.instructions = instructions
        self.result_path = result_path

    def run(self):
        for index, instruction in enumerate(self.instructions):
            print("Instruction {}/{} has started.".format(index+1, len(self.instructions)))
            instruction.result_path = self.result_path
            result = instruction.run()
            print("Instruction {}/{} has finished. Result: {}".format(index+1, len(self.instructions), result))
