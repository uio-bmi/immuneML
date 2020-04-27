from dataclasses import dataclass


@dataclass
class InstructionPresentation:
    path: str
    instruction_class: str
    instruction_name: str
