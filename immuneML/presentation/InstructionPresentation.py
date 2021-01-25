from dataclasses import dataclass
from pathlib import Path

@dataclass
class InstructionPresentation:
    path: Path
    instruction_class: str
    instruction_name: str
