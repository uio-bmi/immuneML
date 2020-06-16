from dataclasses import dataclass
from typing import ClassVar, Any


@dataclass
class DocumentationFormat:
    cls: Any
    cls_name: str
    level_heading: str

    LEVELS: ClassVar = {
        1: "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        2: "''''''''''''''''''''''''''''''''''''''''''''''''''''"
    }
