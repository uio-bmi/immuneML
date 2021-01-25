import sys
from pathlib import Path

from immuneML.dsl.InstructionParser import InstructionParser
from immuneML.dsl.OutputParser import OutputParser
from immuneML.dsl.definition_parsers.DefinitionParser import DefinitionParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings


def generate_docs(docs_path: str):
    docs_path = Path(docs_path)

    DefinitionParser.generate_docs(docs_path)
    InstructionParser.generate_docs(docs_path)
    OutputParser.generate_docs(docs_path)
    print(f"Specification documentation is generated at {docs_path}.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = EnvironmentSettings.specs_docs_path
    generate_docs(path)
