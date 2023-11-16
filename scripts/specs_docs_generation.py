import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from immuneML.dsl.InstructionParser import InstructionParser
from immuneML.dsl.OutputParser import OutputParser
from immuneML.dsl.definition_parsers.DefinitionParser import DefinitionParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def generate_docs(docs_path: str):
    docs_path = PathBuilder.remove_old_and_build(docs_path)
    print(docs_path)

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
