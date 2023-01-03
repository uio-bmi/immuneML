from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.util.ReflectionHandler import ReflectionHandler


class ToolParser:
    keyword = "tools"

    @staticmethod
    def parse(specification, symbol_table):

        # loop keys in tool section of YAML-file. Key is user defined name of tool
        # parse tool specification and add to symbol_table
        if ToolParser.keyword in specification:
            for key in specification[ToolParser.keyword]:
                symbol_table = ToolParser._parse_tool(key, specification[ToolParser.keyword][key], symbol_table)
        else:
            specification[ToolParser.keyword] = {}

        return symbol_table  # , specification[ToolParser.keyword]

    @staticmethod
    def _parse_tool(key: str, tool_item: dict, symbol_table: SymbolTable):
        # assert valid keys
        # path - required
        # type - required
        # name - required?
        # language - optional, default python

        # check that all required parameters are present
        ParameterValidator.assert_keys_present(list(tool_item.keys()), ["type", "path", "name"], ToolParser.__name__, key)

        # check that the value of type is valid
        valid_types = ["MLMethod"]
        ParameterValidator.assert_in_valid_list(tool_item["type"], valid_types, "ToolParser", "type")

        # set default values and create dict with tool_item
        default_params = {'language': 'python'}
        tool_item = {**default_params, **tool_item}

        # TODO:
        #  - check that the tool is reachable on the specified path - reflection handler
        #  - check that name of tool is key in definitions (MLMethod in first version)

        symbol_table.add(key, SymbolType.TOOL, tool_item)

        return symbol_table
