from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.tool_interface.interface_components.DatasetToolComponent import DatasetToolComponent
from immuneML.util.ParameterValidator import ParameterValidator


class ToolParser:
    keyword = "tools"

    @staticmethod
    def parse(specification, symbol_table):

        # loop keys in tool section of YAML-file. Key is user defined name of tool
        # parse tool specification and add to symbol_table
        if ToolParser.keyword in specification:
            for key in specification[ToolParser.keyword]:
                symbol_table = ToolParser._parse_tool(key, specification[ToolParser.keyword][key], symbol_table)
                if specification[ToolParser.keyword][key]['type'] == 'MLMethodTool':
                    specification = ToolParser._add_specs_params(key, specification, symbol_table)
                elif specification[ToolParser.keyword][key]['type'] == 'DatasetTool':
                    specification = ToolParser._get_dataset(key, specification, symbol_table)
        else:
            specification[ToolParser.keyword] = {}

        return symbol_table, specification  # , specification[ToolParser.keyword]

    @staticmethod
    def _parse_tool(key: str, tool_item: dict, symbol_table: SymbolTable):
        # assert valid keys
        # path - required
        # type - required
        # name - required?
        # language - optional, default python

        # check that all required parameters are present
        ParameterValidator.assert_keys_present(list(tool_item.keys()), ["type", "path"], ToolParser.__name__,
                                               key)

        # check that the value of type is valid
        valid_types = ["MLMethodTool", "DatasetTool"]
        ParameterValidator.assert_in_valid_list(tool_item["type"], valid_types, "ToolParser", "type")

        # set default values and create dict with tool_item
        default_params = {'language': 'python'}
        tool_item = {**default_params, **tool_item}

        # TODO:
        #  - check that the tool is reachable on the specified path - reflection handler
        #  - check that name of tool is key in definitions (MLMethod in first version)

        symbol_table.add(key, SymbolType.TOOL, tool_item)

        return symbol_table

    @staticmethod
    def _add_specs_params(key, specification, symbol_table: SymbolTable):
        item = symbol_table.get(key)
        specification["definitions"]["ml_methods"][key] = {item['type']: {'path': item['path']}}
        return specification

    @staticmethod
    def _get_dataset(key, specification, symbol_table: SymbolTable):
        item = symbol_table.get(key)
        tool = DatasetToolComponent(item['path'])
        specification = tool.get_dataset(specification)
        return specification

    @staticmethod
    def get_related_parameters(symbol_table: SymbolTable, specs):
        list_el = symbol_table.get_by_type(SymbolType.TOOL)
        my_item = list(specs.items())[0]
        my_name = my_item[0]
        my_tool_type = my_item[1]

        for el in list_el:
            if el.item['type'] == my_tool_type and el.item['name'] == my_name:
                return el.item
