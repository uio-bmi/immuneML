import copy

from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler


class ReportParser:

    @staticmethod
    def parse_reports(workflow_specification: dict, symbol_table: SymbolTable):
        if "reports" in workflow_specification:
            reports = workflow_specification["reports"]
            for rep_id in reports.keys():
                symbol_table, workflow_specification["reports"][rep_id] = ReportParser._parse(rep_id, reports[rep_id],
                                                                                              symbol_table)
        else:
            reports = {}
        return symbol_table, reports

    @staticmethod
    def _parse(key: str, params: dict, symbol_table: SymbolTable):
        report = ReflectionHandler.get_class_by_name(params["type"])()

        user_params = params["params"] if "params" in params else {}

        if ReflectionHandler.exists("{}Parser".format(params["type"])):
            report_parser = ReflectionHandler.get_class_by_name("{}Parser".format(params["type"]))
            parsed_params, specs = report_parser.parse(user_params, symbol_table)
        else:
            parsed_params = user_params
            specs = copy.deepcopy(parsed_params)

        symbol_table.add(key, SymbolType.REPORT, report, parsed_params)
        return symbol_table, specs
