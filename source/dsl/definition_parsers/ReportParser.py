from source.dsl.DefaultParamsLoader import DefaultParamsLoader
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.logging.Logger import log
from source.util.ReflectionHandler import ReflectionHandler


class ReportParser:

    @staticmethod
    def parse_reports(reports: dict, symbol_table: SymbolTable):
        for rep_id in reports.keys():
            symbol_table, reports[rep_id] = ReportParser._parse_report(rep_id, reports[rep_id], symbol_table)

        return symbol_table, reports

    @staticmethod
    @log
    def _parse_report(key: str, params: dict, symbol_table: SymbolTable):
        # If report is specified without parameters, set to empty parameters
        if type(params) is set:
            params = {param: {} for param in params}

        class_name = list(params.keys())[0]
        report = ReflectionHandler.get_class_by_name(class_name)
        user_params = params[class_name]
        parsed_params = {**DefaultParamsLoader.load("reports/", class_name), **user_params}
        symbol_table.add(key, SymbolType.REPORT, report.build_object(**parsed_params))

        return symbol_table, {**params, **parsed_params}
