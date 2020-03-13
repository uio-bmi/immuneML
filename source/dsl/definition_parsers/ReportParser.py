from source.dsl.ObjectParser import ObjectParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.dsl.symbol_table.SymbolType import SymbolType
from source.logging.Logger import log
from source.reports.Report import Report
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
        classes = ReflectionHandler.get_classes_by_partial_name("", "reports/")
        valid_classes = ReflectionHandler.all_nonabstract_subclasses(Report)
        valid_values = [cls.__name__ for cls in valid_classes]
        report_object, params = ObjectParser.parse_object(params, valid_values, "", "reports/", "ReportParser", key, builder=True,
                                                          return_params_dict=True)

        symbol_table.add(key, SymbolType.REPORT, report_object)

        return symbol_table, params
