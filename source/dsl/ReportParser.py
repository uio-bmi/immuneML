from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.util.ReflectionHandler import ReflectionHandler


class ReportParser:

    @staticmethod
    def parse_reports(workflow_specification: dict, symbol_table: SymbolTable):
        reports = workflow_specification["reports"]
        for rep_id in reports.keys():
            symbol_table = ReportParser._parse(rep_id, reports[rep_id], symbol_table)

        return symbol_table, {}

    @staticmethod
    def _parse(key: str, params: dict, symbol_table: SymbolTable) -> SymbolTable:
        report = ReflectionHandler.get_class_by_name(params["type"])()
        item = {"report": report, "params": params["params"]}

        # TODO: add encodings and models for other types of reports, done so far only for data report
        if "dataset" in params["params"].keys() and symbol_table.contains(params["params"]["dataset"]):
            item["dataset"] = params["params"]["dataset"]

        symbol_table.add(key, SymbolType.REPORT, item)
        return symbol_table
