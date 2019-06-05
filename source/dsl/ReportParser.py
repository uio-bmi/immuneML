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

        if ReflectionHandler.exists("{}Parser".format(params["type"])):
            report_parser = ReflectionHandler.get_class_by_name("{}Parser".format(params["type"]))
            parsed_params, specs = report_parser.parse(params["params"], symbol_table)
        else:
            parsed_params = params["params"]
            specs = copy.deepcopy(parsed_params)

        item = {"report": report, "params": parsed_params}

        # TODO: add encodings and models for other types of reports, done so far only for data report
        if "dataset" in params["params"].keys() and symbol_table.contains(params["params"]["dataset"]):
            item["dataset"] = params["params"]["dataset"]
        elif "encoding" in parsed_params.keys() and symbol_table.contains(parsed_params["encoding"]):
            item["encoding"] = parsed_params["encoding"]

        symbol_table.add(key, SymbolType.REPORT, item)
        return symbol_table, specs
