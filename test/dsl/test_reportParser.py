from unittest import TestCase

from immuneML.dsl.definition_parsers.ReportParser import ReportParser
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution


class TestReportParser(TestCase):
    def test_parse_reports(self):
        reports = {"r1": {"SequenceLengthDistribution": {}}}
        symbol_table = SymbolTable()
        symbol_table, specs = ReportParser.parse_reports(reports, symbol_table)
        self.assertTrue(symbol_table.contains("r1"))
        self.assertTrue(isinstance(symbol_table.get("r1"), SequenceLengthDistribution))
