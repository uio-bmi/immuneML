from unittest import TestCase

from source.dsl.ReportParser import ReportParser
from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution


class TestReportParser(TestCase):
    def test_parse_reports(self):
        reports = {"reports": {"r1": {"type": "SequenceLengthDistribution"}}}
        symbol_table = SymbolTable()
        symbol_table.add("d1", SymbolType.REPORT, {})
        symbol_table, specs = ReportParser.parse_reports(reports, symbol_table)
        self.assertTrue(symbol_table.contains("r1"))
        self.assertTrue(isinstance(symbol_table.get("r1"), SequenceLengthDistribution))
