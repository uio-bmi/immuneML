from unittest import TestCase

from source.dsl_parsers.ReportParser import ReportParser
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution


class TestReportParser(TestCase):
    def test_parse_reports(self):
        reports = ["SequenceLengthDistribution"]
        parsed = ReportParser.parse_reports(reports)
        self.assertTrue(isinstance(parsed, dict))
        self.assertTrue(isinstance(parsed["SequenceLengthDistribution"]["report"], SequenceLengthDistribution))
        self.assertTrue(parsed["SequenceLengthDistribution"]["params"] is None)

