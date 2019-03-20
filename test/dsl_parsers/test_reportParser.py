from unittest import TestCase

from source.dsl_parsers.ReportParser import ReportParser
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution


class TestReportParser(TestCase):
    def test_parse_reports(self):
        reports = ["SequenceLengthDistribution"]
        parsed2 = ReportParser.parse_reports(reports)
        self.assertTrue(isinstance(parsed2, dict))
        self.assertTrue(isinstance(parsed2["SequenceLengthDistribution"]["report"], SequenceLengthDistribution))
        self.assertTrue(parsed2["SequenceLengthDistribution"]["params"] is None)

