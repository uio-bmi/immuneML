from unittest import TestCase

from source.dsl_parsers.ReportParser import ReportParser
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution


class TestReportParser(TestCase):
    def test_parse_reports(self):
        reports = ["SequenceLengthDistribution"]
        parsed_spec = ReportParser.parse_reports(reports)
        self.assertTrue(isinstance(parsed_spec, dict))
        self.assertTrue(isinstance(parsed_spec["SequenceLengthDistribution"]["report"], SequenceLengthDistribution))
        self.assertTrue(parsed_spec["SequenceLengthDistribution"]["params"] is None)

