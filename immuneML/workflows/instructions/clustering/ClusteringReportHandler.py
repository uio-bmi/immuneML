import copy
from pathlib import Path
from typing import List

from immuneML.reports.Report import Report
from immuneML.reports.clustering_reports.ClusteringReport import ClusteringReport
from immuneML.reports.encoding_reports.EncodingReport import EncodingReport
from immuneML.util.Logger import print_log
from immuneML.util.PathBuilder import PathBuilder
from immuneML.workflows.instructions.clustering.ClusteringState import ClusteringState
from immuneML.workflows.instructions.clustering.clustering_run_model import ClusteringItem


class ClusteringReportHandler:
    """Manages report generation for clustering results."""

    def __init__(self, reports: List[Report]):
        self.reports = reports

    def run_clustering_reports(self, state: ClusteringState):
        """Generate overall clustering reports."""
        report_path = PathBuilder.build(state.result_path / f'reports/')
        for report in self.reports:
            if isinstance(report, ClusteringReport):
                tmp_report = copy.deepcopy(report)
                tmp_report.result_path = report_path
                tmp_report.state = state
                state.clustering_report_results.append(tmp_report.generate_report())

        if len(self.reports) > 0:
            gen_rep_count = len(state.clustering_report_results)
            print_log(f"{state.config.name}: generated {gen_rep_count} clustering reports.", True)

        return state

    def run_item_reports(self, cl_item: ClusteringItem, analysis_desc: str, run_id: int, path: Path,
                         state: ClusteringState) -> list:
        """Generate reports for individual clustering items."""
        report_path = PathBuilder.build(path / f'reports/')
        report_results = []
        for report in self.reports:
            if isinstance(report, EncodingReport):
                tmp_report = copy.deepcopy(report)
                tmp_report.result_path = PathBuilder.build(report_path / tmp_report.name)
                tmp_report.dataset = cl_item.dataset
                rep_result = tmp_report.generate_report()
                report_results.append(rep_result)

        if len(self.reports) > 0:
            gen_rep_count = len(report_results)
            print_log(f"{state.config.name}: generated {gen_rep_count} reports for setting "
                      f"{cl_item.cl_setting.get_key()} for {analysis_desc}, run id: {run_id + 1}.", True)

        return report_results
