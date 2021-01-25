from immuneML.reports.Report import Report


class MultiDatasetReport(Report):
    """
    Multi dataset reports are special reports that can be specified when running immuneML with the :py:obj:`~immuneML.api.aggregated_runs.MultiDatasetBenchmarkTool.MultiDatasetBenchmarkTool`.

    When running the :py:obj:`~immuneML.api.aggregated_runs.MultiDatasetBenchmarkTool.MultiDatasetBenchmarkTool`, multi dataset reports can be specified under 'benchmark_reports'.
    """

    @staticmethod
    def get_title():
        return "Multi dataset reports"
