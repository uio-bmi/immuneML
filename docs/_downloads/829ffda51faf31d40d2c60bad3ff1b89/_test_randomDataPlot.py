import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.RandomDataPlot import RandomDataPlot
from immuneML.util.PathBuilder import PathBuilder


class TestRandomDataPlot(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_random_data_plot(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "random_data_plot")

        params = {"n_points_to_plot": 10}
        report = RandomDataPlot.build_object(**params)

        # make sure to set the 'path' manually
        report.result_path = path

        self.assertTrue(report.check_prerequisites())
        result = report._generate()

        # ensure result files are generated
        self.assertTrue(os.path.isfile(result.output_figures[0].path))
        self.assertTrue(os.path.isfile(result.output_tables[0].path))

        # don't forget to remove the temporary path
        shutil.rmtree(path)
