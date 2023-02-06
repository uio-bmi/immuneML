import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.reports.data_reports.SequenceGenerationProbabilityDistribution import \
    SequenceGenerationProbabilityDistribution
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceGenerationProbabilityDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_get_pgen_distribution(self):
        path = EnvironmentSettings.root_path / "test/tmp/datareports/pgen/"
        PathBuilder.build(path)

        repertoire = Repertoire.build(["CSAIGQGKGAFYGYTF", "CASSLDRVSASGANVLTF", "CASSVQPRSEVPNTGELFF"],
                                      v_genes=["TRBV20-1", "TRBV4-1", "TRBV11-3"],
                                      j_genes=["TRBJ1-2", "TRBJ2-6", "TRBJ2-2"],
                                      region_types=[RegionType.IMGT_JUNCTION for _ in range(3)],
                                      counts=[1] * 3,
                                      path=path)

        dataset = RepertoireDataset(repertoires=[repertoire])

        pgen_report = SequenceGenerationProbabilityDistribution(dataset, path, mark_implanted_labels=False)

        result = pgen_report.generate_report()

        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)
