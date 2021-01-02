import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from source.util.PathBuilder import PathBuilder


class TestSequenceLengthDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_get_normalized_sequence_lengths(self):
        path = EnvironmentSettings.root_path / "test/tmp/datareports/"
        PathBuilder.build(path)

        rep1 = Repertoire.build_from_sequence_objects(sequence_objects=[ReceptorSequence(amino_acid_sequence="AAA", identifier="1"),
                                                                        ReceptorSequence(amino_acid_sequence="AAAA", identifier="2"),
                                                                        ReceptorSequence(amino_acid_sequence="AAAAA", identifier="3"),
                                                                        ReceptorSequence(amino_acid_sequence="AAA", identifier="4")],
                                                      path=path, metadata={})
        rep2 = Repertoire.build_from_sequence_objects(sequence_objects=[ReceptorSequence(amino_acid_sequence="AAA", identifier="5"),
                                                                        ReceptorSequence(amino_acid_sequence="AAAA", identifier="6"),
                                                                        ReceptorSequence(amino_acid_sequence="AAAA", identifier="7"),
                                                                        ReceptorSequence(amino_acid_sequence="AAA", identifier="8")],
                                                      path=path, metadata={})

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        sld = SequenceLengthDistribution(dataset, 1, path)

        result = sld.generate()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)
