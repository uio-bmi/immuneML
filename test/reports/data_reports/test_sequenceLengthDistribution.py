import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.reports.data_reports.SequenceLengthDistribution import SequenceLengthDistribution
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceLengthDistribution(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_sequence_lengths_rep_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_rep")

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(2, {100:1}, {4: 0.33, 5: 0.33, 7: 0.33},
                                                                   {"l1": {"a": 0.5, "b": 0.5}}, path / 'dataset')

        default_params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "sequence_length_distribution")
        sld = SequenceLengthDistribution.build_object(**{**default_params, **{"dataset": dataset,
                                                                               "result_path": path,
                                                                               "split_by_label": True}})

        self.assertTrue(sld.check_prerequisites())
        result = sld._generate()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)


    def test_sequence_lengths_seq_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_seq")

        dataset = RandomDatasetGenerator.generate_sequence_dataset(1000, {4: 0.33, 5: 0.33, 7: 0.33}, {"l1": {"a": 0.5, "b": 0.5}}, path / 'dataset')

        default_params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "sequence_length_distribution")
        sld = SequenceLengthDistribution.build_object(**{**default_params, **{"dataset": dataset,
                                                                               "result_path": path,
                                                                               "split_by_label": True,
                                                                              "plot_type": "LINE"}})

        self.assertTrue(sld.check_prerequisites())
        result = sld._generate()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)



    def test_sequence_lengths_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "seq_len_rec")


        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=50,
                                                                   chain_1_length_probabilities={4: 0.33, 5: 0.33, 7: 0.33},
                                                                   chain_2_length_probabilities={7: 0.33, 8: 0.33, 9: 0.33},
                                                                   labels={"l1": {"a": 0.5, "b": 0.5}}, path=path / 'dataset')

        default_params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "reports/", "sequence_length_distribution")
        sld = SequenceLengthDistribution.build_object(**{**default_params, **{"dataset": dataset,
                                                                              "result_path": path,
                                                                              "split_by_label": True,
                                                                              "as_fraction": True}})
        result = sld.generate_report()
        self.assertTrue(os.path.isfile(result.output_figures[0].path))

        shutil.rmtree(path)
