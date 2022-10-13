import os
import shutil
from unittest import TestCase


from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.example_weighting.ExampleWeightingParams import ExampleWeightingParams
from immuneML.example_weighting.importance_weighting.ImportanceWeighting import ImportanceWeighting
from immuneML.util.PathBuilder import PathBuilder


class TestImportanceWeighting(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def _prepare_dataset(self, path, sequences=("AA", "CA", "AC", "CA")):
        sequences = [ReceptorSequence(amino_acid_sequence=sequences[0], identifier="1",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence=sequences[1], identifier="2",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence=sequences[2], identifier="3",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence=sequences[3], identifier="4",
                                      metadata=SequenceMetadata(custom_params={"l1": 1}))]


        PathBuilder.build(path)
        return SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

    def test_compute_weights(self):
        path = EnvironmentSettings.tmp_test_path / "importance_weighting/"
        dataset = self._prepare_dataset(path)

        importance_weighter = ImportanceWeighting.build_object(dataset,
                                                               **{"baseline_dist": "uniform",
                                                                  "dataset_dist": "mutagenesis",
                                                                  "pseudocount_value": 1,
                                                                  "lower_weight_limit": None,
                                                                  "upper_weight_limit": None,
                                                                  "export_weights": True,
                                                                  "name": "my_weighting"}
                                                               )

        w = importance_weighter.compute_weights(dataset, ExampleWeightingParams(result_path=path, learn_model=True))

        expected_pos_freq = {idx: {char: 0.2 for char in EnvironmentSettings.get_sequence_alphabet(SequenceType.AMINO_ACID)} for idx in range(2)}
        expected_pos_freq[0]["A"] = 0.6
        expected_pos_freq[0]["C"] = 0.6
        expected_pos_freq[1]["A"] = 0.8
        expected_pos_freq[1]["C"] = 0.4

        self.assertEqual(importance_weighter.alphabet_size, 20)
        self.assertDictEqual(importance_weighter.dataset_positional_frequences, expected_pos_freq)

        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("AA")), (0.05*0.05)/(0.6*0.8))
        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("AC")), (0.05*0.05)/(0.6*0.4))
        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("CC")), (0.05*0.05)/(0.6*0.4))
        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("CA")), (0.05*0.05)/(0.6*0.8))
        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("FF")), (0.05*0.05)/(0.2*0.2))
        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("CF")), (0.05*0.05)/(0.6*0.2))
        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("AF")), (0.05*0.05)/(0.6*0.2))
        self.assertEqual(importance_weighter._compute_sequence_weight(ReceptorSequence("FC")), (0.05*0.05)/(0.2*0.4))

        self.assertListEqual(w, [(0.05*0.05)/(0.6*0.8), (0.05*0.05)/(0.6*0.8), (0.05*0.05)/(0.6*0.4), (0.05*0.05)/(0.6*0.8)])

        dataset2 = self._prepare_dataset(path, sequences=["FF", "CF", "AF", "FC"])

        w = importance_weighter.compute_weights(dataset2, ExampleWeightingParams(result_path=path, learn_model=False))
        self.assertListEqual(w, [(0.05*0.05)/(0.2*0.2), (0.05*0.05)/(0.6*0.2), (0.05*0.05)/(0.6*0.2), (0.05*0.05)/(0.2*0.4)])

        self.assertTrue((path / f"dataset_{dataset.identifier}_example_weights.tsv").is_file())

        shutil.rmtree(path)

