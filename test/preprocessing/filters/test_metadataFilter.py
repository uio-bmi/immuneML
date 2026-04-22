import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.analysis.criteria_matches.OperationType import OperationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.MetadataFilter import MetadataFilter
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestMetadataRepertoireFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process_repertoire_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "metadata_filter/"
        PathBuilder.remove_old_and_build(path)
        dataset = RepertoireDataset(repertoires=RepertoireBuilder.build([["ACF", "ACF", "ACF"],
                                                                       ["ACF", "ACF"],
                                                                       ["ACF", "ACF", "ACF", "ACF"]], path)[0],
                                    labels={"key1": [0, 1, 2], "key2": [1, 2]})

        df = pd.DataFrame(data={"key1": [0, 1, 2], "key2": [1, 1, 2]})
        df.to_csv(path/"metadata.csv")

        dataset.metadata_file = path/"metadata.csv"

        dataset1 = MetadataFilter(**{
            "criteria": {
                "type": OperationType.GREATER_THAN.name,
                "column": "key2",
                "threshold": 1
            },
            "result_path": path
        }).process_dataset(dataset, path / 'ex1')

        self.assertEqual(1, dataset1.get_example_count())

        dataset1 = MetadataFilter(**{
            'criteria': {
                'type': OperationType.IN.name,
                'values': [1],
                'column': 'key2'
            }
        }).process_dataset(dataset, path / 'ex2')

        self.assertEqual(2, dataset1.get_example_count())
        self.assertEqual([1], dataset1.labels['key2'])

        shutil.rmtree(path)

    def test_process_sequence_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "sequence_dataset/"
        PathBuilder.build(path)

        sequences = [ReceptorSequence(sequence_aa="AAACCC", sequence="AAACCC", sequence_id="1",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="ACACAC", sequence="ACACAC", sequence_id="2",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="CCCAAA", sequence="CCCAAA", sequence_id="3",
                                      metadata={"l1": 1})]

        dataset = SequenceDataset.build_from_objects(sequences, PathBuilder.build(path / 'data'), 'seq_dataset',
                                                     labels={"l1": [1, 2]})

        dataset1 = MetadataFilter(**{
            'criteria': {
                'type': OperationType.IN.name,
                'values': [1],
                'column': 'l1'
            }
        }).process_dataset(dataset, path / 'ex2')

        self.assertEqual(2, dataset1.get_example_count())
        self.assertEqual([1], dataset1.labels['l1'])

        shutil.rmtree(path)

    def test_process_receptor_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "receptor_dataset/"
        PathBuilder.build(path)

        receptors = [
            Receptor(chain_1=ReceptorSequence(sequence_aa="AAACCC", sequence="AAACCC", sequence_id="1a",
                                              locus="TRA", cell_id="cell1"),
                     chain_2=ReceptorSequence(sequence_aa="CCCAAA", sequence="CCCAAA", sequence_id="1b",
                                              locus="TRB", cell_id="cell1"),
                     chain_pair=ChainPair.TRA_TRB, receptor_id="r1", cell_id="cell1",
                     metadata={"l1": "1"}),
            Receptor(chain_1=ReceptorSequence(sequence_aa="ACACAC", sequence="ACACAC", sequence_id="2a",
                                              locus="TRA", cell_id="cell2"),
                     chain_2=ReceptorSequence(sequence_aa="CACACA", sequence="CACACA", sequence_id="2b",
                                              locus="TRB", cell_id="cell2"),
                     chain_pair=ChainPair.TRA_TRB, receptor_id="r2", cell_id="cell2",
                     metadata={"l1": "2"}),
            Receptor(chain_1=ReceptorSequence(sequence_aa="TTTTTT", sequence="TTTTTT", sequence_id="3a",
                                              locus="TRA", cell_id="cell3"),
                     chain_2=ReceptorSequence(sequence_aa="AAAAAA", sequence="AAAAAA", sequence_id="3b",
                                              locus="TRB", cell_id="cell3"),
                     chain_pair=ChainPair.TRA_TRB, receptor_id="r3", cell_id="cell3",
                     metadata={"l1": "1"}),
        ]

        dataset = ReceptorDataset.build_from_objects(receptors, PathBuilder.build(path / 'data'),
                                                     'rec_dataset', labels={"l1": ["1", "2"]})

        dataset1 = MetadataFilter(**{
            'criteria': {
                'type': OperationType.IN.name,
                'values': ["1"],
                'column': 'l1'
            }
        }).process_dataset(dataset, path / 'ex1')

        self.assertEqual(2, dataset1.get_example_count())
        self.assertEqual(["1"], dataset1.labels['l1'])

        dataset2 = MetadataFilter(**{
            'criteria': {
                'type': OperationType.IN.name,
                'values': ["2"],
                'column': 'l1'
            }
        }).process_dataset(dataset, path / 'ex2')

        self.assertEqual(1, dataset2.get_example_count())
        self.assertEqual(["2"], dataset2.labels['l1'])

        shutil.rmtree(path)
