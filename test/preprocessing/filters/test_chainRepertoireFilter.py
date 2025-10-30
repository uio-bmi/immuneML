import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.SequenceSet import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.ChainRepertoireFilter import ChainRepertoireFilter
from immuneML.util.PathBuilder import PathBuilder


class TestChainRepertoireFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "chain_filter/")

        rep1 = Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAA", locus="ALPHA",
                                                                 sequence_id="1")], result_path=path)
        rep2 = Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAC", locus="BETA",
                                                                 sequence_id="2")], result_path=path)

        metadata = pd.DataFrame({"CD": [1, 0]})
        metadata.to_csv(path / "metadata.csv")

        dataset = RepertoireDataset(repertoires=[rep1, rep2], metadata_file=path / "metadata.csv")

        dataset2 = ChainRepertoireFilter(**{"keep_chain": "ALPHA"}).process_dataset(dataset, path / "results")

        self.assertEqual(1, len(dataset2.get_data()))
        self.assertEqual(2, len(dataset.get_data()))

        metadata_dict = dataset2.get_metadata(["CD"])
        self.assertEqual(1, len(metadata_dict["CD"]))
        self.assertEqual(1, metadata_dict["CD"][0])

        for rep in dataset2.get_data():
            self.assertEqual("AAA", rep.sequences()[0].get_sequence())

        self.assertRaises(AssertionError, ChainRepertoireFilter(**{"keep_chain": "GAMMA"}).process_dataset, dataset,
                          path / "results")

        rep1 = Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAA", locus="ALPHA",
                                                                 sequence_id="1"),
                                                ReceptorSequence(sequence_aa="AAA", locus="BETA",
                                                                 sequence_id="1")
                                                ], result_path=path)
        rep2 = Repertoire.build_from_sequences([ReceptorSequence(sequence_aa="AAC", locus="BETA",
                                                                 sequence_id="2")], result_path=path)

        new_input_dataset = RepertoireDataset(repertoires=[rep1, rep2], metadata_file=path / "metadata.csv")

        dataset3 = ChainRepertoireFilter(**{"keep_chain": "BETA", "remove_only_sequences": True}).process_dataset(new_input_dataset, path / "results2")
        self.assertEqual(2, len(dataset3.get_data()))

        shutil.rmtree(path)
