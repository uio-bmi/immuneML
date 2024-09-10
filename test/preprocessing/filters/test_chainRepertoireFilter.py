import os
import shutil
from unittest import TestCase

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.preprocessing.filters.ChainRepertoireFilter import ChainRepertoireFilter
from immuneML.util.PathBuilder import PathBuilder


class TestChainRepertoireFilter(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_process(self):

        path = EnvironmentSettings.root_path / "test/tmp/chain_filter/"
        PathBuilder.build(path)

        rep1 = Repertoire.build_from_sequence_objects([ReceptorSequence("AAA", metadata=SequenceMetadata(locus="A"),
                                                                        sequence_id="1")], path=path, metadata={})
        rep2 = Repertoire.build_from_sequence_objects([ReceptorSequence("AAC", metadata=SequenceMetadata(locus="B"),
                                                                        sequence_id="2")], path=path, metadata={})

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
            self.assertEqual("AAA", rep.sequences[0].get_sequence())

        self.assertRaises(AssertionError, ChainRepertoireFilter(**{"keep_chain": "GAMMA"}).process_dataset, dataset, path / "results")

        shutil.rmtree(path)
