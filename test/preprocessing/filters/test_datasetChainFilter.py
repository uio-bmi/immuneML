import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.SequenceRepertoire import SequenceRepertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.filters.DatasetChainFilter import DatasetChainFilter
from source.util.PathBuilder import PathBuilder


class TestDatasetChainFilter(TestCase):
    def test_process(self):

        path = EnvironmentSettings.root_path + "test/tmp/datasetchainfilter/"
        PathBuilder.build(path)

        rep1 = SequenceRepertoire.build_from_sequence_objects([ReceptorSequence("AAA", metadata=SequenceMetadata(chain="A"),
                                                                                identifier="1")], path=path, metadata={})
        rep2 = SequenceRepertoire.build_from_sequence_objects([ReceptorSequence("AAC", metadata=SequenceMetadata(chain="B"),
                                                                                identifier="2")], path=path, metadata={})

        metadata = pd.DataFrame({"CD": [1, 0]})
        metadata.to_csv(path + "metadata.csv")

        dataset = RepertoireDataset(repertoires=[rep1, rep2], metadata_file=path + "metadata.csv")

        dataset2 = DatasetChainFilter.process(dataset, {"keep_chain": "A", "result_path": path + "results/"})

        self.assertEqual(1, len(dataset2.get_data()))
        self.assertEqual(2, len(dataset.get_data()))

        metadata_dict = dataset2.get_metadata(["CD"])
        self.assertEqual(1, len(metadata_dict["CD"]))
        self.assertEqual(1, metadata_dict["CD"][0])

        for rep in dataset2.get_data():
            self.assertEqual("AAA", rep.sequences[0].get_sequence())

        shutil.rmtree(path)
