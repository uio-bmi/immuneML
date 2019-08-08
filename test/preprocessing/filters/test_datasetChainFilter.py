import pickle
import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.dataset.Dataset import Dataset
from source.data_model.receptor.receptor_sequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.data_model.repertoire.Repertoire import Repertoire
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.preprocessing.filters.DatasetChainFilter import DatasetChainFilter
from source.util.PathBuilder import PathBuilder


class TestDatasetChainFilter(TestCase):
    def test_process(self):
        rep1 = Repertoire(sequences=[ReceptorSequence("AAA", metadata=SequenceMetadata(chain="A"))])
        rep2 = Repertoire(sequences=[ReceptorSequence("AAC", metadata=SequenceMetadata(chain="B"))])

        path = EnvironmentSettings.root_path + "test/tmp/datasetchainfilter/"
        PathBuilder.build(path)
        with open(path + "rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)
        with open(path + "rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        metadata = pd.DataFrame({"CD": [1, 0]})
        metadata.to_csv(path + "metadata.csv")

        dataset = Dataset(filenames=[path + "rep1.pkl", path + "rep2.pkl"], metadata_file=path + "metadata.csv")

        dataset2 = DatasetChainFilter.process(dataset, {"keep_chain": Chain.A})

        self.assertEqual(1, len(dataset2.get_filenames()))
        self.assertEqual(2, len(dataset.get_filenames()))

        metadata_dict = dataset2.get_metadata(["CD"])
        self.assertEqual(1, len(metadata_dict["CD"]))
        self.assertEqual(1, metadata_dict["CD"][0])

        for rep in dataset2.get_data():
            self.assertEqual("AAA", rep.sequences[0].get_sequence())

        shutil.rmtree(path)
