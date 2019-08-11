import pickle
import shutil
from unittest import TestCase

import numpy as np

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.metadata.Sample import Sample
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestKmerFrequencyEncoder(TestCase):
    def test_encode(self):

        path = EnvironmentSettings.root_path + "test/tmp/kmerfreqenc/"

        PathBuilder.build(path)

        rep1 = Repertoire(sequences=[ReceptorSequence("AAA"), ReceptorSequence("ATA"), ReceptorSequence("ATA")],
                          metadata=RepertoireMetadata(sample=Sample(1), custom_params={"l1": 1, "l2": 2}))
        with open(path + "rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)

        rep2 = Repertoire(sequences=[ReceptorSequence("ATA"), ReceptorSequence("TAA"), ReceptorSequence("AAC")],
                          metadata=RepertoireMetadata(sample=Sample(2), custom_params={"l1": 0, "l2": 3}))
        with open(path + "rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(filenames=[path + "rep1.pkl", path + "rep2.pkl"])

        d1 = KmerFrequencyEncoder.encode(dataset, EncoderParams(
            result_path=path + "1/",
            label_configuration=lc,
            batch_size=2,
            learn_model=True,
            model={
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
                "reads": ReadsType.UNIQUE,
                "sequence_encoding": SequenceEncodingType.IDENTITY,
                "k": 3
            },
            filename="dataset.pkl"
        ))

        d2 = KmerFrequencyEncoder.encode(dataset, EncoderParams(
            result_path=path + "2/",
            label_configuration=lc,
            batch_size=2,
            learn_model=True,
            model={
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
                "reads": ReadsType.UNIQUE,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER,
                "k": 3
            },
            filename="dataset.csv"
        ))

        shutil.rmtree(path)

        self.assertTrue(isinstance(d1, RepertoireDataset))
        self.assertTrue(isinstance(d2, RepertoireDataset))
        self.assertEqual(0.67, np.round(d2.encoded_data.examples[0, 2], 2))
