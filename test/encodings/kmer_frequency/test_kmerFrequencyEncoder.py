import pickle
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.data_model.metadata.Sample import Sample
from source.data_model.repertoire.Repertoire import Repertoire
from source.data_model.repertoire.RepertoireMetadata import RepertoireMetadata
from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from source.encodings.kmer_frequency.NormalizationType import NormalizationType
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.IdentitySequenceEncoder import IdentitySequenceEncoder
from source.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestKmerFrequencyEncoder(TestCase):
    def test_encode(self):
        PathBuilder.build("./tmp/")

        rep1 = Repertoire(sequences=[ReceptorSequence("AAA"), ReceptorSequence("ATA"), ReceptorSequence("ATA")],
                          metadata=RepertoireMetadata(Sample(1, custom_params={"l1": 1, "l2": 2})))
        with open("./tmp/rep1.pkl", "wb") as file:
            pickle.dump(rep1, file)

        rep2 = Repertoire(sequences=[ReceptorSequence("ATA"), ReceptorSequence("TAA"), ReceptorSequence("AAC")],
                          metadata=RepertoireMetadata(Sample(2, custom_params={"l1": 0, "l2": 3})))
        with open("./tmp/rep2.pkl", "wb") as file:
            pickle.dump(rep2, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = Dataset(filenames=["./tmp/rep1.pkl", "./tmp/rep2.pkl"])

        d1 = KmerFrequencyEncoder.encode(dataset, {
            "result_path": "./tmp/",
            "label_configuration": lc,
            "batch_size": 2,
            "learn_model": True,
            "vectorizer_path": "./tmp/",
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
            "reads": ReadsType.UNIQUE,
            "sequence_encoding_strategy": SequenceEncodingType.IDENTITY,
            "k": 3
        })

        d2 = KmerFrequencyEncoder.encode(dataset, {
            "result_path": "./tmp/",
            "label_configuration": lc,
            "batch_size": 2,
            "learn_model": True,
            "vectorizer_path": "./tmp/",
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY,
            "reads": ReadsType.UNIQUE,
            "sequence_encoding_strategy": SequenceEncodingType.IDENTITY,
            "k": 3
        })

        shutil.rmtree("./tmp/")

        self.assertTrue(isinstance(d1, Dataset))
        self.assertTrue(isinstance(d2, Dataset))
        self.assertTrue("repertoires" in d1.encoded_data)
        self.assertTrue("repertoires" in d2.encoded_data)
