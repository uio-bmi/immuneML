import pickle
import shutil
from unittest import TestCase

import numpy

from source.analysis.data_manipulation.NormalizationType import NormalizationType
from source.data_model.dataset.SequenceDataset import SequenceDataset
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.KmerFreqSequenceEncoder import KmerFreqSequenceEncoder
from source.encodings.kmer_frequency.ReadsType import ReadsType
from source.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.PathBuilder import PathBuilder


class TestKmerFreqSequenceEncoder(TestCase):

    def test(self):

        sequences = [ReceptorSequence(amino_acid_sequence="AAACCC", identifier="1", metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="ACACAC", identifier="2", metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="CCCAAA", identifier="3", metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="AAACCC", identifier="4", metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="ACACAC", identifier="5", metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="CCCAAA", identifier="6", metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="AAACCC", identifier="7", metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="ACACAC", identifier="8", metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="CCCAAA", identifier="9", metadata=SequenceMetadata(custom_params={"l1": 1}))]

        path = EnvironmentSettings.tmp_test_path + "kmrefreqseqfacencoder/"
        PathBuilder.build(path)
        filename = "{}sequences.pkl".format(path)
        with open(filename, "wb") as file:
            pickle.dump(sequences, file)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = SequenceDataset(params={"l1": [1, 2]}, filenames=[filename], identifier="d1")

        encoder = KmerFreqSequenceEncoder.build_object(dataset, **{
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                "k": 3
            })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path + "2/",
            label_configuration=lc,
            batch_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv"
        ))

        self.assertEqual(9, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4', '5', '6', '7', '8', '9']))
        self.assertTrue(numpy.array_equal(encoded_dataset.encoded_data.examples[0].A, encoded_dataset.encoded_data.examples[3].A))

        shutil.rmtree(path)
