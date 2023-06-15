import os
import shutil
from unittest import TestCase

import numpy

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.SequenceDataset import SequenceDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFreqSequenceEncoder import KmerFreqSequenceEncoder
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType


class TestKmerFreqSequenceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test(self):

        sequences = [ReceptorSequence(amino_acid_sequence="AAACCC", nucleotide_sequence="AAACCC", identifier="1",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="ACACAC", nucleotide_sequence="ACACAC", identifier="2",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="CCCAAA", nucleotide_sequence="CCCAAA", identifier="3",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="AAACCC", nucleotide_sequence="AAACCC", identifier="4",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="ACACAC", nucleotide_sequence="ACACAC", identifier="5",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="CCCAAA", nucleotide_sequence="CCCAAA", identifier="6",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="AAACCC", nucleotide_sequence="AAACCC", identifier="7",
                                      metadata=SequenceMetadata(custom_params={"l1": 1})),
                     ReceptorSequence(amino_acid_sequence="ACACAC", nucleotide_sequence="ACACAC", identifier="8",
                                      metadata=SequenceMetadata(custom_params={"l1": 2})),
                     ReceptorSequence(amino_acid_sequence="CCCAAA", nucleotide_sequence="CCCAAA", identifier="9",
                                      metadata=SequenceMetadata(custom_params={"l1": 1}))]

        path = EnvironmentSettings.tmp_test_path / "kmrefreqseqfacencoder/"
        PathBuilder.remove_old_and_build(path)
        dataset = SequenceDataset.build_from_objects(sequences, 100, PathBuilder.build(path / 'data'), 'd2')

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        encoder = KmerFreqSequenceEncoder.build_object(dataset, **{
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                "sequence_type": SequenceType.NUCLEOTIDE.name,
                "k": 3
            })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "2/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv"
        ))

        self.assertEqual(9, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4', '5', '6', '7', '8', '9']))
        self.assertTrue(numpy.array_equal(encoded_dataset.encoded_data.examples[0].A, encoded_dataset.encoded_data.examples[3].A))

        shutil.rmtree(path)
