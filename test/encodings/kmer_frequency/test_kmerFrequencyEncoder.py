import os
import shutil
from unittest import TestCase

import numpy as np

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.util.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder


class TestKmerFrequencyEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        path = EnvironmentSettings.root_path / "test/tmp/kmerfreqenc/"

        PathBuilder.build(path)

        rep1 = Repertoire.build_from_sequence_objects([ReceptorSequence("AAA", nucleotide_sequence="AAA", identifier="1"),
                                                       ReceptorSequence("ATA", nucleotide_sequence="ATA", identifier="2"),
                                                       ReceptorSequence("ATA", nucleotide_sequence="ATA", identifier='3')],
                                                      metadata={"l1": 1, "l2": 2, "subject_id": "1"}, path=path)

        rep2 = Repertoire.build_from_sequence_objects([ReceptorSequence("ATA", nucleotide_sequence="ATA", identifier="1"),
                                                       ReceptorSequence("TAA", nucleotide_sequence="TAA", identifier="2"),
                                                       ReceptorSequence("AAC", nucleotide_sequence="AAC", identifier="3")],
                                                      metadata={"l1": 0, "l2": 3, "subject_id": "2"}, path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])
        lc.add_label("l2", [0, 3])

        dataset = RepertoireDataset(repertoires=[rep1, rep2])

        encoder = KmerFrequencyEncoder.build_object(dataset, **{
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.IDENTITY.name,
                "sequence_type": SequenceType.AMINO_ACID.name,
                "k": 3
            })

        d1 = encoder.encode(dataset, EncoderParams(
            result_path=path / "1/",
            label_config=lc,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        encoder = KmerFrequencyEncoder.build_object(dataset, **{
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                "sequence_type": SequenceType.AMINO_ACID.name,
                "k": 3
            })

        d2 = encoder.encode(dataset, EncoderParams(
            result_path=path / "2/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
            filename="dataset.csv"
        ))

        encoder3 = KmerFrequencyEncoder.build_object(dataset, **{
            "normalization_type": NormalizationType.BINARY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "sequence_type": SequenceType.NUCLEOTIDE.name,
            "k": 3
        })

        d3 = encoder3.encode(dataset, EncoderParams(
            result_path=path / "3/",
            label_config=lc,
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        shutil.rmtree(path)

        self.assertTrue(isinstance(d1, RepertoireDataset))
        self.assertTrue(isinstance(d2, RepertoireDataset))
        self.assertEqual(0.67, np.round(d2.encoded_data.examples[0, 2], 2))
        self.assertEqual(0.0, np.round(d3.encoded_data.examples[0, 1], 2))
        self.assertTrue(isinstance(encoder, KmerFrequencyEncoder))
