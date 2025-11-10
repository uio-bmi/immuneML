import os
import shutil
from unittest import TestCase

import numpy

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset
from immuneML.data_model.SequenceSet import ReceptorSequence, Receptor
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFreqReceptorEncoder import KmerFreqReceptorEncoder
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.ReadsType import ReadsType


class TestKmerFreqReceptorEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test(self):
        receptors = [Receptor(chain_1=ReceptorSequence(sequence_aa="AAACCC", locus='alpha', cell_id='1'),
                              chain_2=ReceptorSequence(sequence_aa="AAACCC", locus='beta', cell_id="1"),
                              receptor_id="1", cell_id="1", chain_pair=ChainPair.TRA_TRB),
                     Receptor(chain_1=ReceptorSequence(sequence_aa="AAA", locus='alpha', cell_id="2"),
                              chain_2=ReceptorSequence(sequence_aa="CCC", locus='beta', cell_id="2"),
                              receptor_id="2", cell_id="2", chain_pair=ChainPair.TRA_TRB),
                     Receptor(chain_1=ReceptorSequence(sequence_aa="AAACCC", locus='alpha', cell_id="3"),
                              chain_2=ReceptorSequence(sequence_aa="AAACCC", locus='beta', cell_id="3"),
                              receptor_id="3", cell_id="3", chain_pair=ChainPair.TRA_TRB),
                     Receptor(chain_1=ReceptorSequence(sequence_aa="AAA", locus='alpha', cell_id="4"),
                              chain_2=ReceptorSequence(sequence_aa="CCC", locus='beta', cell_id="4"),
                              receptor_id="4", cell_id="4", chain_pair=ChainPair.TRA_TRB)]

        path = EnvironmentSettings.tmp_test_path / "kmer_receptor_frequency/"
        PathBuilder.remove_old_and_build(path / 'data')
        dataset = ReceptorDataset.build_from_objects(receptors, path=path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        encoder = KmerFreqReceptorEncoder.build_object(dataset, **{
            "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
            "reads": ReadsType.UNIQUE.name,
            "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
            "sequence_type": SequenceType.AMINO_ACID.name,
            "k": 3
        })

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "2/",
            label_config=lc,
            pool_size=2,
            learn_model=True,
            model={},
            encode_labels=False
        ))

        self.assertEqual(4, encoded_dataset.encoded_data.examples.shape[0])
        self.assertTrue(all(identifier in encoded_dataset.encoded_data.example_ids
                            for identifier in ['1', '2', '3', '4']))
        self.assertTrue(
            numpy.array_equal(encoded_dataset.encoded_data.examples[0].toarray(), encoded_dataset.encoded_data.examples[2].toarray()))
        print(encoded_dataset.encoded_data.feature_names)
        self.assertTrue(all(feature_name in encoded_dataset.encoded_data.feature_names for feature_name in
                            ["alpha_AAA", "alpha_AAC", "beta_CCC"]))

        shutil.rmtree(path)
