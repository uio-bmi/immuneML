import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.VGeneIMGTKmerEncoder import VGeneIMGTKmerEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestIMGTKmerSequenceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode_sequence(self):
        sequence = ReceptorSequence(sequence_aa="CASSPRERATYEQCASSPRERATYEQCASSPRERATYEQ", sequence=None,
                                    sequence_id='', v_call='V1-1')
        result = VGeneIMGTKmerEncoder.encode_sequence(sequence, EncoderParams(
            model={"k": 3},
            label_config=LabelConfiguration(),
            result_path=None,
            region_type=RegionType.IMGT_CDR3))

        self.assertEqual({'V1-1_CAS_105', 'V1-1_ASS_106', 'V1-1_SSP_107', 'V1-1_SPR_108', 'V1-1_PRE_109',
                          'V1-1_RER_110', 'V1-1_ERA_111',
                          'V1-1_RAT_111.1', 'V1-1_ATY_111.2', 'V1-1_TYE_111.3', 'V1-1_YEQ_111.4', 'V1-1_EQC_111.5',
                          'V1-1_QCA_111.6', 'V1-1_CAS_111.7', 'V1-1_ASS_111.8', 'V1-1_SSP_111.9', 'V1-1_SPR_111.10',
                          'V1-1_PRE_111.11', 'V1-1_RER_111.12', 'V1-1_ERA_111.13', 'V1-1_RAT_112.13', 'V1-1_ATY_112.12',
                          'V1-1_TYE_112.11', 'V1-1_YEQ_112.10', 'V1-1_EQC_112.9', 'V1-1_QCA_112.8', 'V1-1_CAS_112.7',
                          'V1-1_ASS_112.6', 'V1-1_SSP_112.5', 'V1-1_SPR_112.4', 'V1-1_PRE_112.3', 'V1-1_RER_112.2',
                          'V1-1_ERA_112.1', 'V1-1_RAT_112', 'V1-1_ATY_113', 'V1-1_TYE_114', 'V1-1_YEQ_115'},
                         set(result))

        self.assertEqual(len(result), len(sequence.sequence_aa) - 3 + 1)

        sequence = ReceptorSequence(sequence_aa="AHCDE", sequence=None, sequence_id='', v_call='V1-1')
        result = VGeneIMGTKmerEncoder.encode_sequence(sequence, EncoderParams(
            model={"k": 3}, region_type=RegionType.IMGT_CDR3,
            label_config=LabelConfiguration(),
            result_path=""))

        self.assertEqual({'V1-1_AHC_105', 'V1-1_HCD_106', 'V1-1_CDE_107'},
                         set(result))

        self.assertEqual(len(result), len(sequence.sequence_aa) - 3 + 1)
        self.assertEqual(
            VGeneIMGTKmerEncoder.encode_sequence(
                sequence,
                EncoderParams(model={"k": 25}, region_type=RegionType.IMGT_CDR3,
                              label_config=LabelConfiguration(),
                              result_path="")
            ),
            None
        )
