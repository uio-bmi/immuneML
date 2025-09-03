import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.IMGTKmerSequenceEncoder import IMGTKmerSequenceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestIMGTKmerSequenceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode_sequence(self):
        sequence = ReceptorSequence(sequence_aa="CASSPRERATYEQCASSPRERATYEQCASSPRERATYEQ", sequence=None, sequence_id='')
        result = IMGTKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(
            model={"k": 3},
            label_config=LabelConfiguration(),
            result_path="",
            region_type=RegionType.IMGT_CDR3))

        self.assertEqual({'CAS_105', 'ASS_106', 'SSP_107', 'SPR_108', 'PRE_109', 'RER_110', 'ERA_111',
                          'RAT_111.1', 'ATY_111.2', 'TYE_111.3', 'YEQ_111.4', 'EQC_111.5',
                          'QCA_111.6', 'CAS_111.7', 'ASS_111.8', 'SSP_111.9', 'SPR_111.10',
                          'PRE_111.11', 'RER_111.12', 'ERA_111.13', 'RAT_112.13', 'ATY_112.12',
                          'TYE_112.11', 'YEQ_112.10', 'EQC_112.9', 'QCA_112.8', 'CAS_112.7',
                          'ASS_112.6', 'SSP_112.5', 'SPR_112.4', 'PRE_112.3', 'RER_112.2',
                          'ERA_112.1', 'RAT_112', 'ATY_113', 'TYE_114', 'YEQ_115'},
                         set(result))

        self.assertEqual(len(result), len(sequence.sequence_aa) - 3 + 1)

        sequence = ReceptorSequence(sequence_aa="AHCDE", sequence=None, sequence_id='')
        result = IMGTKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(
            model={"k": 3}, region_type=RegionType.IMGT_CDR3,
            label_config=LabelConfiguration(),
            result_path=""))

        self.assertEqual({'AHC_105', 'HCD_106', 'CDE_107'},
                         set(result))

        self.assertEqual(len(result), len(sequence.sequence_aa) - 3 + 1)
        self.assertEqual(
            IMGTKmerSequenceEncoder.encode_sequence(
                sequence,
                EncoderParams(model={"k": 25}, region_type=RegionType.IMGT_CDR3,
                              label_config=LabelConfiguration(),
                              result_path="")
            ),
            None
        )
