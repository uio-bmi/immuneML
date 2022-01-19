import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.IMGTKmerSequenceEncoder import IMGTKmerSequenceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestIMGTKmerSequenceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode_sequence(self):
        sequence = ReceptorSequence("CASSPRERATYEQCASSPRERATYEQCASSPRERATYEQ", None, None)
        result = IMGTKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(
                                                                    model={"k": 3},
                                                                    label_config=LabelConfiguration(),
                                                                    result_path=""))

        self.assertEqual({'CAS-105', 'ASS-106', 'SSP-107', 'SPR-108', 'PRE-109', 'RER-110', 'ERA-111',
                          'RAT-111.001', 'ATY-111.002', 'TYE-111.003', 'YEQ-111.004', 'EQC-111.005',
                          'QCA-111.006', 'CAS-111.007', 'ASS-111.008', 'SSP-111.009', 'SPR-111.01',
                          'PRE-111.011', 'RER-111.012', 'ERA-111.013', 'RAT-112.013', 'ATY-112.012',
                          'TYE-112.011', 'YEQ-112.01', 'EQC-112.009', 'QCA-112.008', 'CAS-112.007',
                          'ASS-112.006', 'SSP-112.005', 'SPR-112.004', 'PRE-112.003', 'RER-112.002',
                          'ERA-112.001', 'RAT-112', 'ATY-113', 'TYE-114', 'YEQ-115'},
                         set(result))

        self.assertEqual(len(result), len(sequence.get_sequence()) - 3 + 1)

        sequence = ReceptorSequence("AHCDE", None, None)
        result = IMGTKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(
                                                                    model={"k": 3},
                                                                    label_config=LabelConfiguration(),
                                                                    result_path=""))

        self.assertEqual({'AHC-105', 'HCD-106', 'CDE-107'},
                         set(result))

        self.assertEqual(len(result), len(sequence.get_sequence()) - 3 + 1)
        self.assertEqual(
            IMGTKmerSequenceEncoder.encode_sequence(
                              sequence,
                              EncoderParams(model={"k": 25},
                                            label_config=LabelConfiguration(),
                                            result_path="")
            ),
            None
        )
