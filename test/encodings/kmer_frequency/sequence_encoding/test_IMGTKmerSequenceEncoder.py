import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from immuneML.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.IMGTKmerSequenceEncoder import IMGTKmerSequenceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestIMGTKmerSequenceEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode_sequence(self):
        sequence = ReceptorSequence("CASSPRERATYEQCASSPRERATYEQCASSPRERATYEQ", None, None, metadata=SequenceMetadata(region_type="IMGT_CDR3"))
        result = IMGTKmerSequenceEncoder.encode_sequence(sequence, EncoderParams(
                                                                    model={"k": 3},
                                                                    label_config=LabelConfiguration(),
                                                                    result_path=""))

        self.assertEqual({'CAS-105', 'ASS-106', 'SSP-107', 'SPR-108', 'PRE-109', 'RER-110', 'ERA-111',
                          'RAT-111.1', 'ATY-111.2', 'TYE-111.3', 'YEQ-111.4', 'EQC-111.5',
                          'QCA-111.6', 'CAS-111.7', 'ASS-111.8', 'SSP-111.9', 'SPR-111.10',
                          'PRE-111.11', 'RER-111.12', 'ERA-111.13', 'RAT-112.13', 'ATY-112.12',
                          'TYE-112.11', 'YEQ-112.10', 'EQC-112.9', 'QCA-112.8', 'CAS-112.7',
                          'ASS-112.6', 'SSP-112.5', 'SPR-112.4', 'PRE-112.3', 'RER-112.2',
                          'ERA-112.1', 'RAT-112', 'ATY-113', 'TYE-114', 'YEQ-115'},
                         set(result))

        self.assertEqual(len(result), len(sequence.get_sequence()) - 3 + 1)

        sequence = ReceptorSequence("AHCDE", None, None, metadata=SequenceMetadata(region_type="IMGT_CDR3"))
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
