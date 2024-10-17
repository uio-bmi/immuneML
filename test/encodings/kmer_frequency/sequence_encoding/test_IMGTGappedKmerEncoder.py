import os
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import RegionType
from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.sequence_encoding.IMGTGappedKmerEncoder import IMGTGappedKmerEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.LabelConfiguration import LabelConfiguration


class TestIMGTGappedKmerEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode_sequence(self):
        sequence = ReceptorSequence(sequence_aa="AHCDE", sequence=None, sequence_id=None)
        kmers = IMGTGappedKmerEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 1, "max_gap": 1},
                                                                              label_config=LabelConfiguration(),
                                                                              region_type=RegionType.IMGT_CDR3,
                                                                              result_path=None))

        self.assertEqual({'AH-105', 'HC-106', 'CD-107', 'DE-116', 'A.C-105', 'H.D-106', 'C.E-107'},
                         set(kmers))

        sequence = ReceptorSequence(sequence_aa="CASSPRERATYEQCAY", sequence=None, sequence_id=None)
        kmers = IMGTGappedKmerEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 1, "max_gap": 1},
                                                                              label_config=LabelConfiguration(),
                                                                              result_path="",
                                                                              region_type=RegionType.IMGT_CDR3))

        self.assertEqual({'CA-105', 'AS-106', 'SS-107', 'SP-108', 'PR-109', 'RE-110', 'ER-111',
                          'RA-111.1', 'AT-112.2', 'TY-112.1', 'YE-112', 'EQ-113', 'QC-114',
                          'CA-115', 'AY-116', 'C.S-105', 'A.S-106', 'S.P-107', 'S.R-108', 'P.E-109',
                          'R.R-110', 'E.A-111', 'R.T-111.1', 'A.Y-112.2', 'T.E-112.1', 'Y.Q-112',
                          'E.C-113', 'Q.A-114', 'C.Y-115'}, set(kmers))
