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
        sequence = ReceptorSequence(sequence_aa="AHCDE", sequence=None, sequence_id='')
        kmers = IMGTGappedKmerEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 1, "max_gap": 1},
                                                                              label_config=LabelConfiguration(),
                                                                              region_type=RegionType.IMGT_CDR3,
                                                                              result_path=None))

        self.assertEqual({'AH_105', 'HC_106', 'CD_107', 'DE_116', 'A.C_105', 'H.D_106', 'C.E_107'},
                         set(kmers))

        sequence = ReceptorSequence(sequence_aa="CASSPRERATYEQCAY", sequence=None, sequence_id='')
        kmers = IMGTGappedKmerEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 1, "max_gap": 1},
                                                                              label_config=LabelConfiguration(),
                                                                              result_path="",
                                                                              region_type=RegionType.IMGT_CDR3))

        self.assertEqual({'CA_105', 'AS_106', 'SS_107', 'SP_108', 'PR_109', 'RE_110', 'ER_111',
                          'RA_111.1', 'AT_112.2', 'TY_112.1', 'YE_112', 'EQ_113', 'QC_114',
                          'CA_115', 'AY_116', 'C.S_105', 'A.S_106', 'S.P_107', 'S.R_108', 'P.E_109',
                          'R.R_110', 'E.A_111', 'R.T_111.1', 'A.Y_112.2', 'T.E_112.1', 'Y.Q_112',
                          'E.C_113', 'Q.A_114', 'C.Y_115'}, set(kmers))
