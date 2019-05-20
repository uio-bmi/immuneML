from unittest import TestCase

from source.data_model.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.encodings.EncoderParams import EncoderParams
from source.encodings.kmer_frequency.sequence_encoding.IMGTGappedKmerEncoder import IMGTGappedKmerEncoder
from source.environment.LabelConfiguration import LabelConfiguration


class TestIMGTGappedKmerEncoder(TestCase):
    def test_encode_sequence(self):
        sequence = ReceptorSequence("AHCDE", None, None)
        kmers = IMGTGappedKmerEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 1, "max_gap": 1},
                                                                              label_configuration=LabelConfiguration(),
                                                                              result_path=""))

        self.assertEqual({'AH///105', 'HC///106', 'CD///107', 'DE///116', 'A.C///105', 'H.D///106', 'C.E///107'},
                         set(kmers.features))
        self.assertEqual(kmers.feature_information_names, ["sequence", "imgt_position"])

        sequence = ReceptorSequence("CASSPRERATYEQCAY", None, None)
        kmers = IMGTGappedKmerEncoder.encode_sequence(sequence, EncoderParams(model={"k_left": 1, "max_gap": 1},
                                                                              label_configuration=LabelConfiguration(),
                                                                              result_path=""))

        self.assertEqual({'CA///105', 'AS///106', 'SS///107', 'SP///108', 'PR///109', 'RE///110', 'ER///111',
                          'RA///111.001', 'AT///112.002', 'TY///112.001', 'YE///112', 'EQ///113', 'QC///114',
                          'CA///115', 'AY///116', 'C.S///105', 'A.S///106', 'S.P///107', 'S.R///108', 'P.E///109',
                          'R.R///110', 'E.A///111', 'R.T///111.001', 'A.Y///112.002', 'T.E///112.001', 'Y.Q///112',
                          'E.C///113', 'Q.A///114', 'C.Y///115'}, set(kmers.features))
