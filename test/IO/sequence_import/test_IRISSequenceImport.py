import os
import shutil
from unittest import TestCase

from source.IO.sequence_import.IRISSequenceImport import IRISSequenceImport
from source.caching.CacheType import CacheType
from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestIRISSequenceImport(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_import_sequences(self):
        path = EnvironmentSettings.tmp_test_path + "importseqsiris/sequences.csv"
        PathBuilder.build(os.path.dirname(path))
        with open(path, "w") as file:
            file.write(
                "Cell type	Clonotype ID	Chain: TRA (1)	TRA - V gene (1)	TRA - D gene (1)	TRA - J gene (1)	Chain: TRA (2)	TRA - V gene (2)	\
                TRA - D gene (2)	TRA - J gene (2)	Chain: TRB (1)	TRB - V gene (1)	TRB - D gene (1)	TRB - J gene (1)	Chain: TRB (2)	TRB - V \
                gene (2)	TRB - D gene (2)	TRB - J gene (2)\n\
                TCR_AB	181	LVGG	TRAV4*01	null	TRAJ4*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	591	AL	TRAV9-2*01	null	TRAJ21*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	1051	VVNII	TRAV12-1*01	null	TRAJ3*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	1341	LNKLT	TRAV2*01	null	TRAJ10*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	1411	AVLY	TRAV8-3*01	null	TRAJ18*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	1421	AT	TRAV12-3*01	null	TRAJ17*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	1671	AVLI	TRAV12*01	null	TRAJ33*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	1901	LVGKLI	TRAV4*01	null	TRAJ4*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	2021	YSSASKII	TRAV2-1*01	null	TRAJ3*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	2251	ARLY	TRAV4/DV5*01	null	TRAJ18*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	2791	IEFN	TRAV26-1*01	null	TRAJ20*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	3031	TLGRLY	TRAV8-3*01	null	TRAJ18*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	3241	AVGLY	TRAV8-3*01	null	TRAJ18*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	3511	KII	TRAV12-1*01	null	TRAJ3*01	null	null	null	null	null	null	null	null	null	null	null	null\n\
                TCR_AB	3821	LVGD	TRAV8*01	null	TRAJ4*01	null	null	null	null	null	null	null	null	null	null	null	null\n")

        sequences = IRISSequenceImport.import_items(path)

        self.assertEqual(15, len(sequences))
        self.assertTrue(all(isinstance(sequence, ReceptorSequence) for sequence in sequences))
        self.assertEqual("LVGG", sequences[0].get_sequence())

        shutil.rmtree(os.path.dirname(path))

    def test_import_paired_sequences(self):
        path = EnvironmentSettings.tmp_test_path + "importseqsiris/sequences.csv"
        PathBuilder.build(os.path.dirname(path))

        with open(path, "w") as file:
            file.write(
                "Cell type	Clonotype ID	Chain: TRA (1)	TRA - V gene (1)	TRA - D gene (1)	TRA - J gene (1)	Chain: TRA (2)	TRA - V gene (2)	\
                TRA - D gene (2)	TRA - J gene (2)	Chain: TRB (1)	TRB - V gene (1)	TRB - D gene (1)	TRB - J gene (1)	Chain: TRB (2)	TRB - V \
                gene (2)	TRB - D gene (2)	TRB - J gene (2)\n\
                TCR_AB	540891	ATDIWSNFGNEKLT	TRAV17*01		TRAJ48*01	null	null	null	null	SARVRNYQETQY	TRBV20-1*01	TRBD1*01	TRBJ2-5*01	null	null	null	null\n\
                TCR_AB	540892	AASAGDDKII	TRAV29/DV5*01		TRAJ30*01	null	null	null	null	ASRPTGTVDYEQY	TRBV5-1*01	TRBD1*01	TRBJ2-7*01	null	null	null	null\n\
                TCR_AB	540893	AAYTSGTYKYI	TRAV8-1*01		TRAJ40*01	null	null	null	null	ASSLTGMNTEAF	TRBV11-1*01	TRBD2*01	TRBJ1-1*01	null	null	null	null\n\
                TCR_AB	54084	ALLSRSGGYQKVT	TRAV12-2*01		TRAJ13*02	null	null	null	null	SARDNQETQY	TRBV20-1*01	TRBD1*01	TRBJ2-5*01	null	null	null	null\n\
                TCR_AB	540895	AYRSRIQGAQKLV	TRAV38-2/DV8*01		TRAJ54*01	null	null	null	null	ASSHGTSGSGEQY	TRBV7-9*01	TRBD2*02	TRBJ2-7*01	null	null	null	null\n")

        paired_sequences = IRISSequenceImport.import_items(path, paired=True)

        self.assertEqual(5, len(paired_sequences))
        self.assertTrue(all(isinstance(sequence, TCABReceptor) for sequence in paired_sequences))

        self.assertEqual("ATDIWSNFGNEKLT", paired_sequences[0].alpha.get_sequence())
        self.assertEqual("SARVRNYQETQY", paired_sequences[0].beta.get_sequence())

        shutil.rmtree(os.path.dirname(path))
