import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.MiXCRImport import MiXCRImport
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestMiXCRLoader(TestCase):

    def create_dummy_dataset(self, path, add_metadata):
        file1_content = """cloneId	cloneCount	cloneFraction	targetSequences	targetQualities	allVHitsWithScore	allDHitsWithScore	allJHitsWithScore	allCHitsWithScore	allVAlignments	allDAlignments	allJAlignments	allCAlignments	nSeqFR1	minQualFR1	nSeqCDR1	minQualCDR1	nSeqFR2	minQualFR2	nSeqCDR2	minQualCDR2	nSeqFR3	minQualFR3	nSeqCDR3	minQualCDR3	nSeqFR4	minQualFR4	aaSeqFR1	aaSeqCDR1	aaSeqFR2	aaSeqCDR2	aaSeqFR3	aaSeqCDR3	aaSeqFR4	refPoints
0	956023.0	0.17165008499706622	TGTGCTCTAGTAACTGACAGCTGGGGGAAATTGCAGTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV6*00(716)		TRAJ24*00(291.3)	TRAC*00(75.5)	615|625|648|0|10||50.0		24|52|83|11|39||140.0												TGTGCTCTAGTAACTGACAGCTGGGGGAAATTGCAGTTT	41								CALVTDSWGKLQF		:::::::::0:-3:10:::::11:-4:39:::
1	102075.0	0.018327155754699974	TGTGCAGAGGCGTTCCTCGAAATACTGGAGGCTTCAAAACTATCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV13-2*00(657.7)		TRAJ9*00(299.4)	TRAC*00(74.9)	586|595|619|0|9||45.0		21|50|81|18|47||145.0												TGTGCAGAGGCGTTCCTCGAAATACTGGAGGCTTCAAAACTATCTTT	41								CAEAFLEI_GGFKTIF		:::::::::0:-4:9:::::18:-1:47:::
2	90101.0	0.016177272208221627	TGTGCTCTAAGGATAACTCAGGGCGGATCTGAAAAGCTGGTCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV16*00(606.1)		TRAJ57*00(319.5)	TRAC*00(74.9)	529|540|563|0|11||55.0		19|52|83|12|45||165.0												TGTGCTCTAAGGATAACTCAGGGCGGATCTGAAAAGCTGGTCTTT	41								CALRITQGGSEKLVF		:::::::::0:-3:11:::::12:1:45:::
3	69706.0	0.012515431976851496	TGCATCCCTAACTTTGGAAATGAGAAATTAACCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV26-2*00(603.6)		TRAJ48*00(299.4)	TRAC*00(75.1)	871|878|906|0|7||35.0		23|52|83|7|36||145.0												TGCATCCCTAACTTTGGAAATGAGAAATTAACCTTT	41								CIPNFGNEKLTF		:::::::::0:-8:7:::::7:-3:36:::
4	56658.0	0.01017271604947138	TGTGCATCCAGGGGCGGCACTGCCAGTAAACTCACCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV29DV5*00(635.9)		TRAJ44*00(279.5)	TRAC*00(76.3)	609|615|642|0|6||30.0		27|52|83|14|39||125.0												TGTGCATCCAGGGGCGGCACTGCCAGTAAACTCACCTTT	41								CASRGGTASKLTF		:::::::::0:-7:6:::::14:-7:39:::
5	55692.0	0.009999274634246887	TGTGCAGCAAGCATCCGGTCAGGAACCTACAAATACATCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV23DV6*00(719.1)		TRAJ40*00(274.6)	TRAC*00(75.2)	597|611|630|0|14||70.0		26|50|81|18|42||120.0												TGTGCAGCAAGCATCCGGTCAGGAACCTACAAATACATCTTT	41								CAASIRSGTYKYIF		:::::::::0:1:14:::::18:-6:42:::
6	43466.0	0.007804145501188235	TGTGCTTATAGGCGGCCTGGGGCTGGGAGTTACCAACTCACTTTC	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV38-2DV8*00(643.4)		TRAJ28*00(299.5)	TRAC*00(74.8)	682|694|718|0|12||60.0		26|55|86|16|45||145.0												TGTGCTTATAGGCGGCCTGGGGCTGGGAGTTACCAACTCACTTTC	41								CAYRRPGAGSYQLTF		:::::::::0:-4:12:::::16:-6:45:::
7	42172.0	0.007571813005017951	TGTGCCGGCTGGGGTCCATCAGGAGGAAGCTACATACCTACATTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV12-2*00(632.1)		TRAJ6*00(299.6)	TRAC*00(75)	631|638|664|0|7||35.0		22|51|82|16|45||145.0												TGTGCCGGCTGGGGTCCATCAGGAGGAAGCTACATACCTACATTT	41								CAGWGPSGGSYIPTF		:::::::::0:-6:7:::::16:-2:45:::
8	41647.0	0.007477551366308987	TGTGCTGTGAGTGAAGATAACTATGGTCAGAATTTTGTCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV8-4*00(338.2)		TRAJ26*00(289.5)	TRAC*00(76.5)	556|570|590|0|14||70.0		22|49|80|15|42||135.0												TGTGCTGTGAGTGAAGATAACTATGGTCAGAATTTTGTCTTT	41								CAVSEDNYGQNFVF		:::::::::0:0:14:::::15:-2:42:::
9	19133.0	0.00343525320651163	TGTGCCGTGAACAGTAGGAGTTACCAGAAAGTTACCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV12-2*00(680.8)		TRAJ13*00(254.6)	TRAC*00(76)	631|644|664|0|13||65.0		32|52|83|19|39||100.0												TGTGCCGTGAACAGTAGGAGTTACCAGAAAGTTACCTTT	41								CAVNSRSYQKVTF		:::::::::0:0:13:::::19:-12:39:::"""

        file2_content = """cloneId	cloneCount	cloneFraction	targetSequences	targetQualities	allVHitsWithScore	allDHitsWithScore	allJHitsWithScore	allCHitsWithScore	allVAlignments	allDAlignments	allJAlignments	allCAlignments	nSeqFR1	minQualFR1	nSeqCDR1	minQualCDR1	nSeqFR2	minQualFR2	nSeqCDR2	minQualCDR2	nSeqFR3	minQualFR3	nSeqCDR3	minQualCDR3	nSeqFR4	minQualFR4	aaSeqFR1	aaSeqCDR1	aaSeqFR2	aaSeqCDR2	aaSeqFR3	aaSeqCDR3	aaSeqFR4	refPoints
10	13954.0	0.002505384583895013	TGTGCTGTGCTGGAAACCAGTGGCTCTAGGTTGACCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV21*00(631.2)		TRAJ58*00(289.6)	TRAC*00(75.4)	624|633|656|0|9||45.0		25|52|83|12|39||135.0												TGTGCTGTGCTGGAAACCAGTGGCTCTAGGTTGACCTTT	41								CAVLETSGSRLTF		:::::::::0:-3:9:::::12:-5:39:::
11	12927.0	0.0023209908639824305	TGTGCCGTGAACGATGCAGGCAACATGCTCACCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV12-2*00(684.1)		TRAJ39*00(269.6)	TRAC*00(75.9)	631|643|664|0|12||60.0		29|52|83|13|36||115.0												TGTGCCGTGAACGATGCAGGCAACATGCTCACCTTT	41								CAVNDAGNMLTF		:::::::::0:-1:12:::::13:-9:36:::
12	9299.0	0.0016695980540088666	TGCATCGTTGGAGATGACAAGATCATCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV26-2*00(617.6)		TRAJ30*00(254.6)	TRAC*00(75.3)	871|877|906|0|6||30.0		26|46|77|10|30||100.0												TGCATCGTTGGAGATGACAAGATCATCTTT	41								CIVGDDKIIF		:::::::::0:-9:6:::::10:-6:30:::
13	8924.0	0.0016022683120738926	TGTGCAGCGGTTTTGTCCTGATTTACTCAAATTCCGGGTATGCACTCAACTTC	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV29DV5*00(558.8)		TRAJ41*00(299.4)	TRAC*00(74.9)	609|617|642|0|8||40.0		22|51|82|24|53||145.0												TGTGCAGCGGTTTTGTCCTGATTTACTCAAATTCCGGGTATGCACTCAACTTC	41								CAAVLS*FT_NSGYALNF		:::::::::0:-5:8:::::24:-2:53:::
14	8589.0	0.0015421204092786489	TGTGCCGTGAGTTCAGGATACAGCACCCTCACCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV12-2*00(678.7)		TRAJ11*00(283.5)	TRAC*00(75.4)	631|641|664|0|10||50.0		20|49|80|7|36|SA23G|129.0												TGTGCCGTGAGTTCAGGATACAGCACCCTCACCTTT	41								CAVSSGYSTLTF		:::::::::0:-3:10:::::7:0:36:::
15	8200.0	0.001472277023644769	TGTGCAATGAGCCAAAACAAAAATGAGAAATTAACCTTT	JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ	TRAV12-3*00(610.2)		TRAJ48*00(249.5)	TRAC*00(76.1)	642|654|675|0|12||60.0		33|52|83|20|39||95.0												TGTGCAATGAGCCAAAACAAAAATGAGAAATTAACCTTT	41								CAMSQNKNEKLTF		:::::::::0:-1:12:::::20:-13:39:::"""

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path / "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        if add_metadata:
            with open(path / "metadata.csv", "w") as file:
                file.writelines("""filename,subject_id
rep1.tsv,1
rep2.tsv,2""")

    def test_load_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "mixcr_rep")
        self.create_dummy_dataset(path, add_metadata=True)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "mixcr")
        params["is_repertoire"] = True
        params["result_path"] = path
        params["path"] = path
        params["metadata_file"] = path / "metadata.csv"

        dataset = MiXCRImport.import_dataset(params, "mixcr_repertoire_dataset")

        self.assertEqual(2, dataset.get_example_count())
        for index, repertoire in enumerate(dataset.get_data()):
            self.assertTrue(all(sequence.metadata.chain == Chain.ALPHA for sequence in repertoire.sequences))
            if index == 0:
                self.assertEqual(9, len(repertoire.sequences))
                self.assertTrue(repertoire.sequences[0].sequence_aa in ["ALVTDSWGKLQ", "AVLETSGSRLT"])  # OSX/windows
                self.assertTrue(repertoire.sequences[0].metadata.v_call in ["TRAV6", "TRAV21"])  # OSX/windows

                self.assertListEqual([Chain.ALPHA for i in range(9)], list(repertoire.get_chains()))
                self.assertListEqual(sorted([956023, 90101, 69706, 56658, 55692, 43466, 42172, 41647, 19133]), sorted(list(repertoire.get_counts())))

            elif index == 1:
                self.assertEqual(5, len(repertoire.sequences))
                self.assertTrue(repertoire.sequences[0].sequence in ["GCTGTGCTGGAAACCAGTGGCTCTAGGTTGACC",
                                                                                "GCTCTAGTAACTGACAGCTGGGGGAAATTGCAG"])  # OSX/windows

        shutil.rmtree(path)

    def test_load_sequence_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "mixcr_seq/")

        self.create_dummy_dataset(path, add_metadata=False)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "mixcr")
        params["is_repertoire"] = False
        params["paired"] = False
        params["result_path"] = path
        params["path"] = path

        dataset = MiXCRImport.import_dataset(params, "mixcr_repertoire_dataset")

        seqs = [sequence for sequence in dataset.get_data()]

        self.assertTrue(seqs[0].sequence_aa in ["AVLETSGSRLT", "ALVTDSWGKLQ"])  # OSX/windows
        self.assertTrue(seqs[0].metadata.v_call in ["TRAV21", "TRAV6"])  # OSX/windows

        shutil.rmtree(path)
