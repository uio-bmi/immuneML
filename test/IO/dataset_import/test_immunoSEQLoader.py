import shutil
from unittest import TestCase

from source.IO.dataset_import.ImmunoSEQLoader import ImmunoSEQLoader
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImmunoSEQLoader(TestCase):
    def test_load(self):
        path = EnvironmentSettings.root_path + "test/tmp/immunoseq/"

        rep1text = """nucleotide	aminoAcid	count (templates/reads)	frequencyCount (%)	cdr3Length	vMaxResolved	vFamilyName	vGeneName	vGeneAllele	vFamilyTies	vGeneNameTies	vGeneAlleleTies	dMaxResolved	dFamilyName	dGeneName	dGeneAllele	dFamilyTies	dGeneNameTies	dGeneAlleleTies	jMaxResolved	jFamilyName	jGeneName	jGeneAllele	jFamilyTies	jGeneNameTies	jGeneAlleleTies	vDeletion	n1Insertion	d5Deletion	d3Deletion	n2Insertion	jDeletion	vIndex	n1Index	dIndex	n2Index	jIndex	estimatedNumberGenomes	sequenceStatus	cloneResolved	vOrphon	dOrphon	jOrphon	vFunction	dFunction	jFunction	fractionNucleated	vAlignLength	vAlignSubstitutionCount	vAlignSubstitutionIndexes	vAlignSubstitutionGeneThreePrimeIndexes	vSeqWithMutations
GCCATCCCCAACCAGACAGCTCTTTACTTCTGTGCCACCAGTGATCAACTTAACCGTTGGGGGACCGGGGAGCTGTTTTTTGGAGAA	CATSDQLNRWGTGELFF	38	0.0017525250196006087	51	TCRBV24	TCRBV24				TCRBV24-01,TCRBV24-or09_02						TCRBD01,TCRBD02	TCRBD01-01,TCRBD02-01		TCRBJ02-02*01	TCRBJ02	TCRBJ02-02	01				3	0	6	1	13	5	30	45	58	-1	63	38	In	VDJ												
GGGTTGGAGTCGGCTGCTCCCTCCCAAACATCTGTGTACTTCTGTGCCAGCAAGGACGGCGACACCGGGGAGCTGTTTTTTGGAGAA	CASKDGDTGELFF	48	0.002213715814232348	39	TCRBV06	TCRBV06				TCRBV06-02,TCRBV06-03						TCRBD01,TCRBD02	TCRBD01-01,TCRBD02-01		TCRBJ02-02*01	TCRBJ02	TCRBJ02-02	01				7	4	1	7	1	3	42	52	53	57	61	48	In	VDJ												
AGGCCCTCACATACCTCTCAGTACCTCTGTGCCAGCAGTGGGGAGGGACAGGGGGTATTTGGTGGCACTGAAGCTTTCTTTGGACAA	CASSGEGQGVFGGTEAFF	37	0.001706405940137435	54	TCRBV25-01*01	TCRBV25	TCRBV25-01	01				TCRBD01-01*01	TCRBD01	TCRBD01-01	01				TCRBJ01-01*01	TCRBJ01	TCRBJ01-01	01				4	10	0	1	4	4	27	40	44	55	65	37	In	VDJ												
GAGTCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTGAGGAGGTAGGGGGCAATCAGCCCCAGCATTTTGGTGAT	CASSEEVGGNQPQHF	53	0.0024443112115482175	45	TCRBV06-01*01	TCRBV06	TCRBV06-01	01								TCRBD01,TCRBD02	TCRBD01-01,TCRBD02-01		TCRBJ01-05*01	TCRBJ01	TCRBJ01-05	01				3	0	5	2	6	2	36	50	56	-1	61	53	In	VDJ												
GAGTCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTGAATTACAGGAAGGTTATGAGACCCAGTACTTCGGGCCA	CASSELQEGYETQYF	28	0.0012913342249688696	45	TCRBV06-01*01	TCRBV06	TCRBV06-01	01				TCRBD01-01*01	TCRBD01	TCRBD01-01	01				TCRBJ02-05*01	TCRBJ02	TCRBJ02-05	01				2	8	3	4	2	5	36	51	53	58	66	28	In	VDJ												
TTGGAGTCGGCTGCTCCCTCCCAAACATCTGTGTACTTCTGTGCCAGCAGTTTCCTAGCGGACCCCGGAGAGCAGTTCTTCGGGCCA	CASSFLADPGEQFF	16	7.379052714107826E-4	42	TCRBV06	TCRBV06				TCRBV06-02,TCRBV06-03		TCRBD02-01	TCRBD02	TCRBD02-01				01,02	TCRBJ02-01*01	TCRBJ02	TCRBJ02-01	01				4	8	4	5	2	10	39	52	54	61	69	16	In	VDJ												
CAGCGCACAGAGCAGGGGGACTCGGCCATGTATCTCTGTGCCAGCAGCTCACTTTGGGGTCGGAGGTATGGCTACACCTTCGGTTCG	CASSSLWGRRYGYTF	72	0.003320573721348522	45	TCRBV07-09	TCRBV07	TCRBV07-09				01,03	TCRBD02-01*02	TCRBD02	TCRBD02-01	02				TCRBJ01-02*01	TCRBJ01	TCRBJ01-02	01				4	0	10	1	12	5	36	49	61	-1	66	72	In	VDJ												
AGCAACATGAGCCCTGAAGACAGCAGCATATATCTCTGCAGCGTTTTGGACCTCCCGACCCAAACAGATACGCAGTATTTTGGCCCA	CSVLDLPTQTDTQYF	14	6.456671124844348E-4	45	TCRBV29-01*01	TCRBV29	TCRBV29-01	01								TCRBD01,TCRBD02	TCRBD01-01,TCRBD02-01		TCRBJ02-03*01	TCRBJ02	TCRBJ02-03	01				5	12	1	7	2	3	36	45	47	51	63	14	In	VDJ												
CAGCGCACACAGCAGGAGGACTCGGCCGTGTATCTCTGTGCCAGCAGCTTAAGGCTAGCGGGAGTGGAGACCCAGTACTTCGGGCCA	CASSLRLAGVETQYF	26	0.0011990960660425217	45	TCRBV07-02*01	TCRBV07	TCRBV07-02	01				TCRBD02-01*02	TCRBD02	TCRBD02-01	02				TCRBJ02-05*01	TCRBJ02	TCRBJ02-05	01				2	2	4	2	3	5	36	51	54	64	66	26	In	VDJ												
CTGGAGTCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGCAGCGGTCCAGGGATGGAGACCCAGTACTTCGGGCCA	CASSSGPGMETQYF	13	5.995480330212608E-4	42	TCRBV06-01*01	TCRBV06	TCRBV06-01	01								TCRBD01,TCRBD02	TCRBD01-01,TCRBD02-01		TCRBJ02-05*01	TCRBJ02	TCRBJ02-05	01				6	3	4	3	8	5	39	50	58	63	66	13	In	VDJ												
TCTAAGAAGCTCCTTCTCAGTGACTCTGGCTTCTATCTCTGTGCCTGGAGTGCTATAGCGGATTACAATGAGCAGTTCTTCGGGCCA	CAWSAIADYNEQFF	8	3.689526357053913E-4	42	TCRBV30-01*01	TCRBV30	TCRBV30-01	01				TCRBD02-01	TCRBD02	TCRBD02-01				01,02	TCRBJ02-01*01	TCRBJ02	TCRBJ02-01	01				1	2	5	5	3	4	39	52	55	61	63	8	In	VDJ												
TCCCTGATTCTGGAGTCCGCCAGCACCAACCAGACATCTATGTACCTCTGTGCCAGCAGTTTAATAGATACGCAGTATTTTGGCCCA	CASSLIDTQYF	16	7.379052714107826E-4	33	TCRBV28-01*01	TCRBV28	TCRBV28-01	01											TCRBJ02-03*01	TCRBJ02	TCRBJ02-03	01				2	2	0	0	0	5	48	-1	-1	63	65	16	In	VJ												
ATCCGGTCCACAAAGCTGGAGGACTCAGCCATGTACTTCTGTGCCAGCAGATCGGGACAGGGATGGGATGAGCAGTTCTTCGGGCCA	CASRSGQGWDEQFF	8	3.689526357053913E-4	42	TCRBV02-01*01	TCRBV02	TCRBV02-01	01				TCRBD01-01*01	TCRBD01	TCRBD01-01	01				TCRBJ02-01*01	TCRBJ02	TCRBJ02-01	01				6	5	0	3	3	8	39	50	53	62	67	8	In	VDJ												
ATCAATTCCCTGGAGCTTGGTGACTCTGCTGTGTATTTCTGTGCCAGCAGCCCTAGCGGAGACACCGGGGAGCTGTTTTTTGGAGAA	CASSPSGDTGELFF	28	0.0012913342249688696	42	TCRBV03	TCRBV03				TCRBV03-01,TCRBV03-02		TCRBD02-01	TCRBD02	TCRBD02-01				01,02	TCRBJ02-02*01	TCRBJ02	TCRBJ02-02	01				4	2	4	5	0	3	39	-1	52	59	61	28	In	VDJ												
GGTCCACAAAGCTGGAGGACTCAGCCATGTACTTCTGTGCCAGCAGTCCCGGGGGACGGGGCTTCATACGAGCAGTACTTCGGGCCG		8	3.689526357053913E-4	46	TCRBV02-01*01	TCRBV02	TCRBV02-01	01				TCRBD02-01*01	TCRBD02	TCRBD02-01	01				TCRBJ02-07*01	TCRBJ02	TCRBJ02-07	01				5	11	8	2	2	4	35	47	49	55	66	8	Out	VDJ												
GAGTCGGCTGCTCCCTCCCAAACATCTGTGTACTTCTGTGCCAGCAGTTCCGACAGCGGTCCCTACAATGAGCAGTTCTTCGGGCCA	CASSSDSGPYNEQFF	7	3.228335562422174E-4	45	TCRBV06	TCRBV06				TCRBV06-02,TCRBV06-03						TCRBD01,TCRBD02	TCRBD01-01,TCRBD02-01		TCRBJ02-01*01	TCRBJ02	TCRBJ02-01	01				4	5	2	5	2	2	36	49	51	56	61	7	In	VDJ												
GGGTTGGAGTCGGCTGCTCCCTCCCAAACATCTGTGTACTTCTGTGCCAGCAGTCCAGGGGACACCGGGGAGCTGTTTTTTGGAGAA	CASSPGDTGELFF	1	4.611907946317391E-5	39	TCRBV06	TCRBV06				TCRBV06-02,TCRBV06-03		TCRBD01-01*01	TCRBD01	TCRBD01-01	01				TCRBJ02-02*01	TCRBJ02	TCRBJ02-02	01				5	0	4	2	1	3	42	54	55	-1	61	1	In	VDJ												
CTGAACATGAGCTCCTTGGAGCTGGGGGACTCAGCCCTGTACTTCTGTGCCAGCAGCTTACGCACAGATACGCAGTATTTTGGCCCA	CASSLRTDTQYF	9	4.1507171516856525E-4	36	TCRBV13-01*01	TCRBV13	TCRBV13-01	01											TCRBJ02-03*01	TCRBJ02	TCRBJ02-03	01				2	1	0	0	0	1	45	-1	-1	60	61	9	In	VJ												
AAGAAGCTCCTTCTCAGTGACTCTGGCTTCTATCTCTGTGCCTGGAGTGTACGTCCGGGCGCAGGGTACGAGCAGTACTTCGGGCCG	CAWSVRPGAGYEQYF	1	4.611907946317391E-5	45	TCRBV30-01*01	TCRBV30	TCRBV30-01	01				TCRBD01-01*01	TCRBD01	TCRBD01-01	01				TCRBJ02-07*01	TCRBJ02	TCRBJ02-07	01				0	0	4	3	11	4	36	50	61	-1	66	1	In	VDJ												"""

        PathBuilder.build(path)

        with open(path + "rep1.tsv", "w") as file:
            file.writelines(rep1text)

        with open(path + "metadata.csv", "w") as file:
            file.writelines(
            """filename,chain,donor,coeliac status (yes/no)
rep1.tsv,TRA,1234,no"""
            )

        dataset = ImmunoSEQLoader().load(path, {"result_path": path,
                                              "dataset_id": "emerson2017",
                                              "batch_size": 1,
                                              "region_type": "CDR3",  # can be loaded from metadata if available?"
                                              "metadata_file": path + "metadata.csv"})

        self.assertEqual(1, dataset.get_example_count())
        for index, rep in enumerate(dataset.get_data()):
            self.assertEqual("rep1", rep.identifier)
            self.assertEqual(19, len(rep.sequences))

        shutil.rmtree(path)
