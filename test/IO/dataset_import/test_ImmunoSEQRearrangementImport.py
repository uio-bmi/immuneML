import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.ImmunoSEQRearrangementImport import ImmunoSEQRearrangementImport
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.data_model.receptor.receptor_sequence.SequenceFrameType import SequenceFrameType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.ImportHelper import ImportHelper
from immuneML.util.PathBuilder import PathBuilder


class TestImmunoSEQRearrangementImport(TestCase):
    def build_dummy_dataset(self, path, add_metadata):

        rep1text = """rearrangement	amino_acid	frame_type	rearrangement_type	templates	reads	frequency	productive_frequency	cdr3_length	v_family	v_gene	v_allele	d_family	d_gene	d_allele	j_family	j_gene	j_allele	v_deletions	d5_deletions	d3_deletions	j_deletions	n2_insertions	n1_insertions	v_index	n1_index	n2_index	d_index	j_index	v_family_ties	v_gene_ties	v_allele_ties	d_family_ties	d_gene_ties	d_allele_ties	j_family_ties	j_gene_ties	j_allele_ties	sequence_tags	v_shm_count	v_shm_indexes	antibody	sample_name	species	locus	product_subtype	kit_pool	total_templates	productive_templates	outofframe_templates	stop_templates	dj_templates	total_rearrangements	productive_rearrangements	outofframe_rearrangements	stop_rearrangements	dj_rearrangements	total_reads	total_productive_reads	total_outofframe_reads	total_stop_reads	total_dj_reads	productive_clonality	productive_entropy	sample_clonality	sample_entropy	sample_amount_ng	sample_cells_mass_estimate	fraction_productive_of_cells_mass_estimate	sample_cells	fraction_productive_of_cells	max_productive_frequency	max_frequency	counting_method	primer_set	release_date	sample_tags	fraction_productive	order_name	kit_id	total_t_cells
ACTCTGACTGTGAGCAACATGAGCCCTGAAGACAGCAGCATATATCTCTGCAGCGTTGAAGAATCCTACGAGCAGTACTTCGGGCCG	CSVEESYEQYF	In	VJ	10	311	7.66134773699554E-5	9.602057989637805E-5	33	TCRBV29	TCRBV29-01	01		unresolved		TCRBJ02	TCRBJ02-07	01	0	0	0	1	1	0	48	-1	62	-1	63										null	null	null	Vb 4	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
GAATGTGAGCACCTTGGAGCTGGGGGACTCGGCCCTTTATCTTTGCGCCAGCAGCATCAAAGGGGCTCACCCCTCCACTTTGGGAAC		Out	VDJ	3088	77915	0.019194016364244615	null	38	TCRBV05	TCRBV05-01	01	TCRBD01	TCRBD01-01	01	TCRBJ01	TCRBJ01-06	01	4	5	2	10	1	5	43	55	65	60	66										null	null	null	Vb 5.1	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
GCTACCAGCTCCCAGACATCTGTGTACTTCTGTGCCACCACGGGTACTAGCGGGGGCCCAAGCCAGAGTACGCAGTATTTTGGCCCA	CATTGTSGGPSQSTQYF	In	VDJ	1772	44434	0.010946119786034077	0.013718901759214348	51	TCRBV10	TCRBV10-03	01	TCRBD02	TCRBD02-01	01	TCRBJ02	TCRBJ02-03	01	10	3	2	8	12	8	30	37	56	45	68										null	null	null	Vb 12	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
ATCCAGCGCACAGAGCAGGGGGACTCGGCCATGTATCTCTGTGCCAGCAGCTTACGAGTCGGGGGCTATGGCTACACCTTCGGTTCG	CASSLRVGGYGYTF	In	VDJ	1763	44008	0.010841176566228286	0.013587375177105484	42	TCRBV07	TCRBV07-09		TCRBD02	TCRBD02-01	01	TCRBJ01	TCRBJ01-02	01	2	8	2	4	0	5	39	54	-1	59	65			01,03							null	null	null	null	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
TGCAGCAAGAAGACTCAGCTGCGTATCTCTGCACCAGCAGCCAAGGGGATCGCGGGGGGCCACTACAATGAGCAGTTCTTCGGGCCA		Out	VDJ	1241	31095	0.007660116009063547	null	52	TCRBV01	TCRBV01-01	01	TCRBD02	TCRBD02-01	01	TCRBJ02	TCRBJ02-01	01	1	7	1	3	3	6	29	45	59	51	62										null	null	null	null	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
GAGTCTGCCAGGCCCTCACATACCTCTCAGTACCTCTGTGCCAGCAGACGCCTCGGAGGGTTGAACACTGAAGCTTTCTTTGGACAA	CASRRLGGLNTEAFF	In	VDJ	NA	24883	0.006129817226355627	0.0076825726352462214	45	TCRBV25	TCRBV25-01	01	TCRBD02	TCRBD02-01	02	TCRBJ01	TCRBJ01-01	01	6	10	0	0	1	7	36	47	60	54	61										null	null	null	Vb 11	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
GTGAGCACCTTGGAGCTGGGGGACTCGGCCCTTTATCTTTGCGCCAGCAGCTTGAGAGGCTCTGGAAACACCATATATTTTGGAGAG	CASSLRGSGNTIYF	In	VDJ	566	14247	0.0035096855694204325	0.0043987305523591575	42	TCRBV05	TCRBV05-01	01	TCRBD02	TCRBD02-01	02	TCRBJ01	TCRBJ01-03	01	1	11	1	0	0	1	39	54	-1	55	59										null	null	null	Vb 5.1	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
AGGCTGGAGTCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGACAGGACGGGAGCACTGAAGCTTTCTTTGGACAA	CASRQDGSTEAFF	In	VDJ	506	12943	0.003188450924756697	0.003996123362054087	39	TCRBV06	TCRBV06-01	01	TCRBD02	TCRBD02-01	02	TCRBJ01	TCRBJ01-01	01	6	8	2	4	0	6	42	53	-1	59	65										null	null	null	null	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
TCCCTGGAGCTTGGTGACTCTGCTGTGTATTTCTGTGCCAGCAGCCGGGCCAGGGTCTTTGGAAACTATGGCTACACCTTCGGTTCG	CASSRARVFGNYGYTF	In	VDJ	398	10151	0.0025006540475318883	0.0031340993779039664	48	TCRBV03	unresolved		TCRBD01	TCRBD01-01	01	TCRBJ01	TCRBJ01-02	01	4	4	3	2	8	4	33	46	55	50	63		TCRBV03-01,TCRBV03-02								null	null	null	null	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
ATCCAGCGCACAGAGCAGGGGGACTCGGCCATGTATCTCTGTGCCAGCAGATAAAAGGGGACGGATCGGGAACTGTTTTTTGGCAGT	CASR*KGTDRELFF	Stop	VDJ	394	10125	0.0024942490622855253	null	42	TCRBV07	TCRBV07-09			unresolved		TCRBJ01	TCRBJ01-04	01	6	0	7	12	8	7	39	50	62	57	70			01,03	TCRBD01,TCRBD02	TCRBD01-01,TCRBD02-01					null	null	null	null	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
GGCTGCTGTCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTTATGGGCCGCCAAGGTGAGCAGTTCTTCGGGCCA		Out	VDJ	388	9745	0.0024006377394540685	null	40	TCRBV06	TCRBV06-05	01	TCRBD01	TCRBD01-01	01	TCRBJ02	TCRBJ02-01	01	3	8	0	9	8	1	41	55	60	56	68										null	null	null	Vb 13.1	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
CACATCAATTCCCTGGAGCTTGGTGACTCTGCTGTGTATTTCTGTGCCAGCAGCCAAGCAGATAATCAGCCCCAGCATTTTGGTGAT	CASSQADNQPQHF	In	VDJ	363	9277	0.0022853480050195377	0.002864253761089065	39	TCRBV03	unresolved		TCRBD01	TCRBD01-01	01	TCRBJ01	TCRBJ01-05	01	1	4	5	4	2	0	42	-1	61	58	63		TCRBV03-01,TCRBV03-02								null	null	null	null	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0
CACATCAATTCCCTGGAGCTTGGTGACTCTGCTGTGTATTTCTGTGCCAGCAGCCAAGCAGATAATCAGCCCCAGCATTTTGGTGAT	CASSQADNQPQHF	In	VDJ	363	9277	0.0022853480050195377	0.002864253761089065	39	nan	unresolved		TCRAD01	TCRAD01-01	01	TCRAJ01	TCRAJ01-05	01	1	4	5	4	2	0	42	-1	61	58	63		TCRBV03-01,TCRBV03-02								null	null	null	null	HIP00110	Human	TCRB	Deep	null	224859	179411	41463	3983	0	130940	104850	24105	1985	0	4059338	3238889	748535	71914	0	0.100719467	14.9981718	0.1101579	15.1260223	3636.47998	559458	0.3206871650776287	0	0.0	0.0137189021	0.0191940162	v2	Human-TCRB-PD1x	2013-12-13 22:23:05.529	Age:55 Years,Biological Sex:Male,Cohort:Cohort 01,Ethnic Group:Unknown Ethnicity,HLA MHC class I:HLA-A*03,HLA MHC class I:HLA-A*24,HLA MHC class I:HLA-B*07,Inferred CMV status (cross-validation): Inferred CMV -,Inferred CMV status: Inferred CMV -,Inferred HLA type:Inferred HLA-A*03,Inferred HLA type:Inferred HLA-A*24,Inferred HLA type:Inferred HLA-B*07,Racial Group:Unknown racial group,Species:Human,Tissue Source:gDNA,Tissue Source:PBMC,Tissue Source:Peripheral blood lymphocytes (PBL),Tissue Source:T cells,Virus Diseases:Cytomegalovirus -	0.7978822284186979	null	null	0"""

        rep2text = """rearrangement	amino_acid	bio_identity	templates	frame_type	rearrangement_type	cdr3_length	frequency	productive_frequency	v_resolved	d_resolved	j_resolved	v_family	v_family_ties	v_gene	v_gene_ties	v_allele	v_allele_ties	d_family	d_family_ties	d_gene	d_gene_ties	d_allele	d_allele_ties	j_family	j_family_ties	j_gene	j_gene_ties	j_allele	j_allele_ties	locus
GCACAGAGCAGGGGGACTCGGCCATGTATCTCTGTGCCAGCAGCTTACTGGTAAACCCCCACTGCTTATGAGCAGTTCTTCGGGCCA	na	X+TCRBV07-09+TCRBJ02-01	170	Out	VDJ	49	0.012084162638612454	na	TCRBV07-09*01	TCRBD02-01	TCRBJ02-01*01	TCRBV07	no data	TCRBV07-09	no data	01	no data	TCRBD02	no data	TCRBD02-01	no data	unknown	01,02	TCRBJ02	no data	TCRBJ02-01	no data	01	no data	TCRB
CCACGGAGTCAGGGGACACAGCACTGTATTTCTGTGCCAGCAGCAAATTTCAGGGGCGGATGGATTGAAAAACTGTTTTTTGGCAGT	na	X+TCRBV21-01+TCRBJ01-04	58	Out	VDJ	49	0.004122831959056014	na	TCRBV21-01*01	TCRBD01-01*01	TCRBJ01-04*01	TCRBV21	no data	TCRBV21-01	no data	01	no data	TCRBD01	no data	TCRBD01-01	no data	01	no data	TCRBJ01	no data	TCRBJ01-04	no data	01	no data	TCRB
CAGCGCACAGAGCAGGGGGACTCGGCCATGTATCTCTGTGCCAGCAGCTTACCGGGGACGAACACCGGGGAGCTGTTTTTTGGAGAA	CASSLPGTNTGELFF	CASSLPGTNTGELFF+TCRBV07-09+TCRBJ02-02	81	In	VDJ	45	0.00575774808075064	0.007861024844720496	TCRBV07-09*01	TCRBD02-01*01	TCRBJ02-02*01	TCRBV07	no data	TCRBV07-09	no data	01	no data	TCRBD02	no data	TCRBD02-01	no data	01	no data	TCRBJ02	no data	TCRBJ02-02	no data	01	no data	TCRB
ACAGTGACCAGTGCCCATCCTGAAGACAGCAGCTTCTACATCTGCAGTTCTTCGACGGGCTCCACTACTGAAGCTTTCTTTGGACAA	CSSSTGSTTEAFF	CSSSTGSTTEAFF+TCRBV20-X+TCRBJ01-01	24	In	VDJ	39	0.0017059994313335229	0.002329192546583851	TCRBV20	unknown	TCRBJ01-01*01	TCRBV20	no data	unresolved	TCRBV20-01,TCRBV20-or09_02	unknown	no data	unknown	TCRBD01,TCRBD02	unresolved	TCRBD01-01,TCRBD02-01	unknown	no data	TCRBJ01	no data	TCRBJ01-01	no data	01	no data	TCRB
CCTCAGAACCCAGGGACTCAGCTGTGTATTTTTGTGCTAGTGGTGACTCTAGCGGGAGAGGGTACAGATACGCAGTATTTTGGCCCA	na	X+TCRBV12-05+TCRBJ02-03	26	Out	VDJ	49	0.0018481660506113164	na	TCRBV12-05*01	TCRBD02-01*02	TCRBJ02-03*01	TCRBV12	no data	TCRBV12-05	no data	01	no data	TCRBD02	no data	TCRBD02-01	no data	02	no data	TCRBJ02	no data	TCRBJ02-03	no data	01	no data	TCRB
AGAGCTTGAGGACTCGGCCGTGTATCTCTGTGCCAGCAGCTCACAAACGGGGATGGGACAGGGCCCAATGAGCAGTTCTTCGGGCCA	na	X+TCRBV11-02+TCRBJ02-01	28	Out	VDJ	53	0.00199033266988911	na	TCRBV11-02	TCRBD01-01*01	TCRBJ02-01*01	TCRBV11	no data	TCRBV11-02	no data	unknown	01,03	TCRBD01	no data	TCRBD01-01	no data	01	no data	TCRBJ02	no data	TCRBJ02-01	no data	01	no data	TCRB
CGGTCCACAAAGCTGGAGGACTCAGCCATGTACTTCTGTGCCAGAAGTAGCGGGGGTACGAACACCGGGGAGCTGTTTTTTGGAGAA	CARSSGGTNTGELFF	CARSSGGTNTGELFF+TCRBV02-01+TCRBJ02-02	19	In	VDJ	45	0.001350582883139039	0.001843944099378882	TCRBV02-01	TCRBD02-01*01	TCRBJ02-02*01	TCRBV02	no data	TCRBV02-01	no data	unknown	01,03	TCRBD02	no data	TCRBD02-01	no data	01	no data	TCRBJ02	no data	TCRBJ02-02	no data	01	no data	TCRB
CTACACGCCCTGCAGCCAGAAGACTCAGCCCTGTATCTCTGCGCCAGCAGCCAAGATCCAACGAACACTGAAGCTTTCTTTGGACAA	CASSQDPTNTEAFF	CASSQDPTNTEAFF+TCRBV04-01+TCRBJ01-01	15	In	VDJ	42	0.0010662496445834517	0.0014557453416149068	TCRBV04-01*01	unknown	TCRBJ01-01*01	TCRBV04	no data	TCRBV04-01	no data	01	no data	unknown	TCRBD01,TCRBD02	unresolved	TCRBD01-01,TCRBD02-01	unknown	no data	TCRBJ01	no data	TCRBJ01-01	no data	01	no data	TCRB
TCTAAGAAGCTCCTTCTCAGTGACTCTGGCTTCTATCTCTGTGCCTACGACAGGGGGGTGGTGAACACTGAAGCTTTCTTTGGACAA	CAYDRGVVNTEAFF	CAYDRGVVNTEAFF+TCRBV30-01+TCRBJ01-01	12	In	VDJ	42	8.529997156667614E-4	0.0011645962732919255	TCRBV30-01	TCRBD01-01*01	TCRBJ01-01*01	TCRBV30	no data	TCRBV30-01	no data	unknown	01,05	TCRBD01	no data	TCRBD01-01	no data	01	no data	TCRBJ01	no data	TCRBJ01-01	no data	01	no data	TCRB
AAGATCCGGTCCACAAAGCTGGAGGACTCAGCCATGTACTTCTGTGCCAGCAATCCATCTGGAGCGATCGAGCAGTACTTCGGGCCG	CASNPSGAIEQYF	CASNPSGAIEQYF+TCRBV02-01+TCRBJ02-07	15	In	VDJ	39	0.0010662496445834517	0.0014557453416149068	TCRBV02-01	TCRBD02-01	TCRBJ02-07*01	TCRBV02	no data	TCRBV02-01	no data	unknown	01,03	TCRBD02	no data	TCRBD02-01	no data	unknown	01,02	TCRBJ02	no data	TCRBJ02-07	no data	01	no data	TCRB
AAGATCCGGTCCACAAAGCTGGAGGACTCAGCCATGTACTTCTGTGCCAGCACATCGACTAGCACAGATACGCAGTATTTTGGCCCA	CASTSTSTDTQYF	CASTSTSTDTQYF+TCRBV02-01+TCRBJ02-03	30	In	VDJ	39	0.0021324992891669035	0.0029114906832298135	TCRBV02-01	TCRBD02-01	TCRBJ02-03*01	TCRBV02	no data	TCRBV02-01	no data	unknown	01,03	TCRBD02	no data	TCRBD02-01	no data	unknown	01,02	TCRBJ02	no data	TCRBJ02-03	no data	01	no data	TCRB
CACCTACACGCCCTGCAGCCAGAAGACTCAGCCCTGTATCTCTGCGCCAGCAGCCAAGACCTGACAGAAGAGCAGTACTTCGGGCCG	CASSQDLTEEQYF	CASSQDLTEEQYF+TCRBV04-01+TCRBJ02-07	20	In	VDJ	39	0.0014216661927779358	0.0019409937888198758	TCRBV04-01*01	TCRBD01-01*01	TCRBJ02-07*01	TCRBV04	no data	TCRBV04-01	no data	01	no data	TCRBD01	no data	TCRBD01-01	no data	01	no data	TCRBJ02	no data	TCRBJ02-07	no data	01	no data	TCRB
TCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTTCATCCTCCGAGCGGGAGGAAGAGACCCAGTACTTCGGGCCA	CASSSSSEREEETQYF	CASSSSSEREEETQYF+TCRBV06-05+TCRBJ02-05	15	In	VDJ	48	0.0010662496445834517	0.0014557453416149068	TCRBV06-05*01	TCRBD02-01*02	TCRBJ02-05*01	TCRBV06	no data	TCRBV06-05	no data	01	no data	TCRBD02	no data	TCRBD02-01	no data	02	no data	TCRBJ02	no data	TCRBJ02-05	no data	01	no data	TCRB
GAGTCCGCCAGCACCAACCAGACATCTATGTACCTCTGTGCCAGCAGACCAGGAACAGGGAGGAAAAACATTCAGTACTTCGGCGCC	CASRPGTGRKNIQYF	CASRPGTGRKNIQYF+TCRBV28-01+TCRBJ02-04	2	In	VDJ	45	1.4216661927779356E-4	1.9409937888198756E-4	TCRBV28-01*01	unknown	TCRBJ02-04*01	TCRBV28	no data	TCRBV28-01	no data	01	no data	unknown	TCRBD01,TCRBD02	unresolved	TCRBD01-01,TCRBD02-01	unknown	no data	TCRBJ02	no data	TCRBJ02-04	no data	01	no data	TCRB
ACAGTGACCAGTGCCCATCCTGAAGACAGCAGCTTCTACATCTGCAGTGCTAGATCCACCTTAGAGTACGAGCAGTACTTCGGGCCG	CSARSTLEYEQYF	CSARSTLEYEQYF+TCRBV20-X+TCRBJ02-07	2	In	VDJ	39	1.4216661927779356E-4	1.9409937888198756E-4	TCRBV20	TCRBD02-01	TCRBJ02-07*01	TCRBV20	no data	unresolved	TCRBV20-01,TCRBV20-or09_02	unknown	no data	TCRBD02	no data	TCRBD02-01	no data	unknown	01,02	TCRBJ02	no data	TCRBJ02-07	no data	01	no data	TCRB"""

        PathBuilder.remove_old_and_build(path)

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(rep1text)

        with open(path / "rep2.tsv", "w") as file:
            file.writelines(rep2text)

        if add_metadata:
            with open(path / "metadata.csv", "w") as file:
                file.writelines(
                    """filename,chain,subject_id,coeliac status (yes/no)
rep1.tsv,TRA,1234,no
rep2.tsv,TRB,1234a,no"""
                )

    def test_repertoire_import(self):
        path = EnvironmentSettings.tmp_test_path / "adaptive/"
        self.build_dummy_dataset(path, True)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "ImmunoSEQRearrangement")
        params["is_repertoire"] = True
        params["result_path"] = path
        params['import_empty_nt_sequences'] = False
        params['import_empty_aa_sequences'] = True
        params["metadata_file"] = path / "metadata.csv"
        params["path"] = path
        params["import_productive"] = True
        params["import_with_stop_codon"] = True
        params["import_out_of_frame"] = True
        params["number_of_processes"] = 1

        dataset_name = "adaptive_dataset_reps"

        dataset = ImmunoSEQRearrangementImport.import_dataset(params, dataset_name)

        self.assertEqual(dataset.repertoires[0].sequences[1].metadata.frame_type, SequenceFrameType.IN)

        self.assertListEqual(list(dataset.repertoires[0].get_counts()), [10, 1772, 1763, None, 566, 506, 398, 394, 363, 363])
        self.assertListEqual(list(dataset.repertoires[0].get_chains()), [Chain.BETA for i in range(10)])

        self.assertEqual(2, dataset.get_example_count())
        for index, rep in enumerate(dataset.get_data()):
            if index == 0:
                self.assertEqual("1234", rep.metadata["subject_id"])
                self.assertEqual(10, len(rep.sequences))
                self.assertEqual(10, rep.sequences[0].metadata.duplicate_count)
                self.assertEqual("TRBV29-1*01", rep.sequences[0].metadata.v_call)
            else:
                self.assertEqual("1234a", rep.metadata["subject_id"])
                self.assertEqual(11, len(rep.sequences))
                self.assertEqual(2, rep.sequences[-1].metadata.duplicate_count)

        dataset_file = path / f"{dataset_name}.{ImportHelper.DATASET_FORMAT}"

        self.assertTrue(dataset_file.is_file())

        shutil.rmtree(path)

    def test_sequence_import(self):
        path = EnvironmentSettings.tmp_test_path / "adaptive_seq"
        self.build_dummy_dataset(path, False)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "ImmunoSEQRearrangement")
        params["is_repertoire"] = False
        params["paired"] = False
        params["result_path"] = path
        params["path"] = path
        params["import_productive"] = True
        params["import_with_stop_codon"] = True
        params["import_out_of_frame"] = True
        params["import_empty_nt_sequences"] = False
        params["import_empty_aa_sequences"] = True

        dataset_name = "adaptive_dataset_seqs"

        dataset = ImmunoSEQRearrangementImport.import_dataset(params, dataset_name)

        self.assertEqual(21, dataset.get_example_count())

        seqs = [sequence for sequence in dataset.get_data()]
        self.assertTrue(seqs[0].sequence_aa in ["ASSLPGTNTGELF", "SVEESYEQY"])  # OSX/windows
        self.assertTrue(seqs[0].sequence in ["GCCAGCAGCTTACCGGGGACGAACACCGGGGAGCTGTTT", 'AGCGTTGAAGAATCCTACGAGCAGTAC'])  # OSX/windows
        self.assertEqual("IN", seqs[0].metadata.frame_type.name)
        self.assertTrue(seqs[0].metadata.v_call in ['TRBV7-9*01', 'TRBV29-1*01'])  # OSX/windows
        self.assertTrue(seqs[0].metadata.j_call in ['TRBJ2-2*01', 'TRBJ2-7*01'])  # OSX/windows

        dataset_file = path / f"{dataset_name}.{ImportHelper.DATASET_FORMAT}"

        self.assertTrue(dataset_file.is_file())

        shutil.rmtree(path)

    def test_alternative_repertoire_import(self):
        path = EnvironmentSettings.tmp_test_path / "immunoseq_alternative"

        rep1text = """sample_name	productive_frequency	templates	amino_acid	rearrangement	v_resolved	d_resolved	j_resolved
LivMet_45	0.014838454958215437	451	CASSLLGLGSEQYF	CTGCTGTCGGCTGCTCCCTCCCAGACATCTGTGTACTTCTGTGCCAGCAGTTTACTCGGGTTAGGGAGCGAGCAGTACTTCGGGCCG	TCRBV06	TCRBD02-01*02	TCRBJ02-07*01
LivMet_45	0.0106928999144568	325	CASSPGQGEGYEQYF	CACGCCCTGCAGCCAGAAGACTCAGCCCTGTATCTCTGCGCCAGCAGCCCGGGACAGGGGGAGGGCTACGAGCAGTACTTCGGGCCG	TCRBV04-01*01	TCRBD01-01*01	TCRBJ02-07*01
LivMet_45	0.0074356780943607296	226	CASSAGETQYF	ACTCTGACGATCCAGCGCACAGAGCAGCGGGACTCGGCCATGTATCGCTGTGCCAGCAGCGCAGGCGAGACCCAGTACTTCGGGCCA	TCRBV07-06*01	TCRBD01-01*01	TCRBJ02-05*01
LivMet_45	0.0072053694808185825	219	CASSGTGEKGEQYF	ATCCGGTCCACAAAGCTGGAGGACTCAGCCATGTACTTCTGTGCCAGCAGTGGGACAGGGGAGAAGGGCGAGCAGTACTTCGGGCCG	TCRBV02-01*01	TCRBD01-01*01	TCRBJ02-07*01
"""
        PathBuilder.remove_old_and_build(path)

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(rep1text)

        with open(path / "metadata.csv", "w") as file:
            file.writelines(
                """filename,chain,subject_id
rep1.tsv,TRB,1234a""")

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "ImmunoSEQRearrangement")
        params["is_repertoire"] = True
        params["result_path"] = path
        params["metadata_file"] = path / "metadata.csv"
        params["path"] = path

        dataset = ImmunoSEQRearrangementImport.import_dataset(params, "alternative")

        self.assertEqual(1, dataset.get_example_count())

        shutil.rmtree(path)
