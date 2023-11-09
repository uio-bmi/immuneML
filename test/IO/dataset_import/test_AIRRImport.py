import shutil
from unittest import TestCase

import pandas as pd

from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter
from immuneML.IO.dataset_import.AIRRImport import AIRRImport
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class TestAIRRImport(TestCase):

    def create_dummy_dataset(self, path, add_metadata):
        file1_content = """rearrangement_id	rearrangement_set_id	sequence_id	sequence	rev_comp	productive	stop_codon	sequence_alignment	germline_alignment	v_call	d_call	j_call	c_call	junction	junction_length	junction_aa	v_score	d_score	j_score	c_score	v_cigar	d_cigar	j_cigar	c_cigar	v_identity	v_evalue	d_identity	d_evalue	j_identity	j_evalue	v_sequence_start	v_sequence_end	v_germline_start	v_germline_end	d_sequence_start	d_sequence_end	d_germline_start	d_germline_end	j_sequence_start	j_sequence_end	j_germline_start	j_germline_end	np1_length	np2_length	duplicate_count	vj_in_frame
IVKNQEJ01BVGQ6	1	IVKNQEJ01BVGQ6	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACTGTCTCCTCA	T	T	F			IGHV4-31*03	IGHD1-7*01,IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGG	36	CASGVAGTFDYW	430	16.4	75.8		22N1S275=	11N280S8=	6N292S32=1X9=		1	1E-122	1	2.7	0.9762	6E-18	0	275	0	317	279	287	10	18	291	333	5	47	4	4	1247	T
IVKNQEJ01AQVWS	1	IVKNQEJ01AQVWS	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCAACTACAACCCCTCCCTCAAGAGTCGAGTCACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA	T	T	F			IGHV4-31*03	IGHD1-7*01,IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGG	36	CASGVAGTFDYW	420	16.4	83.8		22N1S156=1X10=1X17=1X89=	11N280S8=	6N292S42=		0.9891	8E-120	1	2.7	1	2E-20	0	275	0	317	279	287	10	18	291	333	5	47	4	4	4	T
IVKNQEJ01AOYFZ	1	IVKNQEJ01AOYFZ	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGTAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA	T	F	T			IGHV4-31*03	IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGTAACTTTTGACTACTGG	37	CASGVAGNF*LLX	430	20.4	83.8		22N1S275=	11N280S10=	6N293S42=		1	1E-122	1	0.17	1	2E-20	0	275	0	317	279	289	10	20	292	334	5	47	4	3	92	F
IVKNQEJ01EI5S4	1	IVKNQEJ01EI5S4	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA	T	T	F			IGHV4-31*03	IGHD1-7*01,IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGG	36	CASGVAGTFDYW	430	16.4	83.8		22N1S275=	11N280S8=	6N292S42=		1	1E-122	1	2.7	1	2E-20	0	275	0	317	279	287	10	18	291	333	5	47	4	4	2913	T"""

        file2_content = """rearrangement_id	rearrangement_set_id	sequence_id	sequence	rev_comp	productive	sequence_alignment	germline_alignment	v_call	d_call	j_call	c_call	junction	junction_length	junction_aa	v_score	d_score	j_score	c_score	v_cigar	d_cigar	j_cigar	c_cigar	v_identity	v_evalue	d_identity	d_evalue	j_identity	j_evalue	v_sequence_start	v_sequence_end	v_germline_start	v_germline_end	d_sequence_start	d_sequence_end	d_germline_start	d_germline_end	j_sequence_start	j_sequence_end	j_germline_start	j_germline_end	np1_length	np2_length	duplicate_count	vj_in_frame
IVKNQEJ01DGRRI	1	IVKNQEJ01DGRRI	GGCCCAGGACTGGTGAAGCCTTCGGAGACCCTGTCCCTCACCTGCGCTGTCTATGGTGGGTCCTTCAGTGGTTACTACTGGAGCTGGATCCGCCAGCCCCCAGGGAAGGGTCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA	T	T			IGHV4-34*09	IGHD1-7*01,IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGG	36	CASGVAGTFDYW	389	16.4	83.8		22N1S23=2X85=1X15=1X1=1X3=1X2=1X1=1X5=1X6=1X118=	11N274S8=	6N286S42=		0.9628	2E-110	1	2.6	1	2E-20	0	269	0	317	273	281	10	18	285	327	5	47	4	4	1	T
IVKNQEJ01APN5N	1	IVKNQEJ01APN5N	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTAGGGCCAGGGAACCCTGGTCACTGTCTCCTCA	T	F			IGHV4-31*03	IGHD1-7*01,IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTAG	36	CASGVAGTFDY*	430	16.4	67.9		22N1S275=	11N280S8=	6N292S10=1X21=1X9=		1	1E-122	1	2.7	0.9524	1E-15	0	275	0	317	279	287	10	18	291	333	5	47	4	4	1	F
IVKNQEJ01B0TT2	1	IVKNQEJ01B0TT2	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGTAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACTGTCTCCTCA	T	F			IGHV4-31*03	IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGTAACTTTTGACTACTGG	37	CASGVAGNF*LLX	430	20.4	75.8		22N1S275=	11N280S10=	6N293S32=1X9=		1	1E-122	1	0.17	0.9762	6E-18	0	275	0	317	279	289	10	20	292	334	5	47	4	3	30	F
IVKNQEJ01AIS74	1	IVKNQEJ01AIS74	GGCGCAGGACTGTTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGGCGGGGTGGCTGGTAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA	T	F			IGHV4-31*03	IGHD6-19*01	IGHJ4*02		TGTGCGAGGCGGGGTGGCTGGTAACTTTTGACTACTGG	38	CARRGGW*LLTTG	424	20.4	83.8		22N1S3=1X8=1X262=	11N281S10=	6N294S42=		0.9927	9E-121	1	0.17	1	2E-20	0	275	0	317	280	290	10	20	293	335	5	47	5	3	4	F
IVKNQEJ01AJ44V	1	IVKNQEJ01AJ44V	GGCCCAGGACTGGTGAAGCCTTCGGAGACCCTGTCCCTCACCTGCGCTGTCTATGGTGGGTCCTTCAGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACTGTCTCCTCA	T	T			IGHV4-59*06	IGHD1-7*01,IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGG	36	CASGVAGTFDYW	386	16.4	75.8		22N1S45=1X5=2X6=1X3=1X5=1X22=1X4=1X1=1X1=1X165=	11N274S8=	6N286S32=1X9=		0.9625	2E-109	1	2.6	0.9762	5E-18	0	267	0	315	273	281	10	18	285	327	5	47	6	4	12	T"""

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path / "rep2.tsv", "w") as file:
            file.writelines(file2_content)

        if add_metadata:
            pd.DataFrame({'filename': ['rep1.tsv', 'rep2.tsv'], 'subject_id': [1, 2]}).to_csv(str(path / 'metadata.csv'), index=False)

    def get_column_mapping(self):
        column_mapping = {
            "locus": "chain"
        }

        return column_mapping

    def test_import_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "ioairr_repertoire/")
        self.create_dummy_dataset(path, True)

        column_mapping = self.get_column_mapping()
        params = {"is_repertoire": True, "result_path": path, "path": path, "metadata_file": path / "metadata.csv",
                  "import_out_of_frame": False, "import_with_stop_codon": False, "import_illegal_characters": False,
                  "import_productive": True, "region_type": "IMGT_CDR3", "import_empty_nt_sequences": True, "import_empty_aa_sequences": False,
                  "column_mapping": column_mapping,
                  "separator": "\t"}

        dataset = AIRRImport.import_dataset(params, "airr_repertoire_dataset")

        self.assertEqual(2, dataset.get_example_count())
        for index, rep in enumerate(dataset.get_data()):
            if index == 0:
                self.assertEqual(3, len(rep.sequences))
                self.assertListEqual(["IVKNQEJ01BVGQ6", "IVKNQEJ01AQVWS", "IVKNQEJ01EI5S4"], rep.get_sequence_identifiers().tolist())
                self.assertListEqual(['IGHV4-31*03', 'IGHV4-31*03', 'IGHV4-31*03'], rep.get_v_genes().tolist())
                self.assertListEqual([36, 36, 36], rep.get_attribute("junction_length").tolist())
                self.assertListEqual(["ASGVAGTFDY", "ASGVAGTFDY", "ASGVAGTFDY"], rep.get_sequence_aas().tolist())
                self.assertListEqual([1247, 4, 2913], rep.get_counts().tolist())
                self.assertListEqual([Chain.HEAVY for i in range(3)], rep.get_chains().tolist())
            else:
                self.assertEqual(2, len(rep.sequences))

        shutil.rmtree(path)

    def test_sequence_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "ioairr_sequence"
        PathBuilder.remove_old_and_build(path)
        self.create_dummy_dataset(path, False)

        column_mapping = self.get_column_mapping()
        params = {"is_repertoire": False, "result_path": path, "path": path,
                  "import_out_of_frame": False, "import_with_stop_codon": False,
                  "import_productive": True, "region_type": "IMGT_CDR3", "import_empty_nt_sequences": True, "import_empty_aa_sequences": False,
                  "column_mapping": column_mapping, "import_illegal_characters": False,
                  "separator": "\t", "sequence_file_size": 1}

        dataset = AIRRImport.import_dataset(params, "airr_sequence_dataset")

        self.assertEqual(5, dataset.get_example_count())
        self.assertEqual(5, len(dataset.get_filenames()))

        for idx, sequence in enumerate(dataset.get_data()):
            self.assertEqual(sequence.sequence_aa, "ASGVAGTFDY")

        v_genes = sorted(["IGHV4-31*03", "IGHV4-31*03", "IGHV4-31*03", "IGHV4-34*09", "IGHV4-59*06"])
        self.assertListEqual(sorted([sequence.metadata.v_call for sequence in dataset.get_data()]), v_genes)

        shutil.rmtree(path)

    def test_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "io_airr_receptor/")

        file_content = """rearrangement_id	rearrangement_set_id	sequence_id	sequence	rev_comp	productive	sequence_alignment	germline_alignment	v_call	d_call	j_call	c_call	junction	junction_length	junction_aa	v_score	d_score	j_score	c_score	v_cigar	d_cigar	j_cigar	c_cigar	v_identity	v_evalue	d_identity	d_evalue	j_identity	j_evalue	v_sequence_start	v_sequence_end	v_germline_start	v_germline_end	d_sequence_start	d_sequence_end	d_germline_start	d_germline_end	j_sequence_start	j_sequence_end	j_germline_start	j_germline_end	np1_length	np2_length	duplicate_count	cell_id
IVKNQEJ01DGRRI	1	IVKNQEJ01DGRRI	GGCCCAGGACTGGTGAAGCCTTCGGAGACCCTGTCCCTCACCTGCGCTGTCTATGGTGGGTCCTTCAGTGGTTACTACTGGAGCTGGATCCGCCAGCCCCCAGGGAAGGGTCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA	T	T			IGHV4-34*09	IGHD1-7*01,IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTGG	36	CASGVAGTFDYW	389	16.4	83.8		22N1S23=2X85=1X15=1X1=1X3=1X2=1X1=1X5=1X6=1X118=	11N274S8=	6N286S42=		0.9628	2E-110	1	2.6	1	2E-20	0	269	0	317	273	281	10	18	285	327	5	47	4	4	1	1
IVKNQEJ01APN5N	1	IVKNQEJ01APN5N	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTAGGGCCAGGGAACCCTGGTCACTGTCTCCTCA	T	T			IGLV4-31*03	IGLD1-7*01,IGLD6-19*01	IGLJ4*02		TGTGCGAGCGGGGTGGCTGGAACTTTTGACTACTAG	36	CASGVAGTFDY	430	16.4	67.9		22N1S275=	11N280S8=	6N292S10=1X21=1X9=		1	1E-122	1	2.7	0.9524	1E-15	0	275	0	317	279	287	10	18	291	333	5	47	4	4	1	1
IVKNQEJ01B0TT2	1	IVKNQEJ01B0TT2	GGCCCAGGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGCGGGGTGGCTGGTAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACTGTCTCCTCA	T	T			IGHV4-31*03	IGHD6-19*01	IGHJ4*02		TGTGCGAGCGGGGTGGCTGGTAACTTTTGACTACTGG	37	CASGVAGNFLLX	430	20.4	75.8		22N1S275=	11N280S10=	6N293S32=1X9=		1	1E-122	1	0.17	0.9762	6E-18	0	275	0	317	279	289	10	20	292	334	5	47	4	3	30	2
IVKNQEJ01AIS74	1	IVKNQEJ01AIS74	GGCGCAGGACTGTTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATCAGCAGTGGTGGTTACTACTGGAGCTGGATCCGCCAGCACCCAGGGAAGGGCCTGGAGTGGATTGGGTACATCTATTACAGTGGGAGCACCTACTACAACCCGTCCCTCAAGAGTCGAGTTACCATATCAGTAGACACGTCTAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACTGCCGCGGACACGGCCGTGTATTACTGTGCGAGGCGGGGTGGCTGGTAACTTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA	T	T			IGLV4-31*03	IGLD6-19*01	IGLJ4*02		TGTGCGAGGCGGGGTGGCTGGTAACTTTTGACTACTGG	38	CARRGGWLLTTG	424	20.4	83.8		22N1S3=1X8=1X262=	11N281S10=	6N294S42=		0.9927	9E-121	1	0.17	1	2E-20	0	275	0	317	280	290	10	20	293	335	5	47	5	3	4	2
"""
        with open(path / "rep1.tsv", "w") as file:
            file.writelines(file_content)

        column_mapping = self.get_column_mapping()
        params = {"is_repertoire": False, "result_path": path, "path": path,
                  "paired": True, "import_illegal_characters": False,
                  "import_out_of_frame": False, "import_with_stop_codon": False,
                  "import_productive": True, "region_type": "IMGT_CDR3", "import_empty_nt_sequences": True, "import_empty_aa_sequences": False,
                  "column_mapping": column_mapping, "receptor_chains": "IGH_IGL",
                  "separator": "\t", "sequence_file_size": 1}

        dataset = AIRRImport.import_dataset(params, "airr_receptor_dataset")

        self.assertEqual(2, dataset.get_example_count())
        self.assertEqual(2, len(dataset.get_filenames()))

        for idx, receptor in enumerate(dataset.get_data()):
            self.assertTrue(receptor.heavy.sequence_aa in ['ASGVAGTFDY', 'ASGVAGNFLL'])

        shutil.rmtree(path)

    def test_import_exported_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "io_airr/")
        PathBuilder.build(path / 'initial')
        self.create_dummy_dataset(path / 'initial', True)

        column_mapping = self.get_column_mapping()
        params = {"is_repertoire": True, "result_path": path / 'imported', "path": path / 'initial', "metadata_file": path / "initial/metadata.csv",
                  "import_out_of_frame": False, "import_with_stop_codon": False,
                  "import_productive": True, "region_type": "IMGT_CDR3", "import_empty_nt_sequences": True, "import_empty_aa_sequences": False,
                  "column_mapping": column_mapping, "import_illegal_characters": False,
                  "separator": "\t"}

        dataset1 = AIRRImport.import_dataset(params, "airr_repertoire_dataset1")

        path_exported = path / "exported_repertoires"
        AIRRExporter.export(dataset1, path_exported)

        params["path"] = path_exported
        params["metadata_file"] = path_exported / "metadata.csv"
        params["result_path"] = path / "final_output"
        dataset2 = AIRRImport.import_dataset(params, "airr_repertoire_dataset2")

        for attribute in ["amino_acid_sequence", "nucleotide_sequence", "v_call", "j_call", "chain", "frame_type", "custom_params"]:
            self.assertListEqual([sequence.get_attribute(attribute) for sequence in dataset1.repertoires[0].sequences],
                                 [sequence.get_attribute(attribute) for sequence in dataset2.repertoires[0].sequences])

        self.assertEqual(dataset1.repertoires[0].get_region_type(), dataset2.repertoires[0].get_region_type())

        shutil.rmtree(path)

    def test_minimal_dataset(self):
        # test to make sure import works with minimally specified input
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "ioairr_minimal/")
        file1_content = """sequence_id	junction_aa
IVKNQEJ01BVGQ6	CASGVAGTFDYW
IVKNQEJ01AQVWS	CASGVAGTFDYW
IVKNQEJ01AOYFZ	CASGVAGNFLLX
IVKNQEJ01EI5S4	CASGVAGTFDYW"""

        with open(path / "rep1.tsv", "w") as file:
            file.writelines(file1_content)

        with open(path / "metadata.csv", "w") as file:
            file.writelines("""filename,subject_id
rep1.tsv,1""")

        column_mapping = self.get_column_mapping()

        params = {"is_repertoire": True, "result_path": path, "path": path, "metadata_file": path / "metadata.csv",
                  "import_out_of_frame": False, "import_with_stop_codon": False,
                  "import_productive": True, "region_type": "IMGT_CDR3", "import_empty_nt_sequences": True, "import_empty_aa_sequences": False,
                  "column_mapping": column_mapping, "import_illegal_characters": False,
                  "separator": "\t"}

        AIRRImport.import_dataset(params, "airr_minimal_repertoire_dataset")

        shutil.rmtree(path)
