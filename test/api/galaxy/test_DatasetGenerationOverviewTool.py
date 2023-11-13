import os
import shutil
from argparse import Namespace
from unittest import TestCase

import yaml

from immuneML.app.ImmuneMLApp import run_immuneML
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


class TestDatasetGenerationOverviewTool(TestCase):

    def create_dummy_dataset_from_files(self, path):
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

        with open(path / "metadata.csv", "w") as file:
            file.writelines("""filename,subject_id
rep1.tsv,1
rep2.tsv,2""")

    def generate_random_dummy_dataset(self, path):
        RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=2, sequence_count_probabilities={10:1},
                                                           sequence_length_probabilities={10:1}, labels=dict(), path=path, name="dataset")

    def prepare_specs(self, path, dataset_type="files"):
        specs = {
            "definitions": {
                "datasets": {
                    "dataset": dict()
                },
                "reports": {
                    "sequence_length_report": "SequenceLengthDistribution",
                    "vj_gene_report": "VJGeneDistribution"
                }

            },
            "instructions": {
                "my_dataset_generation_instruction": {
                    "type": "ExploratoryAnalysis",
                    "analyses": {
                        "dataset_overview": {
                            "dataset": "dataset",
                            "report": None},
                        "sequence_length_analysis": {
                            "dataset": "dataset",
                            "report": "sequence_length_report"},
                        "vj_gene_analysis": {
                            "dataset": "dataset",
                            "report": "vj_gene_report"}}}
            }
        }


        if dataset_type == "files":
            self.create_dummy_dataset_from_files(path)
            specs["definitions"]["datasets"]["dataset"] = {"format": "AIRR",
                                                           "params": {
                                                               "path": "./",
                                                               "metadata_file": "metadata.csv",
                                                               "is_repertoire": True,
                                                               "result_path": "./"}}
        elif dataset_type == "already_imported":
            self.generate_random_dummy_dataset(path)
            specs["definitions"]["datasets"]["dataset"] = {"format": "ImmuneML", "params": {"path": f"dataset.yaml",
                                                                                            "result_path": "./"}}

        yaml_path = path / "specs.yaml"
        with open(yaml_path, "w") as file:
            yaml.dump(specs, file)

        return yaml_path

    def test_run(self):
        path = EnvironmentSettings.tmp_test_path / "galaxy_api_dataset_generation_overview/"
        PathBuilder.remove_old_and_build(path)

        old_wd = os.getcwd()
        os.chdir(path)

        for dataset_type in ("files", "already_imported"):
            result_path = path / f"tool_results_path_{dataset_type}/"

            yaml_path = self.prepare_specs(path, dataset_type=dataset_type)

            run_immuneML(Namespace(**{"specification_path": yaml_path, "result_path": result_path, 'tool': "DatasetGenerationOverviewTool"}))

            self.assertTrue(os.path.isfile(result_path / "result/dataset_metadata.csv"))
            self.assertTrue(os.path.isfile(result_path / "result/dataset.yaml"))
            self.assertEqual(4, len([name for name in os.listdir(result_path / "result/repertoires/")
                                     if os.path.isfile(os.path.join(result_path / "result/repertoires/", name))]))

        os.chdir(old_wd)
        shutil.rmtree(path)
