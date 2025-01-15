import os
import shutil
import unittest

import pandas as pd

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import Chain
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedReceptorsEncoder import MatchedReceptorsEncoder
from immuneML.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder
from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.reports.encoding_reports.Matches import Matches
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestMatches(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_encoded_matchedreceptors(self, path):
        # Setting up dummy data
        labels = {"subject_id": ["subject_1", "subject_2"],
                  "label": ["yes", "no"]}

        metadata_alpha = {"v_call": "TRAV1", "j_call": "TRAJ1", "locus": Chain.ALPHA.value}
        metadata_beta = {"v_call": "TRBV1", "j_call": "TRBJ1", "locus": Chain.BETA.value}

        dataset = RepertoireBuilder.build_dataset(sequences=[["AAAA", "TTTT"], ["SSSS", "TTTT"]],
                                                  path=path, labels=labels,
                                                  seq_metadata=[[{**metadata_alpha, "duplicate_count": 10},
                                                                 {**metadata_beta, "duplicate_count": 10}],
                                                                [{**metadata_alpha, "duplicate_count": 5},
                                                                 {**metadata_beta, "duplicate_count": 5}]],
                                                  subject_ids=labels["subject_id"])
        label_config = LabelConfiguration()
        label_config.add_label("subject_id", labels["subject_id"])
        label_config.add_label("label", labels["label"])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
100	TRA	AAAAAA	TRAV1	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
100	TRB	TTTTTT	TRBV1	TRBJ1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0"""

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        reference_receptors = {"params": {"path": path / "refs.tsv", "region_type": "FULL_SEQUENCE", "paired": True,
                                          "receptor_chains": "TRA_TRB"},
                               "format": "VDJdb"}

        encoder = MatchedReceptorsEncoder.build_object(dataset, **{
            "reference": reference_receptors,
            "max_edit_distances": 0,
            "reads": "all",
            "sum_matches": False,
            "normalize": False
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config
        ))

        return encoded

    def test_generate_for_matchedreceptors(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "matches_for_matchedreceptors/")

        encoded_data = self.create_encoded_matchedreceptors(path)

        report = Matches(dataset=encoded_data, result_path=path / "report_results/")

        self.assertTrue(report.check_prerequisites())
        report._generate()

        self.assertTrue(os.path.isfile(path / "report_results/complete_match_count_table.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/repertoire_sizes.csv"))

        self.assertTrue(os.path.isdir(path / "report_results/paired_matches"))
        self.assertTrue(os.path.isdir(path / "report_results/receptor_info"))

        self.assertTrue(os.path.isfile(path / "report_results/receptor_info/all_chains.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/receptor_info/all_receptors.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/receptor_info/unique_TRA_chains.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/receptor_info/unique_TRB_chains.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/receptor_info/unique_receptors.csv"))

        matches = pd.read_csv(path / "report_results/complete_match_count_table.csv")
        chains = pd.read_csv(path / "report_results/receptor_info/all_chains.csv")
        receptors = pd.read_csv(path / "report_results/receptor_info/all_receptors.csv")
        unique_alpha_chains = pd.read_csv(path / "report_results/receptor_info/unique_TRA_chains.csv")
        unique_beta_chains = pd.read_csv(path / "report_results/receptor_info/unique_TRB_chains.csv")
        unique_receptors = pd.read_csv(path / "report_results/receptor_info/unique_receptors.csv")

        self.assertListEqual(list(matches["100.TRA"]), [10, 0])
        self.assertListEqual(list(matches["100.TRB"]), [10, 5])

        self.assertListEqual(list(chains["cell_id"]), [100, 100])
        self.assertListEqual(list(receptors["cell_id"]), [100])
        self.assertListEqual(list(unique_alpha_chains["cell_id"]), [100])
        self.assertListEqual(list(unique_beta_chains["cell_id"]), [100])
        self.assertListEqual(list(unique_receptors["cell_id"]), [100])

        self.assertListEqual(list(unique_receptors["sequence_TRA"]), list(unique_alpha_chains["sequence"]))
        self.assertListEqual(list(unique_receptors["sequence_TRB"]), list(unique_beta_chains["sequence"]))

        shutil.rmtree(path)

    def create_encoded_matchedsequences(self, path):
        # Setting up dummy data
        labels = {"label": ["yes", "no"]}

        metadata_beta = {"v_call": "TRBV1", "j_call": "TRBJ1", "locus": Chain.BETA.value}

        dataset = RepertoireBuilder.build_dataset(sequences=[["AAAA", "TTTT"], ["SSSS", "TTTT"]],
                                                  path=path, labels=labels,
                                                  seq_metadata=[[{**metadata_beta, "duplicate_count": 10},
                                                                 {**metadata_beta, "duplicate_count": 10}],
                                                                [{**metadata_beta, "duplicate_count": 5},
                                                                 {**metadata_beta, "duplicate_count": 5}]],
                                                  subject_ids=['subject_1', 'subject_2'])

        label_config = LabelConfiguration()
        label_config.add_label("label", labels["label"])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
100	TRB	AAAAAA	TRBV1	TRBJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
101	TRB	AAAAAA	TRBV1	TRBJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
200	TRB	TTTTTT	TRBV1	TRBJ1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0"""

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        reference_sequences = {"params": {"path": path / "refs.tsv", "paired": False},
                               "format": "VDJdb"}

        encoder = MatchedSequencesEncoder.build_object(dataset, **{
            "reference": reference_sequences,
            "max_edit_distance": 0,
            "reads": "all",
            "sum_matches": False,
            "normalize": False,
            "output_count_as_feature": False
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config,
        ))

        return encoded

    def test_generate_for_matchedsequences(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "matches_for_matchedsequences/")

        encoded_data = self.create_encoded_matchedsequences(path)

        report = Matches(dataset=encoded_data, result_path=path / "report_results/")

        self.assertTrue(report.check_prerequisites())
        report._generate()

        self.assertTrue(os.path.isfile(path / "report_results/complete_match_count_table.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/repertoire_sizes.csv"))

        self.assertTrue(os.path.isfile(path / "report_results/sequence_info/all_chains.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/sequence_info/unique_chains.csv"))

        matches = pd.read_csv(path / "report_results/complete_match_count_table.csv")
        chains = pd.read_csv(path / "report_results/sequence_info/all_chains.csv")
        unique_chains = pd.read_csv(path / "report_results/sequence_info/unique_chains.csv")

        self.assertListEqual(list(matches["TRBV1_AAAA_TRBJ1_100_TRB"]), [10, 0])
        self.assertListEqual(list(matches["TRBV1_AAAA_TRBJ1_101_TRB"]), [10, 0])
        self.assertListEqual(list(matches["TRBV1_TTTT_TRBJ1_200_TRB"]), [10, 5])

        self.assertListEqual(list(chains["sequence_id"]), ["100_TRB", "101_TRB", "200_TRB"])
        self.assertListEqual(list(unique_chains["sequence_id"]), ["100_TRB", "200_TRB"])

        shutil.rmtree(path)

    def create_encoded_matchedregex(self, path):
        # Setting up dummy data
        labels = {"subject_id": ["subject_1", "subject_2", "subject_3"],
                  "label": ["yes", "no", "no"]}

        metadata_alpha = {"v_call": "V1", "j_call": "J1", "locus": Chain.ALPHA.value}
        metadata_beta = {"v_call": "V1", "j_call": "J1", "locus": Chain.BETA.value}

        dataset = RepertoireBuilder.build_dataset(
            sequences=[["FFAGQFGSSNTGKLIFF", "FFAGQFGSSNTGKLIYY", "FFSAGQGETQYFF"],
                       ["ASSFRFF"],
                       ["FFIFFNDYKLSFF", "CCCC", "SSSS", "TTTT"]],
            path=path, labels=labels,
            seq_metadata=[[{**metadata_alpha, "duplicate_count": 10, "v_call": "TRAV35"},
                           {**metadata_alpha, "duplicate_count": 10},
                           {**metadata_beta, "duplicate_count": 10, "v_call": "TRBV29-1"}],
                          [{**metadata_beta, "duplicate_count": 10, "v_call": "TRBV7-3"}],
                          [{**metadata_alpha, "duplicate_count": 5, "v_call": "TRAV26-2"},
                           {**metadata_alpha, "duplicate_count": 2},
                           {**metadata_beta, "duplicate_count": 1},
                           {**metadata_beta, "duplicate_count": 2}]],
            subject_ids=labels["subject_id"])

        label_config = LabelConfiguration()
        label_config.add_label("subject_id", labels["subject_id"])
        label_config.add_label("label", labels["label"])

        file_content = """id	TRAV	TRBV	TRA_regex	TRB_regex
1	TRAV35	TRBV29-1	AGQ.GSSNTGKLI	S[APGFTVML]GQGETQY
2		TRBV7-3		ASS.R.*
3	TRAV26-1		I..NDYKLS	
4	TRAV26-2		I..NDYKLS	
        """

        filepath = path / "reference_motifs.tsv"
        with open(filepath, "w") as file:
            file.writelines(file_content)

        encoder = MatchedRegexEncoder.build_object(dataset, **{
            "motif_filepath": filepath,
            "match_v_genes": False,
            "reads": "all"
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config,
        ))

        return encoded

    def test_generate_for_matchedregex(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "regex_matches_report/")

        encoded_data = self.create_encoded_matchedregex(path / "input_data/")

        report = Matches(dataset=encoded_data, result_path=path / "report_results/")

        self.assertTrue(report.check_prerequisites())
        report._generate()

        self.assertTrue(os.path.isfile(path / "report_results/complete_match_count_table.csv"))
        self.assertTrue(os.path.isfile(path / "report_results/repertoire_sizes.csv"))

        self.assertTrue(os.path.isdir(path / "report_results/paired_matches"))
        self.assertTrue(os.path.isfile(
            path / f"report_results/paired_matches/example_{encoded_data.get_example_ids()[0]}_label_yes_subject_id_subject_1.csv"))
        self.assertTrue(os.path.isfile(
            path / f"report_results/paired_matches/example_{encoded_data.get_example_ids()[1]}_label_no_subject_id_subject_2.csv"))
        self.assertTrue(os.path.isfile(
            path / f"report_results/paired_matches/example_{encoded_data.get_example_ids()[2]}_label_no_subject_id_subject_3.csv"))

        matches = pd.read_csv(path / "report_results/complete_match_count_table.csv")
        subj1_results = pd.read_csv(
            path / f"report_results/paired_matches/example_{encoded_data.get_example_ids()[0]}_label_yes_subject_id_subject_1.csv")

        self.assertListEqual(list(matches["1_TRA"]), [20, 0, 0])
        self.assertListEqual(list(matches["1_TRB"]), [10, 0, 0])
        self.assertListEqual(list(matches["2_TRB"]), [0, 10, 0])
        self.assertListEqual(list(matches["3_TRA"]), [0, 0, 5])

        self.assertListEqual(list(subj1_results["1_TRA"]), [20])
        self.assertListEqual(list(subj1_results["1_TRB"]), [10])

        shutil.rmtree(path)
