import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.data_model.SequenceParams import Chain
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedReceptorsEncoder import MatchedReceptorsEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedReceptorsEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dummy_data(self, path):

        # Setting up dummy data
        labels = {"subject_id": ["subject_1", "subject_1", "subject_2", "subject_2", "subject_3"],
                  "label": ["yes", "yes", "no", "no", "no"]}

        metadata_alpha = {"v_call": "TRAV1", "j_call": "TRAJ1", "locus": Chain.ALPHA.value}
        metadata_beta = {"v_call": "TRBV1", "j_call": "TRBJ1", "locus": Chain.BETA.value}

        dataset = RepertoireBuilder.build_dataset(sequences=[["AAAA"], ["SSSS"], ["AAAA", "CCCC"],
                                                             ["SSSS", "TTTT"], ["AAAA", "CCCC", "SSSS", "TTTT"]],
                                                  path=path, labels=labels,
                                                  seq_metadata=[[{**metadata_alpha, "duplicate_count": 10}],
                                                                [{**metadata_beta, "duplicate_count": 10}],
                                                                [{**metadata_alpha, "duplicate_count": 5},
                                                                 {**metadata_alpha, "duplicate_count": 5}],
                                                                [{**metadata_beta, "duplicate_count": 5},
                                                                 {**metadata_beta, "duplicate_count": 5}],
                                                                [{**metadata_alpha, "duplicate_count": 1},
                                                                 {**metadata_alpha, "duplicate_count": 2},
                                                                 {**metadata_beta, "duplicate_count": 1},
                                                                 {**metadata_beta, "duplicate_count": 2}]],
                                                  subject_ids=labels["subject_id"])

        label_config = LabelConfiguration()
        label_config.add_label("subject_id", labels["subject_id"])
        label_config.add_label("label", labels["label"])

        # clonotype 100 with TRA=AAAA, TRB = SSSS; clonotype 200 with TRA=CCCC, TRB = TTTT
        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
100	TRA	AAAAAA	TRAV1	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
100	TRB	SSSSSS	TRBV1	TRBJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
200	TRA	CCCCCC	TRAV1	TRAJ1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
200	TRB	TTTTTT	TRBV1	TRBJ1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
"""

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        reference_receptors = {"params": {"path": path / "refs.tsv", "region_type": "IMGT_JUNCTION"}, "format": "VDJdb"}

        return dataset, label_config, reference_receptors, labels

    def test__encode_new_dataset(self):
        expected_outcomes = {"all": {True: [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5],
                                            [0.16666666666666666, 0.16666666666666666, 0.3333333333333333,
                                             0.3333333333333333]],
                                     False: [[10, 0, 0, 0], [0, 10, 0, 0], [5, 0, 5, 0], [0, 5, 0, 5], [1, 1, 2, 2]]},
                             "unique": {True: [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5],
                                               [0.25, 0.25, 0.25, 0.25]],
                                        False: [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]]}}

        for reads in ["all", "unique"]:
            for normalize in [True, False]:

                path = EnvironmentSettings.tmp_test_path / "matched_receptors_encoder_all/"

                dataset, label_config, reference_receptors, labels = self.create_dummy_data(path)

                encoder = MatchedReceptorsEncoder.build_object(dataset, **{
                    "reference": reference_receptors,
                    "max_edit_distances": 0,
                    "reads": reads,
                    "sum_matches": False,
                    "normalize": normalize
                })

                encoded = encoder.encode(dataset, EncoderParams(
                    result_path=path,
                    label_config=label_config,
                ))

                expected_outcome = expected_outcomes[reads][normalize]
                for index, row in enumerate(expected_outcome):
                    self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

                self.assertDictEqual(encoded.encoded_data.labels, {"label": ["yes", "yes", "no", "no", "no"],
                                                                   "subject_id": ["subject_1", "subject_1", "subject_2",
                                                                                  "subject_2", "subject_3"]})
                self.assertListEqual(encoded.encoded_data.feature_names,
                                     ["100.TRA", "100.TRB", "200.TRA", "200.TRB"])

                self.assertListEqual(list(encoded.encoded_data.feature_annotations.receptor_id),
                                     ["100", "100", "200", "200"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.locus),
                                     ["TRA", "TRB", "TRA", "TRB"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.sequence),
                                     ["AAAA", "SSSS", "CCCC", "TTTT"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.v_call),
                                     ["TRAV1", "TRBV1", "TRAV1", "TRBV1"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.j_call),
                                     ["TRAJ1", "TRBJ1", "TRAJ1", "TRBJ1"])

                shutil.rmtree(path)

    def test__encode_new_dataset_sum(self):
        expected_outcomes = {"all": {True: [[1, 0], [0, 1], [1, 0], [0, 1], [0.5, 0.5]],
                                     False: [[10, 0], [0, 10], [10, 0], [0, 10], [3, 3]]},
                             "unique": {True: [[1, 0], [0, 1], [1, 0], [0, 1], [0.5, 0.5]],
                                        False: [[1, 0], [0, 1], [2, 0], [0, 2], [2, 2]]}}

        for reads in ["all", "unique"]:
            for normalize in [True, False]:

                path = EnvironmentSettings.tmp_test_path / "matched_receptors_encoder_all_sum/"

                dataset, label_config, reference_receptors, labels = self.create_dummy_data(path)

                encoder = MatchedReceptorsEncoder.build_object(dataset, **{
                    "reference": reference_receptors,
                    "max_edit_distances": 0,
                    "reads": reads,
                    "sum_matches": True,
                    "normalize": normalize
                })

                encoded = encoder.encode(dataset, EncoderParams(
                    result_path=path,
                    label_config=label_config,
                ))

                expected_outcome = expected_outcomes[reads][normalize]

                for index, row in enumerate(expected_outcome):
                    self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

                self.assertDictEqual(encoded.encoded_data.labels, {"label": ["yes", "yes", "no", "no", "no"],
                                                                   "subject_id": ["subject_1", "subject_1", "subject_2",
                                                                                  "subject_2", "subject_3"]})
                self.assertListEqual(encoded.encoded_data.feature_names,
                                     [f"sum_of_{reads}_reads_TRA", f"sum_of_{reads}_reads_TRB"])

                self.assertIsNone(encoded.encoded_data.feature_annotations)

                shutil.rmtree(path)
