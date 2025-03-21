import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import Chain, RegionType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.reference_encoding.MatchedSequencesEncoder import MatchedSequencesEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedSequencesEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dummy_data(self, path):
        # Setting up dummy data
        labels = {"subject_id": ["subject_1", "subject_2", "subject_3"],
                  "label": ["yes", "yes", "no"]}

        metadata = {"v_call": "TRBV1", "j_call": "TRBJ1", "locus": Chain.BETA.value}

        dataset = RepertoireBuilder.build_dataset(sequences=[["AAAA"], ["SSSS"], ["SSSS", "CCCC"]],
                                                  path=path, labels=labels,
                                                  seq_metadata=[[{**metadata, "duplicate_count": 10}],
                                                                [{**metadata, "duplicate_count": 10}],
                                                                [{**metadata, "duplicate_count": 5},
                                                                 {**metadata, "duplicate_count": 5}]],
                                                  subject_ids=labels["subject_id"])

        label_config = LabelConfiguration()
        label_config.add_label("subject_id", labels["subject_id"])
        label_config.add_label("label", labels["label"])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
100	TRB	AAAAAA	TRBV1	TRBJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
200	TRB	SSSSSS	TRBV1	TRBJ1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0"""

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        reference_sequences = {"params": {"path": path / "refs.tsv", "region_type": "FULL_SEQUENCE"}, "format": "VDJdb"}

        return dataset, label_config, reference_sequences, labels

    def test__encode_new_dataset(self):
        expected_outcomes = {"unique": {True: [[1, 0], [0, 1], [0, 0.5]], False: [[1, 0], [0, 1], [0, 1]]},
                             "all": {True: [[1, 0], [0, 1], [0, 0.5]], False: [[10, 0], [0, 10], [0, 5]]}}

        for reads in ["unique", "all"]:
            for normalize in [True, False]:

                path = PathBuilder.remove_old_and_build(
                    EnvironmentSettings.tmp_test_path / "matched_sequences_encoder_all/")

                dataset, label_config, reference_sequences, labels = self.create_dummy_data(path)

                encoder = MatchedSequencesEncoder.build_object(dataset, **{
                    "reference": reference_sequences,
                    "max_edit_distance": 0,
                    "reads": reads,
                    "sum_matches": False,
                    "normalize": normalize,
                    "output_count_as_feature": False
                })

                encoded = encoder.encode(dataset, EncoderParams(
                    result_path=path,
                    label_config=label_config,
                    region_type=RegionType.IMGT_CDR3,
                    sequence_type=SequenceType.AMINO_ACID
                ))

                expected_outcome = expected_outcomes[reads][normalize]
                for index, row in enumerate(expected_outcome):
                    self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

                self.assertDictEqual(encoded.encoded_data.labels, {"label": ["yes", "yes", "no"],
                                                                   "subject_id": ["subject_1", "subject_2",
                                                                                  "subject_3"]})
                self.assertListEqual(encoded.encoded_data.feature_names, ['TRBV1_AAAA_TRBJ1', 'TRBV1_SSSS_TRBJ1'])

                self.assertListEqual(list(encoded.encoded_data.feature_annotations.sequence_id), ["100_TRB", "200_TRB"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.locus), ["TRB", "TRB"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.sequence), ["AAAA", "SSSS"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.v_call), ["TRBV1", "TRBV1"])
                self.assertListEqual(list(encoded.encoded_data.feature_annotations.j_call), ["TRBJ1", "TRBJ1"])

                shutil.rmtree(path)

    def test__encode_new_dataset_sum(self):
        expected_outcomes = {"unique": {True: [[1], [1], [0.5]], False: [[1], [1], [1]]},
                             "all": {True: [[1], [1], [0.5]], False: [[10], [10], [5]]}}

        for reads in ["unique", "all"]:
            for normalize in [True, False]:
                path = EnvironmentSettings.tmp_test_path / "matched_sequences_encoder_all_sum/"

                dataset, label_config, reference_sequences, labels = self.create_dummy_data(path)

                encoder = MatchedSequencesEncoder.build_object(dataset, **{
                    "reference": reference_sequences,
                    "max_edit_distance": 0,
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

                self.assertDictEqual(encoded.encoded_data.labels, {"label": ["yes", "yes", "no"],
                                                                   "subject_id": ["subject_1", "subject_2",
                                                                                  "subject_3"]})
                self.assertListEqual(encoded.encoded_data.feature_names, [f"sum_of_{reads}_reads"])

                self.assertIsNone(encoded.encoded_data.feature_annotations)

                shutil.rmtree(path)
