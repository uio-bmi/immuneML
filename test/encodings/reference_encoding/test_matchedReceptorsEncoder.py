import os
import shutil
from unittest import TestCase

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
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

        metadata_alpha = {"v_gene": "TRAV1", "j_gene": "TRAJ1", "chain": Chain.ALPHA.value}
        metadata_beta = {"v_gene": "TRBV1", "j_gene": "TRBJ1", "chain": Chain.BETA.value}

        repertoires, metadata = RepertoireBuilder.build(sequences=[["AAAA"],
                                                                   ["SSSS"],
                                                                   ["AAAA", "CCCC"],
                                                                   ["SSSS", "TTTT"],
                                                                   ["AAAA", "CCCC", "SSSS", "TTTT"]],
                                                        path=path, labels=labels,
                                                        seq_metadata=[[{**metadata_alpha, "count": 10}],
                                                                      [{**metadata_beta, "count": 10}],
                                                                      [{**metadata_alpha, "count": 5},
                                                                       {**metadata_alpha, "count": 5}],
                                                                      [{**metadata_beta, "count": 5},
                                                                       {**metadata_beta, "count": 5}],
                                                                      [{**metadata_alpha, "count": 1},
                                                                       {**metadata_alpha, "count": 2},
                                                                       {**metadata_beta, "count": 1},
                                                                       {**metadata_beta, "count": 2}]],
                                                        subject_ids=labels["subject_id"])

        dataset = RepertoireDataset(repertoires=repertoires)

        label_config = LabelConfiguration()
        label_config.add_label("subject_id", labels["subject_id"])
        label_config.add_label("label", labels["label"])

        # clonotype 100 with TRA=AAAA, TRB = SSSS; clonotype 200 with TRA=CCCC, TRB = TTTT
#         file_content = """Cell type	Clonotype ID	Chain: TRA (1)	TRA - V gene (1)	TRA - D gene (1)	TRA - J gene (1)	Chain: TRA (2)	TRA - V gene (2)	TRA - D gene (2)	TRA - J gene (2)	Chain: TRB (1)	TRB - V gene (1)	TRB - D gene (1)	TRB - J gene (1)	Chain: TRB (2)	TRB - V gene (2)	TRB - D gene (2)	TRB - J gene (2)	Cells pr. clonotype	Clonotype (Id)	Clonotype (Name)
# TCR_AB	100	AAAA	TRAV1		TRAJ1	null	null	null	null	SSSS	TRBV1		TRBJ1	null	null	null	null	1	1941533	3ca0cd7f-02fd-40bb-b295-7cd5d419e474(101, 102, 103, 104, 105, 108, 109, 127, 128, 130, 131, 132, 133, 134, 174)Size:1
# TCR_AB	200	CCCC	TRAV1		TRAJ1	null	null	null	null	TTTT	TRBV1		TRBJ1	null	null	null	null	1	1941532	1df22bbc-8113-46b9-8913-da95fcf9a568(101, 102, 103, 104, 105, 108, 109, 127, 128, 130, 131, 132, 133, 134, 174)Size:1
# """

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
100	TRA	AAAA	TRAV1	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
100	TRB	SSSS	TRBV1	TRBJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
200	TRA	CCCC	TRAV1	TRAJ1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
200	TRB	TTTT	TRBV1	TRBJ1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
"""

        with open(path / "refs.tsv", "w") as file:
            file.writelines(file_content)

        reference_receptors = {"params": {"path": path / "refs.tsv", "region_type": "IMGT_JUNCTION"}, "format": "VDJdb"}

        return dataset, label_config, reference_receptors, labels

    def test__encode_new_dataset(self):
        path = EnvironmentSettings.root_path / "test/tmp/matched_receptors_encoder/"

        dataset, label_config, reference_receptors, labels = self.create_dummy_data(path)

        encoder = MatchedReceptorsEncoder.build_object(dataset, **{
            "reference": reference_receptors,
            "max_edit_distances": 0
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config,
            filename="dataset.csv"
        ))

        expected_outcome = [[10, 0, 0, 0],[0, 10, 0, 0],[5, 0, 5, 0], [0, 5, 0, 5], [1, 1, 2, 2]]
        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertDictEqual(encoded.encoded_data.labels, {"label": ["yes", "yes", "no", "no", "no"],
                                                           "subject_id": ["subject_1", "subject_1", "subject_2", "subject_2", "subject_3"]})
        self.assertListEqual(encoded.encoded_data.feature_names, ["100.alpha", "100.beta", "200.alpha", "200.beta"])

        self.assertListEqual(list(encoded.encoded_data.feature_annotations.receptor_id), ["100", "100", "200", "200"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.chain), ["alpha", "beta", "alpha", "beta"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.sequence), ["AAAA", "SSSS", "CCCC", "TTTT"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.v_gene), ["TRAV1", "TRBV1", "TRAV1", "TRBV1"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.j_gene), ["TRAJ1", "TRBJ1", "TRAJ1", "TRBJ1"])

        shutil.rmtree(path)