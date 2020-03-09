import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedReceptorsRepertoireEncoder import MatchedReceptorsRepertoireEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedReceptorsEncoder(TestCase):
    def test__encode_new_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/matched_receptors_encoder/"

        # Setting up dummy data
        labels = {"donor": ["donor1", "donor1", "donor2", "donor2", "donor3"],
                  "label": ["yes", "yes", "no", "no", "no"]}

        metadata_alpha = {"v_gene": "v1", "j_gene": "j1", "chain": Chain.A.value}
        metadata_beta = {"v_gene": "v1", "j_gene": "j1", "chain": Chain.B.value}

        repertoires, metadata = RepertoireBuilder.build(sequences=[["AAAA"],
                                                                 ["SSSS"],
                                                                 ["AAAA", "CCCC"],
                                                                 ["SSSS", "TTTT"],
                                                                 ["AAAA", "CCCC", "SSSS", "TTTT"]],
                                                      path=path, labels=labels,
                                                      seq_metadata=[[{**metadata_alpha, "count": 10}],
                                                                    [{**metadata_beta, "count": 10}],
                                                                    [{**metadata_alpha, "count": 5}, {**metadata_alpha, "count": 5}],
                                                                    [{**metadata_beta, "count": 5}, {**metadata_beta, "count": 5}],
                                                                    [{**metadata_alpha, "count": 1}, {**metadata_alpha, "count": 2},
                                                                     {**metadata_beta, "count": 1}, {**metadata_beta, "count": 2}]],
                                                        donors=["donor1", "donor1", "donor2", "donor2", "donor3"])

        dataset = RepertoireDataset(repertoires=repertoires)

        label_config = LabelConfiguration()
        label_config.add_label("donor", labels["donor"])
        label_config.add_label("label", labels["label"])

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
100a	TRA	AAAA	TRAv1	TRAj1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
100a	TRB	SSSS	TRBv1	TRBj1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
200a	TRA	CCCC	TRAv1	TRAj1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
200a	TRB	TTTT	TRBv1	TRBj1	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
        """

        with open(path + "refs.tsv", "w") as file:
            file.writelines(file_content)

        reference_receptors = {"path": path + "refs.tsv", "format": "VDJdb"}

        encoder = MatchedReceptorsRepertoireEncoder.create_encoder(dataset, {
            "reference_receptors": reference_receptors,
            "one_file_per_donor": True
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_configuration=label_config,
            filename="dataset.csv"
        ))

        # General tests: does the tool run and give the correct output?
        expected_outcome = [[10, 0, 0, 0], [0, 10, 0, 0], [5, 0, 5, 0], [0, 5, 0, 5], [1, 1, 2, 2]]
        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertDictEqual(encoded.encoded_data.labels, labels)
        self.assertListEqual(encoded.encoded_data.feature_names, ["100a.alpha", "100a.beta", "200a.alpha", "200a.beta"])

        self.assertListEqual(list(encoded.encoded_data.feature_annotations.id), ['100a', '100a', '200a', '200a'])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.chain), ["alpha", "beta", "alpha", "beta"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.sequence), ["AAAA", "SSSS", "CCCC", "TTTT"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.v_gene), ["v1" for i in range(4)])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.j_gene), ["j1" for i in range(4)])

        encoder = MatchedReceptorsRepertoireEncoder.create_encoder(dataset, {
            "reference_receptors": reference_receptors,
            "one_file_per_donor": False
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_configuration=label_config,
            filename="dataset.csv"
        ))

        expected_outcome = [[10, 10, 0, 0], [5, 5, 5, 5], [1, 1, 2, 2]]
        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertDictEqual(encoded.encoded_data.labels, {"label": ["yes", "no", "no"]})

        # If donor is not a label, one_file_per_donor can not be False (KeyError raised)
        label_config = LabelConfiguration()
        label_config.add_label("label", labels["label"])
        params=EncoderParams(result_path=path, label_configuration=label_config, filename="dataset.csv")

        self.assertRaises(KeyError, encoder.encode, dataset, params)

        # If one_file_per_donor is True, the key "donor" does not need to be specified
        encoder = MatchedReceptorsRepertoireEncoder.create_encoder(dataset, {
            "reference_receptors": reference_receptors,
            "one_file_per_donor": True
        })

        encoder.encode(dataset, params)

        shutil.rmtree(path)

