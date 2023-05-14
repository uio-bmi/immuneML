import os
import shutil
from unittest import TestCase

from immuneML.IO.dataset_import.VDJdbImport import VDJdbImport
from immuneML.caching.CacheType import CacheType
from immuneML.dsl.DefaultParamsLoader import DefaultParamsLoader
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.distance_encoding.TCRdistEncoder import TCRdistEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestTCRdistEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode_receptor_dataset(self):

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
3050	TRB	CASSPPRVYSNGAGLAGVGWRNEQFF	TRBV5-4*01	TRBJ2-1*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
15760	TRB	CASSWTWDAATLWGQGALGGANVLTF	TRBV5-5*01	TRBJ2-6*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
3050	TRA	CAAIYESRGSTLGRLYF	TRAV13-1*01	TRAJ18*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
15760	TRA	CALRLNNQGGKLIF	TRAV9-2*01	TRAJ23*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
3051	TRB	CASSPPRVYSNGAGLAGVGWRNEQFF	TRBV5-4*01	TRBJ2-1*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
15761	TRB	CASSWTWDAATLWGQGALGGANVLTF	TRBV5-5*01	TRBJ2-6*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
3051	TRA	CAAIYESRGSTLGRLYF	TRAV13-1*01	TRAJ18*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
15761	TRA	CALRLNNQGGKLIF	TRAV9-2*01	TRAJ23*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
        """
        if os.path.exists(EnvironmentSettings.root_path / "test/tmp/trcdist_encoder_receptor/"):
            shutil.rmtree(EnvironmentSettings.root_path / "test/tmp/trcdist_encoder_receptor/")

        path = PathBuilder.build(EnvironmentSettings.root_path / "test/tmp/trcdist_encoder_receptor/")

        with open(path / "receptors.tsv", "w") as file:
            file.writelines(file_content)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "vdjdb")
        params["is_repertoire"] = False
        params["paired"] = True
        params["result_path"] = path
        params["path"] = path
        params["receptor_chains"] = "TRA_TRB"
        params['organism'] = 'human'

        dataset = VDJdbImport.import_dataset(params, "vdjdb_receptor_dataset")

        encoder = TCRdistEncoder.build_object(dataset, **{"cores": 2})
        encoded_dataset = encoder.encode(dataset, EncoderParams(path / "result/", LabelConfiguration([Label("epitope", ["AVFDRKSDAK", "KLGGALQAK"])])))

        self.assertTrue(encoded_dataset.encoded_data.examples.shape[0] == encoded_dataset.encoded_data.examples.shape[1]
                        and encoded_dataset.encoded_data.examples.shape[0] == dataset.get_example_count())

        shutil.rmtree(path)

    def test_encode_sequence_dataset(self):
        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
1	TRB	CASSKLASTAGEQYF	TRBV2*01	TRBJ2-7*01	HomoSapiens	HLA-DRA*01:01	HLA-DRB1*11:01	MHCII	DRFYKTLRAEQASQEV	Gag	HIV-1	PMID:27760342	{"frequency": "70/70", "identification": "tetramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "CD8+", "clone.id": "", "donor.MHC": "HLA-A*30:01,HLA-A*30:02,HLA-B*42:01,HLA-B*57:03,HLA-Cw*17,HLA-Cw*18,HLA-DRB1*03:02,HLA-DRB1*11:01,HLA-DPB*01:01,HLA-DPB*39*01,HLA-DQB*04:02,HLA-DQB*05:01", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "viremic-controller", "subject.id": "474723", "tissue": "PBMC"}	{"cdr3": "CASSKLASTAGEQYF", "cdr3_old": "CASSKLASTAGEQYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-7*01", "jStart": 11, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV2*01"}	1
2	TRB	CASSWDSNYGYTF	TRBV5-5*01	TRBJ1-2*01	HomoSapiens	HLA-DRA*01:01	HLA-DRB1*01:01	MHCII	DRFYKTLRAEQASQEV	Gag	HIV-1	PMID:27760342	{"frequency": "23/23", "identification": "tetramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "CD8+", "clone.id": "", "donor.MHC": "HLA-A*11:01,HLA-A*32:01,HLA-B*27:05,HLA-B*27:05,HLA-Cw*01:02,HLA-Cw*02:02,HLA-DRB1*01:01,HLA-DRB1*04:01,HLA-DPB*04:01,HLA-DPB*04:01,HLA-DQB*03:02,HLA-DQB*05:01", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "elite-controller", "subject.id": "388031", "tissue": "PBMC"}	{"cdr3": "CASSWDSNYGYTF", "cdr3_old": "CASSWDSNYGYTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ1-2*01", "jStart": 7, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	1
3	TRB	CASSPLVPYEQYF	TRBV27*01	TRBJ2-7*01	HomoSapiens	HLA-DRA*01:01	HLA-DRB1*11:01	MHCII	DRFYKTLRAEQASQEV	Gag	HIV-1	PMID:27760342	{"frequency": "3/22", "identification": "tetramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "CD4+", "clone.id": "", "donor.MHC": "HLA-A*30:01,HLA-A*30:02,HLA-B*42:01,HLA-B*57:03,HLA-Cw*17,HLA-Cw*18,HLA-DRB1*03:02,HLA-DRB1*11:01,HLA-DPB*01:01,HLA-DPB*39*01,HLA-DQB*04:02,HLA-DQB*05:01", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "viremic-controller", "subject.id": "474723", "tissue": "PBMC"}	{"cdr3": "CASSPLVPYEQYF", "cdr3_old": "CASSPLVPYEQYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-7*01", "jStart": 8, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV27*01"}	1
4	TRB	CASSVGASGLTEQYF	TRBV2*01	TRBJ2-7*01	HomoSapiens	HLA-DRA*01:01	HLA-DRB1*11:01	MHCII	DRFYKTLRAEQASQEV	Gag	HIV-1	PMID:27760342	{"frequency": "3/22", "identification": "tetramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "CD4+", "clone.id": "", "donor.MHC": "HLA-A*30:01,HLA-A*30:02,HLA-B*42:01,HLA-B*57:03,HLA-Cw*17,HLA-Cw*18,HLA-DRB1*03:02,HLA-DRB1*11:01,HLA-DPB*01:01,HLA-DPB*39*01,HLA-DQB*04:02,HLA-DQB*05:01", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "viremic-controller", "subject.id": "474723", "tissue": "PBMC"}	{"cdr3": "CASSVGASGLTEQYF", "cdr3_old": "CASSVGASGLTEQYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-7*01", "jStart": 11, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV2*01"}	1
5	TRB	CASRPLATADTQYF	TRBV2*01	TRBJ2-5*01	HomoSapiens	HLA-DRA*01:01	HLA-DRB1*11:01	MHCII	DRFYKTLRAEQASQEV	Gag	HIV-1	PMID:27760342	{"frequency": "1/22", "identification": "tetramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "CD4+", "clone.id": "", "donor.MHC": "HLA-A*30:01,HLA-A*30:02,HLA-B*42:01,HLA-B*57:03,HLA-Cw*17,HLA-Cw*18,HLA-DRB1*03:02,HLA-DRB1*11:01,HLA-DPB*01:01,HLA-DPB*39*01,HLA-DQB*04:02,HLA-DQB*05:01", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "viremic-controller", "subject.id": "474723", "tissue": "PBMC"}	{"cdr3": "CASRPLATADTQYF", "cdr3_old": "CASRPLATADTQYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-5*01", "jStart": 10, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRBV2*01"}	0
6	TRB	CASRGNHRGNMNTEAFF	TRBV2*01	TRBJ1-1*01	HomoSapiens	HLA-DRA*01:01	HLA-DRB1*11:01	MHCII	DRFYKTLRAEQASQEV	Gag	HIV-1	PMID:27760342	{"frequency": "1/22", "identification": "tetramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "CD4+", "clone.id": "", "donor.MHC": "HLA-A*30:01,HLA-A*30:02,HLA-B*42:01,HLA-B*57:03,HLA-Cw*17,HLA-Cw*18,HLA-DRB1*03:02,HLA-DRB1*11:01,HLA-DPB*01:01,HLA-DPB*39*01,HLA-DQB*04:02,HLA-DQB*05:01", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "viremic-controller", "subject.id": "474723", "tissue": "PBMC"}	{"cdr3": "CASRGNHRGNMNTEAFF", "cdr3_old": "CASRGNHRGNMNTEAFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ1-1*01", "jStart": 11, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRBV2*01"}	0
        """
        if os.path.exists(EnvironmentSettings.root_path / "test/tmp/trcdist_encoder_sequence/"):
            shutil.rmtree(EnvironmentSettings.root_path / "test/tmp/trcdist_encoder_sequence/")

        path = PathBuilder.build(EnvironmentSettings.root_path / "test/tmp/trcdist_encoder_sequence/")

        with open(path / "receptors.tsv", "w") as file:
            file.writelines(file_content)

        params = DefaultParamsLoader.load(EnvironmentSettings.default_params_path / "datasets/", "vdjdb")
        params["is_repertoire"] = False
        params["paired"] = False
        params["result_path"] = path
        params["path"] = path
        params['organism'] = 'human'

        dataset = VDJdbImport.import_dataset(params, "vdjdb_sequence_dataset")

        encoder = TCRdistEncoder.build_object(dataset, **{"cores": 2})
        encoded_dataset = encoder.encode(dataset, EncoderParams(path / "result/", LabelConfiguration([Label("epitope", ["DRFYKTLRAEQASQEV"])])))

        self.assertTrue(encoded_dataset.encoded_data.examples.shape[0] == encoded_dataset.encoded_data.examples.shape[1]
                        and encoded_dataset.encoded_data.examples.shape[0] == 6)

        shutil.rmtree(path)
