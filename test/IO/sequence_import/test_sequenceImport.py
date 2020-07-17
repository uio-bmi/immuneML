import os
import shutil
from unittest import TestCase

from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.caching.CacheType import CacheType
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestSequenceImport(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_import_all_sequences(self):
        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
141	TRB	CASSYVGNTGELFF	TRBV6-5*01	TRBJ2-2*01	HomoSapiens	HLA-A*02:01:48	B2M	MHCI	SLLMWITQV	CTAG1B	HomoSapiens	PMID:15837811	{"frequency": "", "identification": "", "sequencing": "", "singlecell": "", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "2bnq", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "", "tissue": ""}	{"cdr3": "CASSYVGNTGELFF", "cdr3_old": "CASSYVGNTGELFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-2*01", "jStart": 7, "vCanonical": true, "vEnd": 5, "vFixType": "NoFixNeeded", "vId": "TRBV6-5*01"}	3
142	TRB	CASSYVGNTGELFF	TRBV6-5*01	TRBJ2-2*01	HomoSapiens	HLA-A*02:01:48	B2M	MHCI	SLLMWITQC	CTAG1B	HomoSapiens	PMID:15837811	{"frequency": "", "identification": "", "sequencing": "", "singlecell": "", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 2, "structure.id": "2bnr", "studies.found": 2, "study.id": "", "subject.cohort": "", "subject.id": "", "tissue": ""}	{"cdr3": "CASSYVGNTGELFF", "cdr3_old": "CASSYVGNTGELFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-2*01", "jStart": 7, "vCanonical": true, "vEnd": 5, "vFixType": "NoFixNeeded", "vId": "TRBV6-5*01"}	3
145	TRB	CASSYVGNTGELFF	TRBV6-5*01	TRBJ2-2*01	HomoSapiens	HLA-A*02:01:48	B2M	MHCI	SLLMWITQC	CTAG1B	HomoSapiens	PMID:16600963	{"frequency": "", "identification": "", "sequencing": "", "singlecell": "", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 2, "structure.id": "2f53", "studies.found": 2, "study.id": "", "subject.cohort": "", "subject.id": "", "tissue": ""}	{"cdr3": "CASSYVGNTGELFF", "cdr3_old": "CASSYVGNTGELFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-2*01", "jStart": 7, "vCanonical": true, "vEnd": 5, "vFixType": "NoFixNeeded", "vId": "TRBV6-5*01"}	3
146	TRB	CASSYVGNTGELFF	TRBV6-5*01	TRBJ2-2*01	HomoSapiens	HLA-A*02:01:59	B2M	MHCI	SLLMWITQC	CTAG1B	HomoSapiens	PMID:16600963	{"frequency": "", "identification": "", "sequencing": "", "singlecell": "", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "2f54", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "", "tissue": ""}	{"cdr3": "CASSYVGNTGELFF", "cdr3_old": "CASSYVGNTGELFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-2*01", "jStart": 7, "vCanonical": true, "vEnd": 5, "vFixType": "NoFixNeeded", "vId": "TRBV6-5*01"}	3
152	TRB	CASSYLGNTGELFF	TRBV6-5*01	TRBJ2-2*01	HomoSapiens	HLA-A*02:01:48	B2M	MHCI	SLLMWITQC	CTAG1B	HomoSapiens	PMID:17644531	{"frequency": "", "identification": "", "sequencing": "", "singlecell": "", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 3, "structure.id": "2p5e", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "", "tissue": ""}	{"cdr3": "CASSYLGNTGELFF", "cdr3_old": "CASSYLGNTGELFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-2*01", "jStart": 7, "vCanonical": true, "vEnd": 5, "vFixType": "NoFixNeeded", "vId": "TRBV6-5*01"}	3
153	TRB	CASSYLGNTGELFF	TRBV6-5*01	TRBJ2-2*01	HomoSapiens	HLA-A*02:01:48	B2M	MHCI	SLLMWITQC	CTAG1B	HomoSapiens	PMID:17644531	{"frequency": "", "identification": "", "sequencing": "", "singlecell": "", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "2p5w", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "", "tissue": ""}	{"cdr3": "CASSYLGNTGELFF", "cdr3_old": "CASSYLGNTGELFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-2*01", "jStart": 7, "vCanonical": true, "vEnd": 5, "vFixType": "NoFixNeeded", "vId": "TRBV6-5*01"}	3
155	TRB	CASSYLGNTGELFF	TRBV6-5*01	TRBJ2-2*01	HomoSapiens	HLA-A*02:01:48	B2M	MHCI	SLLMWITQC	CTAG1B	HomoSapiens	PMID:17644531	{"frequency": "", "identification": "", "sequencing": "", "singlecell": "", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "2pye", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "", "tissue": ""}	{"cdr3": "CASSYLGNTGELFF", "cdr3_old": "CASSYLGNTGELFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-2*01", "jStart": 7, "vCanonical": true, "vEnd": 5, "vFixType": "NoFixNeeded", "vId": "TRBV6-5*01"}	3"""

        path = EnvironmentSettings.root_path + "test/tmp/sequence_import/"
        PathBuilder.build(path)

        with open(path + "seqs.tsv", "w") as file:
            file.writelines(file_content)

        sequences = VDJdbSequenceImport.import_all_sequences(path + "seqs.tsv")

        self.assertEqual(7, len(sequences))
        self.assertEqual("CASSYVGNTGELFF", sequences[0].get_sequence())
        self.assertEqual("J2-2*01", sequences[3].metadata.j_gene)
        self.assertEqual("V6-5*01", sequences[3].metadata.v_gene)
        self.assertTrue(all([sequence.metadata.chain == Chain.BETA for sequence in sequences]))

        shutil.rmtree(path)

    def test_import_paired_sequences(self):

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
3050	TRB	CASSPPRVYSNGAGLAGVGWRNEQFF	TRBV5-4*01	TRBJ2-1*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
15760	TRB	CASSWTWDAATLWGQGALGGANVLTF	TRBV5-5*01	TRBJ2-6*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
3050	TRA	CAAIYESRGSTLGRLYF	TRAV13-1*01	TRAJ18*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
15760	TRA	CALRLNNQGGKLIF	TRAV9-2*01	TRAJ23*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
"""
        path = EnvironmentSettings.root_path + "test/tmp/receptor_import/"
        PathBuilder.build(path)

        with open(path + "receptors.tsv", "w") as file:
            file.writelines(file_content)

        receptors = VDJdbSequenceImport.import_paired_sequences(path + "receptors.tsv")

        self.assertEqual(2, len(receptors))
        self.assertTrue(receptor.identifier in ["3050", "15760"] for receptor in receptors)
        self.assertNotEqual(receptors[0].identifier, receptors[1].identifier)
        self.assertEqual("CASSPPRVYSNGAGLAGVGWRNEQFF",
                         [receptor for receptor in receptors if receptor.identifier == "3050"][0].beta.amino_acid_sequence)
        self.assertEqual("CALRLNNQGGKLIF",
                         [receptor for receptor in receptors if receptor.identifier == "15760"][0].alpha.amino_acid_sequence)
        self.assertEqual("J2-1*01",
                         [receptor for receptor in receptors if receptor.identifier == "3050"][0].beta.metadata.j_gene)
        self.assertEqual("V9-2*01",
                         [receptor for receptor in receptors if receptor.identifier == "15760"][0].alpha.metadata.v_gene)

        shutil.rmtree(path)
