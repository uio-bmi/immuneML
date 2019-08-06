import shutil
from unittest import TestCase

from source.IO.sequence_import.VDJdbSequenceImport import VDJdbSequenceImport
from source.data_model.receptor_sequence.Chain import Chain
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestSequenceImport(TestCase):
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

        sequences = VDJdbSequenceImport.import_all_sequences(path + "seqs.tsv", VDJdbSequenceImport.COLUMNS)

        self.assertEqual(7, len(sequences))
        self.assertEqual("CASSYVGNTGELFF", sequences[0].get_sequence())
        self.assertEqual("J2-2*01", sequences[3].metadata.j_gene)
        self.assertEqual("V6-5*01", sequences[3].metadata.v_gene)
        self.assertTrue(all([sequence.metadata.chain == Chain.B for sequence in sequences]))

        shutil.rmtree(path)
