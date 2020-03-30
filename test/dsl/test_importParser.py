import csv
import shutil
from unittest import TestCase

import pandas as pd

from source.data_model.dataset.ReceptorDataset import ReceptorDataset
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.dsl.import_parsers.ImportParser import ImportParser
from source.dsl.symbol_table.SymbolTable import SymbolTable
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImportParser(TestCase):

    def test_parse_receptor_dataset(self):
        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
                3050	TRB	CASSPPRVYSNGAGLAGVGWRNEQFF	TRBV5-4*01	TRBJ2-1*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
                15760	TRB	CASSWTWDAATLWGQGALGGANVLTF	TRBV5-5*01	TRBJ2-6*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
                3050	TRA	CAAIYESRGSTLGRLYF	TRAV13-1*01	TRAJ18*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
                15760	TRA	CALRLNNQGGKLIF	TRAV9-2*01	TRAJ23*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
                """
        path = EnvironmentSettings.root_path + "test/tmp/dslimportparservdj/"
        PathBuilder.build(path)

        with open(path + "receptors.tsv", "w") as file:
            file.writelines(file_content)

        st, desc = ImportParser.parse({
            "datasets": {
                "d1": {
                    "format": "VDJDB",
                    "params": {
                        "result_path": path,
                        "path": path
                    }
                }
            }
        }, SymbolTable())

        dataset = st.get("d1")
        self.assertTrue(isinstance(dataset, ReceptorDataset))
        self.assertEqual(2, dataset.get_example_count())

        shutil.rmtree(path)

    def test_parse(self):
        path = EnvironmentSettings.root_path + "test/tmp/parser/"

        PathBuilder.build(path + "tmp_input/")
        with open(path + "tmp_input/CD1_clones_TRA.csv", "w") as file:
            writer = csv.DictWriter(file,
                                    delimiter="\t",
                                    fieldnames=["patient", "dilution", "cloneCount", "allVHitsWithScore",
                                                "allJHitsWithScore", "nSeqCDR1", "nSeqCDR2", "nSeqCDR3", "minQualCDR3",
                                                "aaSeqCDR1", "aaSeqCDR2", "aaSeqCDR3", "sampleID"])
            dicts = [{
                "patient": "CD12",
                "dilution": "108'",
                "cloneCount": 3,
                "allVHitsWithScore": "V13-1*00(735)",
                "allJHitsWithScore": "J15*00(243)",
                "nSeqCDR1": "TGTGCAGCAA",
                "nSeqCDR2": "TGTGCAGCAA",
                "nSeqCDR3": "TGTGCAGCAA",
                "minQualCDR3": 10,
                "aaSeqCDR1": "VFAVFA",
                "aaSeqCDR2": "VFAVFA",
                "aaSeqCDR3": "VFAVFA",
                "sampleID": "2"
            }, {
                "patient": "CD12",
                "dilution": "108'",
                "cloneCount": 5,
                "allVHitsWithScore": "V14-1*00(735)",
                "allJHitsWithScore": "J12*00(243)",
                "nSeqCDR1": "CAATGTGA",
                "nSeqCDR2": "CAATGTGA",
                "nSeqCDR3": "CAATGTGA",
                "minQualCDR3": 10,
                "aaSeqCDR1": "CASCAS",
                "aaSeqCDR2": "CASCAS",
                "aaSeqCDR3": "CASCAS",
                "sampleID": "3"
            }]

            writer.writeheader()
            writer.writerows(dicts)

        with open(path + "tmp_input/HC2_clones_TRB.csv", "w") as file:
            writer = csv.DictWriter(file,
                                    delimiter="\t",
                                    fieldnames=["patient", "dilution", "cloneCount", "allVHitsWithScore",
                                                "allJHitsWithScore", "nSeqCDR1", "nSeqCDR2", "nSeqCDR3", "minQualCDR3",
                                                "aaSeqCDR1", "aaSeqCDR2", "aaSeqCDR3", "sampleID"])
            dicts = [{
                "patient": "CD12",
                "dilution": "108'",
                "cloneCount": 3,
                "allVHitsWithScore": "TRAV13-1*00(735)",
                "allJHitsWithScore": "TRAJ15*00(243)",
                "nSeqCDR1": "TGTGCAGCAA",
                "nSeqCDR2": "TGTGCAGCAA",
                "nSeqCDR3": "TGTGCAGCAA",
                "minQualCDR3": 10,
                "aaSeqCDR1": "CAASNQA",
                "aaSeqCDR2": "CAASNQA",
                "aaSeqCDR3": "CAASNQA",
                "sampleID": "1"
            }, {
                "patient": "CD12",
                "dilution": "108'",
                "cloneCount": 6,
                "allVHitsWithScore": "TRAV19-1*00(735)",
                "allJHitsWithScore": "TRAJ12*00(243)",
                "nSeqCDR1": "CAATGTGA",
                "nSeqCDR2": "CAATGTGA",
                "nSeqCDR3": "CAATGTGA",
                "minQualCDR3": 10,
                "aaSeqCDR1": "CAASNTTA",
                "aaSeqCDR2": "CAASNTTA",
                "aaSeqCDR3": "CAASNTTA",
                "sampleID": 1
            }, {
                "patient": "CD12",
                "dilution": "108'",
                "cloneCount": 6,
                "allVHitsWithScore": "TRAV19-1*00(735)",
                "allJHitsWithScore": "TRAJ12*00(243)",
                "nSeqCDR1": "CAATGTGA",
                "nSeqCDR2": "CAATGTGA",
                "nSeqCDR3": "CAATGTGA",
                "minQualCDR3": 10,
                "aaSeqCDR1": "CAASNTTA",
                "aaSeqCDR2": "CAASNTTA",
                "aaSeqCDR3": "CAASNTTA",
                "sampleID": 1
            }]

            writer.writeheader()
            writer.writerows(dicts)

        metadata = pd.DataFrame({"filename": ["HC2_clones_TRB.csv", "CD1_clones_TRA.csv"], "donor": ["HC2", "CD1"], "CD": [False, True]})
        metadata.to_csv(path + "metadata.csv")
        specs = {
            "datasets": {
                "d1": {
                    "format": "MiXCR",
                    "params": {
                        "path": path + "tmp_input/",
                        "result_path": path + "tmp_output/",
                        "metadata_file": path + "metadata.csv",
                        "batch_size": 2,
                    }
                }
            }
        }

        st, desc = ImportParser.parse(specs, SymbolTable())
        self.assertTrue(isinstance(st.get("d1"), RepertoireDataset))
        self.assertEqual(2, len(st.get("d1").get_data()))

        shutil.rmtree(path)
