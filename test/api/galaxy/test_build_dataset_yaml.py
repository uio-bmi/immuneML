import os
import shutil
import unittest

import pandas as pd
import yaml

from immuneML.api.galaxy.build_dataset_yaml import build_metadata_column_mapping
from immuneML.api.galaxy.build_dataset_yaml import main as yamlbuilder_main
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.dsl.ImmuneMLParser import ImmuneMLParser
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


class MyTestCase(unittest.TestCase):
    def create_dummy_dataset(self, path, write_metadata):
        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
3050	TRB	CASSPPRVYSNGAGLAGVGWRNEQFF	TRBV5-4*01	TRBJ2-1*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CASSPPRVYSNGAGLAGVGWRNEQFF", "cdr3_old": "CASSPPRVYSNGAGLAGVGWRNEQFF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-1*01", "jStart": 21, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-4*01"}	0
15760	TRB	CASSWTWDAATLWGQGALGGANVLTF	TRBV5-5*01	TRBJ2-6*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CASSWTWDAATLWGQGALGGANVLTF", "cdr3_old": "CASSWTWDAATLWGQGALGGANVLTF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRBJ2-6*01", "jStart": 19, "vCanonical": true, "vEnd": 4, "vFixType": "NoFixNeeded", "vId": "TRBV5-5*01"}	0
3050	TRA	CAAIYESRGSTLGRLYF	TRAV13-1*01	TRAJ18*01	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/11684", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "1", "tissue": ""}	{"cdr3": "CAAIYESRGSTLGRLYF", "cdr3_old": "CAAIYESRGSTLGRLYF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ18*01", "jStart": 7, "oldVEnd": -1, "oldVFixType": "FailedBadSegment", "oldVId": null, "vCanonical": true, "vEnd": 3, "vFixType": "ChangeSegment", "vId": "TRAV13-1*01"}	0
15760	TRA	CALRLNNQGGKLIF	TRAV9-2*01	TRAJ23*01	HomoSapiens	HLA-A*03:01	B2M	MHCI	KLGGALQAK	IE1	CMV	https://www.10xgenomics.com/resources/application-notes/a-new-way-of-exploring-immunity-linking-highly-multiplexed-antigen-recognition-to-immune-repertoire-and-phenotype/#	{"frequency": "1/25584", "identification": "dextramer-sort", "sequencing": "rna-seq", "singlecell": "yes", "verification": ""}	{"cell.subset": "", "clone.id": "", "donor.MHC": "", "donor.MHC.method": "", "epitope.id": "", "replica.id": "", "samples.found": 1, "structure.id": "", "studies.found": 1, "study.id": "", "subject.cohort": "", "subject.id": "3", "tissue": ""}	{"cdr3": "CALRLNNQGGKLIF", "cdr3_old": "CALRLNNQGGKLIF", "fixNeeded": false, "good": true, "jCanonical": true, "jFixType": "NoFixNeeded", "jId": "TRAJ23*01", "jStart": 6, "vCanonical": true, "vEnd": 3, "vFixType": "NoFixNeeded", "vId": "TRAV9-2*01"}	0
"""
        number_of_repertoires = 2

        for i in range(number_of_repertoires):
            with open(path / "receptors_{}.tsv".format(i + 1), "w") as file:
                file.writelines(file_content)

        if write_metadata:
            metadata = {
                "filename": ["receptors_{}.tsv".format(i + 1) for i in range(number_of_repertoires)],
                "label1": [i % 2 for i in range(number_of_repertoires)]
            }

            pd.DataFrame(metadata).to_csv(path / "metadata.csv", index=False)

    def test_build_metadata_column_mapping(self):
        self.assertDictEqual({}, build_metadata_column_mapping("''"))
        self.assertDictEqual({"a": "a"}, build_metadata_column_mapping("'a',"))
        self.assertDictEqual({"a": "a", "b": "b"}, build_metadata_column_mapping("'b',a"))

    def test_sequencedataset(self):
        path = EnvironmentSettings.tmp_test_path / "sequencedataset_yaml/"
        PathBuilder.remove_old_and_build(path)
        self.create_dummy_dataset(path, write_metadata=False)

        old_wd = os.getcwd()

        try:
            os.chdir(path)

            yamlbuilder_main(["-r", "VDJdb", "-o", str(path), "-f", "sequence.yaml", "-p", "False", "-a", "a,b", "-i", "False"])

            with open(path / "sequence.yaml", "r") as file:
                loaded_receptor = yaml.load(file, Loader=yaml.FullLoader)

                self.assertDictEqual(loaded_receptor["definitions"]["datasets"], {"dataset": {"format": "VDJdb", "params":
                    {"path": "./", "is_repertoire": False, "paired": False, "metadata_column_mapping": {"a": "a", "b": "b"},
                     "region_type": RegionType.IMGT_CDR3.name, "result_path": "./"}}})

            ImmuneMLParser.parse_yaml_file(path / "sequence.yaml")

        finally:
            os.chdir(old_wd)

        shutil.rmtree(path)

    def test_receptordataset(self):

        path = EnvironmentSettings.tmp_test_path / "receptordataset_yaml/"
        PathBuilder.remove_old_and_build(path)
        self.create_dummy_dataset(path, write_metadata=False)

        old_wd = os.getcwd()

        try:
            os.chdir(path)

            yamlbuilder_main(["-r", "VDJdb", "-o", str(path), "-f", "receptor.yaml", "-p", "True", "-c", "TRA_TRB", "-a", "'c'", "-i", "False"])

            with open(path / "receptor.yaml", "r") as file:
                loaded_receptor = yaml.load(file, Loader=yaml.FullLoader)

                self.assertDictEqual(loaded_receptor["definitions"]["datasets"], {"dataset": {"format": "VDJdb", "params":
                    {"path": "./", "is_repertoire": False, "paired": True, "receptor_chains": "TRA_TRB", "metadata_column_mapping": {"c": "c"},
                     "region_type": RegionType.IMGT_CDR3.name, "result_path": "./"}}})

            ImmuneMLParser.parse_yaml_file(path / "receptor.yaml")

        finally:
            os.chdir(old_wd)

        shutil.rmtree(path)

    def test_repertoiredataset(self):

        path = EnvironmentSettings.tmp_test_path / "repertoiredataset_yaml"
        PathBuilder.remove_old_and_build(path)
        self.create_dummy_dataset(path, write_metadata=True)

        old_wd = os.getcwd()

        try:
            os.chdir(path)

            yamlbuilder_main(["-r", "VDJdb", "-o", str(path), "-f", "repertoire.yaml", "-m", "metadata.csv", "-i", "True"])

            with open(path / "repertoire.yaml", "r") as file:
                loaded_receptor = yaml.load(file, Loader=yaml.FullLoader)

                self.assertDictEqual(loaded_receptor["definitions"]["datasets"], {"dataset": {"format": "VDJdb", "params":
                    {"path": "./", "metadata_file": "metadata.csv", "is_repertoire": True, "region_type": RegionType.IMGT_CDR3.name,
                     "result_path": "./"}}})

            ImmuneMLParser.parse_yaml_file(path / "repertoire.yaml")

        finally:
            os.chdir(old_wd)

        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
