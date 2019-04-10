import csv
import shutil
from unittest import TestCase

from source.data_model.dataset.Dataset import Dataset
from source.dsl.ImportParser import ImportParser
from source.dsl.SymbolTable import SymbolTable
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImportParser(TestCase):
    def test_parse(self):
        path = EnvironmentSettings.root_path + "test/tmp/parser/"

        PathBuilder.build(path + "tmp_input/")
        with open(path + "tmp_input/CD1_TRA.csv", "w") as file:
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

        with open(path + "tmp_input/HC2_TRB.csv", "w") as file:
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
            }]

            writer.writeheader()
            writer.writerows(dicts)

        specs = {
            "dataset_import": {
                "d1": {
                    "path": path + "tmp_input/",
                    "format": "MiXCR",
                    "params": {
                        "additional_columns": ["minQualCDR3"],
                        "sequence_type": "CDR2+CDR3",
                        "result_path": path + "tmp_output/",
                        "batch_size": 2,
                        "extension": "csv",
                        "custom_params": [{
                            "name": "CD",
                            "location": "filepath_binary",
                            "alternative": "HC"
                        }]
                    },
                    "result_path": path + "tmp_output/",
                    "assessment_type": "CV",
                    "CV_folds_number": "LOOCV"
                }
            }
        }

        st, desc = ImportParser.parse(specs, SymbolTable())
        self.assertTrue(isinstance(st.get("d1")["dataset"], Dataset))
        shutil.rmtree(path)
