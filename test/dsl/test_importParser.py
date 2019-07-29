import csv
import shutil
from unittest import TestCase

from helpers.metadata_converter import convert_metadata
from source.data_model.dataset.Dataset import Dataset
from source.dsl.SymbolTable import SymbolTable
from source.dsl.import_parsers.ImportParser import ImportParser
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImportParser(TestCase):
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

        convert_metadata(path + "tmp_input/", path + "metadata.csv", "CD", "HC")

        specs = {
            "datasets": {
                "d1": {
                    "path": path + "tmp_input/",
                    "format": "MiXCR",
                    "params": {
                        "sequence_type": "CDR2+CDR3",
                        "result_path": path + "tmp_output/",
                        "extension": "csv",
                        "metadata_file": path + "metadata.csv",
                        "batch_size": 2,
                        "additional_columns": []
                    },
                    "preprocessing": {
                        "filter_out_short_reps": {
                            "params": {"lower_limit": 3},
                            "type": "ClonotypeCountFilter"
                        }
                    }
                }
            }
        }

        st, desc = ImportParser.parse(specs, SymbolTable())
        self.assertTrue(isinstance(st.get("d1"), Dataset))
        self.assertEqual(1, len(st.get("d1").get_filenames()))

        specs = {
            "datasets": {
                "d1": {
                    "path": path + "tmp_input/",
                    "format": "MiXCR",
                    "params": {
                        "sequence_type": "CDR2+CDR3",
                        "result_path": path + "tmp_output/",
                        "extension": "csv",
                        "metadata_file": path + "metadata.csv",
                        "additional_columns": [],
                        "batch_size": 2
                    },
                    "preprocessing": {
                        "filter_out_short_reps": {
                            "type": "ClonotypeCountFilter"
                        }
                    }
                }
            }
        }

        st, desc = ImportParser.parse(specs, SymbolTable())
        self.assertTrue(isinstance(st.get("d1"), Dataset))
        self.assertEqual(0, len(st.get("d1").get_filenames()))

        self.assertEqual(100, desc["d1"]["preprocessing"]["filter_out_short_reps"]["params"]["lower_limit"])

        specs = {
            "datasets": {
                "d1": {
                    "path": path + "tmp_input/",
                    "format": "MiXCR",
                    "params": {
                        "sequence_type": "CDR2+CDR3",
                        "result_path": path + "tmp_output/",
                        "extension": "csv",
                        "metadata_file": path + "metadata.csv",
                        "additional_columns": [],
                        "batch_size": 2
                    },
                    "preprocessing": {
                        "filter_cd": {
                            "type": "MetadataFilter",
                            "params": {
                                "criteria": {
                                    "type": "in",
                                    "value": {
                                        "type": "column",
                                        "name": "donor"
                                    },
                                    "allowed_values": ["CD1"]
                                }
                            }
                        }
                    }
                }
            }
        }

        st, desc = ImportParser.parse(specs, SymbolTable())
        self.assertTrue(isinstance(st.get("d1"), Dataset))
        self.assertEqual(1, len(st.get("d1").get_filenames()))

        shutil.rmtree(path)
