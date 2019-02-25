import csv
import shutil
from unittest import TestCase

from source.IO.MiXCRLoader import MiXCRLoader
from source.util.PathBuilder import PathBuilder


class TestMiXCRLoader(TestCase):
    def test_load(self):

        PathBuilder.build("./tmp_input/")
        with open("./tmp_input/CD1_TRA.csv", "w") as file:
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

        with open("./tmp_input/HC2_TRB.csv", "w") as file:
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

        dataset = MiXCRLoader.load("./tmp_input/", {
            "additional_columns": ["minQualCDR3"],
            "sequence_type": "CDR1+CDR2+CDR3",
            "result_path": "./tmp_output/",
            "batch_size": 2,
            "extension": "csv",
            "custom_params": [{
                "name": "CD",
                "location": "filepath_binary",
                "alternative": "HC"
            }]
        })

        self.assertEqual(2, dataset.get_repertoire_count())

        for index, repertoire in enumerate(dataset.get_data()):
            if index == 0:
                self.assertTrue(repertoire.sequences[0].amino_acid_sequence == "VFAVFAVFAVFAVFAVFA")
                self.assertTrue(repertoire.sequences[1].metadata.v_gene == "V14-1")
                self.assertTrue(repertoire.metadata.sample.custom_params["CD"])
            else:
                self.assertEqual("TGTGCAGCAATGTGCAGCAATGTGCAGCAA", repertoire.sequences[0].nucleotide_sequence)
                self.assertEqual(6, repertoire.sequences[1].metadata.count)
                self.assertFalse(repertoire.metadata.sample.custom_params["CD"])

        shutil.rmtree("./tmp_input/")
        shutil.rmtree("./tmp_output/")
