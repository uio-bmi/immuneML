import csv
import shutil
from unittest import TestCase

import pandas as pd

from source.IO.dataset_import.MiXCRImport import MiXCRImport
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestMiXCRLoader(TestCase):
    def test_load(self):
        path = EnvironmentSettings.root_path + "test/tmp/mixcr/"

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
                "allVHitsWithScore": "TRAV29DV5*00(553.8)",
                "allJHitsWithScore": "TRAJ15*00(243)",
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
                "allVHitsWithScore": "TRAV14-1*00(735)",
                "allJHitsWithScore": "TRAJ12*00(243)",
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

        output_path = path + "tmp_output/"

        dataset = MiXCRImport.import_dataset({
            "path": path + "tmp_input/",
            "region_type": "CDR3",
            "result_path": output_path,
            "batch_size": 2, "separator": "\t",
            "region_definition": "IMGT",
            "metadata_file": path + "metadata.csv",
            "column_mapping": {
                "cloneCount": "counts",
                "allVHitsWithScore": "v_genes",
                "allJHitsWithScore": "j_genes"
            }
        }, "mixcr_dataset")

        self.assertEqual(2, dataset.get_example_count())

        for index, repertoire in enumerate(dataset.get_data()):
            self.assertTrue(all(sequence.metadata.chain == Chain.ALPHA for sequence in repertoire.sequences))
            if index == 1:
                self.assertTrue(repertoire.sequences[0].amino_acid_sequence == "FAVF")
                self.assertTrue(repertoire.sequences[0].metadata.v_gene == "TRAV29/DV5")
                self.assertTrue(repertoire.sequences[1].metadata.v_gene == "TRAV14-1")
                self.assertTrue(repertoire.metadata["CD"])
            elif index == 0:
                self.assertEqual(5, len(repertoire.sequences))
                self.assertEqual("GCAG", repertoire.sequences[0].nucleotide_sequence)
                self.assertEqual(6, repertoire.sequences[1].metadata.count)
                self.assertFalse(repertoire.metadata["CD"])

        shutil.rmtree(output_path)

        dataset = MiXCRImport.import_dataset({
            "path": path + "tmp_input/",
            "region_type": "CDR3",
            "result_path": path + "tmp_output/",
            "batch_size": 2, "separator": "\t",
            "metadata_file": path + "metadata.csv",
            "column_mapping": {
                "cloneCount": "counts",
                "allVHitsWithScore": "v_genes",
                "allJHitsWithScore": "j_genes"
            }
        }, "mixcr_dataset")

        for index, repertoire in enumerate(dataset.get_data()):
            self.assertTrue(all(sequence.metadata.chain == Chain.ALPHA for sequence in repertoire.sequences))
            if index == 1:
                self.assertTrue(repertoire.sequences[0].amino_acid_sequence == "VFAVFA")
                self.assertTrue(repertoire.sequences[0].metadata.v_gene == "TRAV29/DV5")
                self.assertTrue(repertoire.sequences[1].metadata.v_gene == "TRAV14-1")
                self.assertTrue(repertoire.metadata["CD"])
            elif index == 0:
                self.assertEqual(5, len(repertoire.sequences))
                self.assertEqual("TGTGCAGCAA", repertoire.sequences[0].nucleotide_sequence)
                self.assertEqual(6, repertoire.sequences[1].metadata.count)
                self.assertFalse(repertoire.metadata["CD"])

        shutil.rmtree(path)
