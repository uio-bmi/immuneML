import shutil
from unittest import TestCase

from source.dsl_parsers.ImportParser import ImportParser
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestImportParser(TestCase):
    def test_parse(self):

        path = EnvironmentSettings.root_path + "test/tmp/importparser/"
        PathBuilder.build(path)

        content = """cloneId	cloneCount	cloneFraction	clonalSequence	clonalSequenceQuality	allVHitsWithScore	allDHitsWithScore	allJHitsWithScore	allCHitsWithScore	allVAlignments	allDAlignments	allJAlignments	allCAlignments	nSeqFR1	minQualFR1	nSeqCDR1	minQualCDR1	nSeqFR2	minQualFR2	nSeqCDR2	minQualCDR2	nSeqFR3	minQualFR3	nSeqCDR3	minQualCDR3	nSeqFR4	minQualFR4	aaSeqFR1	aaSeqCDR1	aaSeqFR2	aaSeqCDR2	aaSeqFR3	aaSeqCDR3	aaSeqFR4	refPoints
0	2	1	TGTGCTGTGAGAGGAGGTGCTGACGGACTCACCTTT	NA	TRAV21		TRAJ45	TRAC	324|335|356|0|11||55.0	NA	29|55|86|10|36||130.0	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	TGTGCTGTGAGAGGAGGTGCTGACGGACTCACCTTT	NA	NA	NA	NA	NA	NA	NA	NA	CAVRGGADGLTF	NA	:::::::::0:-1:11:::::10:-9:36:::
0	2	1	TGTGCTGTGAGCGGCAACATGCTCACCTTT	NA	TRAV21		TRAJ39	TRAC	324|335|356|0|11||55.0	NA	34|52|83|12|30||90.0	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	TGTGCTGTGAGCGGCAACATGCTCACCTTT	NA	NA	NA	NA	NA	NA	NA	NA	CAVSGNMLTF	NA	:::::::::0:-1:11:::::12:-14:30:::
0	3	1	TGTGCTGTGAGACATTCAGATGGCCAGAAGCTGCTCTTT	NA	TRAV21		TRAJ16	TRAC	324|335|356|0|11||55.0	NA	24|49|80|14|39||125.0	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	TGTGCTGTGAGACATTCAGATGGCCAGAAGCTGCTCTTT	NA	NA	NA	NA	NA	NA	NA	NA	CAVRHSDGQKLLF	NA	:::::::::0:-1:11:::::14:-4:39:::
0	2	1	TGTGCTGTGAGAATGTCAGGAGGAAGCTACATACCTACATTT	NA	TRAV21		TRAJ6	TRAC	324|335|356|0|11||55.0	NA	24|51|82|15|42||135.0	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	TGTGCTGTGAGAATGTCAGGAGGAAGCTACATACCTACATTT	NA	NA	NA	NA	NA	NA	NA	NA	CAVRMSGGSYIPTF	NA	:::::::::0:-1:11:::::15:-4:42:::
0	1	1	TGTGCTGTGAGAGCAGACAGCTGGGGGAAATTGCAGTTT	NA	TRAV21		TRAJ24	TRAC	324|335|356|0|11||55.0	NA	28|52|83|15|39||120.0	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	NA	TGTGCTGTGAGAGCAGACAGCTGGGGGAAATTGCAGTTT	NA	NA	NA	NA	NA	NA	NA	NA	CAVRADSWGKLQF	NA	:::::::::0:-1:11:::::15:-8:39:::"""

        with open(path + "CD1_TRA.csv", "w") as file:
            file.writelines(content)
        with open(path + "HC1_TRA.csv", "w") as file:
            file.writelines(content)

        workflow_specification = {
            "result_path": path,
            "dataset_import": {
                "path": path,
                "format": "MiXCR",
                "params": {
                    "sequence_type": "CDR3",
                    "additional_columns": ["sampleID"],
                    "extension": "csv",
                    "batch_size": 2,
                    "custom_params": [{
                        "name": "CD",
                        "location": "filepath_binary",
                        "alternative": "HC"
                    }]
                }
            }
        }

        dataset = ImportParser.parse(workflow_specification)

        self.assertEqual(2, dataset.get_repertoire_count())
        self.assertEqual(2, len(dataset.params["CD"]))
        for rep in dataset.get_data():
            self.assertEqual(5, len(rep.sequences))
            self.assertTrue("CD" in rep.metadata.custom_params)

        shutil.rmtree(path)
