import glob
import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.encoding_reports.MatchingSequenceDetails import MatchingSequenceDetails
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchingSequenceDetails(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/encrepmatchingseq/"
        repertoires = RepertoireBuilder.build([["AAA", "CCC"], ["AAC", "ASDA"], ["CCF", "ATC"]], path,
                                              {"default": [1, 0, 0]})[0]

        file_content = """complex.id	Gene	CDR3	V	J	Species	MHC A	MHC B	MHC class	Epitope	Epitope gene	Epitope species	Reference	Method	Meta	CDR3fix	Score
        100a	TRA	AAA	TRAV1	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	                
        100a	TRA	CCF	TRAV1	TRAJ1	HomoSapiens	HLA-A*11:01	B2M	MHCI	AVFDRKSDAK	EBNA4	EBV	                
        """

        with open(path + "refs.tsv", "w") as file:
            file.writelines(file_content)

        references = {"path": path + "refs.tsv", "format": "VDJdb"}

        dataset = RepertoireDataset(repertoires=repertoires,
                                    params={"default": [0, 1]},
                                    encoded_data=EncodedData(
                                                      examples=np.array([[2], [1], [1]]),
                                                      labels={"default": [1, 0, 0]},
                                                      feature_names=["percentage"]
                                                  ))

        report = MatchingSequenceDetails.build_object(**{
            "dataset": dataset,
            "result_path": path + "result/",
            "reference_sequences": references,
            "max_edit_distance": 1
        })

        report.generate()

        self.assertTrue(os.path.isfile(path + "result/matching_sequence_overview.tsv"))
        self.assertEqual(4, len([name for name in glob.glob(path + "result/*.tsv") if os.path.isfile(name)]))

        df = pd.read_csv(path + "result/matching_sequence_overview.tsv", sep="\t")
        self.assertTrue(all([key in df.keys() for key in ["repertoire_identifier", "percentage", "repertoire_size", "max_levenshtein_distance"]]))
        self.assertEqual(3, df.shape[0])

        shutil.rmtree(path)
