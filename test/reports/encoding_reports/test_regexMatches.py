import os
import shutil
import unittest

import pandas as pd
import numpy as np

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.encoded_data.EncodedData import EncodedData
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.reports.encoding_reports.PairedReceptorMatches import PairedReceptorMatches
from source.reports.encoding_reports.RegexMatches import RegexMatches
from source.util.RepertoireBuilder import RepertoireBuilder


class TestPairedReceptorMatches(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_encoded_dummy_data(self, path):
        # Setting up dummy data
        labels = {"subject_id": ["subject_1", "subject_2", "subject_3"],
                  "label": ["yes", "no", "no"]}

        metadata_alpha = {"v_gene": "V1", "j_gene": "J1", "chain": Chain.ALPHA.value}
        metadata_beta = {"v_gene": "V1", "j_gene": "J1", "chain": Chain.BETA.value}

        repertoires, metadata = RepertoireBuilder.build(sequences=[["XXAGQXGSSNTGKLIXX", "XXAGQXGSSNTGKLIYY", "XXSAGQGETQYXX"],
                                                                   ["ASSXRXX"],
                                                                   ["XXIXXNDYKLSXX", "CCCC", "SSSS", "TTTT"]],
                                                        path=path, labels=labels,
                                                        seq_metadata=[[{**metadata_alpha, "count": 10, "v_gene": "TRAV35"},
                                                                       {**metadata_alpha, "count": 10},
                                                                       {**metadata_beta, "count": 10, "v_gene": "TRBV29-1"}],
                                                                      [{**metadata_beta, "count": 10, "v_gene": "TRBV7-3"}],
                                                                      [{**metadata_alpha, "count": 5, "v_gene": "TRAV26-2"},
                                                                       {**metadata_alpha, "count": 2},
                                                                       {**metadata_beta, "count": 1},
                                                                       {**metadata_beta, "count": 2}]],
                                                        subject_ids=labels["subject_id"])

        dataset = RepertoireDataset(repertoires=repertoires)

        feature_annotations_content = """receptor_id,chain_id,regex,v_gene
1,1_TRA,AGQ.GSSNTGKLI,TRAV35
1,1_TRB,S[APGFTVML]GQGETQY,TRBV29-1
2,2_TRB,ASS.R.*,TRBV7-3
3,3_TRA,I..NDYKLS,TRAV26-1
4,4_TRA,I..NDYKLS,TRAV26-2"""

        with open(path + "feature_annotations.tsv", "w") as file:
            file.writelines(feature_annotations_content)

        feature_annotations = pd.read_csv(path+"feature_annotations.tsv", sep=",")

        encoded = RepertoireDataset(repertoires=repertoires,
                                    encoded_data=EncodedData(
                                        encoding="MatchedRegexEncoder",
                                        example_ids=["subject_1", "subject_2", "subject_3"],
                                        examples=np.array([[10, 10, 0, 0, 0],
                                                           [0, 0, 10, 0, 0],
                                                           [0, 0, 0, 0, 5]]),
                                        labels={"label": ["yes", "no", "no"]},
                                        feature_names=["1_TRA", "1_TRB", "2_TRB", "3_TRA", "4_TRA"],
                                        feature_annotations=feature_annotations
                                    ))

        return encoded

    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/regex_matches_report/"

        encoded_data = self.create_encoded_dummy_data(path + "input_data/")

        report = RegexMatches(dataset=encoded_data, result_path=path + "report_results/")

        report.check_prerequisites()
        report.generate()

        self.assertTrue(os.path.isfile(path + "report_results/complete_match_count_table.csv"))
        # self.assertTrue(os.path.isfile(path + "report_results/repertoire_sizes.csv"))

        self.assertTrue(os.path.isdir(path + "report_results/paired_matches"))
        self.assertTrue(os.path.isfile(path + "report_results/paired_matches/example_subject_1_label_yes.csv"))
        self.assertTrue(os.path.isfile(path + "report_results/paired_matches/example_subject_2_label_no.csv"))
        self.assertTrue(os.path.isfile(path + "report_results/paired_matches/example_subject_3_label_no.csv"))


        shutil.rmtree(path)