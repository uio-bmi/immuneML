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
from source.util.RepertoireBuilder import RepertoireBuilder


class TestPairedReceptorMatches(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_encoded_dummy_data(self, path):
        # Setting up dummy data
        labels = {"subject_id": ["subject_1", "subject_1", "subject_2", "subject_2", "subject_3"],
                  "label": ["yes", "yes", "no", "no", "no"]}

        metadata_alpha = {"v_gene": "V1", "j_gene": "J1", "chain": Chain.ALPHA.value}
        metadata_beta = {"v_gene": "V1", "j_gene": "J1", "chain": Chain.BETA.value}

        repertoires, metadata = RepertoireBuilder.build(sequences=[["AAAA"],
                                                                   ["SSSS"],
                                                                   ["AAAA", "CCCC"],
                                                                   ["SSSS", "TTTT"],
                                                                   ["AAAA", "CCCC", "SSSS", "TTTT"]],
                                                        path=path, labels=labels,
                                                        seq_metadata=[[{**metadata_alpha, "count": 10}],
                                                                      [{**metadata_beta, "count": 10}],
                                                                      [{**metadata_alpha, "count": 5},
                                                                       {**metadata_alpha, "count": 5}],
                                                                      [{**metadata_beta, "count": 5},
                                                                       {**metadata_beta, "count": 5}],
                                                                      [{**metadata_alpha, "count": 1},
                                                                       {**metadata_alpha, "count": 2},
                                                                       {**metadata_beta, "count": 1},
                                                                       {**metadata_beta, "count": 2}]],
                                                        subject_ids=labels["subject_id"])

        dataset = RepertoireDataset(repertoires=repertoires)

        feature_annotations_content = """receptor_id	clonotype_id	chain	dual_chain_id	sequence	v_gene	j_gene
100-A0-B0	100	alpha	1	AAAA	V1	J1
100-A0-B0	100	beta	1	SSSS	V1	J1
200-A0-B0	200	alpha	1	CCCC	V1	J1
200-A0-B0	200	beta	1	TTTT	V1	J1
300-A0-B0	300	alpha	1	NONO	V1	J1
300-A0-B0	300	beta	1	NONO	V1	J1
400-A0-B0	400	alpha	1	NONO	V1	J1
400-A0-B0	400	beta	1	NONO	V1	J1"""

        with open(path + "feature_annotations.tsv", "w") as file:
            file.writelines(feature_annotations_content)

        feature_annotations = pd.read_csv(path+"feature_annotations.tsv", sep="\t")

        encoded = RepertoireDataset(repertoires=repertoires,
                                    encoded_data=EncodedData(
                                        encoding="MatchedReceptorsEncoder",
                                        example_ids=['donor1', 'donor2', 'donor3'],
                                        examples=np.array([[10, 10, 0, 0, 0, 0, 0, 0],
                                                           [5, 5, 5, 5, 0, 0, 0, 0],
                                                           [1, 1, 2, 2, 0, 0, 0, 0]]),
                                        labels={"label": ["yes", "no", "no"]},
                                        feature_names=['100-A0-B0.alpha', '100-A0-B0.beta', '200-A0-B0.alpha', '200-A0-B0.beta', '300-A0-B0.alpha', '300-A0-B0.beta', '400-A0-B0.alpha', '400-A0-B0.beta'],
                                        feature_annotations=feature_annotations
                                    ))

        return encoded

    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/matched_paired_reference_report/"

        encoded_data = self.create_encoded_dummy_data(path + "input_data/")

        report = PairedReceptorMatches(dataset=encoded_data, result_path=path + "report_results/")

        report.check_prerequisites()
        report.generate()

        self.assertTrue(os.path.isfile(path + "report_results/complete_match_count_table.csv"))
        self.assertTrue(os.path.isfile(path + "report_results/repertoire_sizes.csv"))

        self.assertTrue(os.path.isdir(path + "report_results/paired_matches"))
        self.assertTrue(os.path.isdir(path + "report_results/receptor_info"))

        self.assertTrue(os.path.isfile(path + "report_results/receptor_info/all_chains.csv"))
        self.assertTrue(os.path.isfile(path + "report_results/receptor_info/all_receptors.csv"))
        self.assertTrue(os.path.isfile(path + "report_results/receptor_info/unique_alpha_chains.csv"))
        self.assertTrue(os.path.isfile(path + "report_results/receptor_info/unique_beta_chains.csv"))
        self.assertTrue(os.path.isfile(path + "report_results/receptor_info/unique_receptors.csv"))

        chains = pd.read_csv(path + "report_results/receptor_info/all_chains.csv")
        receptors = pd.read_csv(path + "report_results/receptor_info/all_receptors.csv")
        unique_alpha_chains = pd.read_csv(path + "report_results/receptor_info/unique_alpha_chains.csv")
        unique_beta_chains = pd.read_csv(path + "report_results/receptor_info/unique_beta_chains.csv")
        unique_receptors = pd.read_csv(path + "report_results/receptor_info/unique_receptors.csv")

        self.assertListEqual(list(chains["clonotype_id"]), [100, 100, 200, 200, 300, 300, 400, 400])
        self.assertListEqual(list(receptors["clonotype_id"]), [100, 200, 300, 400])
        self.assertListEqual(list(unique_alpha_chains["clonotype_id"]), [100, 200, 300])
        self.assertListEqual(list(unique_beta_chains["clonotype_id"]), [100, 200, 300])
        self.assertListEqual(list(unique_receptors["clonotype_id"]), [100, 200, 300])

        self.assertListEqual(list(unique_receptors["sequence_alpha"]), list(unique_alpha_chains["sequence"]))
        self.assertListEqual(list(unique_receptors["sequence_beta"]), list(unique_beta_chains["sequence"]))

        shutil.rmtree(path)