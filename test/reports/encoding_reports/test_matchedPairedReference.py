import os
import shutil
import unittest

import pandas as pd

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedReceptorsRepertoireEncoder import MatchedReceptorsRepertoireEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.reports.encoding_reports.MatchedPairedReference import MatchedPairedReference
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedPairedReference(unittest.TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_encoded_dummy_data(self, path):
        # Setting up dummy data
        labels = {"donor": ["donor1", "donor1", "donor2", "donor2", "donor3"],
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
                                                        donors=labels["donor"])

        dataset = RepertoireDataset(repertoires=repertoires)

        label_config = LabelConfiguration()
        label_config.add_label("donor", labels["donor"])
        label_config.add_label("label", labels["label"])

        # clonotype 100 with TRA=AAAA, TRB = SSSS; clonotype 200 with TRA=CCCC, TRB = TTTT, adn 300&400 with both=NONO
        file_content = """Cell type	Clonotype ID	Chain: TRA (1)	TRA - V gene (1)	TRA - D gene (1)	TRA - J gene (1)	Chain: TRA (2)	TRA - V gene (2)	TRA - D gene (2)	TRA - J gene (2)	Chain: TRB (1)	TRB - V gene (1)	TRB - D gene (1)	TRB - J gene (1)	Chain: TRB (2)	TRB - V gene (2)	TRB - D gene (2)	TRB - J gene (2)	Cells pr. clonotype	Clonotype (Id)	Clonotype (Name)
TCR_AB	100	AAAA	TRAV1		TRAJ1	null	null	null	null	SSSS	TRBV1		TRBJ1	null	null	null	null	1	1941533	3ca0cd7f-02fd-40bb-b295-7cd5d419e474(101, 102, 103, 104, 105, 108, 109, 127, 128, 130, 131, 132, 133, 134, 174)Size:1
TCR_AB	200	CCCC	TRAV1		TRAJ1	null	null	null	null	TTTT	TRBV1		TRBJ1	null	null	null	null	1	1941532	1df22bbc-8113-46b9-8913-da95fcf9a568(101, 102, 103, 104, 105, 108, 109, 127, 128, 130, 131, 132, 133, 134, 174)Size:1
TCR_AB	300	NONO	TRAV1		TRAJ1	null	null	null	null	NONO	TRBV1		TRBJ1	null	null	null	null	1	1941532	1df22bbc-8113-46b9-8913-da95fcf9a568(101, 102, 103, 104, 105, 108, 109, 127, 128, 130, 131, 132, 133, 134, 174)Size:1
TCR_AB	400	NONO	TRAV1		TRAJ1	null	null	null	null	NONO	TRBV1		TRBJ1	null	null	null	null	1	1941532	1df22bbc-8113-46b9-8913-da95fcf9a568(101, 102, 103, 104, 105, 108, 109, 127, 128, 130, 131, 132, 133, 134, 174)Size:1
        """

        with open(path + "refs.tsv", "w") as file:
            file.writelines(file_content)

        reference_receptors = {"path": path + "refs.tsv", "format": "IRIS"}

        encoder = MatchedReceptorsRepertoireEncoder.build_object(dataset, **{
            "reference_receptors": reference_receptors,
            "max_edit_distances": 0
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_configuration=label_config,
            filename="dataset.csv"
        ))

        return encoded

    def test_generate(self):
        path = EnvironmentSettings.root_path + "test/tmp/matched_paired_reference_report/"

        encoded_data = self.create_encoded_dummy_data(path + "input_data/")

        report = MatchedPairedReference(dataset=encoded_data, result_path=path + "report_results/")

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
