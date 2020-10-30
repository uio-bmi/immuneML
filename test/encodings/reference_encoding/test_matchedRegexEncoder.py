import os
import shutil
from unittest import TestCase

from source.caching.CacheType import CacheType
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedRegexEncoder import MatchedRegexEncoder
from source.environment.Constants import Constants
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedReceptorsEncoder(TestCase):

    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def create_dummy_data(self, path):

        # Setting up dummy data
        labels = {"subject_id": ["subject_1", "subject_2", "subject_3"],
                  "label": ["yes", "no", "no"]}

        metadata_alpha = {"v_gene": "V1", "j_gene": "J1", "chain": Chain.LIGHT.value}
        metadata_beta = {"v_gene": "V1", "j_gene": "J1", "chain": Chain.HEAVY.value}

        repertoires, metadata = RepertoireBuilder.build(sequences=[["XXAGQXGSSNTGKLIXX", "XXAGQXGSSNTGKLIYY", "XXSAGQGETQYXX"],
                                                                   ["ASSXRXX"],
                                                                   ["XXIXXNDYKLSXX", "CCCC", "SSSS", "TTTT"]],
                                                        path=path, labels=labels,
                                                        seq_metadata=[[{**metadata_alpha, "count": 10, "v_gene": "IGLV35"},
                                                                       {**metadata_alpha, "count": 10},
                                                                       {**metadata_beta, "count": 10, "v_gene": "IGHV29-1"}],
                                                                      [{**metadata_beta, "count": 10, "v_gene": "IGHV7-3"}],
                                                                      [{**metadata_alpha, "count": 5, "v_gene": "IGLV26-2"},
                                                                       {**metadata_alpha, "count": 2},
                                                                       {**metadata_beta, "count": 1},
                                                                       {**metadata_beta, "count": 2}]],
                                                        subject_ids=labels["subject_id"])

        dataset = RepertoireDataset(repertoires=repertoires)

        label_config = LabelConfiguration()
        label_config.add_label("subject_id", labels["subject_id"])
        label_config.add_label("label", labels["label"])

        file_content = """id	IGLV	IGHV	IGL_regex	IGH_regex
1	IGLV35	IGHV29-1	AGQ.GSSNTGKLI	S[APGFTVML]GQGETQY
2		IGHV7-3		ASS.R.*
3	IGLV26-1		I..NDYKLS	
4	IGLV26-2		I..NDYKLS	
"""

        filepath = path + "reference_motifs.tsv"
        with open(filepath, "w") as file:
            file.writelines(file_content)

        return dataset, label_config, filepath, labels

    def test_encode_no_v(self):
        path = EnvironmentSettings.root_path + "test/tmp/regex_matches_encoder/"

        dataset, label_config, motif_filepath, labels = self.create_dummy_data(path)

        encoder = MatchedRegexEncoder.build_object(dataset, **{
            "motif_filepath": motif_filepath,
            "match_v_genes": False,
            "sum_counts": True
        })


        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config,
            filename="dataset.csv"
        ))

        expected_outcome = [[20, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 5]]

        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertListEqual(["1_IGL", "1_IGH", "2_IGH", "3_IGL"], encoded.encoded_data.feature_names)
        self.assertListEqual(["subject_1", "subject_2", "subject_3"], encoded.encoded_data.example_ids)

        shutil.rmtree(path)


    def test_encode_no_v_no_count(self):
        path = EnvironmentSettings.root_path + "test/tmp/regex_matches_encoder/"

        dataset, label_config, motif_filepath, labels = self.create_dummy_data(path)

        encoder = MatchedRegexEncoder.build_object(dataset, **{
            "motif_filepath": motif_filepath,
            "match_v_genes": False,
            "sum_counts": False
        })


        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config,
            filename="dataset.csv"
        ))

        expected_outcome = [[2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertListEqual(["1_IGL", "1_IGH", "2_IGH", "3_IGL"], encoded.encoded_data.feature_names)
        self.assertListEqual(["subject_1", "subject_2", "subject_3"], encoded.encoded_data.example_ids)

        shutil.rmtree(path)


    def test_encode_with_v(self):
        path = EnvironmentSettings.root_path + "test/tmp/regex_matches_encoder/"

        dataset, label_config, motif_filepath, labels = self.create_dummy_data(path)

        encoder = MatchedRegexEncoder.build_object(dataset, **{
            "motif_filepath": motif_filepath,
            "match_v_genes": True,
            "sum_counts": True
        })


        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_config=label_config,
            filename="dataset.csv"
        ))

        expected_outcome = [[10, 10, 0, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 5]]

        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertListEqual(["1_IGL", "1_IGH", "2_IGH", "3_IGL", "4_IGL"], encoded.encoded_data.feature_names)
        self.assertListEqual(["subject_1", "subject_2", "subject_3"], encoded.encoded_data.example_ids)

        shutil.rmtree(path)