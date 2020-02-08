import shutil
from unittest import TestCase

from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.data_model.receptor.TCABReceptor import TCABReceptor
from source.data_model.receptor.receptor_sequence.Chain import Chain
from source.data_model.receptor.receptor_sequence.ReceptorSequence import ReceptorSequence
from source.data_model.receptor.receptor_sequence.SequenceMetadata import SequenceMetadata
from source.encodings.EncoderParams import EncoderParams
from source.encodings.reference_encoding.MatchedReceptorsRepertoireEncoder import MatchedReceptorsRepertoireEncoder
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.environment.LabelConfiguration import LabelConfiguration
from source.util.RepertoireBuilder import RepertoireBuilder


class TestMatchedReceptorsEncoder(TestCase):
    def test__encode_new_dataset(self):
        path = EnvironmentSettings.root_path + "test/tmp/matched_receptors_encoder/"

        # Setting up dummy data
        labels = {"donor": ["donor1", "donor1", "donor2", "donor2", "donor3"],
                  "label": ["yes", "yes", "no", "no", "no"]}

        metadata_alpha = {"v_gene": "v1", "j_gene": "j1", "chain": Chain.A.value}
        metadata_beta = {"v_gene": "v1", "j_gene": "j1", "chain": Chain.B.value}

        repertoires, metadata = RepertoireBuilder.build(sequences=[["AAAA"],
                                                                 ["SSSS"],
                                                                 ["AAAA", "CCCC"],
                                                                 ["SSSS", "TTTT"],
                                                                 ["AAAA", "CCCC", "SSSS", "TTTT"]],
                                                      path=path, labels=labels,
                                                      seq_metadata=[[{**metadata_alpha, "count":10}],
                                                                    [{**metadata_beta, "count": 10}],
                                                                    [{**metadata_alpha, "count": 5}, {**metadata_alpha, "count": 5}],
                                                                    [{**metadata_beta, "count": 5}, {**metadata_beta, "count": 5}],
                                                                    [{**metadata_alpha, "count": 1}, {**metadata_alpha, "count": 2},
                                                                     {**metadata_beta, "count": 1}, {**metadata_beta, "count": 2}]],
                                                        donors=["donor1", "donor1", "donor2", "donor2", "donor3"])

        dataset = RepertoireDataset(repertoires=repertoires)

        label_config = LabelConfiguration()
        label_config.add_label("donor", labels["donor"])
        label_config.add_label("label", labels["label"])

        reference_receptors = [TCABReceptor(alpha=ReceptorSequence("AAAA", metadata=SequenceMetadata(**metadata_alpha)),
                                            beta=ReceptorSequence("SSSS", metadata=SequenceMetadata(**metadata_beta)),
                                            identifier=str(100)),
                               TCABReceptor(alpha=ReceptorSequence("CCCC", metadata=SequenceMetadata(**metadata_alpha)),
                                            beta=ReceptorSequence("TTTT", metadata=SequenceMetadata(**metadata_beta)),
                                            identifier=str(200))]

        encoder = MatchedReceptorsRepertoireEncoder.create_encoder(dataset, {
            "reference_sequences": reference_receptors,
            "one_file_per_donor": True
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_configuration=label_config,
            filename="dataset.csv"
        ))

        # General tests: does the tool run and give the correct output?
        expected_outcome = [[10, 0, 0, 0],[0, 10, 0, 0],[5, 0, 5, 0], [0, 5, 0, 5], [1, 1, 2, 2]]
        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertDictEqual(encoded.encoded_data.labels, labels)
        self.assertListEqual(encoded.encoded_data.feature_names, ["100.alpha", "100.beta", "200.alpha", "200.beta"])

        self.assertListEqual(list(encoded.encoded_data.feature_annotations.id), ['100', '100', '200', '200'])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.chain), ["alpha", "beta", "alpha", "beta"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.sequence), ["AAAA", "SSSS", "CCCC", "TTTT"])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.v_gene), ["v1" for i in range(4)])
        self.assertListEqual(list(encoded.encoded_data.feature_annotations.j_gene), ["j1" for i in range(4)])

        encoder = MatchedReceptorsRepertoireEncoder.create_encoder(dataset, {
            "reference_sequences": reference_receptors,
            "one_file_per_donor": False
        })

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path,
            label_configuration=label_config,
            filename="dataset.csv"
        ))

        expected_outcome = [[10, 10, 0, 0], [5, 5, 5, 5], [1, 1, 2, 2]]
        for index, row in enumerate(expected_outcome):
            self.assertListEqual(list(encoded.encoded_data.examples[index]), expected_outcome[index])

        self.assertDictEqual(encoded.encoded_data.labels, {"label": ["yes", "no", "no"]})

        # If donor is not a label, one_file_per_donor can not be False (KeyError raised)
        label_config = LabelConfiguration()
        label_config.add_label("label", labels["label"])
        params=EncoderParams(result_path=path, label_configuration=label_config, filename="dataset.csv")

        self.assertRaises(KeyError, encoder.encode, dataset, params)

        # If one_file_per_donor is True, the key "donor" does not need to be specified
        encoder = MatchedReceptorsRepertoireEncoder.create_encoder(dataset, {
            "reference_sequences": reference_receptors,
            "one_file_per_donor": True
        })

        encoder.encode(dataset, params)

        shutil.rmtree(path)

