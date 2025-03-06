import numpy as np

from immuneML.data_model.SequenceSet import ReceptorSequence
from immuneML.data_model.datasets.ElementDataset import SequenceDataset
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.TCRBertEncoder import TCRBertEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestTCRBertEncoder:

    def _prepare_sequence_test_dataset(self, path):
        sequences = [ReceptorSequence(sequence_aa="AAC", sequence="AAACCC", sequence_id="1",
                                      metadata={"l1": 1}),
                     ReceptorSequence(sequence_aa="ACA", sequence="ACACAC", sequence_id="2",
                                      metadata={"l1": 2}),
                     ReceptorSequence(sequence_aa="TCAA", sequence="CCCAAA", sequence_id="3",
                                      metadata={"l1": 1})]

        PathBuilder.build(path)

        lc = LabelConfiguration()
        lc.add_label("l1", [1, 2])

        dataset = SequenceDataset.build_from_objects(sequences, path)
        return dataset, lc

    def _prepare_receptor_test_dataset(self, path):
        dataset = RandomDatasetGenerator.generate_receptor_dataset(receptor_count=10,
                                                                   chain_1_length_probabilities={3: 1},
                                                                   chain_2_length_probabilities={3: 1},
                                                                   labels={"label": {True: 0.5, False: 0.5}}, path=path)
        lc = LabelConfiguration()
        lc.add_label("label", [True, False])
        return dataset, lc

    def test_encode_sequence_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "tcr_bert/"
        PathBuilder.remove_old_and_build(path)

        dataset, lc = self._prepare_sequence_test_dataset(path)

        encoder = TCRBertEncoder.build_object(dataset=dataset, **{"model": "tcr-bert", 'layers': [-1],
                                                                  "method": "mean", "batch_size": 1})

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True

        ))
        assert isinstance(encoded_dataset.encoded_data.examples, np.ndarray), "The embeddings are not of type numpy.ndarray"
        assert encoded_dataset.encoded_data.examples.shape == (3, 768), f"The array shape is {encoded_dataset.encoded_data.examples.shape}, expected (3, 768)"

    def test_encode_receptor_test_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "tcr_bert/"
        PathBuilder.remove_old_and_build(path)

        dataset, lc = self._prepare_receptor_test_dataset(path)

        encoder = TCRBertEncoder.build_object(dataset=dataset, **{"model": "tcr-bert", 'layers': [-1],
                                                                  "method": "mean", "batch_size": 1})

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True

        ))
        assert isinstance(encoded_dataset.encoded_data.examples, np.ndarray), "The embeddings are not of type numpy.ndarray"
        assert encoded_dataset.encoded_data.examples.shape == (10, 1536), f"The array shape is {encoded_dataset.encoded_data.examples.shape}, expected (10, 1536)"
        assert len(encoded_dataset.encoded_data.example_ids) == 10, f"The number of example ids is {len(encoded_dataset.encoded_data.example_ids)}, expected 10"
        assert all(isinstance(label, bool) for label in encoded_dataset.encoded_data.labels['label']), "All labels are not of type bool"
        assert encoded_dataset.encoded_data.encoding == "TCRBertEncoder(tcr-bert)", f"The encoding is {encoded_dataset.encoded_data.encoding}, expected TCRBertEncoder(tcr-bert)"

    def test_encode_repertoire_test_dataset(self):
        path = EnvironmentSettings.tmp_test_path / "tcr_bert/"
        PathBuilder.remove_old_and_build(path)

        dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=10,
                                                                     sequence_count_probabilities={10: 1},
                                                                     sequence_length_probabilities={3: 1},
                                                                     labels={"label": {True: 0.5, False: 0.5}},
                                                                     path=path)

        lc = LabelConfiguration()
        lc.add_label("label", [True, False])

        encoder = TCRBertEncoder.build_object(dataset=dataset, **{"model": "tcr-bert", 'layers': [-1],
                                                                  "method": "mean", "batch_size": 1})

        encoded_dataset = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True

        ))
        assert isinstance(encoded_dataset.encoded_data.examples, np.ndarray), "The embeddings are not of type numpy.ndarray"
        assert encoded_dataset.encoded_data.examples.shape == (10, 10), f"The array shape is {encoded_dataset.encoded_data.examples.shape}, expected (10, 10)"
        assert all(isinstance(label, bool) for label in encoded_dataset.encoded_data.labels['label']), "All labels are not of type bool"