import shutil

import numpy as np

from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.protein_embedding.ESMCEncoder import ESMCEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.util.PathBuilder import PathBuilder


def test_encode_sequence_dataset():
    path = EnvironmentSettings.tmp_test_path / "esmc_seq/"
    PathBuilder.remove_old_and_build(path)

    dataset = RandomDatasetGenerator.generate_sequence_dataset(sequence_count=10, length_probabilities={3: 0.5, 4: 0.5},
                                                               path=path, labels={"label": {True: 0.5, False: 0.5}})

    encoder = ESMCEncoder.build_object(dataset=dataset, region_type="IMGT_CDR3")

    encoded_dataset = encoder.encode(dataset, EncoderParams(
        result_path=path / "encoded",
        label_config=LabelConfiguration([Label("label", [True, False])]),
        learn_model=True

    ))
    assert isinstance(encoded_dataset.encoded_data.examples, np.ndarray), "The embeddings are not of type numpy.ndarray"
    assert encoded_dataset.encoded_data.examples.shape == (10, 960), \
        f"The array shape is {encoded_dataset.encoded_data.examples.shape}, expected (10, 960)"

    shutil.rmtree(path)


def test_encode_repertoire_dataset():
    path = EnvironmentSettings.tmp_test_path / "esmc_rep/"
    PathBuilder.remove_old_and_build(path)

    dataset = RandomDatasetGenerator.generate_repertoire_dataset(repertoire_count=5,
                                                                 sequence_count_probabilities={4: 1},
                                                                 sequence_length_probabilities={3: 1},
                                                                 labels={"label": {True: 0.5, False: 0.5}},
                                                                 path=path)

    lc = LabelConfiguration()
    lc.add_label("label", [True, False])

    encoder = ESMCEncoder.build_object(dataset=dataset, region_type="IMGT_CDR3")

    encoded_dataset = encoder.encode(dataset, EncoderParams(
        result_path=path / "encoded",
        label_config=lc,
        learn_model=True

    ))
    assert isinstance(encoded_dataset.encoded_data.examples, np.ndarray), "The embeddings are not of type numpy.ndarray"
    assert encoded_dataset.encoded_data.examples.shape == (5, 960), \
        f"The array shape is {encoded_dataset.encoded_data.examples.shape}, expected (5, 960)"
    assert all(isinstance(label, bool) for label in
               encoded_dataset.encoded_data.labels['label']), "All labels are not of type bool"

    shutil.rmtree(path)
