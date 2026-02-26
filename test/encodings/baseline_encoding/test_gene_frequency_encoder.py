import shutil

import numpy as np
from numpy.testing import assert_array_equal

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.baseline_encoding.GeneFrequencyEncoder import GeneFrequencyEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


def test_gene_frequency_encoder():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'test_gene_frequency_encoder')

    dataset = RepertoireBuilder.build_dataset([['AA', 'CC'], ['CC'], ['TT']],
                                              path / 'dataset',
                                              labels={'label': [0, 1, 0]},
                                              seq_metadata=[[{'v_call': 'TRBV1*01', 'j_call': 'TRBJ2-1*01'},
                                                             {'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                            [{'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                            [{'v_call': 'TRBV2*01', 'j_call': 'TRBJ2-1*01'}]])

    encoder = GeneFrequencyEncoder(genes=['V', 'J'], normalization_type=NormalizationType.RELATIVE_FREQUENCY,
                                   scale_to_zero_mean=False, scale_to_unit_variance=False)

    label_config = LabelConfiguration(labels=[Label('label', [0, 1])])
    encoded_dataset = encoder.encode(dataset,
                                     EncoderParams(label_config=label_config, learn_model=True))

    assert encoded_dataset.encoded_data.examples.shape == (3, 5)

    assert_array_equal(encoded_dataset.encoded_data.examples, np.array([[0.5, 0.5, 0., 0.5, 0.5],
                                                                        [0., 1., 0., 0., 1.],
                                                                        [0., 0., 1., 1., 0.]]))

    assert encoded_dataset.encoded_data.feature_names == ['TRBV1', 'TRBV3', 'TRBV2', 'TRBJ2-1', 'TRBJ2-2']
    assert 'label' in encoded_dataset.encoded_data.labels and encoded_dataset.encoded_data.labels['label'] == [0, 1, 0]

    dataset2 = RepertoireBuilder.build_dataset([['AA', 'CC'], ['CC'], ['TT']],
                                               path / 'dataset2',
                                               labels={'label': [0, 1, 0]},
                                               seq_metadata=[[{'v_call': 'TRBV1*01', 'j_call': 'TRBJ2-1*01'},
                                                              {'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                             [{'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                             [{'v_call': 'TRBV2*01', 'j_call': 'TRBJ4-1*01'}]])

    encoded_dataset2 = encoder.encode(dataset2,
                                     EncoderParams(label_config=label_config, learn_model=False))

    assert encoded_dataset2.encoded_data.examples.shape == (3, 5)
    assert encoded_dataset2.encoded_data.feature_names == ['TRBV1', 'TRBV3', 'TRBV2', 'TRBJ2-1', 'TRBJ2-2']

    shutil.rmtree(path)


def test_gene_frequency_encoder_receptor():
    path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / 'test_gene_frequency_encoder_receptor')

    def make_receptor(cell_id, tra_v, tra_j, trb_v, trb_j):
        return Receptor(
            chain_pair=ChainPair.TRA_TRB,
            chain_1=ReceptorSequence(sequence_aa='AA', locus='TRA', v_call=tra_v, j_call=tra_j),
            chain_2=ReceptorSequence(sequence_aa='CC', locus='TRB', v_call=trb_v, j_call=trb_j),
            cell_id=cell_id
        )

    receptors = [
        make_receptor('cell1', 'TRAV1*01', 'TRAJ1*01', 'TRBV1*01', 'TRBJ1*01'),
        make_receptor('cell2', 'TRAV2*01', 'TRAJ1*01', 'TRBV1*01', 'TRBJ2*01'),
        make_receptor('cell3', 'TRAV1*01', 'TRAJ2*01', 'TRBV2*01', 'TRBJ1*01'),
    ]

    dataset = ReceptorDataset.build_from_objects(receptors, PathBuilder.build(path / 'dataset'), name='receptor_dataset')

    encoder = GeneFrequencyEncoder(genes=['V', 'J'], normalization_type=NormalizationType.NONE,
                                   scale_to_zero_mean=False, scale_to_unit_variance=False)

    label_config = LabelConfiguration(labels=[Label('label', [0, 1])])
    encoded = encoder.encode(dataset, EncoderParams(label_config=label_config, learn_model=True,
                                                    encode_labels=False))

    # V: TRAV1, TRAV2, TRBV1, TRBV2 (alphabetical); J: TRAJ1, TRAJ2, TRBJ1, TRBJ2
    assert encoded.encoded_data.examples.shape == (3, 8)
    assert encoded.encoded_data.feature_names == ['TRAV1', 'TRAV2', 'TRBV1', 'TRBV2',
                                                  'TRAJ1', 'TRAJ2', 'TRBJ1', 'TRBJ2']
    assert_array_equal(encoded.encoded_data.examples, np.array([
        [1., 0., 1., 0.,  1., 0., 1., 0.],  # cell1
        [0., 1., 1., 0.,  1., 0., 0., 1.],  # cell2
        [1., 0., 0., 1.,  0., 1., 1., 0.],  # cell3
    ]))

    # at predict time: TRAV3 is novel → ignored (0); TRBV1 known → 1
    receptors2 = [make_receptor('cell1', 'TRAV3*01', 'TRAJ1*01', 'TRBV1*01', 'TRBJ1*01')]
    dataset2 = ReceptorDataset.build_from_objects(receptors2, PathBuilder.build(path / 'dataset2'), name='receptor_dataset2')

    encoded2 = encoder.encode(dataset2, EncoderParams(label_config=label_config, learn_model=False,
                                                      encode_labels=False))

    assert encoded2.encoded_data.examples.shape == (1, 8)
    assert encoded2.encoded_data.feature_names == ['TRAV1', 'TRAV2', 'TRBV1', 'TRBV2',
                                                   'TRAJ1', 'TRAJ2', 'TRBJ1', 'TRBJ2']
    assert_array_equal(encoded2.encoded_data.examples, np.array([
        [0., 0., 1., 0.,  1., 0., 1., 0.],  # TRAV3 unknown → TRAV1=0, TRAV2=0
    ]))

    shutil.rmtree(path)


def test_gene_frequency_encoder_sequence():
    path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / 'test_gene_frequency_encoder_sequence')

    sequences = [
        ReceptorSequence(sequence_aa='AA', locus='TRB', v_call='TRBV1*01', j_call='TRBJ1*01'),
        ReceptorSequence(sequence_aa='CC', locus='TRB', v_call='TRBV2*01', j_call='TRBJ1*01'),
        ReceptorSequence(sequence_aa='GG', locus='TRB', v_call='TRBV1*01', j_call='TRBJ2*01'),
    ]

    dataset = SequenceDataset.build_from_objects(sequences, PathBuilder.build(path / 'dataset'), name='sequence_dataset')

    encoder = GeneFrequencyEncoder(genes=['V', 'J'], normalization_type=NormalizationType.NONE,
                                   scale_to_zero_mean=False, scale_to_unit_variance=False)

    label_config = LabelConfiguration(labels=[Label('label', [0, 1])])
    encoded = encoder.encode(dataset, EncoderParams(label_config=label_config, learn_model=True,
                                                    encode_labels=False))

    # V: TRBV1, TRBV2 (alphabetical); J: TRBJ1, TRBJ2
    assert encoded.encoded_data.examples.shape == (3, 4)
    assert encoded.encoded_data.feature_names == ['TRBV1', 'TRBV2', 'TRBJ1', 'TRBJ2']
    assert_array_equal(encoded.encoded_data.examples, np.array([
        [1., 0., 1., 0.],
        [0., 1., 1., 0.],
        [1., 0., 0., 1.],
    ]))

    # at predict time: TRBV3 is novel → ignored (0)
    sequences2 = [ReceptorSequence(sequence_aa='TT', locus='TRB', v_call='TRBV3*01', j_call='TRBJ1*01')]
    dataset2 = SequenceDataset.build_from_objects(sequences2, PathBuilder.build(path / 'dataset2'), name='sequence_dataset2')

    encoded2 = encoder.encode(dataset2, EncoderParams(label_config=label_config, learn_model=False,
                                                      encode_labels=False))

    assert encoded2.encoded_data.examples.shape == (1, 4)
    assert encoded2.encoded_data.feature_names == ['TRBV1', 'TRBV2', 'TRBJ1', 'TRBJ2']
    assert_array_equal(encoded2.encoded_data.examples, np.array([[0., 0., 1., 0.]]))

    shutil.rmtree(path)
