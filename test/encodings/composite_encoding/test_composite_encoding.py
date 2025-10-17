import shutil

import numpy as np
from numpy.testing import assert_array_equal

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.baseline_encoding.GeneFrequencyEncoder import GeneFrequencyEncoder
from immuneML.encodings.baseline_encoding.MetadataEncoder import MetadataEncoder
from immuneML.encodings.composite_encoding.CompositeEncoder import CompositeEncoder
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


def test_metadata_encoder():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'test_metadata_encoder')

    dataset = RepertoireBuilder.build_dataset([['AA', 'CC'], ['CC'], ['TT']],
                                              path / 'dataset',
                                              seq_metadata=[[{'v_call': 'TRBV1*01', 'j_call': 'TRBJ2-1*01'},
                                                             {'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                            [{'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                            [{'v_call': 'TRBV2*01', 'j_call': 'TRBJ2-1*01'}]],
                                              labels={'label': [0, 1, 0],
                                                      'HLA': ['HLAA1,HLAB2', "HLAA2,HLAB3", "HLAA1,HLAB3"]}, )

    encoder_hla = MetadataEncoder(metadata_fields=['HLA'], name='metadata_encoding')
    encoder_genes = GeneFrequencyEncoder(genes=['V', 'J'], normalization_type=NormalizationType.RELATIVE_FREQUENCY,
                                   scale_to_zero_mean=False, scale_to_unit_variance=False)

    encoder = CompositeEncoder(encoders=[encoder_hla, encoder_genes], name='composite_encoder')

    label_config = LabelConfiguration(labels=[Label('label', [0, 1])])
    encoded_dataset = encoder.encode(dataset,
                                     EncoderParams(label_config=label_config, learn_model=True))

    assert encoded_dataset.encoded_data.feature_names == ['HLA_HLAA1', 'HLA_HLAA2', 'HLA_HLAB2', 'HLA_HLAB3', 'TRBV1', 'TRBV3', 'TRBV2', 'TRBJ2-1', 'TRBJ2-2']
    assert_array_equal(encoded_dataset.encoded_data.get_examples_as_np_matrix(), np.array([[1, 0, 1, 0, 0.5, 0.5, 0., 0.5, 0.5],
                                                                                           [0, 1, 0, 1, 0., 1., 0., 0., 1.],
                                                                                           [1, 0, 0, 1, 0., 0., 1., 1., 0.]]))

    assert 'label' in encoded_dataset.encoded_data.labels and encoded_dataset.encoded_data.labels['label'] == [0, 1, 0]

    dataset2 = RepertoireBuilder.build_dataset([['AA', 'CC'], ['CC'], ['TT']],
                                               path / 'dataset2',
                                               labels={'label': [0, 1, 0],
                                                       'HLA': ['HLAA1,HLAB4', "HLAA2,HLAB3", "HLAA1,HLAB3"]},
                                               seq_metadata=[[{'v_call': 'TRBV1*01', 'j_call': 'TRBJ2-1*01'},
                                                              {'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                             [{'v_call': 'TRBV3*01', 'j_call': 'TRBJ2-2*01'}],
                                                             [{'v_call': 'TRBV2*01', 'j_call': 'TRBJ4-1*01'}]]
                                               )

    encoded_dataset2 = encoder.encode(dataset2, EncoderParams(label_config=label_config, learn_model=False))

    assert encoded_dataset2.encoded_data.feature_names == encoded_dataset.encoded_data.feature_names
    assert_array_equal(encoded_dataset2.encoded_data.get_examples_as_np_matrix(), np.array([[1, 0, 0, 0, 0.5, 0.5, 0., 0.5, 0.5],
                                                                                            [0, 1, 0, 1, 0., 1., 0., 0., 1.],
                                                                                            [1, 0, 0, 1, 0., 0., 1., 0., 0.]]))
    shutil.rmtree(path)
