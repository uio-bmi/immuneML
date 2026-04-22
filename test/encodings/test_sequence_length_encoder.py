import os
import shutil

import numpy as np
import pytest

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence, Repertoire
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.sequence_length_encoding.SequenceLengthEncoder import SequenceLengthEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


@pytest.fixture(autouse=True)
def use_test_cache():
    os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

def test_encode_sequence_dataset_amino_acid():
    path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / "seq_len_seq_aa/")

    sequences = [
        ReceptorSequence(sequence_aa="ACDE", sequence_id="s1", metadata={"label": 1}),
        ReceptorSequence(sequence_aa="FGHI", sequence_id="s2", metadata={"label": 2}),
        ReceptorSequence(sequence_aa="KLM",  sequence_id="s3", metadata={"label": 3}),
    ]
    dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)
    lc = LabelConfiguration([Label("label", [1, 2, 3])])

    encoder = SequenceLengthEncoder.build_object(
        dataset, region_type="imgt_cdr3", sequence_type="amino_acid",
        scale_to_zero_mean=False, scale_to_unit_variance=False)

    encoded = encoder.encode(dataset, EncoderParams(
        result_path=path / "encoded",
        label_config=lc,
        learn_model=True,
    ))

    assert isinstance(encoded, SequenceDataset)
    ex = encoded.encoded_data.examples
    assert ex.shape == (3, 1)
    np.testing.assert_array_equal(ex[:, 0], [4.0, 4.0, 3.0])
    assert encoded.encoded_data.example_ids == ["s1", "s2", "s3"]
    assert encoded.encoded_data.labels["label"] == [1, 2, 3]
    assert encoded.encoded_data.feature_names == ["sequence_length"]

    shutil.rmtree(path)


# ------------------------------------------------------------------
# SequenceDataset — nucleotide
# ------------------------------------------------------------------

def test_encode_sequence_dataset_nucleotide():
    path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / "seq_len_seq_nt/")

    # nucleotide sequences are deliberately different lengths from their
    # amino acid counterparts to confirm the correct field is selected
    sequences = [
        ReceptorSequence(sequence_aa="AC", sequence="ACGTAC",    sequence_id="s1", metadata={"label": 1}),
        ReceptorSequence(sequence_aa="FG", sequence="TTGCAA",    sequence_id="s2", metadata={"label": 2}),
        ReceptorSequence(sequence_aa="KL", sequence="AAACCCGGG", sequence_id="s3", metadata={"label": 1}),
    ]
    dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)
    lc = LabelConfiguration([Label("label", [1, 2])])

    encoder = SequenceLengthEncoder.build_object(
        dataset, region_type="imgt_cdr3", sequence_type="nucleotide",
        scale_to_zero_mean=False, scale_to_unit_variance=False)

    encoded = encoder.encode(dataset, EncoderParams(
        result_path=path / "encoded",
        label_config=lc,
        learn_model=True,
    ))

    assert isinstance(encoded, SequenceDataset)
    ex = encoded.encoded_data.examples
    assert ex.shape == (3, 1)
    # lengths come from nucleotide sequences (6, 6, 9), not amino acid (2, 2, 2)
    np.testing.assert_array_equal(ex[:, 0], [6.0, 6.0, 9.0])
    assert encoded.encoded_data.feature_names == ["sequence_length"]

    shutil.rmtree(path)


# ------------------------------------------------------------------
# ReceptorDataset — per chain
# ------------------------------------------------------------------

def test_encode_receptor_dataset():
    path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / "seq_len_receptor/")

    receptors = [
        Receptor(chain_1=ReceptorSequence(sequence_aa="ACDE",  locus="alpha"),
                 chain_2=ReceptorSequence(sequence_aa="FGHI",  locus="beta"),
                 metadata={"label": 1}, receptor_id="r1", cell_id="c1",
                 chain_pair=ChainPair.TRA_TRB),
        Receptor(chain_1=ReceptorSequence(sequence_aa="KLM",   locus="alpha"),
                 chain_2=ReceptorSequence(sequence_aa="NPQRS", locus="beta"),
                 metadata={"label": 2}, receptor_id="r2", cell_id="c2",
                 chain_pair=ChainPair.TRA_TRB),
    ]
    dataset = ReceptorDataset.build_from_objects(receptors, path)
    lc = LabelConfiguration([Label("label", [1, 2])])

    encoder = SequenceLengthEncoder.build_object(
        dataset, region_type="imgt_cdr3", sequence_type="amino_acid",
        scale_to_zero_mean=False, scale_to_unit_variance=False)

    encoded = encoder.encode(dataset, EncoderParams(
        result_path=path / "encoded",
        label_config=lc,
        learn_model=True,
    ))

    assert isinstance(encoded, ReceptorDataset)
    ex = encoded.encoded_data.examples
    # 2 receptors × 2 chains (alpha, beta) — chains sorted alphabetically
    assert ex.shape == (2, 2)
    assert encoded.encoded_data.feature_names == ["alpha_length", "beta_length"]

    # find row for receptor c1: alpha="ACDE" (4), beta="FGHI" (4)
    receptor_ids = encoded.encoded_data.example_ids
    idx_c1 = receptor_ids.index("c1")
    idx_c2 = receptor_ids.index("c2")
    np.testing.assert_array_equal(ex[idx_c1], [4.0, 4.0])
    # receptor c2: alpha="KLM" (3), beta="NPQRS" (5)
    np.testing.assert_array_equal(ex[idx_c2], [3.0, 5.0])

    shutil.rmtree(path)


# ------------------------------------------------------------------
# Scaling
# ------------------------------------------------------------------

def test_encode_with_scaling():
    path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / "seq_len_scaling/")

    sequences = [
        ReceptorSequence(sequence_aa="A",    sequence_id="s1", metadata={"label": 1}),
        ReceptorSequence(sequence_aa="AC",   sequence_id="s2", metadata={"label": 2}),
        ReceptorSequence(sequence_aa="ACE",  sequence_id="s3", metadata={"label": 1}),
        ReceptorSequence(sequence_aa="ACDE", sequence_id="s4", metadata={"label": 2}),
    ]
    dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)
    lc = LabelConfiguration([Label("label", [1, 2])])

    encoder = SequenceLengthEncoder.build_object(
        dataset, region_type="imgt_cdr3", sequence_type="amino_acid",
        scale_to_zero_mean=True, scale_to_unit_variance=True)

    encoded = encoder.encode(dataset, EncoderParams(
        result_path=path / "encoded",
        label_config=lc,
        learn_model=True,
    ))

    ex = encoded.encoded_data.examples
    assert ex.shape == (4, 1)
    np.testing.assert_allclose(ex[:, 0].mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(ex[:, 0].std(),  1.0, atol=1e-10)

    shutil.rmtree(path)


# ------------------------------------------------------------------
# Unsupported dataset type
# ------------------------------------------------------------------

def test_encode_rejects_repertoire_dataset():
    path = PathBuilder.remove_old_and_build(
        EnvironmentSettings.tmp_test_path / "seq_len_repertoire_rejected/")

    rep = Repertoire.build_from_sequences(
        [ReceptorSequence(sequence_aa="ACDE", sequence_id="s1")],
        metadata={"label": True}, result_path=path)
    dataset = RepertoireDataset.build_from_objects(repertoires=[rep], path=path)
    lc = LabelConfiguration([Label("label", [True, False])])

    encoder = SequenceLengthEncoder.build_object(
        dataset, region_type="imgt_cdr3", sequence_type="amino_acid",
        scale_to_zero_mean=False, scale_to_unit_variance=False)

    with pytest.raises(RuntimeError):
        encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
        ))

    shutil.rmtree(path)