import os
import shutil
from unittest import TestCase

import numpy as np
from sklearn.preprocessing import StandardScaler

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.SequenceParams import ChainPair
from immuneML.data_model.SequenceSet import Receptor, ReceptorSequence, Repertoire
from immuneML.data_model.datasets.ElementDataset import ReceptorDataset, SequenceDataset
from immuneML.data_model.datasets.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.amino_acid_property_encoding.AminoAcidPropertyEncoder import AminoAcidPropertyEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder


class TestAminoAcidPropertyEncoder(TestCase):

    def setUp(self):
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    # ------------------------------------------------------------------
    # Helper: expected average factor vector for a sequence string
    # ------------------------------------------------------------------

    def _expected_avg(self, sequence: str, encoder: AminoAcidPropertyEncoder) -> np.ndarray:
        vecs = [encoder.factor_table[aa] for aa in sequence if aa in encoder.factor_table]
        return np.mean(vecs, axis=0)

    # ------------------------------------------------------------------
    # SequenceDataset
    # ------------------------------------------------------------------

    def test_encode_sequence_dataset_atchley(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "aa_prop_seq_atchley/")

        sequences = [
            ReceptorSequence(sequence_aa="ACDE", sequence_id="s1", metadata={"label": True}),
            ReceptorSequence(sequence_aa="FGHI", sequence_id="s2", metadata={"label": False}),
            ReceptorSequence(sequence_aa="KLM",  sequence_id="s3", metadata={"label": True}),
        ]
        dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)
        lc = LabelConfiguration([Label("label", [True, False])])

        encoder = AminoAcidPropertyEncoder.build_object(scale_to_zero_mean=False, scale_to_unit_variance=False,
            dataset=dataset, factors="atchley", region_type="imgt_cdr3")

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
        ))

        self.assertIsInstance(encoded, SequenceDataset)
        ex = encoded.encoded_data.examples
        self.assertEqual(ex.shape, (3, 5))

        np.testing.assert_allclose(ex[0], self._expected_avg("ACDE", encoder), rtol=1e-6)
        np.testing.assert_allclose(ex[1], self._expected_avg("FGHI", encoder), rtol=1e-6)

        self.assertEqual(encoded.encoded_data.example_ids, ["s1", "s2", "s3"])
        self.assertEqual(encoded.encoded_data.labels["label"], [True, False, True])
        self.assertEqual(encoded.encoded_data.feature_names,
                         [f"atchley_{n}" for n in encoder.factor_names])

        shutil.rmtree(path)

    def test_encode_sequence_dataset_kidera(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "aa_prop_seq_kidera/")

        sequences = [
            ReceptorSequence(sequence_aa="ACDE", sequence_id="s1", metadata={"label": 1}),
            ReceptorSequence(sequence_aa="FGHI", sequence_id="s2", metadata={"label": 2}),
        ]
        dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)
        lc = LabelConfiguration([Label("label", [1, 2])])

        encoder = AminoAcidPropertyEncoder.build_object(scale_to_zero_mean=False, scale_to_unit_variance=False,
            dataset=dataset, factors="kidera", region_type="imgt_cdr3")

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
        ))

        self.assertIsInstance(encoded, SequenceDataset)
        ex = encoded.encoded_data.examples
        self.assertEqual(ex.shape, (2, 10))

        np.testing.assert_allclose(ex[0], self._expected_avg("ACDE", encoder), rtol=1e-6)

        self.assertEqual(len(encoded.encoded_data.feature_names), 10)
        self.assertEqual(encoded.encoded_data.feature_names,
                         [f"kidera_{n}" for n in encoder.factor_names])

        shutil.rmtree(path)

    def test_encode_sequence_dataset_amino_acid_property(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "aa_prop_seq_aap/")

        sequences = [
            ReceptorSequence(sequence_aa="ACDE", sequence_id="s1", metadata={"label": 1}),
            ReceptorSequence(sequence_aa="FGHI", sequence_id="s2", metadata={"label": 2}),
        ]
        dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)
        lc = LabelConfiguration([Label("label", [1, 2])])

        encoder = AminoAcidPropertyEncoder.build_object(scale_to_zero_mean=False, scale_to_unit_variance=False,
            dataset=dataset, factors="amino_acid_property", region_type="imgt_cdr3")

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
        ))

        self.assertIsInstance(encoded, SequenceDataset)
        ex = encoded.encoded_data.examples
        self.assertEqual(ex.shape, (2, 14))

        np.testing.assert_allclose(ex[0], self._expected_avg("ACDE", encoder), rtol=1e-6)

        self.assertEqual(len(encoded.encoded_data.feature_names), 14)
        self.assertEqual(encoded.encoded_data.feature_names,
                         [f"amino_acid_property_{n}" for n in encoder.factor_names])

        shutil.rmtree(path)

    # ------------------------------------------------------------------
    # ReceptorDataset
    # ------------------------------------------------------------------

    def test_encode_receptor_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "aa_prop_receptor/")

        receptors = [
            Receptor(chain_1=ReceptorSequence(sequence_aa="ACDE", locus="alpha"),
                     chain_2=ReceptorSequence(sequence_aa="FGHI", locus="beta"),
                     metadata={"label": 1}, receptor_id="r1", cell_id="c1",
                     chain_pair=ChainPair.TRA_TRB),
            Receptor(chain_1=ReceptorSequence(sequence_aa="KLM", locus="alpha"),
                     chain_2=ReceptorSequence(sequence_aa="NPQ", locus="beta"),
                     metadata={"label": 2}, receptor_id="r2", cell_id="c2",
                     chain_pair=ChainPair.TRA_TRB),
        ]
        dataset = ReceptorDataset.build_from_objects(receptors, path)
        lc = LabelConfiguration([Label("label", [1, 2])])

        encoder = AminoAcidPropertyEncoder.build_object(scale_to_zero_mean=False, scale_to_unit_variance=False,
            dataset=dataset, factors="atchley", region_type="imgt_cdr3")

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
        ))

        self.assertIsInstance(encoded, ReceptorDataset)
        ex = encoded.encoded_data.examples
        self.assertEqual(ex.shape, (2, 10))

        expected_alpha_r1 = self._expected_avg("ACDE", encoder)
        expected_beta_r1  = self._expected_avg("FGHI", encoder)
        np.testing.assert_allclose(ex[0, :5],  expected_alpha_r1, rtol=1e-6)
        np.testing.assert_allclose(ex[0, 5:], expected_beta_r1,  rtol=1e-6)

        self.assertEqual(len(encoded.encoded_data.feature_names), 10)
        self.assertTrue(all("alpha" in fn or "beta" in fn
                            for fn in encoded.encoded_data.feature_names))

        shutil.rmtree(path)

    # ------------------------------------------------------------------
    # RepertoireDataset
    # ------------------------------------------------------------------

    def test_encode_repertoire_dataset(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "aa_prop_repertoire/")

        rep1 = Repertoire.build_from_sequences(
            [ReceptorSequence(sequence_aa="ACDE", sequence_id="s1"),
             ReceptorSequence(sequence_aa="FGHI", sequence_id="s2")],
            metadata={"label": True}, result_path=path)
        rep2 = Repertoire.build_from_sequences(
            [ReceptorSequence(sequence_aa="KLM", sequence_id="s3")],
            metadata={"label": False}, result_path=path)

        dataset = RepertoireDataset.build_from_objects(repertoires=[rep1, rep2], path=path)
        lc = LabelConfiguration([Label("label", [True, False])])

        encoder = AminoAcidPropertyEncoder.build_object(scale_to_zero_mean=False, scale_to_unit_variance=False,
            dataset=dataset, factors="atchley", region_type="imgt_cdr3")

        encoded = encoder.encode(dataset, EncoderParams(
            result_path=path / "encoded",
            label_config=lc,
            learn_model=True,
        ))

        self.assertIsInstance(encoded, RepertoireDataset)
        ex = encoded.encoded_data.examples
        self.assertEqual(ex.shape, (2, 5))

        expected_rep1 = np.mean([self._expected_avg("ACDE", encoder),
                                  self._expected_avg("FGHI", encoder)], axis=0)
        np.testing.assert_allclose(ex[0], expected_rep1, rtol=1e-6)

        expected_rep2 = self._expected_avg("KLM", encoder)
        np.testing.assert_allclose(ex[1], expected_rep2, rtol=1e-6)

        self.assertEqual(encoded.encoded_data.labels["label"], [True, False])
        self.assertEqual(encoded.encoded_data.feature_names,
                         [f"atchley_{n}" for n in encoder.factor_names])

        shutil.rmtree(path)

    # ------------------------------------------------------------------
    # build_object parameter validation
    # ------------------------------------------------------------------

    def test_build_object_rejects_invalid_factors(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "aa_prop_invalid/")
        sequences = [ReceptorSequence(sequence_aa="ACDE", sequence_id="s1", metadata={"label": 1})]
        dataset = SequenceDataset.build_from_objects(sequences=sequences, path=path)

        with self.assertRaises(AssertionError):
            AminoAcidPropertyEncoder.build_object(dataset, factors="blosum", region_type="imgt_cdr3")

        shutil.rmtree(path)