import shutil
from unittest import TestCase

import bionumpy as bnp
import numpy as np
import pandas as pd

from immuneML.data_model.receptor.receptor_sequence.Chain import Chain
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.generative_models.GenModelAsTSV import GenModelAsTSV
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.rejection_sampling.RejectionSampler import RejectionSampler
from immuneML.util.PathBuilder import PathBuilder


class TestRejectionSampler(TestCase):

    def make_sampler(self):
        motif1 = Motif(identifier='m1', instantiation=GappedKmerInstantiation(), seed='AA')
        motif2 = Motif(identifier='m2', instantiation=GappedKmerInstantiation(), seed='EA')

        signal1 = Signal('s1', [motif1], None)
        signal2 = Signal('s2', [motif2], None)

        return RejectionSampler(LIgOSimulationItem([signal1], repertoire_implanting_rate=0.5, number_of_examples=5,
                                                   number_of_receptors_in_repertoire=6,
                                                   generative_model=OLGA.build_object(default_model_name="humanTRB", chain=Chain.BETA,
                                                                                      model_path=None, use_only_productive=True)),
                                SequenceType.AMINO_ACID, [signal1, signal2], 40, 100)

    def test__generate_sequences(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_gen_seqs')

        sampler = self.make_sampler()
        sampler._make_background_sequences(path, 15, {'s1': 15})

        for filename in ['sequences_no_signal.tsv', 'sequences_with_signal_s1.tsv']:
            seqs = pd.read_csv(path / filename, sep='\t')
            assert seqs.shape[0] == 15, seqs
            for index, row in seqs.iterrows():
                assert len(row['s1_positions']) == len(row['s2_positions'])
                assert len(row['sequence_aa']) + 1 == len(row['s1_positions']), (row['sequence_aa'], row['s1_positions'])
                assert len(row['sequence_aa']) + 1 == len(row['s2_positions']), (row['sequence_aa'], row['s2_positions'])

        shutil.rmtree(path)

    def test__update_seqs_without_signal(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_update_seqs_no_sig')

        sampler = self.make_sampler()  # s1: AA, s2: EA
        signal_matrix = np.array([[True, False], [False, False], [True, True], [False, True]])
        pd.DataFrame({**{key: ['', '', '', ''] for key in sampler.sim_item.generative_model.OUTPUT_COLUMNS},
                      **{'sequence_aa': ['AA', 'DFG', 'AAEA', 'DGEAFT']}}) \
            .to_csv(path / 'tmp.tsv', sep='\t', index=False, header=True)
        background_seqs = bnp.open(path / 'tmp.tsv',
                                   buffer_type=bnp.delimited_buffers.get_bufferclass_for_datatype(GenModelAsTSV, has_header=True)).read()
        sampler.seqs_no_signal_path = path / 'no_sig.tsv'
        count = sampler._update_seqs_without_signal(5, signal_matrix, background_seqs)

        assert count == 4, count

        df = pd.read_csv(path / 'no_sig.tsv', sep='\t')
        for i, row in df.iterrows():
            assert len(row['sequence_aa']) + 1 == len(row['s1_positions']), (row['sequence_aa'], row['s1_positions'])
            assert len(row['sequence_aa']) + 1 == len(row['s2_positions']), (row['sequence_aa'], row['s2_positions'])

        shutil.rmtree(path)

    def test__update_seqs_with_signal(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_update_seqs_with_sig')

        sampler = self.make_sampler()
        signal_matrix = np.array([[True, False], [False, False], [False, True]])
        signal_positions = pd.DataFrame({'s1_positions': ['m10', 'm000', 'm000000'],
                                         's2_positions': ['m00', 'm000', 'm001000']})
        pd.DataFrame({**{key: ['', '', ''] for key in sampler.sim_item.generative_model.OUTPUT_COLUMNS},
                      **{'sequence_aa': ['AA', 'DFG', 'DGEAFT']}}) \
            .to_csv(path / 'tmp.tsv', sep='\t', index=False, header=True)
        background_seqs = bnp.open(path / 'tmp.tsv',
                                   buffer_type=bnp.delimited_buffers.get_bufferclass_for_datatype(GenModelAsTSV, has_header=True)).read()
        sampler.seqs_with_signal_path = {'s1': path / 'with_sig.tsv'}
        count = sampler._update_seqs_with_signal({'s1': 5}, signal_matrix, background_seqs, signal_positions)

        assert count['s1'] == 4, count

        df = pd.read_csv(path / 'with_sig.tsv', sep='\t')
        for i, row in df.iterrows():
            assert len(row['sequence_aa']) + 1 == len(row['s1_positions']), (row['sequence_aa'], row['s1_positions'])
            assert len(row['sequence_aa']) + 1 == len(row['s2_positions']), (row['sequence_aa'], row['s2_positions'])

            assert row['s1_positions'] == 'm10', row['s1_positions']
            assert row['s2_positions'] == 'm00', row['s2_positions']

        shutil.rmtree(path)

    def test_make_sequences(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_make_seqs')

        sampler = self.make_sampler()

        sequences = sampler.make_sequences(path)

        shutil.rmtree(path)

        assert len(sequences) == 5
        for seq in sequences:
            assert len(seq.amino_acid_sequence) + 1 == len(seq.metadata.custom_params['s1_positions']), (
                seq.amino_acid_sequence, seq.metadata.custom_params['s1_positions'])
            assert len(seq.amino_acid_sequence) + 1 == len(seq.metadata.custom_params['s2_positions']), (
                seq.amino_acid_sequence, seq.metadata.custom_params['s2_positions'])

    def test_filter_out_illegal_sequences(self):
        signal_matrix = np.array([[True, False], [False, False], [True, True], [True, False], [False, True]])

        sampler = self.make_sampler()

        legal_indices = sampler.filter_out_illegal_sequences(signal_matrix)
        expected_indices = np.array([True, True, False, True, False])
        assert np.array_equal(legal_indices, expected_indices), legal_indices

    def test_get_signal_matrix(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_signal_matrix')

        motif1 = Motif(identifier='m1', instantiation=GappedKmerInstantiation(), seed='AA', v_call='V1')
        motif2 = Motif(identifier='m2', instantiation=GappedKmerInstantiation(), seed='AA')
        motif3 = Motif(identifier='m3', instantiation=GappedKmerInstantiation(), seed='AC', v_call='V1-1')
        motif4 = Motif(identifier='m4', instantiation=GappedKmerInstantiation(), seed='EA', j_call='J3')

        signal1 = Signal('s1', [motif1, motif2], None)
        signal2 = Signal('s2', [motif3, motif4], None)

        sampler = RejectionSampler(LIgOSimulationItem([signal1, signal2], repertoire_implanting_rate=0.5, number_of_examples=5,
                                                      number_of_receptors_in_repertoire=5,
                                                      generative_model=OLGA(default_model_name="humanTRB", chain=Chain.BETA, model_path=None)),
                                   sequence_type=SequenceType.AMINO_ACID, all_signals=[signal1, signal2], sequence_batch_size=40, max_iterations=100)

        sequences = pd.DataFrame({'sequence_aa': ['AAACCC', 'EEAAF'], 'sequence': ['A', 'CCA'], 'v_call': ['V1-1', 'V2'], 'j_call': ['J2', 'J3-2'],
                                  'region_type': ['JUNCTION', 'JUNCTION'], 'frame_type': ['in', 'in']})
        sequences.to_csv(path / 'sequences.tsv', sep='\t', index=False)

        sequences = bnp.open(path / 'sequences.tsv',
                             buffer_type=bnp.delimited_buffers.get_bufferclass_for_datatype(GenModelAsTSV, delimiter="\t", has_header=True)).read()

        signal_matrix, signal_positions = sampler.get_signal_matrix(sequences)

        assert np.array_equal(signal_matrix, [[True, True], [True, True]])
        assert np.array_equal(signal_positions.values, [['m110000', 'm001000'], ['m00100', 'm01000']])

        shutil.rmtree(path)
