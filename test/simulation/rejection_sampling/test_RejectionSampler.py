import shutil
from unittest import TestCase

import bionumpy as bnp
import numpy as np
import pandas as pd
from bionumpy import as_encoded_array, DNAEncoding, AminoAcidEncoding
from bionumpy.encodings import BaseEncoding
from bionumpy.io import delimited_buffers
from npstructures.testing import assert_raggedarray_equal

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
from immuneML.simulation.util.util import annotate_sequences, filter_out_illegal_sequences, make_bnp_annotated_sequences
from immuneML.util.PathBuilder import PathBuilder


class TestRejectionSampler(TestCase):

    def make_sampler(self):
        motif1 = Motif(identifier='m1', instantiation=GappedKmerInstantiation(), seed='AA')
        motif2 = Motif(identifier='m2', instantiation=GappedKmerInstantiation(), seed='EA')

        signal1 = Signal('s1', [motif1], None)
        signal2 = Signal('s2', [motif2], None)

        return RejectionSampler(LIgOSimulationItem([signal1], repertoire_implanting_rate=0.5, number_of_examples=5,
                                                   receptors_in_repertoire_count=6,
                                                   generative_model=OLGA.build_object(default_model_name="humanTRB", chain=Chain.BETA,
                                                                                      model_path=None, use_only_productive=True)),
                                SequenceType.AMINO_ACID, [signal1, signal2], 40, 100)

    def make_data_and_signals(self, path):
        motif1 = Motif(identifier='m1', instantiation=GappedKmerInstantiation(), seed='AA', v_call='V1')
        motif2 = Motif(identifier='m2', instantiation=GappedKmerInstantiation(), seed='AA')
        motif3 = Motif(identifier='m3', instantiation=GappedKmerInstantiation(), seed='AC', v_call='V1-1')
        motif4 = Motif(identifier='m4', instantiation=GappedKmerInstantiation(), seed='EA', j_call='J3')

        signal1 = Signal('s1', [motif1, motif2], None)
        signal2 = Signal('s2', [motif3, motif4], None)

        sequences = pd.DataFrame({'sequence_aa': ['AAACCC', 'EEAAF', 'EEFAF'], 'sequence': ['A', 'CCA', 'CTA'], 'v_call': ['V1-1', 'V2', 'V7'],
                                  'j_call': ['J2', 'J3-2', 'J1'],
                                  'region_type': ['JUNCTION', 'JUNCTION', 'JUNCTION'], 'frame_type': ['in', 'in', 'in']})
        sequences.to_csv(path / 'sequences.tsv', sep='\t', index=False)

        sequences = bnp.open(path / 'sequences.tsv',
                             buffer_type=delimited_buffers.get_bufferclass_for_datatype(GenModelAsTSV, delimiter="\t", has_header=True)).read()

        return sequences, [signal1, signal2]

    def make_annotated_data(self, path):
        seqs, signals = self.make_data_and_signals(path)
        seqs = annotate_sequences(seqs, True, signals)
        seqs = filter_out_illegal_sequences(seqs, LIgOSimulationItem(signals=signals), signals, 1)
        return seqs, signals

    def test__generate_sequences(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_gen_seqs')

        sampler = self.make_sampler()
        sequence_counts = {'s1': 10, 'no_signal': 5}
        sampler._make_background_sequences(path, sequence_counts)

        for filename, expected_count in zip(['sequences_no_signal.tsv', 'sequences_with_signal_s1.tsv'], [5, 10]):
            seqs = pd.read_csv(path / f"tmp_/{filename}", sep='\t')
            assert seqs.shape[0] == expected_count, seqs

            if 'no_signal.tsv' in filename:
                assert seqs['s1'].values.sum() == 0, f"sum in no signal: {seqs['s1'].values.sum()}"
            else:
                assert seqs['s1'].values.sum() == seqs.shape[0], f"sum in signal: {seqs['s1'].values.sum()}, expected {seqs.shape[0]}"

            for index, row in seqs.iterrows():
                assert len(row['s1_positions']) == len(row['s2_positions'])
                assert len(row['sequence_aa']) + 1 == len(row['s1_positions']), (row['sequence_aa'], row['s1_positions'])
                assert len(row['sequence_aa']) + 1 == len(row['s2_positions']), (row['sequence_aa'], row['s2_positions'])

        shutil.rmtree(path)

    def test__update_seqs_without_signal(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_update_seqs_no_sig')

        sampler = self.make_sampler()  # s1: AA, s2: EA
        signal_matrix = np.array([[True, False], [False, False], [True, True], [False, True]])
        background_seqs = GenModelAsTSV(**{**{key: as_encoded_array(['A', 'A', 'A', 'A'], BaseEncoding if 'sequence' not in key else DNAEncoding)
                                              for key in sampler.sim_item.generative_model.OUTPUT_COLUMNS},
                                           **{'sequence_aa': as_encoded_array(['AA', 'DFG', 'AAEA', 'DGEAFT'], AminoAcidEncoding)}})
        seqs = make_bnp_annotated_sequences(background_seqs, [Signal('s1', None, None), Signal("s2", None, None)], signal_matrix,
                                            {"s1_positions": ['m', 'm', 'm', 'm'], "s2_positions": ['m', 'm', 'm', 'm']})
        sampler.seqs_no_signal_path = path / 'no_sig.tsv'
        count = sampler._update_seqs_without_signal(5, seqs)

        assert count == 4, count

        shutil.rmtree(path)

    def test__update_seqs_with_signal(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_update_seqs_with_sig')

        sampler = self.make_sampler()
        signal_matrix = np.array([[True, False], [False, False], [False, True]])
        signal_positions = {'s1_positions': ['m10', 'm000', 'm000000'], 's2_positions': ['m00', 'm000', 'm001000']}
        pd.DataFrame({**{key: ['', '', ''] for key in sampler.sim_item.generative_model.OUTPUT_COLUMNS},
                      **{'sequence_aa': ['AA', 'DFG', 'DGEAFT']}}) \
            .to_csv(path / 'tmp.tsv', sep='\t', index=False, header=True)
        background_seqs = bnp.open(path / 'tmp.tsv',
                                   buffer_type=delimited_buffers.get_bufferclass_for_datatype(GenModelAsTSV, has_header=True)).read()
        annotated_sequences = make_bnp_annotated_sequences(background_seqs, sampler.all_signals, signal_matrix, signal_positions)
        sampler.seqs_with_signal_path = {'s1': path / 'with_sig.tsv'}

        print(annotated_sequences)

        count = sampler._update_seqs_with_signal({'s1': 5}, annotated_sequences)

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
        sampler.sim_item.receptors_in_repertoire_count = None

        sequences = sampler.make_sequences(path)

        shutil.rmtree(path)

        assert len(sequences) == 5
        for seq in sequences:
            assert len(seq.amino_acid_sequence) + 1 == len(seq.metadata.custom_params['s1_positions']), (
                seq.amino_acid_sequence, seq.metadata.custom_params['s1_positions'])
            assert len(seq.amino_acid_sequence) + 1 == len(seq.metadata.custom_params['s2_positions']), (
                seq.amino_acid_sequence, seq.metadata.custom_params['s2_positions'])

    def test_filter_out_illegal_sequences(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_filter')

        sequences, signals = self.make_data_and_signals(path)
        annotated_sequences = annotate_sequences(sequences, True, signals)
        filtered_sequences = filter_out_illegal_sequences(annotated_sequences, LIgOSimulationItem(signals=signals), signals, 1)
        expected_sequences = annotated_sequences[[False, False, True]]

        for field_name in vars(filtered_sequences):
            assert_raggedarray_equal(getattr(filtered_sequences, field_name), getattr(expected_sequences, field_name)), filtered_sequences

        shutil.rmtree(path)

    def test_annotate_sequences(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'rej_sampling_signal_matrix')

        sequences, signals = self.make_data_and_signals(path)

        annotated_seqs = annotate_sequences(sequences, True, signals)

        assert np.array_equal(annotated_seqs.get_signal_matrix(), [[1, 1], [1, 1], [0, 0]])

        assert np.array_equal(annotated_seqs.s1, [True, True, False])
        assert np.array_equal(annotated_seqs.s2, [True, True, False])
        assert np.array_equal([s.to_string() for s in annotated_seqs.s1_positions], ['m110000', 'm00100', 'm00000'])
        assert np.array_equal([s.to_string() for s in annotated_seqs.s2_positions], ['m001000', 'm01000', 'm00000'])

        shutil.rmtree(path)
