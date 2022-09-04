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


def test_make_repertoires():
    assert False


def test__make_repertoire_from_sequences():
    assert False


def test__get_signal_sequence_count():
    assert False


def test__get_no_signal_sequences():
    assert False


def test__add_signal_sequences():
    assert False


def test__make_sequences_from_generative_model():
    assert False


def test__generate_sequences():
    assert False


def test__update_seqs_without_signal():
    assert False


def test__update_seqs_with_signal():
    assert False


def test__store_sequences():
    assert False


def test_make_receptors():
    assert False


def test_make_sequences():
    assert False


def test_filter_out_illegal_sequences():
    assert False


def test_get_signal_matrix():
    path = PathBuilder.build(EnvironmentSettings.tmp_test_path / 'rej_sampling_signal_matrix')

    motif1 = Motif(identifier='m1', instantiation=GappedKmerInstantiation(), seed='AA', v_call='V1')
    motif2 = Motif(identifier='m2', instantiation=GappedKmerInstantiation(), seed='AA')
    motif3 = Motif(identifier='m3', instantiation=GappedKmerInstantiation(), seed='AC', v_call='V1-1')
    motif4 = Motif(identifier='m4', instantiation=GappedKmerInstantiation(), seed='EA', j_call='J3')

    signal1 = Signal('s1', [motif1, motif2], None)
    signal2 = Signal('s2', [motif3, motif4], None)

    sampler = RejectionSampler(LIgOSimulationItem([signal1, signal2], repertoire_implanting_rate=0.5, number_of_examples=5,
                                                  number_of_receptors_in_repertoire=5,
                                                  generative_model=OLGA(default_model_name="humanTRB", chain=Chain.BETA, model_path=None)),
                               SequenceType.AMINO_ACID, [signal1, signal2], 40, 100)

    sequences = pd.DataFrame({'sequence_aa': ['AAACCC', 'EEAAF'], 'sequence': ['A', 'CCA'], 'v_call': ['V1-1', 'V2'], 'j_call': ['J2', 'J3-2'],
                              'region_type': ['JUNCTION', 'JUNCTION'], 'frame_type': ['in', 'in']})
    sequences.to_csv(path / 'sequences.tsv', sep='\t', index=False)

    sequences = bnp.open(path / 'sequences.tsv', mode='full',
                         buffer_type=bnp.delimited_buffers.get_bufferclass_for_datatype(GenModelAsTSV, delimiter="\t"),
                         has_header=True)

    signal_matrix, signal_positions = sampler.get_signal_matrix(sequences)

    assert np.array_equal(signal_matrix, [[True, True], [True, True]])
    assert np.array_equal(signal_positions, [['110000', '001000'], ['00100', '01000']])


def test__match_genes():
    assert False


def test__match_motif():
    assert False
