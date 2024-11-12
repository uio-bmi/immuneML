from immuneML.environment.SequenceType import SequenceType
from immuneML.ml_methods.generative_models.BackgroundSequences import BackgroundSequences
from immuneML.simulation.SimConfigItem import SimConfigItem
from immuneML.simulation.implants.SeedMotif import SeedMotif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.simulation_strategy.ImplantingStrategy import ImplantingStrategy


def test_implanting():
    s1 = Signal('s1', [SeedMotif('m1', 'TTT')], {'104': 0, '105': 0})
    seqs = BackgroundSequences(sequence=["CCCCC", "CCCCCCCCC"], sequence_aa=["A", "AA"], v_call=["", ""],
                               j_call=["", ""], region_type=["IMGT_JUNCTION", "IMGT_JUNCTION"], frame_type=["", ""],
                               p_gen=[-1., -1.], from_default_model=[1, 1],
                               duplicate_count=[1,1], locus=["TRB", "TRB"])
    processed_seqs = ImplantingStrategy().process_sequences(seqs, {'s1': 2}, False, SequenceType.NUCLEOTIDE,
                                                            SimConfigItem({s1: 1.}), [s1], False)
    print(processed_seqs)
