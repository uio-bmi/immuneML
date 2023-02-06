from unittest import TestCase

from immuneML.simulation.SequenceDispenser import SequenceDispenser
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation


class TestSequenceDispenser(TestCase):
    def test_generate_mutation(self):
        seq_disp = SequenceDispenser(occurrence_limit_pgen_range={1e-10: 2})

        seed_motif = Motif("motif1", GappedKmerInstantiation(max_gap=0), "CASSLPSYQNTEAFF",
                           v_call="TRBV5-1", j_call="TRBJ1-1",
                           mutation_position_possibilities={7: 1})

        seq_disp.add_seed_sequence(seed_motif)

        new_mutated_motif = seq_disp.generate_mutation(repertoire_id=1)

        self.assertEqual(Motif, type(new_mutated_motif))
        self.assertEqual(seed_motif.v_call, new_mutated_motif.v_call)
        self.assertEqual(seed_motif.j_call, new_mutated_motif.j_call)
        self.assertEqual(len(seed_motif.seed), len(new_mutated_motif.seed))
