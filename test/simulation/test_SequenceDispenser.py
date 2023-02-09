import shutil
from unittest import TestCase

from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.SequenceDispenser import SequenceDispenser
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.util.PathBuilder import PathBuilder


class TestSequenceDispenser(TestCase):
    def test_generate_mutation(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "decoy_implanting/")

        repertoire = Repertoire.build(["CSAIGQGKGAFYGYTF"], v_genes=["TRBV20-1"], j_genes=["TRBJ1-2"],
                                      region_types=[RegionType.IMGT_JUNCTION for _ in range(3)],
                                      counts=[1] * 3, path=path)

        dataset = RepertoireDataset(repertoires=[repertoire])

        seq_disp = SequenceDispenser(dataset=dataset, occurrence_limit_pgen_range={1e-10: 2},
                                     mutation_hamming_distance=1)

        seed = Motif("motif1", GappedKmerInstantiation(max_gap=0), "CASSLPSYQNTEAFF",
                     v_call="TRBV5-1", j_call="TRBJ1-1",
                     mutation_position_possibilities={7: 1})

        seq_disp.add_seed_sequence(seed)

        new_mutated_seed = seq_disp.generate_mutation(repertoire_id=1)

        self.assertEqual(Motif, type(new_mutated_seed))
        self.assertEqual(seed.v_call, new_mutated_seed.v_call)
        self.assertEqual(seed.j_call, new_mutated_seed.j_call)
        self.assertEqual(len(seed.seed), len(new_mutated_seed.seed))

        shutil.rmtree(path)
