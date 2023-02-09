import math
import shutil
from unittest import TestCase

from immuneML.data_model.receptor.RegionType import RegionType
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.signal_implanting_strategy.DecoyImplanting import DecoyImplanting
from immuneML.util.PathBuilder import PathBuilder


class TestDecoyImplanting(TestCase):
    def test_implant_in_repertoire(self):
        path = PathBuilder.build(EnvironmentSettings.tmp_test_path / "decoy_implanting/")

        implanting_strategy = DecoyImplanting()
        implanting_strategy.nr_of_decoys = 5
        implanting_strategy.repertoire_implanting_rate_per_decoy = 0.33
        implanting_strategy.dataset_implanting_rate_per_decoy = 1
        implanting_strategy.overwrite_sequences = False
        implanting_strategy.default_model_name = "humanTRB"

        signal_sequence = "CASSLPSYQNTEAFF"
        signal = Signal("sig1", [Motif("motif1", GappedKmerInstantiation(max_gap=0), signal_sequence,
                                       v_call="TRBV5-1", j_call="TRBJ1-1")],
                        implanting_strategy)

        repertoire = Repertoire.build(["CSAIGQGKGAFYGYTF", "CASSLDRVSASGANVLTF", "CASSVQPRSEVPNTGELFF"],
                                      v_genes=["TRBV20-1", "TRBV4-1", "TRBV11-3"],
                                      j_genes=["TRBJ1-2", "TRBJ2-6", "TRBJ2-2"],
                                      region_types=[RegionType.IMGT_JUNCTION for _ in range(3)],
                                      counts=[1] * 3,
                                      path=path)

        repertoire_implanting_rate = 0.33
        new_repertoire = signal.implant_to_repertoire(repertoire, repertoire_implanting_rate, path)

        self.assertEqual(len(repertoire.sequences) +
                         math.ceil(len(repertoire.sequences)*repertoire_implanting_rate) +
                         math.ceil(len(repertoire.sequences)*implanting_strategy.repertoire_implanting_rate_per_decoy)*implanting_strategy.nr_of_decoys,
                         len(new_repertoire.sequences))

        for seq in list(repertoire.get_sequence_aas()) + [signal_sequence]:
            self.assertIn(seq, new_repertoire.get_sequence_aas())

        self.assertEqual(new_repertoire.get_region_type(), RegionType.IMGT_JUNCTION)

        shutil.rmtree(path)
