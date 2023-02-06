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
        signal = Signal("sig1", [Motif("motif1", GappedKmerInstantiation(max_gap=0), "CASSLPSYQNTEAFF",
                                       v_call="TRBV5-1", j_call="TRBJ1-1")],
                        implanting_strategy)

        repertoire = Repertoire.build(["CSAIGQGKGAFYGYTF", "CASSLDRVSASGANVLTF", "CASSVQPRSEVPNTGELFF"],
                                      v_genes=["TRBV20-1", "TRBV4-1", "TRBV11-3"],
                                      j_genes=["TRBJ1-2", "TRBJ2-6", "TRBJ2-2"],
                                      region_types=[RegionType.IMGT_JUNCTION for _ in range(3)],
                                      counts=[1] * 3,
                                      path=path)

        new_repertoire = signal.implant_to_repertoire(repertoire, 0.33, path)

        self.assertEqual(len(repertoire.sequences), len(new_repertoire.sequences))
        self.assertEqual(1, len([seq for seq in new_repertoire.sequences if
                                 seq.amino_acid_sequence not in repertoire.get_sequence_aas()]))
        self.assertEqual(2, len([seq for seq in new_repertoire.sequences if
                                 seq.amino_acid_sequence in repertoire.get_sequence_aas()]))
        self.assertEqual(new_repertoire.get_region_type(), RegionType.IMGT_JUNCTION)

        shutil.rmtree(path)
