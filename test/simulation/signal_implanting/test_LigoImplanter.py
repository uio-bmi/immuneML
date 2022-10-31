import shutil
from unittest import TestCase

from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.SequenceType import SequenceType
from immuneML.simulation.LIgOSimulationItem import LIgOSimulationItem
from immuneML.simulation.generative_models.OLGA import OLGA
from immuneML.simulation.implants.Motif import Motif
from immuneML.simulation.implants.Signal import Signal
from immuneML.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation
from immuneML.simulation.signal_implanting.LigoImplanter import LigoImplanter
from immuneML.util.PathBuilder import PathBuilder


class TestLigoImplanter(TestCase):

    def test_make_repertoires(self):

        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'ligo_implanter')

        signals = [Signal('s1', [Motif('m1', GappedKmerInstantiation(), 'AA')], None)]
        implanter = LigoImplanter(LIgOSimulationItem(signals, 'sim_item1', 0.3, is_noise=False, seed=1,
                                                     generative_model=OLGA.build_object(**{"default_model_name": 'humanTRB', 'model_path': None,
                                                                                           'use_only_productive': False}),
                                                     number_of_examples=5, receptors_in_repertoire_count=20), SequenceType.AMINO_ACID, signals,
                                  sequence_batch_size=1000, seed=1, export_p_gens=True, keep_p_gen_dist=True, remove_seqs_with_signals=True,
                                  max_iterations=100, p_gen_bin_count=5)
        repertoires = implanter.make_repertoires(path)

        self.assertEqual(len(repertoires), 5)
        for repertoire in repertoires:
            self.assertTrue(isinstance(repertoire, Repertoire))
            self.assertEqual(len(repertoire.sequences), 30)
            self.assertTrue(all(repertoire.get_attribute("p_gen") > 0))

        shutil.rmtree(path)
