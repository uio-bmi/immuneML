from unittest import TestCase

from source.simulation.motif_instantiation_strategy.GappedKmerInstantiation import GappedKmerInstantiation


class TestGappedKmerInstantiation(TestCase):
    def test_instantiate_motif(self):

        strategy = GappedKmerInstantiation({"max_hamming_distance": 2, "max_gap": 2, "min_gap": 1})
        # TODO: specify in the docs the values position_weights can take
        instance = strategy.instantiate_motif("C/AS", {
            "position_weights": {0: 1, 1: 0, 2: 0},
            "alphabet_weights": {"T": 0.5, "F": 0.5, "A": 0, "C": 0, "D": 0, "E": 0, "G": 0, "H": 0, "K": 0, "I": 0, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "V": 0, "W": 0, "Y": 0}
        })

        self.assertTrue("AS" in instance.instance)
        self.assertEqual(len(instance.instance), 4)
        self.assertTrue(1 <= instance.gap <= 2)

        strategy = GappedKmerInstantiation({"max_hamming_distance": 1})
        instance = strategy.instantiate_motif("CAS", params={
            "position_weights": {0: 1, 1: 0, 2: 0},
            "alphabet_weights": {"T": 0.5, "F": 0.5, "A": 0, "C": 0, "D": 0, "E": 0, "G": 0, "H": 0, "K": 0, "I": 0, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "V": 0, "W": 0, "Y": 0}
        })

        self.assertTrue("AS" in instance.instance)
        self.assertEqual(len(instance.instance), 3)
        self.assertTrue(instance.gap == 0)
        self.assertTrue(instance.instance[0] in ["C", "T", "F"])
