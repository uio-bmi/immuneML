from unittest import TestCase

from source.encodings.pipeline.steps.FisherExactWrapper import FisherExactWrapper


class TestFisherExactWrapper(TestCase):
    def test_fisher_exact(self):
        fe = FisherExactWrapper()
        p_value = fe.fisher_exact([[8, 2], [1, 5]], FisherExactWrapper.TWO_SIDED)
        self.assertEqual(0.03497, round(p_value, 5))
