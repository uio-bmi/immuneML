import unittest

import numpy.testing as nptest
import numpy as np

from ..io import get_residue_data, get_structure
from ..testing import get_pdb_path

from tmtools import tm_align


def _coords_from_pdb(sname):
    pdb = get_pdb_path(sname)
    s = get_structure(pdb)
    c = next(s.get_chains())
    return get_residue_data(c)


class TestBindings(unittest.TestCase):
    def test_call_identical(self):
        # Given
        coords, seq = _coords_from_pdb("2gtl")

        # When
        res = tm_align(coords, coords, seq, seq)

        # Then
        nptest.assert_array_almost_equal(res.t, np.zeros(3))
        nptest.assert_array_almost_equal(res.u, np.eye(3))

    def test_call_different(self):
        # Given
        coords1, seq1 = _coords_from_pdb("2gtl")
        coords2, seq2 = _coords_from_pdb("7ok9")

        # Verified by running TMalign manually
        t_expected = np.array([10.42888594, 42.96954856, 74.43889102])
        u_expected = np.array(
            [
                [-0.93809129, -0.30032785, 0.17259179],
                [-0.21695917, 0.12102283, -0.96864968],
                [0.27002492, -0.94612719, -0.17868934],
            ]
        )
        tm_norm2 = 0.15158
        tm_norm1 = 0.38759

        # When
        res = tm_align(coords1, coords2, seq1, seq2)

        # Then
        nptest.assert_array_almost_equal(res.t, t_expected)
        nptest.assert_array_almost_equal(res.u, u_expected)
        self.assertAlmostEqual(res.tm_norm_chain1, tm_norm1, places=4)
        self.assertAlmostEqual(res.tm_norm_chain2, tm_norm2, places=4)
