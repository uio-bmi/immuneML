from __future__ import division

import subprocess
import tmscoring
import numpy as np
from numpy.testing import assert_almost_equal, TestCase
from nose.exc import SkipTest
from shutil import which


class TestAligningBase(TestCase):
    def test_matrix(self):
        align_object = tmscoring.Aligning('pdb1.pdb', 'pdb2.pdb')
        np.random.seed(124)
        for _ in range(100):
            theta, phi, psi = 2 * np.pi * np.random.random(3)
            dx, dy, dz = 10 * np.random.random(3)

            matrix = align_object.get_matrix(theta, phi, psi, dx, dy, dz)
            rotation = matrix[:3, :3]
            assert_almost_equal(1, np.linalg.det(rotation), 6)
            assert_almost_equal(1, np.linalg.det(matrix), 6)

    def test_tm_valuex(self):
        align_object = tmscoring.Aligning('pdb1.pdb', 'pdb2.pdb')
        np.random.seed(124)
        for _ in range(100):
            theta, phi, psi = 2 * np.pi * np.random.random(3)
            dx, dy, dz = 10 * np.random.random(3)

            tm = align_object._tm(theta, phi, psi, dx, dy, dz)

            assert np.all(0 <= -tm / align_object.N)

    def test_load_data_alignment(self):
        align_object = tmscoring.Aligning('pdb1.pdb', 'pdb2.pdb', mode='align')
        assert align_object.coord1.shape[0] == 4
        assert align_object.coord2.shape[0] == 4
        assert align_object.coord1.shape == align_object.coord2.shape

    def test_load_data_index(self):
        align_object = tmscoring.Aligning('pdb1.pdb', 'pdb2.pdb', mode='index')
        assert align_object.coord1.shape[0] == 4
        assert align_object.coord2.shape[0] == 4
        assert align_object.coord1.shape == align_object.coord2.shape

def test_identity():
    sc = tmscoring.TMscoring('pdb1.pdb', 'pdb1.pdb')
    assert sc.tmscore(0, 0, 0, 0, 0, 0) == 1

    sc = tmscoring.RMSDscoring('pdb1.pdb', 'pdb1.pdb')
    assert sc.rmsd(0, 0, 0, 0, 0, 0) == 0.0


def test_tm_output():
    if which("TMscore") is None:
        raise SkipTest('TMscore is not installed in the system.')

    pdb1, pdb2 = 'pdb1.pdb', 'pdb2.pdb'
    sc = tmscoring.TMscoring(pdb1, pdb2)
    _, tm, rmsd = sc.optimise()

    p = subprocess.Popen('TMscore {} {} | grep TM-score | grep d0'.format(pdb1, pdb2), stdout=subprocess.PIPE, shell=True)
    ref_tm = float(p.communicate()[0].decode('utf-8').split('=')[1].split('(')[0])
    assert_almost_equal(ref_tm, tm, decimal=2)

    p = subprocess.Popen('TMscore {} {} | grep RMSD | grep common'.format(pdb1, pdb2),
                         stdout=subprocess.PIPE, shell=True)
    ref_rmsd = float(p.communicate()[0].decode('utf-8').split('=')[1])
    assert abs(ref_rmsd - rmsd) < 0.1

def test_repeated():
    pdb1, pdb2 = 'pdbrep_1.pdb', 'pdbrep_2.pdb'
    sc = tmscoring.TMscoring(pdb1, pdb2)
    _, tm, rmsd = sc.optimise()

    assert_almost_equal(tm, 0.27426501120343644)
    assert_almost_equal(rmsd, 15.940038528551929)
