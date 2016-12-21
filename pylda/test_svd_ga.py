import unittest
import numpy as np
from fit.data import Data
from fit.svd_ga import SVD_GA

class TestSVD_GA(unittest.TestCase):
    def setUp(self):
        print("in setUp()")
        self.data = Data("data/test_mat.csv")
        self.data.updateIRF(0, 0, 0, 0)
        self.svd_ga = SVD_GA(self.data)
        self.U = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        self.S = np.diag([5, 4, 3, 2])
        self.Vt = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        self.taus = np.array([0.5, 1, 2, 4])
        self.D = np.array([[1, 1, 1, 1], [0.015625, 0.125, 0.353553390593273, 0.297301778750680],
                           [0.00390625, 0.0625, 0.25, 0.5], [0.0009765625, 0.03125, 0.176776695296636, 0.420448207626857]])

    def tearDown(self):
        print("in tearDown()")
        del self.data
        del self.svd_ga
        del self.U
        del self.S
        del self.Vt
        del self.taus
        del self.D

    def test_load(self):
        print("In test_load()")
        self.assertIsInstance(self.data, Data)

    def test_shape(self):
        print("In test_shape()")
        self.assertEqual(self.data.data.shape, (4, 4))

    def test_SVD_U(self):
        print("Testing correct SVD calculation")
        print("In test_SVD_U()")
        self.assertEqual(self.svd_ga.U.all(), self.U.all())

    def test_SVD_Vt(self):
        print("In test_SVD_Vt()")
        self.assertEqual(self.svd_ga.Vt.all(), self.Vt.all())
    
    def test_SVD_S(self):
        print("In test_SVD_S()")
        self.assertEqual(self.svd_ga.S.all(), self.S.all())

    def test_genD(self):
        print("In test_genD()")
        D = self.svd_ga._genD(self.taus, self.data.get_T())
        self.assertEqual(D.all(), self.D.all())


if __name__ == '__main__':
    unittest.main()
