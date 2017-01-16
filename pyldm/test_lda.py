import unittest
import numpy as np
from fit.data import Data
from fit.svd_ga import SVD_GA

class TestSVD_GA(unittest.TestCase):
    ##############################################################
    # These tests are run using a 4x4 matrix for which explicit  #
    # solutions and decompositions are known, and compared with  #
    # the package's solutions.  See the github wiki for details. #
    ##############################################################
    def setUp(self):
        print("in setUp()")
        self.data = Data("data/test_mat.csv")
        self.data.updateIRF(0, 0, 0, 0)
        self.svd_ga = SVD_GA(self.data)
        self.U = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        self.S = np.diag([5, 4, 3, 2])
        self.Vt = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        self.taus = np.array([0.5, 1, 2, 4])
        #self.D = np.array([[1, 1, 1, 1], [0.015625, 0.125, 0.353553390593273, 0.297301778750680],
        #                   [0.00390625, 0.0625, 0.25, 0.5], [0.0009765625, 0.03125, 0.176776695296636, 0.420448207626857]])
        self.D = np.array([[1, 1, 1, 1], [1./64., 1./8., 2**(0.5)/4., 2**(0.25)/4.],
                           [1./256., 1./16., 1./4., 1./2.], [1./1024., 1./32., 2**(0.5)/8., 8**(0.25)/4]])

    def tearDown(self):
        print("in tearDown()")
        del self.data
        del self.svd_ga
        del self.U
        del self.S
        del self.Vt
        del self.taus
        del self.D
        del self.wLSVs

    def test_load(self):
        print("In test_load()")
        self.assertIsInstance(self.data, Data)

    def test_shape(self):
        print("In test_shape()")
        self.assertEqual(self.data.data.shape, (4, 4))

if __name__ == '__main__':
    unittest.main()
