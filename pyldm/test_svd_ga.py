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
        self.taus = np.array([0.5, 1, 2, 4]) # Defined (not solved for) lifetimes for test cases
        self.D = np.array([[1, 1, 1, 1], [1./64., 1./8., 2**(0.5)/4., 2**(0.25)/2.],
                           [1./256., 1./16., 1./4., 1./2.], [1./1024., 1./32., 2**(0.5)/8., 8**(0.25)/4]])
        self.wLSVs = self.U.dot(self.S)

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

    def test_SVD_U(self):
        print("Testing correct SVD calculation")
        print("In test_SVD_U()")
        self.assertEqual(self.svd_ga.U[3, 1], self.U[3, 1])

    def test_SVD_U2(self):
        print("In test_SVD_U2()")
        self.assertEqual(self.svd_ga.U[-2, -2].all(), self.U[-2, -2])

    def test_SVD_Vt(self):
        print("In test_SVD_Vt()")
        self.assertAlmostEqual(self.svd_ga.Vt[1, 3], self.Vt[1, 3])

    def test_SVD_Vt(self):
        print("In test_SVD_Vt2()")
        self.assertAlmostEqual(self.svd_ga.Vt[0, 0], self.Vt[0, 0])
    
    def test_SVD_S(self):
        print("In test_SVD_S()")
        self.assertAlmostEqual(self.svd_ga.S[1, 0], self.S[1, 0])

    def test_SVD_S2(self):
        print("In test_SVD_S2()")
        self.assertAlmostEqual(self.svd_ga.S[0, 2], self.S[0, 2])

    def test_genD(self):
        print("In test_genD()")
        D = self.svd_ga._genD(self.taus, self.data.get_T())
        self.assertAlmostEqual(D[1, 1], self.D[1, 1])

    def test_genD2(self):
        print("In test_genD2()")
        D = self.svd_ga._genD(self.taus, self.data.get_T())
        self.assertAlmostEqual(D[-1, 3], self.D[-1, 3])

    def test_getDAS(self):
        print("\n\nTesting Matrix Calculations")
        print("in test_getDAS()")
        DAS = self.svd_ga._getDAS(self.svd_ga._genD(self.taus, self.data.get_T()), self.wLSVs, 0)
        approx = np.array([[-7.2300635698046480214, -480.32543356902219514, -162.65183225843831625, -139.13202790668617981],
                           [3.4575175629506989349, 931.32338000329043751, 279.90829362276597186, 293.15748848168345893],
                           [-1.7034983388326247186,-660.67013276722984717, -167.07728420200191694,-236.93548696561566506],
                           [0.47604434568657380511,209.67218633296160479, 49.820822837674261333,82.910026390618385946]])
        self.assertAlmostEqual(DAS[3,0], approx[3, 0], delta=1e-8)

    def test_getDAS_2(self):
        print("in test_getDAS_2()")
        DAS = self.svd_ga._getDAS(self.svd_ga._genD(self.taus, self.data.get_T()), self.wLSVs, 0)
        approx = np.array([[-7.2300635698046480214, -480.32543356902219514, -162.65183225843831625, -139.13202790668617981],
                           [3.4575175629506989349, 931.32338000329043751, 279.90829362276597186, 293.15748848168345893],
                           [-1.7034983388326247186,-660.67013276722984717, -167.07728420200191694,-236.93548696561566506],
                           [0.47604434568657380511,209.67218633296160479, 49.820822837674261333, 82.910026390618385946]])
        self.assertAlmostEqual(DAS[-1,-1], approx[-1, -1], delta=1e-8)

    def test_getDAS2(self):
        print("in test_getDAS2()")
        I = np.identity(4)
        DAS = self.svd_ga._getDAS(self.svd_ga._genD(self.taus, self.data.get_T()), I, 0)
        approx = np.array([[1.4460127139609296043, -54.217277419479438751, 120.08135839225554879, -69.566013953343089904],
                           [-0.69150351259013978697, 93.302764540921990619, -232.83084500082260938, 146.57874424084172946],
                           [0.34069966776652494372, -55.692428067333972313, 165.16753319180746179, -118.46774348280783253],
                           [-0.095208869137314761022, 16.606940945891420444, -52.418046583240401199, 41.455013195309192973]])
        self.assertAlmostEqual(DAS[2, 2], approx[2, 2], delta=1e-8)

    def test_getDAS2_2(self):
        print("in test_getDAS2_2()")
        I = np.identity(4)
        DAS = self.svd_ga._getDAS(self.svd_ga._genD(self.taus, self.data.get_T()), I, 0)
        approx = np.array([[1.4460127139609296043, -54.217277419479438751, 120.08135839225554879, -69.566013953343089904],
                           [-0.69150351259013978697, 93.302764540921990619, -232.83084500082260938, 146.57874424084172946],
                           [0.34069966776652494372, -55.692428067333972313, 165.16753319180746179, -118.46774348280783253],
                           [-0.095208869137314761022, 16.606940945891420444, -52.418046583240401199, 41.455013195309192973]])
        self.assertAlmostEqual(DAS[0, 0], approx[0, 0], delta=1e-8)

    def test_getDAS3(self):
        print("in test_getDAS3()")
        negI = -1*np.identity(4)
        DAS = self.svd_ga._getDAS(self.svd_ga._genD(self.taus, self.data.get_T()), negI, 0)
        approx = -1*np.array([[1.4460127139609296043, -54.217277419479438751, 120.08135839225554879, -69.566013953343089904],
                              [-0.69150351259013978697, 93.302764540921990619, -232.83084500082260938, 146.57874424084172946],
                              [0.34069966776652494372, -55.692428067333972313, 165.16753319180746179, -118.46774348280783253],
                              [-0.095208869137314761022, 16.606940945891420444, -52.418046583240401199, 41.455013195309192973]])
        self.assertAlmostEqual(DAS[1, 2], approx[1, 2], delta=1e-8)

    def test_getDAS3_2(self):
        print("in test_getDAS3_2()")
        negI = -1*np.identity(4)
        DAS = self.svd_ga._getDAS(self.svd_ga._genD(self.taus, self.data.get_T()), negI, 0)
        approx = -1*np.array([[1.4460127139609296043, -54.217277419479438751, 120.08135839225554879, -69.566013953343089904],
                              [-0.69150351259013978697, 93.302764540921990619, -232.83084500082260938, 146.57874424084172946],
                              [0.34069966776652494372, -55.692428067333972313, 165.16753319180746179, -118.46774348280783253],
                              [-0.095208869137314761022, 16.606940945891420444, -52.418046583240401199, 41.455013195309192973]])
        self.assertAlmostEqual(DAS[3, 1], approx[3, 1], delta=1e-8)

    def test_min(self):
        print ("in test_min()")
        res = self.svd_ga._min(self.taus, self.wLSVs, self.data.get_T(), 0)
        self.assertAlmostEqual(res, 1.58062e-23, delta=1e-8)

    def test_min2(self):
        print("in test_min2()")
        I = np.identity(4)
        res = self.svd_ga._min(self.taus, self.wLSVs, self.data.get_T(), 0)
        self.assertAlmostEqual(res, 0, delta=1e-8)

    def test_GA(self):
        print("in test_GA()")
        # Not tested as the only new feature is the scipy minimize function which
        # has been extensively tested.

    def test_get_wLSVs_for_fit(self):
        print("in test_get_wLSVs_for_fit()")
        indices, wLSVs = self.svd_ga._get_wLSVs_for_fit("1")
        self.assertEqual(wLSVs[1], self.wLSVs[1, 0])

    def test_get_wLSVs_for_fit2(self):
        print("in test_get_wLSVs_for_fit2()")
        indices, wLSVs = self.svd_ga._get_wLSVs_for_fit("1 2")
        self.assertEqual(wLSVs[1, 1], self.wLSVs[1, 1])

    def test_get_wLSVs_for_fit3(self):
        print("in test_get_wLSVs_for_fit3()")
        indices, wLSVs = self.svd_ga._get_wLSVs_for_fit("1 2 4")
        self.assertEqual(indices[-1], 4)



if __name__ == '__main__':
    unittest.main()
