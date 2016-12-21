import unittest
import numpy as np
from fit.data import Data

"""
These tests are to verify that data is loaded properly and simple manipulations are performed correctly.
"""

class TestData(unittest.TestCase):
    def setUp(self):
        print("in setUp()")
        self.D = Data("data/fulldata_noise10.csv")
        self.D.updateBounds(11, 50, 1, 40) #Define bounds of working data set (wl0, wl1, t0, t1)

    def tearDown(self):
        print("In tearDown()")
        del self.D

    def test_load(self):
        print("In test_load()")
        self.assertIsInstance(self.D, Data)
    
    def test_shape(self):
        print("In test_shape()")
        self.assertEqual(self.D.data.shape, (100, 300))

    def test_update(self):
        print("In test_update()")
        self.assertEqual(self.D.data_work[3, 10], self.D.data[4, 21])

    def test_update2(self):
        print("In test_update2()")
        self.assertEqual(self.D.data_work[-5, -4], self.D.data[35, 46])

    def test_update3(self):
        print("In test_update3()")
        self.D.updateBounds(0, -1, 0, -1) #See if you can return to original full data set
        self.assertEqual(self.D.data_work[10, 10], self.D.data[10, 10])

if __name__ == '__main__':
    unittest.main()
