#!/usr/bin/env python
"""
    PyLDA - Lifetime Density Analysis
    Copyright (C) 2016 Gabriel Dorlhiac, Clyde Fare

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *
from scipy.optimize import curve_fit

class Data(object):
    def __init__(self, f_in):
        self.times = []
        self.data = []
        with open(f_in) as f:
            i = 0
            for line in f:
                if i == 0:
                    line = line.rstrip()
                    self.wls = line.split(',')
                    self.wls = list(map(float,self.wls[1:]))
                    i += 1
                else:
                    temp = line.split(',')
                    self.times.append(float(temp[0]))
                    self.data.append(list(map(float, temp[1:])))
	self.data = np.array(self.data)
 	self.times = np.array(self.times)
    	self.wls = np.array(self.wls)

	self.izero = np.where(self.times==0)[0]
	self.wls_work = self.wls
	self.data_work = self.data[self.izero:,:]
	self.times_work = self.times[self.izero:]
	self.U, self.S, self.Vt = np.linalg.svd(self.data_work, full_matrices=False)

    def truncData(self, wLSVs):
	U, S, Vt = np.linalg.svd(self.data, full_matrices=False)
	if wLSVs == None:
	    print('Error enter wLSVs')
	elif len(wLSVs) == 1:
	    S = np.diag(self.S)
	    self.data = U[:, :wLSVs[0]].dot(S[:wLSVs[0], :wLSVs[0]]).dot(Vt[:wLSVs[0],:]) 
	else:
	    Uprime = np.zeros([len(self.times), len(wLSVs)])
	    Vtprime = np.zeros([len(wLSVs), len(self.wls)])
	    Sprime = np.zeros([len(wLSVs)])
	    for j in range(len(wLSVs)):
		Uprime[:, j] = U[:, wLSVs[j]]
		Vtprime[j, :] = Vt[wLSVs[j], :]
		Sprime[j] = S[wLSVs[j]]
	    Sprime = diag(Sprime)
	    self.data = Uprime.dot(Sprime).dot(Vtprime)
	    
    
    def display(self):
        self.fig_raw_data = plt.figure()
        self.fig_raw_data.canvas.set_window_title('Raw Data')
	d = plt.contourf(self.wls, self.times, self.data)
        plt.yscale('symlog', linthreshy=1)
        plt.ylabel('Time')
        plt.xlabel('Wavelength')
        plt.colorbar()
        plt.draw()

    def updateBounds(self, wl_lb, wl_ub, t0, t):
	self.wls_work = self.wls[wl_lb:wl_ub]
	self.times_work = self.times[t0:t]
	self.data_work = self.data[t0:t, wl_lb:wl_ub]
	self.U, self.S, self.Vt = np.linalg.svd(self.data_work, full_matrices=False)

    def updateIRF(self, order, fwhm, munot, lamnot):
        self.irforder = order
        self.FWHM = fwhm
        self.munot = munot
        self.lamnot = lamnot

    def fitIRF(self):
        pass

    def _get_fit_func(self, x, order, mu_0, mu_i, lam_c):
        pass

    def get_SVD(self):
	return self.U, self.S, self.Vt

    def get_IRF(self):
	return self.irforder, self.FWHM, self.munot, self.lamnot

    def get_T(self):
	return self.times_work

    def get_wls(self):
	return self.wls_work

    def get_data(self):
	return self.data_work



    def set_GA_taus(self, taus):
	pass
	
