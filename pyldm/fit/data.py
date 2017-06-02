#!/usr/bin/env python
"""
    PyLDM - Lifetime Density Analysis
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
from scipy.signal import gaussian
from scipy.interpolate import interp1d,interp2d
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
        self.data_dechirped = np.copy(self.data)
 	self.times = np.array(self.times)
    	self.wls = np.array(self.wls)

        if 0 in self.times:
            self.izero = np.where(self.times==0)[0]
        else:
            print("Data set does not contain 0 time point. Setting start time to first positive time point.")
            self.izero = np.where(self.times > 0)[0][0]
	self.wls_work = np.copy(self.wls)
	self.data_work = np.copy(self.data_dechirped[self.izero:,:])
	self.times_work = np.copy(self.times[self.izero:])
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
	d = plt.contourf(self.wls_work, self.times_work, self.data_work)
        plt.yscale('symlog', linthreshy=1)
        plt.ylabel('Time')
        plt.xlabel('Wavelength')
        plt.colorbar()
        plt.draw()

    def updateBounds(self, wl_lb, wl_ub, t0, t):
	self.wls_work = np.copy(self.wls[wl_lb:wl_ub])
	self.times_work = np.copy(self.times[t0:t])
	self.data_work = np.copy(self.data_dechirped[t0:t, wl_lb:wl_ub])
	self.U, self.S, self.Vt = np.linalg.svd(self.data_work, full_matrices=False)

    def updateIRF(self, order, fwhm, munot, mus, lamnot):
        self.chirporder = order
        self.FWHM = fwhm
        self.munot = munot
        self.mu = mus
        self.lamnot = lamnot

    def fitchirp(self):
        f = interp2d(self.wls, self.times, self.data)
        if self.times[-1] < 400:
            n = 400
        else:
            n = self.times[-1]
        time_interp = np.linspace(self.times[0], self.times[-1], 2*n)
        spacing = time_interp[1]-time_interp[0]
        data_interp = f(self.wls, time_interp)
        sig = 0.1*spacing
        IRF_norm = np.exp(-time_interp**2/(2*sig**2))*np.amax(np.absolute(data_interp))
        delay_shift = np.zeros([len(self.wls_work)])
        for j in range(len(self.wls_work)):
            cor = np.correlate(data_interp[:, j], IRF_norm, "full")
            delay_shift[j] = np.argmax(np.diff(np.absolute(cor)))

        delay_shift -= np.amax([len(data_interp), len(IRF_norm)])
        delay_shift += 1
        delay_shift *= spacing

        params = [self.munot, self.lamnot]
        for i in range(self.chirporder):
            params.append(self.mu[i])
        if self.chirporder == 1:
            p_opt, p_cov = curve_fit(self._fit_func1, self.wls_work, delay_shift, p0=params, maxfev=10000)
        if self.chirporder == 2:
            p_opt, p_cov = curve_fit(self._fit_func2, self.wls_work, delay_shift, p0=params, maxfev=10000)
        if self.chirporder == 3:
            p_opt, p_cov = curve_fit(self._fit_func3, self.wls_work, delay_shift, p0=params, maxfev=10000)
        if self.chirporder == 4:
            p_opt, p_cov = curve_fit(self._fit_func4, self.wls_work, delay_shift, p0=params, maxfev=10000)
        if self.chirporder == 5:
            p_opt, p_cov = curve_fit(self._fit_func5, self.wls_work, delay_shift, p0=params, maxfev=10000)
        self.munot = p_opt[0]
        self.lamnot = p_opt[1]
        self.mu = p_opt[2:]
        self._chirp_correct()
        return delay_shift

    def _fit_func1(self, wl, mu_0, lam_0, mu_i):
        f = mu_0 + mu_i*((wl - lam_0)/100)
        return f

    def _fit_func2(self, wl, mu_0, lam_0, mu_i, mu_i2):
        f = mu_0 + mu_i*((wl - lam_0)/100) + mu_i2*((wl - lam_0)/100)**2
        return f

    def _fit_func3(self, wl, mu_0, lam_0, mu_i, mu_i2, mu_i3):
        f = mu_0 + mu_i*((wl - lam_0)/100) + mu_i2*((wl - lam_0)/100)**2 + mu_i3*((wl - lam_0)/100)**3
        return f

    def _fit_func4(self, wl, mu_0, lam_0, mu_i, mu_i2, mu_i3, mu_i4):
        f = mu_0 + mu_i*((wl - lam_0)/100) + mu_i2*((wl - lam_0)/100)**2 + mu_i3*((wl - lam_0)/100)**2 + mu_i4*((wl - lam_0)/100)**4
        return f

    def _fit_func5(self, wl, mu_0, lam_0, mu_i, mu_i2, mu_i3, mu_i4, mu_i5):
        f = mu_0 + mu_i*((wl - lam_0)/100) + mu_i2*((wl - lam_0)/100)**2 + mu_i3*((wl - lam_0)/100)**2 + mu_i4*((wl - lam_0)/100)**2 + mu_i5*((wl - lam_0)/100)**5
        return f

    def plot_chirp(self, delay_shift):
        self.fig_chirp_fit = plt.figure()
        self.fig_chirp_fit.canvas.set_window_title('Fit of Chirp to Autocorrelation')
        chirp = self._get_chirp()
        plt.plot(self.wls_work, delay_shift)
        plt.plot(self.wls_work, chirp)
        plt.draw()
        plt.show()

    def _chirp_correct(self):
        chirp = self._get_chirp()
        for i in range(len(self.wls)):
            chirped_time = self.times - chirp[i]
            f = interp1d(chirped_time, self.data[:,i], kind="linear", bounds_error=False, fill_value=(0,0))
            self.data_dechirped[:,i] = f(self.times)

    def _get_chirp(self):
        if self.chirporder == 1:
            chirp = self._fit_func1(self.wls, self.munot, self.lamnot, self.mu[0])
        if self.chirporder == 2:
            chirp = self._fit_func2(self.wls, self.munot, self.lamnot, self.mu[0], self.mu[1])
        if self.chirporder == 3:
            chirp = self._fit_func3(self.wls, self.munot, self.lamnot, self.mu[0], self.mu[1], self.mu[2])
        if self.chirporder == 4:
            chirp = self._fit_func4(self.wls, self.munot, self.lamnot, self.mu[0], self.mu[1], self.mu[2], self.mu[3])
        if self.chirporder == 5:
            chirp = self._fit_func5(self.wls, self.munot, self.lamnot, self.mu[0], self.mu[1], self.mu[2], self.mu[3], self.mu[4])
        return chirp

    def get_SVD(self):
	return self.U, self.S, self.Vt

    def get_IRF(self):
	return self.chirporder, self.FWHM, self.munot, self.mu, self.lamnot

    def get_T(self):
	return self.times_work

    def get_wls(self):
	return self.wls_work

    def get_data(self):
	return self.data_work

    def set_GA_taus(self, taus):
	pass
	
