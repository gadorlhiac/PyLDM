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
import numpy as np
from numpy.linalg import *
from scipy import *
from scipy.sparse.linalg import eigs
#from scipy.interpolate import *
#from scipy.special import *
#from scipy.linalg import rq
from matplotlib import *
import matplotlib.pyplot as plt
#import scipy.optimize as sciopt
from matplotlib.widgets import Slider


class LDA(object):
    def __init__(self, data):
        self.taus = np.logspace(-1, 4, 100)
        self.L = np.identity(len(self.taus))

        self.updateData(data)
        self.reg = 'L2'
        self.simfit = True #Fit all wavelengths simultaneously or individually

        #For hyperparameter selection in Elastic Net
        self.rhos = np.linspace(0.1, 0.9, 9)

    def run_LDA(self, GA_taus=None):
        if self.reg == 'L2':
            GCVs, Cps = self._L2()
            lcurve_x, lcurve_y, k = self._lcurve()
            self._plot_lcurve(lcurve_x, lcurve_y, k)
            self._plot_GCV_Cp(Cps, GCVs)
        elif self.reg == 'L1':
            Cps = self._L1()
            l1x, l1y, k = self._l1curve()
            self._plot_lcurve(l1x, l1y, k)
            self._plot_GCV_Cp(Cps)
        elif self.reg == 'elnet':
            self._elnet()
        self._plot_LDM(GA_taus)

    def replot(self, GA_taus=None, num_c=10):
        plt.close(self.fig_ldm)
        self._plot_LDM(GA_taus, num_c)

    #####################################
    # Data and Initialization Functions #
    #####################################

    # Get data and IRF parameters
    def updateData(self, data):
        self.A = data.get_data()
        self.times = data.get_T()
        self.wls = data.get_wls()
        self.chirporder, self.FWHM, self.munot, self.mu, self.lamnot = data.get_IRF()
        self.FWHM_mod = self.FWHM/(2*sqrt(log(2)))
        if self.FWHM != 0:
            self.wl_mus = self._calc_mu()
        self.genD()

    # Get matrix LDA parameters
    def updateParams(self, taus, alphas, reg, L, simfit):
        self.taus = taus
        self.alphas = alphas
        self.reg = reg
        self.L = L
        self.simfit = simfit
        self.genD()
        self.x_opts = np.zeros([len(self.taus), len(self.wls), len(self.alphas)])

    # Calculate Wavelength Dependent mu for chirp correction
    def _calc_mu(self):
        mu = np.tile(self.munot, len(self.wls))
        for i in range(len(self.mu)):
            mu += self.mu[i]*((self.wls - self.lamnot))**(i+1)
        return mu

    # Matrix of Exponential Decays
    def genD(self):
        D = np.zeros([len(self.times), len(self.taus)])
        for i in range(len(D)):
            for j in range(len(D[i])):
                t = self.times[i]
                tau = self.taus[j]
                if self.FWHM_mod != 0:
                    One = 0.5*(exp(-t/tau)*exp(self.FWHM_mod**2/(2*tau))/tau)
                    Two = 1 + erf((t-(self.FWHM_mod**2/tau))/(sqrt(2)*self.FWHM_mod))
                    D[i, j] = One*Two
                else:
                    D[i, j] = exp(-t/tau)
        self.D = np.nan_to_num(D)

    ######################
    # Tikhonov Functions #
    ######################

    # Calculate Tikhonov solutions for all wavelengths, and all alphas
    # Runs GCV and Cp, either independently or for all wavelengths simultaneously
    def _L2(self):
        if self.simfit:
            GCVs = np.zeros([len(self.alphas)])
            Cps = np.zeros([len(self.alphas)])
        else:
            GCVs = np.zeros([len(self.wls), len(self.alphas)])
            Cps = np.zeros([len(self.wls), len(self.alphas)])

        for alpha in range(len(self.alphas)):
            self.x_opts[:, :, alpha] = self._solve_L2(self.alphas[alpha])
            H, S = self._calc_H_and_S(self.alphas[alpha])
            if alpha == 0:
                n = len(self.times)
                self.var = sum((self.D.dot(self.x_opts[:, :, 0])-self.A)**2)/n
            GCVs[alpha] = self._calc_GCV(alpha, H)
            Cps[alpha] = self._calc_Cp(alpha, S)
        return GCVs, Cps

    def _solve_L2(self, alpha):
        if alpha != 0:
            D_aug = np.concatenate((self.D, alpha**(0.5)*self.L))
            A_aug = np.concatenate((self.A, np.zeros([len(self.L), len(self.wls)])))
        else:
            D_aug = self.D
            A_aug = self.A
        U, S, Vt = np.linalg.svd(D_aug, full_matrices=False)
        V = np.transpose(Vt)
        Ut = np.transpose(U)
        Sinv = np.diag(1/S)
        x_opt = V.dot(Sinv).dot(Ut).dot(A_aug)
        return x_opt

    def _calc_H_and_S(self, alpha):
        X = np.transpose(self.D).dot(self.D) + alpha*np.transpose(self.L).dot(self.L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Xinv = np.transpose(Vt).dot(np.diag(1/S)).dot(np.transpose(U))
        H = self.D.dot(Xinv).dot(np.transpose(self.D))
        S = Xinv.dot(np.transpose(self.D).dot(self.D))
        return H, S

    # Calculates GCV
    def _calc_GCV(self, alpha, H):
        n = len(self.times)
        I = np.identity(len(H))
        tr = (np.trace(I - H)/n)**2
        if self.simfit:
            res = self._calc_res(alpha)
        else:
            res = np.array([self._calc_res(alpha, wl) for wl in range(len(self.wls))])
        return res/tr

    # Calculates Cp
    def _calc_Cp(self, alpha, S, wl=None):
        n = len(self.times)
        if wl is not None:
            self.var = sum((self.D.dot(self.x_opts[:, wl, 0])-self.A[:, wl])**2)/n
        res = self._calc_res(alpha, wl)
        df = np.trace(S)
        return res + 2*self.var*df

    # Stores the L-Curve and MPM values
    def _lcurve(self):
        if self.simfit:
            lcurve_x = np.array([sqrt(self._calc_res(a)) for a in range(len(self.alphas))])
            lcurve_y = np.array([self._calc_smoothNorm(a) for a in range(len(self.alphas))])
        else:
            lcurve_x = np.array([[sqrt(self._calc_res(a, wl)) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
            lcurve_y = np.array([[self._calc_smoothNorm(a, wl) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
        k = self._calc_k(lcurve_x, lcurve_y)

        return lcurve_x, lcurve_y,  k

    # Curvature function, for finding optimal alpha on L-curve
    def _calc_k(self, lx, ly):
        #dx = np.gradient(lx)
        #dy = np.gradient(ly, dx)
        #d2y = np.gradient(dy, dx)
        #k = abs(d2y)/(1+dy**2)**(1.5)
        da = np.gradient(self.alphas)
        dx = np.gradient(lx, da)
        dy = np.gradient(ly, da)
        d2x = np.gradient(dx, da)
        d2y = np.gradient(dy, da)
        k = (dx*d2y - d2x*dy)/(dx**2 + dy**2)**(1.5)
        return k

    # Residuals and norms
    def _calc_res(self, alpha, wl=None):
        if wl == None:
            return sum((self.D.dot(self.x_opts[:, :, alpha])-self.A)**2)
        else:
            return sum((self.D.dot(self.x_opts[:, wl, alpha])-self.A[:, wl])**2)

    def _calc_smoothNorm(self, alpha, wl=None):
        if wl == None:
            return sum((self.L.dot(self.x_opts[:, :, alpha]))**2)**(0.5)
        else:
            return sum((self.L.dot(self.x_opts[:, wl, alpha]))**2)**(0.5)


    ###################
    # Lasso Functions #
    ###################

    # Find LASSO for each alpha
    def _L1(self):
        if self.simfit:
            Cps = np.zeros([len(self.alphas)])
        else:
            Cps = np.zeros([len(self.wls), len(self.alphas)])

        G,C = self._L2() # Throw away G, C, simply initialize x_opts to have a start guess that is the Tikhonov solution
        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            self.x_opts[:, :, i] = self._L1_min(self.D, self.A, alpha)
            Cps[i] = self._calc_L1_Cp(i)
        return Cps

    # Giving same result for first and last alpha ???
    # Does the regularized least squares after converting to orthogonal design matrix
    def _L1_min(self, D, A, alpha):
        p = len(D[0])
        Dt = np.transpose(D)
        cov = Dt.dot(D)
        g, v = eigs(cov, k=1, ncv=len(D))
        I = np.identity(p)
        B = g*I - cov
        if self.reg == 'elnet':
            x = self.x_opts[:, :, 0, 0]
        else:
            x = self.x_opts[:, :, 0]
        cond = np.array([1])
        for i in range(len(x)):
            for j in range(len(x[0])):
                cond = np.array([1])
                while cond > 1e-8 and x[i, j] != 0: # Can change tolerance here
                    x_old = np.copy(x)
                    U = Dt.dot(A[:, j]) + B.dot(x_old[:, j])
                    sgn = np.sign(U[i])
                    absolute = np.absolute(U[i])
                    x_new = sgn*np.maximum((absolute-alpha)/g, 0)
                    x[i, j] = np.real(x_new)
                    cond = (x[i, j]-x_old[i, j])/x_old[i, j]
        return x

    def _calc_L1_Cp(self, alpha, wl=None):
        n = len(self.times)
        if wl is not None:
            self.var = sum((self.D.dot(self.x_opts[:, wl, 0])-self.A[:, wl])**2)/n
        res = self._calc_res(alpha, wl)
        X = np.transpose(self.D).dot(self.D) + self.alphas[alpha]*np.transpose(self.L).dot(self.L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Xinv = np.transpose(Vt).dot(np.diag(1/S)).dot(np.transpose(U))
        S = Xinv.dot(np.transpose(self.D).dot(self.D))
        df = np.trace(S)
        return res + 2*self.var*df

    def _l1curve(self):
        if self.simfit:
            l1x = np.array([sqrt(self._calc_res(a)) for a in range(len(self.alphas))])
            l1y = np.array([self._calc_L1Norm(a) for a in range(len(self.alphas))])
        else:
            l1x = np.array([[sqrt(self._calc_res(a, wl)) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
            l1y = np.array([[self._calc_L1Norm(a, wl) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
        k = self._calc_k(l1x, l1y)
        return l1x, l1y, k

    def _calc_L1Norm(self, alpha, wl=None):
        if wl == None:
            return sum(abs(self.L.dot(self.x_opts[:, :, alpha])))
        else:
            return sum(abs(self.L.dot(self.x_opts[:, wl, alpha])))


    #########################
    # Elastic Net Functions #
    #########################

    # Calculate Elastic Net Solution for every alpha and rho
    # First creates augmented matrices to remove L2 penalty, reducing problem to a LASSO regularization
    def _elnet(self):
        G, C = self._L2()
        x = self.x_opts[:, :, 0]
        self.x_opts = np.zeros([len(self.taus), len(self.wls), len(self.alphas), len(self.rhos)])
        self.x_opts[:, :, 0, 0] = x
        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            for j in range(len(self.rhos)):
                rho = self.rhos[j]
                a1 = rho*alpha
                a2 = (1-rho)*alpha
                atil = a1/(sqrt(1+a2))

                D_aug = np.concatenate((self.D, sqrt(a2)*self.L))
                D_aug *= (1 + sqrt(a2))**(-.5)

                A_aug = np.concatenate((self.A, np.zeros([len(self.L), len(self.wls)])))

                x_naive = self._L1_min(D_aug, A_aug, atil)
                self.x_opts[:, :, i, j] = (1 + a2)*x_naive

    ##################
    # TSVD Functions #
    ##################

    # Public function for running truncated svd regularization
    def run_tsvd(self, k, t1, t2, nt, GA_taus):
        self.taus = np.logspace(t1, t2, nt)
        self.genD()
        x = self._tsvd(k)
        fig_tsvd = plt.figure()
        fig_tsvd.canvas.set_window_title('TSVD LDM')
        max_c = np.max(np.absolute(x))
        num_c = 12
        C_pos = np.linspace(0, max_c, num_c)
        C_neg = np.linspace(-max_c, 0, num_c, endpoint=False)
        Contour_Levels = np.concatenate((C_neg, C_pos))
        ax = fig_tsvd.add_subplot(111)
        C = ax.contourf(self.wls, self.taus, x, cmap=plt.cm.seismic, levels=Contour_Levels)
        if GA_taus is not None:
            for i in range(len(GA_taus)):
                ax.axhline(GA_taus[i], linestyle='dashed', color='k')
        ax.set_yscale('log')
        plt.colorbar(C)
        ax.set_ylabel(r'$\tau$', fontsize=14)
        ax.set_xlabel('Wavelength', fontsize=14)
        ax.set_title('K = %i' % k)
        plt.draw()

    # Actual solution
    def _tsvd(self, k):
        D_plus = self._tsvdInv(k)
        x_k = D_plus.dot(self.A)
        return x_k

    # Truncated inverse
    def _tsvdInv(self, k):
        U, S, Vt = np.linalg.svd(self.D, full_matrices=False)
        V = np.transpose(Vt)
        Ut = np.transpose(U)
        S = 1/S
        S = np.array([S[i] if i < k else 0 for i in range(len(S))])
        S = np.diag(S)
        return V.dot(S).dot(Ut)

    # Picard condition for K selection
    def _picard(self):
        pass


    #################################
    # Output and Plotting Functions #
    #################################

    def display(self):
        pass

    def _plot_lcurve(self, lx, ly, k):
        fig_lcurve = plt.figure()
        fig_lcurve.canvas.set_window_title('L-Curve')
        kmax = k.argmax()
        ax = fig_lcurve.add_subplot(121)
        ax.plot(lx, ly, 'bo-')
        ax.plot(lx[kmax], ly[kmax], 'ro')
        ax.set_xlabel('Residual Norm', fontsize=14)
        ax.set_ylabel('Smoothing Norm', fontsize=14)
        ax.annotate(self.alphas[kmax], (lx[kmax], ly[kmax]))
        ax.set_title('L-curve', fontsize=16)
        ax2 = fig_lcurve.add_subplot(122)
        ax2.plot(self.alphas, k, 'bo-')
        ax2.plot(self.alphas[kmax], k[kmax], 'ro')
        ax2.set_xlabel('Alpha', fontsize=14)
        ax2.set_title('Curvature', fontsize=16)
        ax2.annotate(self.alphas[kmax], (self.alphas[kmax], k[kmax]))
        plt.draw()

    def _plot_GCV_Cp(self, Cps, GCVs=None):
        fig_gcv = plt.figure()
        fig_gcv.canvas.set_window_title('Cps')
        ax = fig_gcv.add_subplot(121)
        ax.plot(self.alphas, Cps, 'bo-')
        Cpmin = Cps.argmin()
        ax.plot(self.alphas[Cpmin], Cps[Cpmin], 'ro')
        ax.annotate(self.alphas[Cpmin], (self.alphas[Cpmin], Cps[Cpmin]))
        ax.set_xlabel('Alpha', fontsize=14)
        ax.set_title('Cp', fontsize=16)
        if GCVs is not None:
            fig_gcv.canvas.set_window_title('Cps and GCVs')
            ax2 = fig_gcv.add_subplot(122)
            ax2.plot(self.alphas, GCVs, 'bo-')
            GCVmin = GCVs.argmin()
            ax2.plot(self.alphas[GCVmin], GCVs[GCVmin], 'ro')
            ax2.annotate(self.alphas[GCVmin], (self.alphas[GCVmin], GCVs[GCVmin]))
            ax2.set_title('GCV', fontsize=16)
            ax2.set_xlabel('Alpha', fontsize=14)
        plt.draw()

    def _plot_LDM(self, GA_taus=None, num_c=10):
        self.fig_ldm = plt.figure()
        self.fig_ldm.canvas.set_window_title('LDM')

        # LDM
        self.ax = self.fig_ldm.add_subplot(121)
        self.ax.set_yscale('log')
        self.ax.set_ylabel(r'$\tau$', fontsize=14)
        self.ax.set_xlabel('Wavelength', fontsize=14)
        self.ax.set_title('Alpha = %f' % self.alphas[0])

        if self.reg == 'elnet':
            C_pos = np.linspace(0, np.max(self.x_opts[:, :, 0, 6]), num_c)
            C_neg = np.linspace(np.min(self.x_opts[:, :, 0, 6]), 0, num_c, endpoint=False)
            Contour_Levels = np.concatenate((C_neg, C_pos))
            self.C = self.ax.contourf(self.wls, self.taus, self.x_opts[:, :, 0, 6], cmap=plt.cm.seismic, levels=Contour_Levels)
        else:
            C_pos = np.linspace(0, np.max(self.x_opts[:, :, 0]), num_c)
            C_neg = np.linspace(np.min(self.x_opts[:, :, 0]), 0, num_c, endpoint=False)
            Contour_Levels = np.concatenate((C_neg, C_pos))
            self.C = self.ax.contourf(self.wls, self.taus, self.x_opts[:, :, 0], cmap=plt.cm.seismic, levels=Contour_Levels)

        # Need to hold on to the colorbar, and it's axis.
        # Because it is the second axis created, it will be second in the fig
        # axes.  It needs to be manually cleared and reupdated.
        self.cbar = self.fig_ldm.colorbar(self.C)
        self.cbar_ax = self.fig_ldm.axes[1]

        # Wavelength trace plot
        self.ax2 = self.fig_ldm.add_subplot(122)
        self.ax2.set_title('Wavelength = %f' % self.wls[0])
        self.ax2.set_xscale('log')
        self.ax2.set_xlabel(r'$\tau$', fontsize=14)
        self.ax2.set_ylabel('Amplitude', fontsize=14)
        self.ax2.yaxis.set_label_position('right')
        self.ax2.yaxis.tick_right()
        self.ax2.yaxis.label.set_rotation(270)

        if self.reg == 'elnet':
            self.ax2.plot(self.taus, self.x_opts[:, 0, 0, 6])
        else:
            self.ax2.plot(self.taus, self.x_opts[:, 0, 0])

        # Add GA lifetimes to plot
        if GA_taus is not None:
            for i in range(len(GA_taus)):
    	        self.ax.axhline(GA_taus[i], linestyle='dashed', color='k')

        # Add contour levels
        if len(Contour_Levels) > 0:
            for i in range(len(Contour_Levels)):
                self.ax2.axhline(Contour_Levels[i], linestyle='dashed', color='k')

        # Make space for sliders
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Alpha slider
        self.axS = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.S = Slider(self.axS, 'alpha', 0, len(self.alphas), valinit=0)
        self.S.valtext.set_visible(False)

        # Wavelength slider
        self.axS2 = plt.axes([0.25, 0.015, 0.65, 0.03])
        self.S2 = Slider(self.axS2, 'Wavelength', 0, len(self.wls), valinit=0)
        self.S2.valtext.set_visible(False)

        # Rho slider
        if self.reg == 'elnet':
            self.axS3 = plt.axes([0.25, 0.055, 0.65, 0.03])
        if self.reg == 'elnet':
            self.S3 = Slider(self.axS3, 'rho', 0, len(self.rhos), valinit=6)
            self.S3.valtext.set_visible(False)

        r = 6
        a = 0
        wl = 0
        def update(val):
            a = int(self.S.val)
            wl = int(self.S2.val)

            self.ax.clear()
            self.ax2.clear()
            self.cbar_ax.clear()

            if self.reg == 'elnet':
                C_pos = np.linspace(0, np.max(self.x_opts[:, :, a, r]), num_c)
                C_neg = np.linspace(np.min(self.x_opts[:, :, a, r]), 0, num_c, endpoint=False)
                Contour_Levels = np.concatenate((C_neg, C_pos))
                self.C = self.ax.contourf(self.wls, self.taus, self.x_opts[:, :, a, r], cmap=plt.cm.seismic, levels=Contour_Levels)
                self.ax2.plot(self.taus, self.x_opts[:, wl, a, r])
            else:
                C_pos = np.linspace(0, np.max(self.x_opts[:, :, a]), num_c)
                C_neg = np.linspace(np.min(self.x_opts[:, :, a]), 0, num_c, endpoint=False)
                Contour_Levels = np.concatenate((C_neg, C_pos))
                self.C = self.ax.contourf(self.wls, self.taus, self.x_opts[:, :, a], cmap=plt.cm.seismic, levels=Contour_Levels)
                self.ax2.plot(self.taus, self.x_opts[:, wl, a])

            self.cbar = self.fig_ldm.colorbar(self.C, self.cbar_ax)

            self.ax.set_title('Alpha = %f' % self.alphas[a])
            self.ax.set_ylabel(r'$\tau$', fontsize=16)
            self.ax.set_xlabel('Wavelength', fontsize=16)
            self.ax.set_yscale('log')

            self.ax2.set_title('Wavelength = %f' % self.wls[wl])
            self.ax2.set_xscale('log')
            self.ax2.set_xlabel(r'$\tau$', fontsize=14)
            self.ax2.set_ylabel('Amplitude', fontsize=14)
            self.ax2.yaxis.set_label_position('right')
            self.ax2.yaxis.tick_right()
            self.ax2.yaxis.label.set_rotation(270)

            if GA_taus is not None:
                for i in range(len(GA_taus)):
                    self.ax.axhline(GA_taus[i], linestyle='dashed', color='k')

            if len(Contour_Levels) > 1:
                for i in range(len(Contour_Levels)):
                    self.ax2.axhline(Contour_Levels[i], linestyle='dashed', color='k')
            self.fig_ldm.canvas.draw_idle()

        self.S.on_changed(update)
        self.S2.on_changed(update)
        if self.reg == 'elnet':
            self.S3.on_changed(update)
        plt.show()
