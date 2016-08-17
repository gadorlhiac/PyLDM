#!/usr/bin/env python

import numpy as np
from numpy.linalg import *
from scipy import *
from scipy.interpolate import *
from scipy.special import *
from scipy.linalg import solve_triangular
from matplotlib import *
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
from matplotlib.widgets import Slider


class LDA(object):
    def __init__(self, data):
	self.taus = np.logspace(-1, 4, 100)
        self.L = np.identity(len(self.taus))

	self.updateData(data)
	self.reg = 'L2'
	self.simfit = True #Fit all wavelengths simultaneously or individually

        #For hyperparameter selection
	self.rhos = np.linspace(0, 1, 10)

    def run_LDA(self):
	if self.reg == 'L2':
	    self._L2()
	    #GCVs, Cps = self._L2_GCV_Cp()
	    lcurve_x, lcurve_y, mpm_y, k = self._lcurve_MPM()
            self._plot_lcurve_MPM(lcurve_x, lcurve_y, mpm_y, k.argmin(), mpm_y.argmin())
            print self.x_opts[:, :, 0]
	    self._plot_LDM()

	elif self.reg == 'L1':
	    pass
	elif self.reg == 'elnet':
	    pass
	return

    ##################
    # Data Functions #
    ##################

    def updateData(self, data):
	self.A = data.get_data()
	self.times = data.get_T()
	self.wls = data.get_wls()
	self.irforder, self.FWHM, self.munot, self.lamnot = data.get_IRF()
        self.FWHM_mod = self.FWHM/(2*sqrt(log(2)))
	self.genD()

    def updateParams(self, taus, alphas, reg, L, simfit):
	self.taus = taus
	self.alphas = alphas
        print self.alphas
	self.reg = reg
	self.L = L
	self.simfit = simfit
	self.genD()
        self.x_opts = np.zeros([len(self.taus), len(self.wls), len(self.alphas)])


    ############################
    # Initialization Functions #
    ############################

    # Matrix of Exponential Decays
    def genD(self):
        D = np.zeros([len(self.times), len(self.taus)])
        for i in range(len(D)):
            for j in range(len(D[i])):
                t = self.times[i]
                tau = self.taus[j]
                #One = 0.5*(exp(-t/tau)*exp((self.munot + (self.FWHM_mod**2/(2*tau)))/tau))
                #Two = 1 + erf((t-(self.munot+(self.FWHM_mod**2/tau)))/(sqrt(2)*self.FWHM_mod))
                #D[i, j] = One*Two
		D[i, j] = exp(-t/tau)
	self.D = D

    ######################
    # Tikhonov Functions #
    ######################

    def _L2(self):
	for alpha in range(len(self.alphas)):
            D_aug = np.concatenate((self.D, alpha**(0.5)*self.L))
            A_aug = np.concatenate((self.A, np.zeros([len(self.taus), len(self.wls)])))
            Q, R = np.linalg.qr(D_aug)
            Qt = np.transpose(Q)
            QtA = Qt.dot(A_aug)
	    R_inv = self._R_inv(R)
	    #R_inv = solve_triangular(R, np.identity(len(R)))

	    self.x_opts[:, :, alpha] = R_inv.dot(QtA)
	    H = self.D.dot(R_inv.dot(Qt))
	    #self.x_opts[:, :, alpha] = solve_triangular(R, QtA)

            #x_alpha = np.zeros([len(D_aug[0]),len(A_aug[0])])

            #x_alpha[-1, :] = QtA[-1, :]/R[-1, -1]
            #for i in range(len(x_alpha)-2, -1, -1):
            #    s = 0
            #    for k in range(i+1, len(x_alpha)):
            #        s += R[i, k]*x_alpha[k, :]
            #    x_alpha[i, :] = (QtA[i, :] - s)/R[i, i]
            #self.x_opts[:, :, alpha] = x_alpha

    def _R_inv(self, R):
	I = np.identity(len(R))
	R_inv = np.zeros([len(R), len(R)])
	R_inv[-1, :] = I[-1, :]/R[-1, -1]
        for i in range(len(R_inv)-2, -1, -1):
            s = 0
            for k in range(i+1, len(R_inv)):
                s += R[i, k]*R_inv[k, :]
            R_inv[i, :] = (I[i, :] - s)/R[i, i]
        return R_inv

    def _calc_H(self, alpha):
	#D_aug = np.concatenate((self.D, np.zeros([len(self.times), len(self.times)])), axis=1)
	H = self.D.dot(self._TikRegInv(self.alphas[alpha]))
	return H

    def _L2_GCV_Cp(self):
        n = len(self.times)
        if self.simfit:
            #As an estimator of true variance, the error variance of most complex model is usually used (the lowest alpha)
	    self.var = sum((self.D.dot(self.x_opts[:, :, 0])-self.A)**2)/n
	    GCVs = np.array([self._calc_GCV(a) for a in range(len(self.alphas))])
	    Cps = np.array([self._calc_Cp(a) for a in range(len(self.alphas))])
	    return GCVs, Cps
	else:
            GCVs = np.array([[self._calc_GCV(a, wl) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
	    Cps = np.array([[self._calc_Cp(a, wl) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
	    return GCVs, Cps

    def _calc_GCV(self, alpha, wl=None):
	H = self._calc_H(alpha)
        n = len(self.times)
	df = np.trace(H)
	tr = (1 - df/n)**2
	res = self._calc_res(alpha, wl)
	return (res/tr)/n

    def _calc_Cp(self, alpha, wl=None):
        n = len(self.times)
	if wl != None:
	    self.var = sum((self.D.dot(self.x_opts[:, wl, 0])-self.A[:, wl])**2)/n
	res = self._calc_res(alpha, wl)
	H = self._calc_H(alpha)
	df = np.trace(H)
	return res + 2*self.var*df

    def _lcurve_MPM(self):
	if self.simfit:
	    lcurve_x = np.array([sqrt(self._calc_res(a)) for a in range(len(self.alphas))])
	    lcurve_y = np.array([self._calc_smoothNorm(a) for a in range(len(self.alphas))])
	    mpm_y = lcurve_x*lcurve_y
	else:
	    lcurve_x = np.array([[self._calc_res(a, wl) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
	    lcurve_y = np.array([[self._calc_smoothNorm(a, wl) for wl in range(len(self.wls))] for a in range(len(self.alphas))])
	    mpm_y = lcurve_x*lcurve_y
        k = self._calc_k(lcurve_x, lcurve_y)
        return lcurve_x, lcurve_y, mpm_y, k

    #Curvature
    def _calc_k(self, lx, ly):
        d1y = np.array([(ly[a]-ly[a+1])/(lx[a]-lx[a+1]) for a in range(len(ly)-1)])
        d2y = np.array([(d1y[a]-d1y[a+1])/(lx[a]-lx[a+1]) for a in range(len(ly)-2)])
        k = abs(d2y)/(1+d1y[:-1]**2)**(3/2)
        return k

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

    def _L1(self):
	for alpha in range(len(self.alphas)):
            D_aug = np.concatenate((self.D, alpha**(0.5)*self.L))
            A_aug = np.concatenate((self.A, np.zeros([len(self.taus), len(self.wls)])))
            Q, R = np.linalg.qr(D_aug)
            Qt = np.transpose(Q)
            QtA = Qt.dot(A_aug)

            x_alpha = np.zeros([len(D_aug[0]),len(A_aug[0])])

            x_alpha[-1, :] = QtA[-1, :]/R[-1, -1]
            for i in range(len(x_alpha)-2, -1, -1):
                s = 0
                for k in range(i+1, len(x_alpha)):
                    s += R[i, k]*x_alpha[k, :]
            x_alpha[i, :] = (QtA[i, :] - s)/R[i, i]
            self.x_opts[:, :, alpha] = x_alpha
        return self.x_opt, self.alpha_list

    def L1_min(self, P, wl, alpha):
        A = np.transpose(np.array(self.data[self.t1:self.t2,wl], ndmin=2))
        x = np.transpose(np.array(P, ndmin=2))
        res_1 = sum((self.matA.dot(x) - A)**2)
        res_2 = norm(alpha*self.L.dot(x))
        return res_1+res_2

    def L1_GCV(self, wl):
        S = np.transpose(np.array(self.data[self.t1:self.t2,wl], ndmin=2))
        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            result = sciopt.minimize(self.L1_min, x0=self.x_opt[:,wl], args=(wl, alpha))
            x_alpha = np.transpose(np.array(result.x, ndmin=2))
            self.x_opts[:, i] = squeeze(x_alpha)

            W = diag(abs(x_alpha[:,0]))
            W_ = inv(W)
            Sg = self.matA.dot(x_alpha)
    
            n = len(S)
            H = self.matA.dot(inv(np.transpose(self.matA).dot(self.matA) + alpha*W_)).dot(np.transpose(self.matA))
            p = np.trace(H)
            res_1 = sum((Sg-S)**2)
            tr = (1-p/n)**2
            GCV = (res_1/tr)/n
            self.GCVs[i, wl] = GCV
        return self.alphas[self.GCVs[:, wl].argmin()], self.x_opts[:, self.GCVs.argmin()]

    # In progress
    def L1_Cp(self, wl):
        A = np.transpose(np.array(self.data[self.t1:self.t2,wl], ndmin=2))
        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            result = sciopt.minimize(self.L1_min, x0=self.x_opt[:,wl], args=(wl, alpha))
            x_alpha = np.transpose(np.array(result.x, ndmin=2))
            self.x_opts[:, i] = squeeze(x_alpha)

            Ag = self.matA.dot(x_alpha)
            W = diag(abs(x_alpha[:,0]))
            W_ = inv(W)
            Ag = self.matA.dot(x_alpha)

            n = len(A)
            df = np.trace(inv(np.transpose(self.matA).dot(self.matA) + alpha*W_).dot(np.transpose(self.matA)))
            #df = np.trace(np.cov(zip(squeeze(Ag),squeeze(A))))/var
            var = sum((A - Ag)**2)/len(A)
            
            Cp = sum((A-Ag)**2) + 2*var*df
            self.CPs[i, wl] = Cp
        return self.alphas[self.CPs[:, wl].argmin()], self.x_opts[:,self.CPs.argmin()]


    #########################
    # Elastic Net Functions #
    #########################

    def elastic_net(self, alpha, rho):
        for wl in self.wls_fit:
            S = np.transpose(np.array(self.data[self.t1:self.t2,wl], ndmin=2))
            result = sciopt.minimize(self.elnet_min, x0=self.x_opt[:,wl], args=(wl, alpha, rho, self.matA, S))
            self.x_opt[:, wl] = result.x
            #if wl % 10 == 0:
            #    print wl
        return self.x_opt

    def elnet_CV(self):
        alphas = np.linspace(0.0001, 0.01, 100)
        rhos = np.linspace(0.1,0.9,30)
        CVs = np.zeros([len(alphas),len(rhos)])
        for wl in self.wls_fit:
            for i in range(len(alphas)):
                for j in range(len(rhos)):
                    CVs[i, j] = self.CV_min(wl, alphas[i], rhos[j])
            
            a, r = np.unravel_index(CVs.argmin(), CVs.shape)
            alpha_hat = alphas[a]
            rho_hat = rhos[r]
            self.alpha_list[wl] = alpha_hat
            self.rho_list[wl] = rho_hat

            S = np.transpose(np.array(self.data[self.t1:self.t2,wl], ndmin=2))
            result = sciopt.minimize(self.elnet_min, x0=self.x_opt[:,wl], args=(wl, alpha_hat, rho_hat, self.matA, S))
            self.x_opt[:,wl] = result.x
            #if wl % 10 == 0:
            #print alpha_hat
            #print rho_hat
            #print wl
        return self.x_opt, self.alpha_list, self.rho_list

    def CV_min(self, wl, alpha, rho):
        CV = 0
        for i in range(self.t2-self.t1):
            A_validation = self.matA[i,:]
            A_train = np.delete(self.matA, (i), axis=0)
            S_validation = self.data[self.t1+i,wl]
            S_train = np.transpose(np.array(self.data[self.t1:self.t2,wl], ndmin=2))
            S_train = np.delete(S_train, (i), axis=0)
            result = sciopt.minimize(self.elnet_min, x0=self.x_opt[:,wl], args=(wl, alpha, rho, A_train, S_train))
            x = result.x
            self.x_opt[:,wl] = x
            Ax_validation = A_validation.dot(x)
            CV += squeeze((Ax_validation - S_validation)**2)
        return CV/(self.t2-self.t1)

    def elnet_min(self, P, wl, alpha, rho, A, S):
        x = np.transpose(np.array(P, ndmin=2))
        OLS = sum((A.dot(x) - S)**2)
        L1 = rho*norm(self.L.dot(x))
        L2 = (1-rho)*(sum(self.L.dot(x)**2))
        return OLS + alpha*(L1+L2)


    ##################
    # TSVD Functions #
    ##################

    def run_tsvd(self, k, t1, t2, nt):
        self.taus = np.logspace(t1, t2, nt)
        self.genD()
        x = self._tsvd(k)
        fig_tsvd = plt.figure()
	fig_tsvd.canvas.set_window_title('TSVD LDM')
	Contour_Levels=[-0.001,-0.00075, -0.0005, -0.00025, -0.0001, -0.000075, -0.00005,  0,
                0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
	ax = fig_tsvd.add_subplot(111)
        ax.contourf(self.wls, self.taus, x, cmap=plt.cm.seismic)#, levels=Contour_Levels)
	ax.set_yscale('log')
        plt.draw()

    def _tsvd(self, k):
	D_plus = self._tsvdInv(k)	
	x_k = D_plus.dot(self.A)
	return x_k

    def _tsvdInv(self, k):
	U, S, Vt = np.linalg.svd(self.D, full_matrices=True)
        U = U[:, :len(self.times)]
        Vt = Vt[:len(self.times), :]
	V = np.transpose(Vt)
	Ut = np.transpose(U)
        print len(S)
        print len(U)
        print len(V)
	S = 1/S
	S = np.array([S[i] if i < k else 0 for i in range(len(S))])
	S = np.diag(S)
	return V.dot(S).dot(Ut)

    def _picard(self):
        pass


    #################################
    # Output and Plotting Functions #
    #################################
    
    def display(self):
        pass



    def _plot_lcurve_MPM(self, lx, ly, my, kmin, mymin):
        fig_lcurve = plt.figure()
        fig_lcurve.canvas.set_window_title('L-Curve and MPM')
        ax = fig_lcurve.add_subplot(121)
        ax.plot(lx, ly, 'bo-')
        ax.plot(lx[kmin], ly[kmin], 'ro')
        ax.annotate(self.alphas[kmin], (lx[kmin], ly[kmin]))
        ax2 = fig_lcurve.add_subplot(122)
        ax2.plot(self.alphas, my)
        ax2.plot(self.alphas[mymin], my[mymin])
        ax2.annotate(self.alphas[mymin], (self.alphas[mymin], my[mymin]))
        plt.draw()

    def _plot_GCV_Cp(self, GCVs, Cps):
	pass

    def _plot_LDM(self):
	fig_ldm = plt.figure()
	fig_ldm.canvas.set_window_title('LDM and Contour Selection')
	ax = fig_ldm.add_subplot(121)
	Contour_Levels=[-0.001,-0.00075, -0.0005, -0.00025, -0.0001, -0.000075, -0.00005,  0,
                0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
        ax.contourf(self.wls, self.taus, self.x_opts[:,:,0], cmap=plt.cm.seismic, levels=Contour_Levels)
	ax.set_yscale('log')
	ax2 = fig_ldm.add_subplot(122)
	plt.subplots_adjust(left=0.25, bottom=0.25)
	axS = plt.axes([0.25, 0.1, 0.65, 0.03])
        S = Slider(axS, 'alpha', 1, len(self.alphas), valinit=1, valfmt='%0.0f')
        def update(val):
            n = int(S.val)
	    ax.clear()
	    ax.contourf(self.wls, self.taus, self.x_opts[:, :, n], cmap=plt.cm.seismic, levels=Contour_Levels)
	    ax.set_yscale('log')
	    ax.legend(loc=0, frameon=False)
	    plt.draw()
        S.on_changed(update)
	plt.draw()

