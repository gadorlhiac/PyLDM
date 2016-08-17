#!/usr/bin/env python

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Tkinter as tk
import re
from tkFileDialog import askopenfilename
from Fit.lda import *
from Fit.svd import *
from Fit.data import *

class Main(tk.Frame):
    def __init__(self, master=None):
	self.GA_taus = None
	self.font_l = "Verdana 9"
        tk.Frame.__init__(self, master)
	self.mainFrame = tk.Frame(master)
	self.mainFrame.pack(side='top', fill='both', expand=True)
	self.topFrame = tk.Frame(self.mainFrame)
	self.topFrame.pack(side='top', fill='both', expand=True)
	self.bottomFrame = tk.Frame(self.mainFrame)
	self.bottomFrame.pack(side='bottom', fill='both', expand=True)

        self.dataWidgets()
	self.ldaWidgets()
	self.svdWidgets()
    
    def dataWidgets(self):
        self.tnotvar = tk.StringVar()
        self.tmaxvar = tk.StringVar()
        self.wl1var = tk.StringVar()
        self.wl2var = tk.StringVar()

        self.fwhmvar = tk.StringVar()
        self.fwhmvar.set('200')
        self.munotvar = tk.StringVar()
        self.munotvar.set('0')
        self.lamnotvar = tk.StringVar()
        self.lamnotvar.set('0')
        self.irfordervar = tk.StringVar()
        self.irfordervar.set('1')

	self.dataButtons = tk.Frame(self.topFrame)
	self.dataButtons.pack(side='top', fill='x')
        self.loadButton = tk.Button(self.dataButtons, text="Load File", command=self.load)
        self.loadButton.pack(side='left', anchor='w')
        self.quitButton = tk.Button(self.dataButtons, text="Quit", command=self.quit)
        self.quitButton.pack(side='right', anchor='e')

	self.dataFrame = tk.Frame(self.topFrame)
	self.dataFrame.pack(side='top', fill='both', expand=True)
	self.boundsFrame = tk.Frame(self.dataFrame)
	self.boundsFrame.pack(side='left', padx=10, pady=10)
	self.boundsTitle = tk.Label(self.boundsFrame, text='Time and Wavelength Bounds', font='%s underline' % self.font_l)
	self.boundsTitle.pack(side='top', fill='x')
	self.boundsLabels = tk.Frame(self.boundsFrame)
	self.boundsLabels.pack(side='left')
	self.boundsEntries = tk.Frame(self.boundsFrame)
	self.boundsEntries.pack(side='left')
	self.updateBoundsButton = tk.Button(self.boundsFrame, text="Update Bounds", command=self.updateBounds)
	self.updateBoundsButton.pack(side='left')

	self.tnotLabel = tk.Label(self.boundsLabels, text="Time Min", font=self.font_l)
	self.tnotLabel.pack(side='top', anchor='w')
	self.tmaxLabel = tk.Label(self.boundsLabels, text="Time Max", font=self.font_l)
	self.tmaxLabel.pack(side='top', anchor='w')
	self.wl1Label = tk.Label(self.boundsLabels, text="WL Min", font=self.font_l)
	self.wl1Label.pack(side='top', anchor='w')
	self.wl2Label = tk.Label(self.boundsLabels, text='WL Max', font=self.font_l)
	self.wl2Label.pack(side='top', anchor='w')

	self.tnot = tk.Entry(self.boundsEntries, textvariable=self.tnotvar, width=5)
	self.tnot.pack(side='top', anchor='w')
	self.tmax = tk.Entry(self.boundsEntries, textvariable=self.tmaxvar, width=5)
	self.tmax.pack(side='top', anchor='w')
	self.wl1 = tk.Entry(self.boundsEntries, textvariable=self.wl1var, width=5)
	self.wl1.pack(side='top', anchor='w')
	self.wl2 = tk.Entry(self.boundsEntries, textvariable=self.wl2var, width=5)
	self.wl2.pack(side='top', anchor='w')

	self.irfFrame = tk.Frame(self.dataFrame)
	self.irfFrame.pack(side='left', fill='both', padx=10, pady=10)
	self.irfTitle = tk.Label(self.irfFrame, text="IRF and Chirp Parameters", font='%s underline' % self.font_l)
	self.irfTitle.pack(side='top', fill='x')
	self.irfLabels = tk.Frame(self.irfFrame)
	self.irfLabels.pack(side='left', fill='both', expand=True)
	self.irfEntries = tk.Frame(self.irfFrame)
	self.irfEntries.pack(side='left', fill='both', expand=True)
	self.irfButtons = tk.Frame(self.irfFrame)
	self.irfButtons.pack(side='left', fill='both', expand=True)
	
	self.fwhmLabel = tk.Label(self.irfLabels, text='FWHM', font=self.font_l)
	self.fwhmLabel.pack(side='top', anchor='w')
	self.munotLabel = tk.Label(self.irfLabels, text=u'\u03bc 0')
	self.munotLabel.pack(side='top', anchor='w')
	self.lamnotLabel = tk.Label(self.irfLabels, text=u'\u03bb 0')
	self.lamnotLabel.pack(side='top', anchor='w')
	self.irforderLabel = tk.Label(self.irfLabels, text='Order')
	self.irforderLabel.pack(side='top', anchor='w')

	self.fwhmEntry = tk.Entry(self.irfEntries, textvariable=self.fwhmvar, width=5)
	self.fwhmEntry.pack(side='top', anchor='w')
	self.munotEntry = tk.Entry(self.irfEntries, textvariable=self.munotvar, width=5)
	self.munotEntry.pack(side='top', anchor='w')
	self.lamnotEntry = tk.Entry(self.irfEntries, textvariable=self.lamnotvar, width=5)
	self.lamnotEntry.pack(side='top', anchor='w')
	self.irforderEntry = tk.Entry(self.irfEntries, textvariable=self.irfordervar, width=5)
	self.irforderEntry.pack(side='top', anchor='w')

	self.updateIRFButton = tk.Button(self.irfButtons, text='Fix Values', command=self.updateIRF)
	self.updateIRFButton.pack(side='top', anchor='w')
	self.fitIRFButton = tk.Button(self.irfButtons, text='Fit IRF', command=self.fitIRF)
	self.fitIRFButton.pack(side='top', anchor='w')

    def ldaWidgets(self):
	self.regvar = tk.StringVar()
	self.regvar.set('L2')
	self.matvar = tk.StringVar()
	self.matvar.set('Id')
	self.taulinlogvar = tk.StringVar()
	self.taulinlogvar.set('log')
	self.simfitvar = tk.BooleanVar()
	self.simfitvar.set(True)

	self.ldaFrame = tk.Frame(self.bottomFrame)
	self.ldaFrame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
	
	self.ldaTitle = tk.Label(self.ldaFrame, text="LDA Options", font="%s underline" % self.font_l)
	self.ldaTitle.pack(side='top', fill='both')
	self.ldaRegandMat = tk.Frame(self.ldaFrame)
	self.ldaRegandMat.pack(side='top', fill='both', expand=True)

	self.ldaRegOptions = tk.Frame(self.ldaRegandMat)
	self.ldaRegOptions.pack(side='left', fill='both', expand=True)
	self.regLabel = tk.Label(self.ldaRegOptions, text='Regularization Method', font='Verdana 9')
	self.regLabel.pack(side='top')
	self.r1 = tk.Radiobutton(self.ldaRegOptions, text="Tikhonov", variable=self.regvar, value='L2',  anchor='w')
	self.r1.pack(side='top', anchor='w')
	self.r2 = tk.Radiobutton(self.ldaRegOptions, text="LASSO", variable=self.regvar, value='L1',  anchor='w')
	self.r2.pack(side='top', anchor='w')
	self.r3 = tk.Radiobutton(self.ldaRegOptions, text='Elastic Net', variable=self.regvar, value='elnet',  anchor='w')
	self.r3.pack(side='top', anchor='w')

	self.ldaMatOptions = tk.Frame(self.ldaRegandMat)
	self.ldaMatOptions.pack(side='right', fill='both', expand=True)
	self.matLabel = tk.Label(self.ldaMatOptions, text='Regularization Matrix', font='Verdana 9')
	self.matLabel.pack(side='top')

	self.m1 = tk.Radiobutton(self.ldaMatOptions, text='Identity', variable=self.matvar, value='Id')
	self.m1.pack(side='top', anchor='w')
	self.m2 = tk.Radiobutton(self.ldaMatOptions, text='1D', variable=self.matvar, value='1D')
	self.m2.pack(side='top', anchor='w')
	self.m3 = tk.Radiobutton(self.ldaMatOptions, text='2D', variable=self.matvar, value='2D')
	self.m3.pack(side='top', anchor='w')
        self.m4 = tk.Radiobutton(self.ldaMatOptions, text='Fused', variable=self.matvar, value='fused')
        self.m4.pack(side='top', anchor='w')

	self.ldaLifetimeOptions = tk.Frame(self.ldaFrame)
	self.ldaLifetimeOptions.pack(side='top', fill='both', expand=True, pady=10)
	self.ldaLifetimeOptionsTitle = tk.Label(self.ldaLifetimeOptions, text='Lifetime Settings')
	self.ldaLifetimeOptionsTitle.pack(side='top')
	self.ldaLifetimeLabels = tk.Frame(self.ldaLifetimeOptions)
	self.ldaLifetimeLabels.pack(side='left', fill='both', expand=True)
	self.ldaLifetimeEntries = tk.Frame(self.ldaLifetimeOptions)
	self.ldaLifetimeEntries.pack(side='left', fill='both', expand=True)
	self.ldaLifetimeLinvLog = tk.Frame(self.ldaLifetimeOptions)
	self.ldaLifetimeLinvLog.pack(side='left', fill='both', expand=True)

	self.tauminLabel = tk.Label(self.ldaLifetimeLabels, text='Min Tau', font=self.font_l)
	self.tauminLabel.pack(side='top', anchor='w')
	self.taumaxLabel = tk.Label(self.ldaLifetimeLabels, text='Max Tau', font=self.font_l)
	self.taumaxLabel.pack(side='top', anchor='w')
	self.numtauLabel = tk.Label(self.ldaLifetimeLabels, text='Num Taus', font=self.font_l)
	self.numtauLabel.pack(side='top', anchor='w')

	self.taumin = tk.Entry(self.ldaLifetimeEntries, text='taumin', width=8)
	self.taumin.pack(side='top', anchor='w')
	self.taumax = tk.Entry(self.ldaLifetimeEntries, text='taumax', width=8)
	self.taumax.pack(side='top', anchor='w')
	self.numtau = tk.Entry(self.ldaLifetimeEntries, text='numtau', width=8)
	self.numtau.pack(side='top', anchor='w')

	self.taulogbutton = tk.Radiobutton(self.ldaLifetimeLinvLog, text='Log Scale', font=self.font_l, variable=self.taulinlogvar, value='log')
	self.taulogbutton.pack(side='top', anchor='w')
	self.taulinbutton = tk.Radiobutton(self.ldaLifetimeLinvLog, text='Linear Scale', font=self.font_l, variable=self.taulinlogvar, value='lin')
	self.taulinbutton.pack(side='top', anchor='w')

	self.ldaAlphaOptions = tk.Frame(self.ldaFrame)
	self.ldaAlphaOptions.pack(side='top', fill='both', expand=True, pady=10)
	self.ldaAlphaOptionsTitle = tk.Label(self.ldaAlphaOptions, text='Alpha Settings')
	self.ldaAlphaOptionsTitle.pack(side='top')
	self.ldaAlphaLabels = tk.Frame(self.ldaAlphaOptions)
	self.ldaAlphaLabels.pack(side='left', fill='both', expand=True)
	self.ldaAlphaEntries = tk.Frame(self.ldaAlphaOptions)
	self.ldaAlphaEntries.pack(side='left', fill='both', expand=True)
	self.ldaAlphaExtra = tk.Frame(self.ldaAlphaOptions)
	self.ldaAlphaExtra.pack(side='left', fill='both', expand=True)

	self.alphaminLabel = tk.Label(self.ldaAlphaLabels, text='Alpha Min', font=self.font_l)
	self.alphaminLabel.pack(side='top', anchor='w')
	self.alphamaxLabel = tk.Label(self.ldaAlphaLabels, text='Alpha Max', font=self.font_l)
	self.alphamaxLabel.pack(side='top', anchor='w')
	self.numalphasLabel = tk.Label(self.ldaAlphaLabels, text='Num Alphas', font=self.font_l)
	self.numalphasLabel.pack(side='top', anchor='w')

	self.alphamin = tk.Entry(self.ldaAlphaEntries, text='alphamin', width=8)
	self.alphamin.pack(side='top', anchor='w')
	self.alphamax = tk.Entry(self.ldaAlphaEntries, text='alphamax', width=8)
	self.alphamax.pack(side='top', anchor='w')
	self.numalphas = tk.Entry(self.ldaAlphaEntries, text='numalphas', width=8)
	self.numalphas.pack(side='top', anchor='w')

	self.alphasimfitYes = tk.Radiobutton(self.ldaAlphaExtra, text='Simultaneous', font=self.font_l, variable=self.simfitvar, value=True)
	self.alphasimfitYes.pack(side='top', anchor='w')
	self.alphasimfitNo = tk.Radiobutton(self.ldaAlphaExtra, text='Independently', font=self.font_l, variable=self.simfitvar, value=False)
	self.alphasimfitNo.pack(side='top', anchor='w')
	self.mat_ldaButton = tk.Button(self.ldaAlphaExtra, text='Run LDA', command=self.mat_lda)
	self.mat_ldaButton.pack(side='top', anchor='w')

	self.ldaTSVDOptions = tk.Frame(self.ldaFrame)
	self.ldaTSVDOptions.pack(side='top', fill='both', expand=True, pady=10)
	self.ldaTSVDTitle = tk.Label(self.ldaTSVDOptions, text='TSVD')
	self.ldaTSVDTitle.pack(side='top')
	self.ldaTSVDLabels = tk.Frame(self.ldaTSVDOptions)
	self.ldaTSVDLabels.pack(side='left', fill='both', expand=True)
	self.ldaTSVDEntries = tk.Frame(self.ldaTSVDOptions)
	self.ldaTSVDEntries.pack(side='left', fill='both', expand=True)
	self.ldaTSVDExtra = tk.Frame(self.ldaTSVDOptions)
	self.ldaTSVDExtra.pack(side='left', fill='both', expand=True)
	
	self.tsvdKLabel = tk.Label(self.ldaTSVDLabels, text='K')
	self.tsvdKLabel.pack(side='top', anchor='w')
        self.tsvdTauminLabel = tk.Label(self.ldaTSVDLabels, text='Tau Min')
        self.tsvdTauminLabel.pack(side='top', anchor='w')
        self.tsvdTaumaxLabel = tk.Label(self.ldaTSVDLabels, text='Tau Max')
        self.tsvdTaumaxLabel.pack(side='top', anchor='w')
        self.tsvdNtLabel = tk.Label(self.ldaTSVDLabels, text='Num Taus')
        self.tsvdNtLabel.pack(side='top', anchor='w')
	self.tsvdK = tk.Entry(self.ldaTSVDEntries, text='tsvdK', width=8)
	self.tsvdK.pack(side='top', anchor='w')
        self.tsvdTaumin = tk.Entry(self.ldaTSVDEntries, width=8)
        self.tsvdTaumin.pack(side='top', anchor='w')
        self.tsvdTaumax = tk.Entry(self.ldaTSVDEntries, width=8)
        self.tsvdTaumax.pack(side='top', anchor='w')
        self.tsvdNt = tk.Entry(self.ldaTSVDEntries, width=8)
        self.tsvdNt.pack(side='top', anchor='w')
	self.tsvdButton = tk.Button(self.ldaTSVDExtra, text='Run TSVD LDA', command=self.tsvd_lda)
	self.tsvdButton.pack(side='top', anchor='w')

    def svdWidgets(self):
	self.svdAlphavar = tk.StringVar()
	self.svdAlphavar.set('0')

	self.svdFrame = tk.Frame(self.bottomFrame)
	self.svdFrame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        self.svdTitle = tk.Label(self.svdFrame, text="SVD and GA Options", font="%s underline" % self.font_l)
	self.svdTitle.pack(side='top', fill='both')

	self.svdButtons = tk.Frame(self.svdFrame)
	self.svdButtons.pack(side='top', fill='both')
        self.svdButton = tk.Button(self.svdButtons, text="SVD", command=self.svd_init)
        self.svdButton.pack(side='left')
	self.GAButton = tk.Button(self.svdButtons, text="GA", command=self.svd_GA)
	self.GAButton.pack(side='right')

	self.svdOptions = tk.Frame(self.svdFrame)
	self.svdOptions.pack(side='top', fill='both')
	self.svdLabels = tk.Frame(self.svdOptions)
	self.svdLabels.pack(side='left', fill='both', expand=True)
	self.svdAlphaLabel = tk.Label(self.svdLabels, text="Alpha", font=self.font_l)
	self.svdAlphaLabel.pack(side='top', anchor='w')
	self.svdwLSVsLabel = tk.Label(self.svdLabels, text="wLSVs", font=self.font_l)
	self.svdwLSVsLabel.pack(side='top', anchor='w')
	self.svdInitLabel = tk.Label(self.svdLabels, text="Initial Guess", font=self.font_l)
	self.svdInitLabel.pack(side='top', anchor='w')
	self.svdBoundsLabel = tk.Label(self.svdLabels, text="Bounds", font=self.font_l)
	self.svdBoundsLabel.pack(side='top', anchor='w')

	self.svdEntries = tk.Frame(self.svdOptions)
	self.svdEntries.pack(side='left', fill='both', expand=True)
	self.svdAlpha = tk.Entry(self.svdEntries, textvariable=self.svdAlphavar)
	self.svdAlpha.pack(side='top', fill='x')
	self.svdwLSVs = tk.Entry(self.svdEntries, text="wLSVs")
	self.svdwLSVs.pack(side='top', fill='x')
	self.svdInit = tk.Entry(self.svdEntries, text="Initial")
	self.svdInit.pack(side='top', fill='x')
	self.svdBounds = tk.Entry(self.svdEntries, text="Bounds")
	self.svdBounds.pack(side='top', fill='x')

	self.truncDataOptions = tk.Frame(self.svdFrame)
	self.truncDataOptions.pack(side='top', fill='both', pady=10)
	self.truncDataTitle = tk.Label(self.truncDataOptions, text="Low Rank Data Approximation", font='%s underline' % self.font_l)
	self.truncDataTitle.pack(side='top', fill='x')
	self.truncDataButton = tk.Button(self.truncDataOptions, text="Truncate", command=self._trunc_data)
	self.truncDataButton.pack(side='top')

    def load(self):    
        f = askopenfilename()
        self.data = Data(f)
        self.tnotvar.set(str(self.data.izero[0]))
        self.tmaxvar.set('-1')
        self.wl1var.set('0')
        self.wl2var.set('-1')
        self.updateIRF()
	self.SVD = SVD(self.data)
	self.LDAnalyzer = LDA(self.data)
        self.data.display()
	plt.show()

    def updateBounds(self):
	self.data.updateBounds(int(self.wl1var.get()), int(self.wl2var.get()), int(self.tnotvar.get()), int(self.tmaxvar.get()))
	self.LDAnalyzer.updateData(self.data)
	self.SVD.updateData(self.data)
	plt.close(self.data.fig_raw_data)
	self.data.display()
	plt.show()
	plt.close()

    def svd_init(self):
	self.SVD.display()
	plt.show()
	plt.close()

    def svd_GA(self):
	plt.close()
	bounds = self.svdBounds.get()
	if bounds != '':
	    bounds = [(float(re.split('[( \s )]', x)[1]), float(re.split('[( \s )]', x)[2])) for x in bounds.split(',')]
	else:
	    bounds = None
	x0 = map(float, self.svdInit.get().split())
	print x0[0]
        self.GA_taus = self.SVD.Global(self.svdwLSVs.get(), x0, bounds, float(self.svdAlphavar.get()))
	plt.show()
	plt.close()

    def _trunc_data(self):
	wLSVs = self.svdwLSVs.get()
	wLSVs = map(int, wLSVs.split(' '))
	self.data.truncData(wLSVs)
	self.updateBounds()

    def mat_lda(self):
	if self.taulinlogvar.get() == 'log':
	    taus = np.logspace(float(self.taumin.get()), float(self.taumax.get()), float(self.numtau.get()))
	else:
	    taus = np.linspace(float(self.taumin.get()), float(self.taumax.get()), float(self.numtau.get()))
	
	alphas = np.linspace(float(self.alphamin.get()), float(self.alphamax.get()), float(self.numalphas.get()))
	reg = self.regvar.get()
	L = self.getL(taus)
	simfit = self.simfitvar.get()
	self.LDAnalyzer.updateParams(taus, alphas, reg, L, simfit)
	self.LDAnalyzer.run_LDA(self.GA_taus)
	plt.show()
	plt.close()

    def tsvd_lda(self):
	self.LDAnalyzer.run_tsvd(int(self.tsvdK.get()), float(self.tsvdTaumin.get()), float(self.tsvdTaumax.get()), float(self.tsvdNt.get()), self.GA_taus)
        plt.show()
	plt.close()

    def updateIRF(self):
	self.data.updateIRF(int(self.irfordervar.get()), float(self.fwhmvar.get()), float(self.munotvar.get()), float(self.lamnotvar.get()))
    
    def fitIRF(self):
	self.data.updateIRF(int(self.irfordervar.get()), float(self.fwhmvar.get()), float(self.munotvar.get()), float(self.lamnotvar.get()))
	self.data.fitIRF()

    def getL(self, taus):
	I = np.identity(len(taus))
	if self.matvar.get() == 'Id':
	    return I
	elif self.matvar.get() == '1D':
	    L = I
	    for j in range(len(I)-2):
		L[j, j+1] = -1
	    return (0.5)*L
	elif self.matvar.get() == '2D':
	    L = (-2)*I
	    for j in range(len(I)-2):
		L[j+1, j] = 1
		L[j, j+1] = 1
	    return (0.25)*L
	elif self.matvar.get() == 'fused':
	    F = I
	    for j in range(len(I)-2):
		F[j, j+1] = -1
	    L = np.concatenate((I, F))
	    return L
	

