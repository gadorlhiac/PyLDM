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
	self.font_l = "Verdana 11"
        tk.Frame.__init__(self, master)
	self.mainFrame = tk.Frame(self)
	self.mainFrame.pack(side='top', fill='both', expand=True)
	self.topFrame = tk.Frame(self.mainFrame)
	self.topFrame.pack(side='top', fill='both', expand=True)
	self.bottomFrame = tk.Frame(self.mainFrame)
	self.bottomFrame.pack(side='bottom', fill='both', expand=True)

        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)
        self.dataWidgets()
	self.ldaWidgets()
	self.svdWidgets()
    
    def dataWidgets(self):
        self.dataFrame = tk.Frame(self.main)
	self.dataFrame.pack(side='top', fill='x', expand=True)

        self.quitButton = tk.Button(self.dataFrame, text="Quit", command=self.quit)
        self.quitButton.grid(row=0, column=10)

        self.loadButton = tk.Button(self.dataFrame, text="Load File", command=self.load, width=10)
        self.loadButton.grid(row=0, column=0, columnspan=2)

	self.tnotLabel = tk.Label(self.dataFrame, text="T0", pady=3, font=self.font_l)
	self.tnotLabel.grid(row=1, column=0, columnspan=2) 
	self.tnot = tk.Entry(text="T0", width=5)
	self.tnot.grid(row=1, column=5, columnspan=2)

    def svdWidgets(self):
	self.svdFrame = tk.Frame(

        self.svdLabel = tk.Label(self, text="SVD and GA Options", pady=3, font="%s underline" % self.font_l)
	self.svdLabel.grid(row=10, column=55, columnspan=5, rowspan=2)
        self.svdButton = tk.Button(self, text="SVD", command=self.svd_init, width=10)
        self.svdButton.grid(row=12,column=55)
	self.GAButton = tk.Button(self, text="GA", command=self.svd_GA, width=10)
	self.GAButton.grid(row=12, column=58)

	self.svdAlphaLabel = tk.Label(self, text="Alpha", pady=3, font=self.font_l)
	self.svdAlphaLabel.grid(row=13, column=55, columnspan=3)
	self.svdAlpha = tk.Entry(self, text="Alpha", width=10)
	self.svdAlpha.grid(row=13, column=58, columnspan=3)
	self.svdwLSVsLabel = tk.Label(self, text="wLSVs", pady=3, font=self.font_l)
	self.svdwLSVsLabel.grid(row=14, column=55, columnspan=3)
	self.svdwLSVs = tk.Entry(self, text="wLSVs", width=10)
	self.svdwLSVs.grid(row=14, column=58, columnspan=10)
	self.svdBoundsLabel = tk.Label(self, text="Bounds", pady=3, font=self.font_l)
	self.svdBoundsLabel.grid(row=15, column=55, columnspan=3)
	self.svdBounds = tk.Entry(self, text="Bounds", width=50)
	self.svdBounds.grid(row=15, column=58, columnspan=50)

    def ldaWidgets(self):	
	self.ldaLabel = tk.Label(self, text="LDA Options", pady=3, font="%s underline" % self.font_l)
	self.ldaLabel.grid(row=10, column=0, rowspan=2, columnspan=50)
	self.regLabel = tk.Label(self, text='Regularization Method', font='Verdana 9')
	self.regLabel.grid(row=12, column=0, rowspan=2, columnspan=20)
	self.matLabel = tk.Label(self, text='Regularization Matrix', font='Verdana 9')
	self.matLabel.grid(row=12, column=30, rowspan=2, columnspan=20)

	self.regvar = tk.StringVar()
	self.regvar.set('L2')
	self.r1 = tk.Radiobutton(self, text="Tikhonov", variable=self.regvar, value='L2',  anchor='w')
	self.r1.grid(row=14, column=0)
	self.r2 = tk.Radiobutton(self, text="LASSO", variable=self.regvar, value='L1',  anchor='w')
	self.r2.grid(row=15, column=0)
	self.r3 = tk.Radiobutton(self, text='Elastic Net', variable=self.regvar, value='elnet',  anchor='w')
	self.r3.grid(row=16, column=0)

	self.r4 = tk.Radiobutton(self, text='Truncated SVD', variable=self.regvar, value='tsvd',  anchor='w')
	self.r4.grid(row=17, column=0)

	self.matvar = tk.StringVar()
	self.matvar.set('Id')
	self.m1 = tk.Radiobutton(self, text='Identity', variable=self.matvar, value='Id')
	self.m1.grid(row=14, column=30, columnspan=20)
	self.m2 = tk.Radiobutton(self, text='1D', variable=self.matvar, value='1D')
	self.m2.grid(row=15, column=30, columnspan=20)
	self.m3 = tk.Radiobutton(self, text='2D', variable=self.matvar, value='2D')
	self.m3.grid(row=16, column=30, columnspan=20)

    def load(self):    
        f = askopenfilename()
        self.data = Data(f)
	self.SVD = SVD(self.data)
	#self.LDAnalyzer = LDA(self.data)
        self.data.display()
	plt.show()

    def svd_init(self):
	self.SVD.display()
	plt.show()

    def svd_GA(self):
	bounds = self.svdBounds.get()
	bounds = [(float(re.split('[( \s )]', x)[1]), float(re.split('[( \s )]', x)[2])) for x in bounds.split(',')]
        self.SVD.Global(self.svdwLSVs.get(), bounds, float(self.svdAlpha.get()))
	plt.show()

    def lda_init(self):
	pass


