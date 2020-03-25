import warnings
import os, contextlib, sys
import glob

import numpy as np
import pandas as pd
import pylab

import pyspeckit
import json
import csv


from astropy.io import ascii, fits
from astropy.table import Table


from EQW import EQW
from Restframe import Restframe
from LoadData import LoadData
from ColumnExtinction import CE
from LoadNEWData import LoadNEWData

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns

from scipy.stats import norm, binned_statistic
from scipy.optimize import least_squares, curve_fit, nnls
from scipy.interpolate import griddata
from datetime import datetime
from extinction import fm07, apply

from sklearn.linear_model import Lasso
import statistics
import math

# Sets the directory to the current directory

os.chdir(sys.path[0])

# Setting the frequency of the ASTROWAKEUP sound

duration = 1 
freq = 440  

# A function to smooth the fluxes from noise. Written by Kasper Heinz

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# A function to determine the model quasar. Based on equation 1 ...
#  in Fitzpatrick Massa and a script written by Kasper Heinz

def drude(x, x0, gam):
    return (x**2.)/(((x**2. - x0**2.)**2.) + (x**2.)*(gam**2))

def extinction_curve(wl_mod):
    x = 1/(1e-4 * wl_mod)
    k = np.zeros_like(wl_mod)
    D = drude(x, x0, gam)
    mask = (x <= c5)
    k[mask] = c1 + c2 * x[mask] + c3 * D[mask]
    k[~mask] = c1 + c2 * x[~mask] + c3 * D[~mask] + c4*((x[~mask] - c5)**2.)
    return -0.4*(k + rv)

def extinction_absorption(wl_mod, ebv):
    k = extinction_curve(wl_mod)
    return ebv*k 

# The Voigt-Hjerting profile based on the numerical approximation by Garcia

def H(a,x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a / np.sqrt(np.pi) /\
    P * ( H0 ** 2 * (4. * P**2 + 7. * P + 4. + Q) - Q - 1.0 )

def addAbs(wl_mod, t):
    C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
    a = lamb * 1.E-8 * gamma / (4.*np.pi * broad)
    dl_D = broad/c * lamb
    x = (wl_mod/(zabs+1.0) - lamb)/dl_D+0.01

    # Optical depth
    tau = C_a * t * H(a,x)
    return np.exp(-tau)



# Setting the constants as given by Kasper Heinz 

c1 = -4.959
c2 = 2.264
c3 = 0.389
c4 = 0.461
c5 = 5.9
gam = 1.
rv = 2.74
x0 = 4.6
m_e = 9.1095e-28
e = 4.8032e-10
c = 2.998e10
lamb = 1215.67
f = 0.416
gamma = 6.265e8
broad = 1


# Setting an initial guess for the column density

nion = 2.0e20


print('This script is written by Simone Vejlgaard Nielsen.')



options =  ['Lines and equivalent widths', 'Redshift determination from lines', 'Column density and extinction']

option = int(input(f'Please choose an option: {[(i, name) for i, name in enumerate(options)]}'))



# Lines and EQW

if option == 0:
    data_type = int(input('Are you loading ESO data [0] or PypeIt data [1]? '))
    if data_type == 0:
        total_wave, total_flux, total_err, pltname = LoadData()
    if data_type == 1:
        total_wave, total_flux, total_err, pltname = LoadNEWData()
    EQW(total_wave, total_flux, total_err, pltname)


# Converting to restframe and determining redshift


elif option == 1:
    check = 'n'
    filename, folder = None, None 
    while filename is None:
    
        folders = glob.glob(f"{folder}/*") if folder is not None else glob.glob(f"*")
        view = '\n'.join([str((i, name)) for i, name in enumerate(folders)])
        folder = input(f'Choose file or directory: \n {view} \n by inputting index [0..]: ')
        
        choice = folders[int(folder)]
        if os.path.isfile(choice):
            filename = choice
        else:
            folder = choice 

    with open(filename, 'r') as infile:
        data = infile.read()
    start = len(r"""
\begin{table}[H] 
\begin{tabular}{cccccccc}
Transition & Obs. wavelength [Å] & EW [Å] & Redshift\\
\hline
""")
    end = len(r"""
\hline
\end{tabular}
\end{table}
""")
    data = data[start:-end]
    Restframe(data, filename)

# Determine extinction and column density

elif option == 2:
    total_wave, total_flux, total_err, pltname = LoadData()
    CE(total_wave, total_flux, total_err, pltname)
