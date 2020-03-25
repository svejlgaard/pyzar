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

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns

from scipy.stats import norm, binned_statistic
from scipy.optimize import least_squares, curve_fit, nnls
from scipy.interpolate import griddata
from datetime import datetime
from extinction import fm07, apply

from sklearn.linear_model import Lasso

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

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns

from scipy.stats import norm, binned_statistic
from scipy.optimize import least_squares, curve_fit, nnls
from scipy.interpolate import griddata
from datetime import datetime
from extinction import fm07, apply

from sklearn.linear_model import Lasso



# Setting the frequency of the ASTROWAKEUP sound

duration = 1 
freq = 440  

# Sets the directory to the current directory

os.chdir(sys.path[0])


# Setting the frequency of the ASTROWAKEUP sound

duration = 1 
freq = 440  

# Sets the directory to the current directory

os.chdir(sys.path[0])

def CE(wave, flux, err, pltname):

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

    def addAbs(wl_mod, t, integer):
        C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
        a = lamb * 1.E-8 * gamma / (4.*np.pi * broad)
        dl_D = broad/c * lamb
        x = (wl_mod/(z_abs[integer]+1.0) - lamb)/dl_D+0.01

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

    CIV = 1549
    CIII = 1900
    MgII = 2799 
    SIV = 1402


    z_qso = float(input('Insert the redshift of the quasar: '))
    n_abs = int(input('Insert the number of DLAs: '))
    z_abs = list()
    for number in range(n_abs):
        z_abs.append(float(input(f'Insert the redshift of DLA {number+1}: ')))


    # Loading the composite QSO model

    model_data = np.genfromtxt("compoM.data")
    model_dict = {"MODwave": model_data[:,0]*(1+z_qso), "MODflux": model_data[:,1]}

    # Preparing the loaded data
    total_wave = wave.copy()
    total_flux = flux.copy()

    model_wave = model_dict["MODwave"]
    model_flux = model_dict["MODflux"]


    fit_wave = griddata(model_wave, model_wave, total_wave, fill_value = 0)
    fit_flux = griddata(model_wave, model_flux, total_wave, fill_value=0)


    # Removing the noisy flux
    selec_wave = (fit_wave !=0) & (fit_flux > 0)

    total_wave = total_wave[selec_wave]
    total_flux = total_flux [selec_wave]
    fit_flux = fit_flux[selec_wave]
    fit_wave = fit_wave[selec_wave]
    total_error = err[selec_wave]

    selec_t_flux = total_flux > 0

    total_wave = total_wave[selec_t_flux]
    fit_wave = fit_wave[selec_t_flux] 
    fit_flux = fit_flux[selec_t_flux]
    total_flux = total_flux[selec_t_flux]
    total_error = total_error[selec_t_flux]

    ratio = total_flux/fit_flux




   # An area with the Lyman alpha absorbtion line
    Lya_abs_min = list()
    Ly_abs_max = list()

    for dla in range(len(z_abs)):
        Lya_abs_min.append((1+z_abs[dla]) * lamb - 50)
        Ly_abs_max.append((1+z_abs[dla]) * lamb +50)

    # An area, typically dominated by the continuum

    x_CIV = (1+z_qso) * CIV -100
    x_CIII = (1+z_qso) * CIII
    x_MgII = (1+z_qso) * MgII -100
    x_SIV = (1+z_qso) * SIV +100
    print(x_SIV, x_MgII)
    # Remove telluric lines

    tel_dict = {
    'A' : [7600, 7630],
    'B' : [6860, 6890],
    'C' : [7170, 7350],
    'D' : [5450, 5650],
    'E' : [10000, 10500],
    'F' : [12600, 12800],
    'G': [13500, 14500],
    'H': [18000, 19500],
    }

    #Selecting wanted area


    wanted = np.ones(len(fit_wave), dtype = bool)
    wanted[np.argwhere((fit_wave > x_MgII) | (fit_wave < x_SIV))] = False
    for k in tel_dict.keys():
        wanted[np.argwhere((tel_dict[f'{k}'][0] < fit_wave) | (np.argwhere(tel_dict[f'{k}'][1]) > fit_wave))] = False
    for m in range(len(z_abs)):
        wanted[np.argwhere((Lya_abs_min[m] < fit_wave) & (Ly_abs_max[m] > fit_wave))] = True
    wanted[np.argwhere(((1+z_qso)*lamb -50 < wave) & ((1+z_qso)*lamb + 50 > wave))] = True

    flux_for_fit = total_flux.copy()*10**(17)



    # Making it possible to fit up to three DLAs
    if len(z_abs) == 1:
        def fit_func(x, e, n: list):
            return fit_flux[wanted] * 10**(extinction_absorption(x,e)) * addAbs(x,n,0)
        popt, pcov = curve_fit(fit_func, fit_wave[wanted], flux_for_fit[wanted],sigma=total_error[wanted], bounds= (np.array([0,1e19]),np.array([3,3e21])), check_finite= False)
    elif len(z_abs) == 2:
        def fit_func(x, e, n1, n2):
            return fit_flux[wanted] * 10**(extinction_absorption(x,e)) * addAbs(x,n1,0) * addAbs(x,n2,1)
        popt, pcov = curve_fit(fit_func, fit_wave[wanted], flux_for_fit[wanted],sigma=total_error[wanted], bounds= (np.array([0,1e19, 1e19]),np.array([3,3e21, 3e21])), check_finite= False)
    elif len(z_abs) == 3:
        def fit_func(x, e, n1, n2, n3):
            return fit_flux[wanted] * 10**(extinction_absorption(x,e)) * addAbs(x,n1,0) * addAbs(x,n2,1) * addAbs(x,n3,2)
        popt, pcov = curve_fit(fit_func, fit_wave[wanted], flux_for_fit[wanted],sigma=total_error[wanted], bounds= (np.array([0,1e19, 1e19, 1e19]),np.array([3,3e21, 3e21, 3e21])), check_finite= False)


    perr = np.sqrt(np.diag(pcov))

    # Saving the constants from the fit
    ebv = popt[0]
    nion = list()
    for n in range(len(z_abs)):
        nion.append(popt[n+1] ) 

    # Creating intervals around the Ly-alpha abs. lines
    xmin = list()
    xmax = list()
    for x in range(len(z_abs)):
        xmin.append((1+z_abs[x]) * lamb - 5)
        xmax.append((1+z_abs[x]) * lamb + 5)

    ext_comp = 10**(extinction_absorption(model_dict["MODwave"], [ebv]))
    dla_mod = 1

    for another in range(len(z_abs)):
        dla_mod = dla_mod * addAbs(model_dict["MODwave"], np.array([nion[another]], dtype=np.float64), another)

    
    # Adding the DLAs to the model

    red_mod = model_dict["MODflux"] * ext_comp * dla_mod


    filters = ['UVB', 'VIS', 'NIR']

    # Creating plot
    figure = plt.figure(figsize = (10,10))
    axs = list()


        
    figure.suptitle(f'{pltname}')

    axs.append(
        figure.add_subplot(2, 1, 1, )
    )
    axs[0].plot(wave, flux, 'k-', lw=1)
    axs[0].plot(model_dict["MODwave"],model_dict["MODflux"]*3e-17,color= 'tab:blue', lw=0.5, label = f'E(B-V)=0')
    axs[0].plot(model_dict["MODwave"],smooth(red_mod*3e-17,2),color='tab:red', lw=1, label = f'E(B-V)={ebv}')


    axs[0].set_xscale('log')
    axs[0].axhline(0.0,color='k',linestyle='--')


    # Set limits

    axs[0].set_ylim(-0.3e-17,1.75e-16)
    axs[0].set_xlim(3200,22200)

    #   Insert the zero value of the flux

    axs[0].axhline(0.0,color='k',linestyle='--')

    # Mask bad regions of spectra

    for j in tel_dict.keys():
        axs[0].axvspan(float(tel_dict[f"{j}"][0]) , tel_dict[f"{j}"][1] , color="grey", alpha = 0.3)

    # Labels

    axs[0].set_xlabel(r"Observed wavelength  [$\mathrm{\AA}$]",fontsize=12)
    axs[0].set_ylabel(r'Flux [$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-1}\,\mathrm{\AA}^{-1}$]',fontsize=12)

    # Create a small subplot for the Lyman alpha absorption line from the DLA
    for y in np.arange(1,len(z_abs)+1,step = 1):
        axs.append(
            figure.add_subplot(2, len(z_abs), len(z_abs) + y)
        )
        axs[y].plot(wave,flux, 'k-', lw=1)
        axs[y].plot(model_dict["MODwave"], smooth(red_mod*1e-17,2),color='tab:red', lw=1)
        axs[y].set_xbound( lower = xmin[y-1]-90, upper = xmax[y-1]+90)
        
        plot_flux = flux.copy()
        plot_flux = plot_flux[ (wave > xmin[y-1]-100) & (xmax[y-1]+100 > wave)]
        axs[y].set_ybound([0,np.amax(plot_flux)])
    # Save the figure as a pdf


    time_signature = datetime.now().strftime("%m%d-%H%M")

    plt.savefig(f'{pltname}/{pltname}_{time_signature}.pdf')

    print(f'The extinction is {ebv} +/- {perr[0]} and the column density of hydrogen is {nion} +/- {perr[1]}')


    plt.show()