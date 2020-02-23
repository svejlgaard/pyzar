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
    # Setting the constants as given by ???

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


    # Setting an initial guess for the column density

    #nion = 1e20
    z_qso = float(input('Insert the redshift of the quasar: '))
    n_abs = int(input('Insert the number of DLAs: '))
    z_abs = list()
    for number in range(n_abs):
        z_abs.append(float(input(f'Insert the redshift of DLA {number+1}: ')))


    model_data = np.genfromtxt("compoM.data")
    model_dict = {"MODwave": model_data[:,0]*(1+z_qso), "MODflux": model_data[:,1]}

    #total_wave = np.hstack([data_dict["UVBwave"],data_dict["VISwave"],data_dict["NIRwave"]])
    #total_flux = np.hstack([data_dict["UVBflux"],data_dict["VISflux"],data_dict["NIRflux"]])

    total_wave = wave.copy()
    total_flux = flux.copy()

    model_wave = model_dict["MODwave"]
    model_flux = model_dict["MODflux"]


    fit_wave = griddata(model_wave, model_wave, total_wave, fill_value = 0)
    fit_flux = griddata(model_wave, model_flux, total_wave, fill_value=0)




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


    #print(np.all(np.isfinite(fit_func(fit_wave,0, 1e20))))
    #print(np.all(np.isfinite(fit_wave)))
    #print(np.all(np.isfinite(total_flux*10**17)))
    #plt.plot(fit_wave, fit_func(fit_wave,0, 1e20))
    #plt.plot(fit_wave, total_flux*10**(17))
    #plt.show()
    #print(len(mask),len(fit_wave),len(flux_for_fit), len(err))
    #popt, pcov = curve_fit(fit_func, fit_wave[wanted], flux_for_fit[wanted],sigma=total_error[wanted], bounds= (np.array([0,1e19]),np.array([3,3e21])), check_finite= False)
    #print(popt, pcov)
    perr = np.sqrt(np.diag(pcov))
    #print(perr)
    #def ebv_coefficient_grid_search(search_interval: tuple, dla: np.array,  N_searches = 5, N_partitions = 30):
    #    search_array = np.linspace(*search_interval, num = N_partitions)

    #    losses = np.zeros_like(search_array)

     #   for i, ebv_coefficient in enumerate(search_array):
     #       ext_comp = 10**(extinction_absorption(fit_wave, [ebv_coefficient]))
     #       red_mod = fit_flux * ext_comp * dla_mod
        
     #       prediction = smooth(red_mod,2)

            #Selecting data to fit
            #QSOselec = (fit_wave > x_CIV) & (fit_wave < x_MgII)
            #print(type(QSOselec))
     #       mask = np.ones(len(fit_wave), dtype = bool)
     #       mask[np.argwhere((fit_wave < x_CIV) | (fit_wave > x_MgII))] = False
     #       mask[np.argwhere((fit_wave > xmin_lamb) & (fit_wave < xmax_lamb))] = True
     #       for k in tel_dict.keys():
     #           mask[np.argwhere((tel_dict[f'{k}'][0] < fit_wave) | (np.argwhere(tel_dict[f'{k}'][1]) > fit_wave))] = False
                #tel_list.append((fit_wave < tel_dict[f'{k}'][0]) | (fit_wave > A[1]))            
            #Aselec = (fit_wave < A[0]) | (fit_wave > A[1])
            #Bselec = (fit_wave < B[0]) | (fit_wave > B[1])
            #Cselec = (fit_wave < C[0]) | (fit_wave > C[1])
            #lineselec = (Aselec) & (Bselec) & (Cselec)
            #print(type(lineselec))
            #finalselec = [(QSOselec) & (lineselec)]

            #Use mean squared error as loss function
     #       losses[i]= np.sum(
     #           ( prediction[mask] - total_flux[mask])**2
     #       )
     #   i_opti = np.argmin(losses)
     #   if N_searches > 0:
        

            #Search again in the area around the optimum
     #       a, b = i_opti-2, i_opti+2
     #       if a < 0: a =0 
      #      elif b>len(search_array)-1: b = len(search_array)-1

       #     new_search_interval = (search_array[a], search_array[b] )
        
     #       return ebv_coefficient_grid_search(new_search_interval, dla, N_searches=N_searches-1, N_partitions=N_partitions)

      #  else:
      #      return search_array[i_opti], (1 / N_partitions)**N_searches * abs(search_interval[0]-search_interval[1]) 

    #dla_for_fit = dla_mod = addAbs(fit_wave, nion)
    #found_ebv, grid_size = ebv_coefficient_grid_search((0,3), dla_for_fit)
    #print(f"found: {found_ebv}, with the size {grid_size}")
    ebv = popt[0]
    nion = list()
    for n in range(len(z_abs)):
        nion.append(popt[n+1] ) 

    xmin = list()
    xmax = list()
    for x in range(len(z_abs)):
        xmin.append((1+z_abs[x]) * lamb - 5)
        xmax.append((1+z_abs[x]) * lamb + 5)


    #def nion_coefficient_grid_search(search_interval: tuple, dla: np.array,  N_searches = 5, N_partitions = 100):
    #    search_array = np.linspace(*search_interval, num = N_partitions)

    #    losses = np.zeros_like(search_array)

    #    for i, nion_coefficient in enumerate(search_array):
    #        dla_mod = addAbs(fit_wave, np.array([nion_coefficient], dtype=np.float64))
            #ext_comp = 10**(extinction_absorption(fit_wave, [ebv_coefficient]))
    #        red_mod = fit_flux * ext_comp * dla_mod
        
    #        prediction = smooth(red_mod,2)

            #Use mean squared error as loss function
    #        DLAselec = [(fit_wave > xmin) & (fit_wave < xmax)]
    #        losses[i]= np.sum(
    #            ( prediction[DLAselec] - total_flux[DLAselec])**2
    #        )
    #    i_opti = np.argmin(losses)
    #    if N_searches > 0:
        

            #Search again in the area around the optimum
    #        a, b = i_opti-2, i_opti+2
    #        if a < 0: a =0 
    #        elif b>len(search_array)-1: b = len(search_array)-1

    #        new_search_interval = (search_array[a], search_array[b] )
        
    #        return nion_coefficient_grid_search(new_search_interval, dla, N_searches=N_searches-1, N_partitions=N_partitions)

    #    else:
    #        return search_array[i_opti], (1 / N_partitions)**N_searches * abs(search_interval[0]-search_interval[1]) 

    #ext_for_fit = ext_comp = 10**(extinction_absorption(fit_wave, ebv))
    #found_nion, grid_size_nion = nion_coefficient_grid_search((0,5e21), ext_for_fit)
    #print(f"found: {found_nion:.2E}, with the size {grid_size_nion:.2E}")
    #nion = found_nion

    #nion = 3e20

    #ebv = 0.65





    #curve_fit(extinction_absorption,  model_dict["MODwave"], total_flux)

    ext_comp = 10**(extinction_absorption(model_dict["MODwave"], [ebv]))
    dla_mod = 1

    for another in range(len(z_abs)):
        dla_mod = dla_mod * addAbs(model_dict["MODwave"], np.array([nion[another]], dtype=np.float64), another)

    #Column = addAbs(model_dict["MODwave"], nion)[1]

    red_mod = model_dict["MODflux"] * ext_comp * dla_mod


    filters = ['UVB', 'VIS', 'NIR']
    print(len(z_abs))


    #plt.rcParams["font.family"] = "monospace"

    #fig, subplots = plt.subplots(nrows = len(z_abs)+1, figsize=(10,6))
    figure = plt.figure(figsize = (10,10))
    axs = list()


        
    figure.suptitle(f'{pltname}')

    #for val in filters:
    #    ax1.plot(data_dict[f"{val}wave"],smooth(data_dict[f"{val}flux"],45),'k-',lw=1)

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
    #axs[0].xscale('log')

    #   Insert the zero value of the flux

    axs[0].axhline(0.0,color='k',linestyle='--')

    # Mask bad regions of spectra

    for j in tel_dict.keys():
        axs[0].axvspan(float(tel_dict[f"{j}"][0]) , tel_dict[f"{j}"][1] , color="grey", alpha = 0.3)
    #plt.axvspan(5450, 5650, color="grey", alpha=0.3)
    #plt.axvspan(10000, 10500, color="grey", alpha=0.3)
    #plt.axvspan(12600, 12800, color="grey", alpha=0.3)
    #plt.axvspan(13500, 14500, color="grey", alpha=0.3)
    #plt.axvspan(18000, 19500, color="grey", alpha=0.3)

    # Labels

    axs[0].set_xlabel(r"Observed wavelength  [$\mathrm{\AA}$]",fontsize=12)
    axs[0].set_ylabel(r'Flux [$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-1}\,\mathrm{\AA}^{-1}$]',fontsize=12)
    #plt.tight_layout()

    # Create a small subplot for the Lyman alpha absorption line from the DLA

    #left, bottom, width, height = [0.57, 0.60, 0.37, 0.3]
    for y in np.arange(1,len(z_abs)+1,step = 1):
        axs.append(
            figure.add_subplot(2, len(z_abs), len(z_abs) + y)
        )
        #axs[y] = fig.add_axes([left, bottom, width, height])
        axs[y].plot(wave,flux, 'k-', lw=1)
        axs[y].plot(model_dict["MODwave"], smooth(red_mod*1e-17,2),color='tab:red', lw=1)
        
 #         axs[y] = plt.gca()
#        axs[y].set_xlim([xmin[y-1]-90,xmax[y-1]+90])
        axs[y].set_xbound( lower = xmin[y-1]-90, upper = xmax[y-1]+90)
        
        plot_flux = flux.copy()
        plot_flux = plot_flux[ (wave > xmin[y-1]-100) & (xmax[y-1]+100 > wave)]
        axs[y].set_ybound([0,np.amax(plot_flux)])
#        start, end = axs[y].get_xlim()
        
 #       axs[y].xaxis.set_ticks(np.arange(start, end, 100))
       
    # Save the figure as a pdf


    time_signature = datetime.now().strftime("%m%d-%H%M")

    plt.savefig(f'{pltname}/{pltname}_{time_signature}.pdf')

    print(f'The extinction is {ebv} +/- {perr[0]} and the column density of hydrogen is {nion} +/- {perr[1]}')


    plt.show()