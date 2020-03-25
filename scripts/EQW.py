import warnings
import os, contextlib, sys
import time

import numpy as np
import pandas as pd
import pylab

import pyspeckit

from astropy.io import ascii, fits
from astropy.table import Table

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns

from scipy.stats import norm
from datetime import datetime
import cProfile

duration = 1 
freq = 440  

os.chdir(sys.path[0])

def EQW(wave, flux, error, pltname):

    # Telluric lines
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

    # Loading the data

    dirName = f'{pltname}'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
    time_signature = datetime.now().strftime("%m%d-%H%M")
    output_name = f"latex_table_{time_signature}_{pltname}.txt"
    
    # Creating the spectrum needed to use the pyspeckit tools

    sp = pyspeckit.Spectrum(xarr = wave, data= flux, error = error)

    # Plotting the data to use as a search tool for the user

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.plot(wave, flux, linewidth=0.5,color='k')

    # Mask bad regions of spectra

    for j in tel_dict.keys():
        plt.axvspan(float(tel_dict[f"{j}"][0]) , tel_dict[f"{j}"][1] , color="grey", alpha = 0.3)

    plt.show(block=False)
    tex_document = r"""
\begin{table}[H] 
\begin{tabular}{cccccccc}
Transition & Obs. wavelength [Å] & EW [Å] & Redshift\\
\hline
"""

    number_of_lines = float(input('Please insert the number of lines you want to fit: '))
    number_of_MCI = 1000
    # The user chooses the lines and appropiate areas around the lines for the baseline fit
    for i in np.arange(number_of_lines):
        given_wave = input('Plese insert a wavelength value below and above the absorbtion/emission line for the baseline fitting (wavelength_min, wavelength_max): ').split(",")
        given_line = input('Plese insert a wavelength value below and above the absorbtion/emission line for the line fitting (wavelength_min, wavelength_max): ').split(",")
        line_area = np.array(given_line,dtype=np.float64)
        wavelength_area = np.array(given_wave, dtype=np.float64)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        sp.plotter(xmin=wavelength_area[0],xmax=wavelength_area[1],color='k')
        
        # The baseline fit
        sp.baseline(order=0, xmin=wavelength_area[0],xmax=wavelength_area[1],exclude=[line_area[0],line_area[1]], interactive=False, subtract=True, selec_region=True, linestyle='--')
        sp.specfit(xmin=line_area[0], xmax=line_area[1], interactive=False, fittype='voigt',color='r' , annotate = True)
        sp.specfit.annotate(loc='center right')
   
        line_xmin = np.argmax(wave > line_area[0])
        line_xmax = np.argmax(wave > line_area[1])

        # The equivalent width measurement
        eq = sp.specfit.EQW(xmin=line_xmin, xmax=line_xmax, plot=True, annotate=True, fitted=True, components = False, plotcolor = 'blue', loc='center left', midpt_location= 'fitted', continuum_as_baseline= True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

        # Saving the figure for the user to check afterwards
        plt.savefig(f"{dirName}/{pltname}_{time_signature}_EQW.png")
        
        #Monte Carlo iterations
        peak_MCI = []
        eq_error = []

        for j in np.arange(number_of_MCI):
            sp_MCI = sp.copy()
            if not j % 10: print(j)
            sp_MCI.data = sp.data + np.random.randn(sp.data.size)*sp.error
            

            # Reduce the warnings to shorten the run time of the script
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore","Warning: 'partition' will ignore the 'mask' of the MaskedArray.")
                warnings.filterwarnings("ignore","Passing the drawstyle with the linestyle as a single string is deprecated since Matplotlib 3.1 and support will be removed in 3.3; please pass the drawstyle separately using the drawstyle keyword argument to Line2D or set_drawstyle() method (or ds/set_ds()).")
                warnings.filterwarnings("ignore","divide by zero encountered in true_divide")
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    sp_MCI.specfit(xmin=line_area[0], xmax=line_area[1], interactive=False)

            
            new_eq = sp_MCI.specfit.EQW(xmin=line_xmin, xmax=line_xmax, plot=False, fitted=False, components = False )
            new_peak = sp_MCI.specfit.parinfo.values[1]

            peak_MCI.append(new_peak)
            eq_error.append(new_eq)

        # This part is inserted to avoid a RunTimeError, if something else is already playing on the speakers, when executing the astro-wake-up
        try:
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        except:
            print("DOOT!") # An alternative astro-wake-up
        plt.figure()
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        sns.distplot(eq_error,rug=False,kde=False,fit=norm,hist_kws={"label":"Monte Carlo simulation data","color":"grey"},fit_kws={"label":"Gaussian"},axlabel='Equivalent width [nm]')
        plt.legend()

        # Saving the results from the Monte Carlo simulation
        output_name_eq = f"figure_{pltname}_{i}_{time_signature}.png"
        plt.savefig(f"{dirName}/{output_name_eq}")


        EQW_mean, EQW_error = norm.fit(eq_error)
        SHIFT_mean, SHIFT_error = norm.fit(peak_MCI)
    
        # Saving the data as a table ready to insert in LaTeX
        tex_document += f" & ${SHIFT_mean:.6f}\pm {SHIFT_error:.6f}$ & ${EQW_mean:.4f}\pm {EQW_error:.4f}$ & \\\\"


    
        tex_footer = r"""
\hline
\end{tabular}
\end{table}
"""
        total_name = dirName + '/' + output_name 
        with open(total_name, "w+") as outfile:
            outfile.write(tex_document+tex_footer) 

    #Prints the table for the user to inspect
    print(tex_document+tex_footer)
