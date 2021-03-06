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

# Sets the directory to the current directory

os.chdir(sys.path[0])

# A function to smooth the fluxes from noise. Written by Kasper Heinz

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Loading data

def LoadNEWData():

    loading = input('Are you loading new data? [y/n]')

    time_signature = datetime.now().strftime("%m%d-%H%M")

    filters = ['UVB', 'VIS', 'NIR']

    # Setting the intervals for the XSHOOTER arms where the electical noise ...
    #  is lower than the observational noise

    selec_dict = {"UVB":[3200,5500], "VIS": [5600,10000], "NIR": [10000, 30000]}


    # Load new data

    data_dict = {}

    if loading == 'y':
        fig, ax1 = plt.subplots(figsize=(10,6))

        for i, fil in enumerate(filters):
            check = 'n'    
            folder = None

            while check == 'n':
                folders = glob.glob(f"{folder}/*") if folder is not None else glob.glob(f"*")
                view = "\n".join([str((i, name)) for i, name in enumerate(folders)])
                folder = input(f'Choose directory for {fil}-file: \n {view} \n by inputting index [0..]: ')
                folder = folders[int(folder)]
                check = input(f'Is your {fil}-file in this list: {glob.glob(f"{folder}/*")} [y/n]')
    
            filename = f'{glob.glob(f"{folder}/{fil}OB1skysuboptext.dat")[0]}'
            data = np.genfromtxt(filename)
            flux_error = data[:,3]
            flux = data[:,2]
            wavelength = data[:,1]

            if i == 0:
                pltname = pltname = filename.split(sep='/')[0]
                output_name = f"table_{time_signature}_{pltname}"
            selection = (wavelength > selec_dict[f"{fil}"][0]) & (wavelength < selec_dict[f"{fil}"][1])
            flux = flux[selection]
            wavelength = wavelength[selection]
            flux_error = flux_error[selection]
            
            # Saving the data in a dictionary
            data_dict.update({f"{fil}wave": wavelength})
            data_dict.update({f"{fil}flux": smooth(flux,45)})
            data_dict.update({f"{fil}err": smooth(flux_error,45)})

            # Add the plot to see the loaded data
            ax1.plot(wavelength,smooth(flux,45),'k-',lw=1)
        dirName = f'{pltname}'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")
            
        # Saves all the data to save time when reviewing the data 
        output_name = f'{dirName}/{output_name}'
        np.savez(output_name, **data_dict)

        # Shows the newly loaded data
        plt.show()

        # Combines all filters to one dataset 
        total_wave = np.hstack([data_dict["UVBwave"],data_dict["VISwave"],data_dict["NIRwave"]])
        total_flux = np.hstack([data_dict["UVBflux"],data_dict["VISflux"],data_dict["NIRflux"]])
        total_err = np.hstack([data_dict["UVBerr"],data_dict["VISerr"],data_dict["NIRerr"]])
        return total_wave, total_flux, total_err, pltname
        

    elif loading == 'n':
        # Search for the data
        check = 'n'
        filename, folder = None, None 
        while filename is None:
        
            folders = glob.glob(f"{folder}/*") if folder is not None else glob.glob(f"*")
            view = "\n".join([str((i, name)) for i, name in enumerate(folders)])
            folder = input(f'Choose file or directory: \n {view} \n by inputting index [0..]: ')
            
            choice = folders[int(folder)]
            if os.path.isfile(choice):
                filename = choice
            else:
                folder = choice 
        
        # Save the data in a python dictionary
        data_dict = {}
        with np.load(filename) as data_object:
            for key in data_object: 
                data_dict[key] = data_object[key]
        pltname = filename.split(sep='_')[2]
        pltname = pltname.replace('.npz','')
        

        # Combines all filters to one dataset 
        total_wave = np.hstack([data_dict["UVBwave"],data_dict["VISwave"],data_dict["NIRwave"]])
        total_flux = np.hstack([data_dict["UVBflux"],data_dict["VISflux"],data_dict["NIRflux"]])
        total_err = np.hstack([data_dict["UVBerr"],data_dict["VISerr"],data_dict["NIRerr"]])
        return total_wave, total_flux, total_err, pltname
