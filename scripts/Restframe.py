import numpy as np
import uncertainties as unc  
import uncertainties.unumpy as unumpy  

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


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns

from scipy.stats import norm, binned_statistic
from scipy.optimize import least_squares, curve_fit, nnls
from scipy.interpolate import griddata
from datetime import datetime
from extinction import fm07, apply

from sklearn.linear_model import Lasso

from datetime import datetime

from pyperclip import paste, copy
import statistics
import math
import collections


def Restframe(string, name):
    dirName = f'{name}'
    dirName = dirName.split('/')[0]

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")



    #Creating a line dict from NIST
    line_dict = {'CIV_1548': [1548.202, 0.01, r'C \Rn{4} $\lambda$ 1548'],
    'SiIV_1402': [1402.77,0.1, r'Si \Rn{4} $\lambda$ 1403'],
    'Ly_a': [1215.6701, 0.0021, r'Ly-$\alpha$ $\lambda$ 1216'],
    'CIII_1897': [1897.57, 0.1,r'C \Rn{3} $\lambda$ 1898'],
    'MgII_2796': [2795.528, 0.01, r'Mg \Rn{2} $\lambda$ 2796'],
    'MgII_2803':[2802.705,0.01, r'Mg \Rn{2} $\lambda$ 2803'],
    'OIII_5007': [5006.843,0.01, r'O \Rn{3} $\lambda$ 5007'],
    'H_b': [4861.283363,0.000024, r'H-$\beta$ $\lambda$ 4861'],
    'H_a':[6562.85117,0.00003,r'H-$\alpha$ $\lambda$ 6563'],
    'SiII_989': [989.87,0.1,r'Si \Rn{2} $\lambda$ 989'],
    'BeIII_4487': [4487.30,0.10,r'Be \Rn{3} $\lambda$ 4487'],
    'AlII_1670': [1670.7874,0.001, r'Al \Rn{2} $\lambda$ 1670'],
    'NV_1239': [1238.8210,0.01,r'N \Rn{5} $\lambda$ 1239'],
    'CuII_10852': [10852.401, 0.004, r'Cu \Rn{2} $\lambda$ 10852'],
    'CuII_5007': [5006.79978, 0.00015, r'Cu \Rn{2} $\lambda$ 5007'],
    'TiIII_4971': [4971.194, 0.010, r'Ti \Rn{3} $\lambda$ 4971'],
    'ZnII_2026': [2025.4845,0.001, r'Zn \Rn{2} $\lambda$ 2026'],
    'ZnII_2062': [2062.0011,0.0010, r'Zn \Rn{2} $\lambda$ 2062'],
    'ZnII_2064': [2064.2266, 0.0010, r'Zn \Rn{2} $\lambda$ 2064'],
    'SiII_1808': [1808.00, 0.10, r'Si \Rn{2} $\lambda$ 1808'],
    'FeII_1611': [1610.9229, 0.0005, r'Fe \Rn{2} $\lambda$ 1611'],
    'FeII_2249': [2249.05864, 0.00011, r'Fe \Rn{2} $\lambda$ 2249'],
    'FeII_2260': [2260.5173, 0.0019, r'Fe \Rn{2} $\lambda$ 2260'],
    'SiII_1526': [1526.734, 0.10, r'Si \Rn{2} $\lambda$ 1526'],
    'FeI_3021': [3021.0725, 0.0003, r'Fe \Rn{2} $\lambda$ 3021'],
    'FeII_2344': [2344.9842, 0.0001, r'Fe \Rn{2} $\lambda$ 2344'],
    'FeII_2374': [2374.6530, 0.0001, r'Fe \Rn{2} $\lambda$ 2374'],
    'FeII_2382': [2382.0373, 0.0001, r'Fe \Rn{2} $\lambda$ 2382'],
    'FeII_2586': [2585.8756, 0.0001, r'Fe \Rn{2} $\lambda$ 2586'],
    'CrII_2026': [2025.6186, 0.0001, r'Cr \Rn{2} $\lambda$ 2026'],
    'CrII_2062': [2061.57673, 0.00007, r'Cr \Rn{2} $\lambda$ 2062'],
    'CrII_2056': [2055.59869, 0.00006, r'Cr \Rn{2} $\lambda$ 2056'],
    'CrII_2066': [2065.50389, 0.00007, r'Cr \Rn{2} $\lambda$ 2066'],
    'MnII_1162': [1162.0150, 0.0001, r'Mn \Rn{2} $\lambda$ 1162'],
    'MnII_1197': [1197.1840, 0.0001, r'Mn \Rn{2} $\lambda$ 1197'],
    'MnII_1199': [1199.3910, 0.0001, r'Mn \Rn{2} $\lambda$ 1199'],
    'MnII_1201': [1201.1180, 0.0001, r'Mn \Rn{2} $\lambda$ 1201'],
    'MnII_2576': [2576.8770, 0.0001, r'Mn \Rn{2} $\lambda$ 2576'],
    'MnII_2594': [2594.4990, 0.0001, r'Mn \Rn{2} $\lambda$ 2594'],
    'MnII_2606': [2606.4630, 0.0001, r'Mn \Rn{2} $\lambda$ 2606'],
    'NiII_1317': [1317.2170, 0.0001, r'Ni \Rn{2} $\lambda$ 1317'],
    'NiII_1345': [1345.8780, 0.0001, r'Ni \Rn{2} $\lambda$ 1345'],
    'NiII_1370': [1370.1320, 0.0001, r'Ni \Rn{2} $\lambda$ 1370'],
    'NiII_1393': [1393.3240, 0.0001, r'Ni \Rn{2} $\lambda$ 1393'],
    'NiII_1454': [1454.8420, 0.0001, r'Ni \Rn{2} $\lambda$ 1454'],
    'NiII_1502': [1502.1480, 0.0001, r'Ni \Rn{2} $\lambda$ 1502'],
    'NiII_1703': [1703.4050, 0.0001, r'Ni \Rn{2} $\lambda$ 1703'],
    'NiII_1709': [1709.6000, 0.0001, r'Ni \Rn{2} $\lambda$ 1709'],
    'NiII_1741': [1741.5490, 0.0001, r'Ni \Rn{2} $\lambda$ 1741'],
    'NiII_1751': [1751.9100, 0.0001, r'Ni \Rn{2} $\lambda$ 1751'],
    'NiII_1773': [1773.9490,0.0001, r'Ni \Rn{2} $\lambda$ 1773'],
    'NiII_1804': [1804.4730,0.0001, r'Ni \Rn{2} $\lambda$ 1804'],
    'CrII_1058': [1058.7320,0.0001, r'Cr \Rn{2} $\lambda$ 1058'],
    'CrII_1059': [1059.7320,0.0001, r'Cr \Rn{2} $\lambda$ 1059'],
    'CrII_1064': [1064.1240,0.0001, r'Cr \Rn{2} $\lambda$ 1064'],
    'CrII_1066': [1066.7760,0.0001, r'Cr \Rn{2} $\lambda$ 1066'],
    'SiII_1020': [1020.6989, 0.1, r'Si \Rn{2} $\lambda$ 1020'],
    'SiIV_1062': [1062.66, 0.1, r'Si \Rn{4} $\lambda$ 1062'],
    'SiII_1190': [1190.4158, 0.1, r'Si \Rn{2} $\lambda$ 1190'],
    'SiII_1193': [1193.2897, 0.1, r'Si \Rn{2} $\lambda$ 1193'],
    'SiIII_1206': [1206.5000, 0.1, r'Si \Rn{3} $\lambda$ 1206'],
    'SiII_1260': [1260.4221, 0.1, r'Si \Rn{2} $\lambda$ 1260'],
    'SiII_1304': [1304.3702, 0.1, r'Si \Rn{2} $\lambda$ 1304'],
    'SiI_1554': [1554.2960, 0.1, r'Si \Rn{1} $\lambda$ 1554'],
    'SiI_1562': [1562.0020, 0.1, r'Si \Rn{1} $\lambda$ 1562'],
    'SiI_1625': [1625.7051, 0.1, r'Si \Rn{1} $\lambda$ 1625'],
    'SiI_1631': [1631.1705, 0.1, r'Si \Rn{1} $\lambda$ 1631'],
    'SiIV_1693': [1693.2935, 0.1, r'Si \Rn{4} $\lambda$ 1693'],
    'SiI_2515': [2515.0725, 0.1, r'Si \Rn{1} $\lambda$ 2515'],
    'SiI_2208': [2208.6665, 0.1, r'Si \Rn{1} $\lambda$ 2208'],
    'SiIII_1892': [1892.0300, 0.1, r'Si \Rn{3} $\lambda$ 1892'],
    'SiI_1845': [1845.5202, 0.1, r'Si \Rn{1} $\lambda$ 1845'],
    'FeII_935': [935.5175, 0.001, r'Fe \Rn{2} $\lambda$ 935'],
    'FeII_937': [937.6520, 0.001, r'Fe \Rn{2} $\lambda$ 937'],
    'FeII_1055': [1055.2617, 0.001, r'Fe \Rn{2} $\lambda$ 1055'],
    'FeII_1062': [1062.1520, 0.001, r'Fe \Rn{2} $\lambda$ 1062'],
    'FeII_1081': [1081.8750, 0.001, r'Fe \Rn{2} $\lambda$ 1081'],
    'FeII_1083': [1083.4200, 0.001, r'Fe \Rn{2} $\lambda$ 1083'],
    'FeII_1096': [1096.8769, 0.001, r'Fe \Rn{2} $\lambda$ 1096'],
    'FeIII_1122': [1122.5360, 0.001, r'Fe \Rn{3} $\lambda$ 1122'],
    'FeII_1144': [1144.9379, 0.001, r'Fe \Rn{2} $\lambda$ 1144'],
    'FeII_1260': [1260.5330, 0.001, r'Fe \Rn{2} $\lambda$ 1260'],
    'FeII_1608': [1608.4511, 0.001, r'Fe \Rn{2} $\lambda$ 1608'],
    'FeII_1611': [1611.2005, 0.001, r'Fe \Rn{2} $\lambda$ 1611'],
    'FeI_1934': [1934.5351, 0.001, r'Fe \Rn{1} $\lambda$ 1934'],
    'FeI_1937': [1937.2682, 0.001, r'Fe \Rn{1} $\lambda$ 1937'],
    'FeI_2167': [2167.4531, 0.001, r'Fe \Rn{1} $\lambda$ 2167'],
    'FeII_2344': [2344.2140, 0.001, r'Fe \Rn{2} $\lambda$ 2344'],
    'FeII_2382': [2382.7650, 0.001, r'Fe \Rn{2} $\lambda$ 2382'],
    'FeII_2484': [2484.0211, 0.001, r'Fe \Rn{2} $\lambda$ 2484'],
    'FeI_2523': [2523.6083, 0.001, r'Fe \Rn{1} $\lambda$ 2523'],
    'FeII_2600': [2600, 0.001, r'Fe \Rn{2} $\lambda$ 2600'],
    'FeI_2719': [2719.8331, 0.001, r'Fe \Rn{1} $\lambda$ 2719'],
    'FeI_3021': [3021.5189, 0.001, r'Fe \Rn{1} $\lambda$ 3021'],
    'AlIII_1854': [1854.7164, 0.1, r'Al \Rn{3} $\lambda$ 1854'],
    'AlIII_1862': [1862.7895, 0.1, r'Al \Rn{3} $\lambda$ 1862'],
    'MgI_2852': [2852.1370, 0.1, r'Mg \Rn{1} $\lambda$ 2852'],
    'MgI_2026': [2026.4768, 0.1, r'Mg \Rn{1} $\lambda$ 2026'],
    'PII_1152': [1152.8180, 0.001, r'P \Rn{2} $\lambda$ 1152'],
    'CuII_1358': [1358.7730, 0.001, r'Cu \Rn{2} $\lambda$ 1358']
    }
    time_signature = datetime.now().strftime("%m%d-%H%M")
    name = name.replace('.txt','')
    output_name = f'Final_table_{time_signature}_{dirName}.txt'

    numify = lambda x: float(''.join(char for char in x if char.isdigit() or char == "." or char =="-"))

    tex_table = string


    tex_document = r"""
\begin{table}[H] 
\begin{tabular}{cccccccc}
Transition & EW$_r$ [Å] & Redshift\\
\hline
"""


    approx_z = float(input('What is the approximate redshift?'))

    lines = tex_table.split(r"\\")[0:-1]

    data = []

    text = []

    # Interactive way of categorising the lines
    for line in lines:
        [_, wave, eq, _] = line.strip().split("&")
        [wave, wave_err] = wave.split(r"\pm") 
        [eq, eq_err] = eq.split(r"\pm") 

        line_exp = unc.ufloat(numify(wave), numify(wave_err))/(1+approx_z)
        ordered_dict = collections.OrderedDict(sorted(line_dict.items(), key= lambda v: v[1] ))
        line_true = input(f'\n What is this line {line_exp}? Please choose from: \n {ordered_dict.keys()}: ')

        data.append(
            [
                unc.ufloat(numify(wave), numify(wave_err)),
                unc.ufloat(numify(eq), numify(eq_err)),
                unc.ufloat(line_dict[f'{line_true}'][0],line_dict[f'{line_true}'][1])
            ] 
        )   

        text.append(line_dict[f'{line_true}'][2])



    data = unumpy.matrix(data)

    z = (data[:,0] / data[:,2] - 1)

    # The errors and their contribution to the weighted redshift avarage 
    z_avg = np.mean(z)
    var  = sum(pow(i-z_avg,2) for i in z) / len(z)
    var = var[0,0]
    z_avg_std = math.sqrt(var.nominal_value)

    ew_r = (data[:,1] / (z_avg + 1))

    # Creating a LaTeX table for the results

    tex_begin = "\\noindent The weighted avarage of the redshift is $z = {:.1uL}$".format(z_avg)


    for i in range(len(z)):
        eq_value = ew_r[i,0]

        z_value = z[i,0]
    
        tex_document += " {0} & ${1:.1uL}$ & ${2:.1uL}$ \\\\".format(text[i],eq_value, z_value)

    tex_footer = r"""
\hline
\end{tabular}
\end{table}
"""
    output_name = dirName + "/" + output_name
    with open(output_name, "w+") as outfile:
        outfile.write(tex_begin+tex_document+tex_footer) 

    # When the file is saved, it is also printed and inserted in the clip board 
    print(tex_begin+tex_document+tex_footer)
    copy(tex_begin+tex_document+tex_footer)
