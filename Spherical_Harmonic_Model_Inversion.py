import numpy as np
from scipy.linalg import lstsq
import pandas as pd
from scipy.special import lpmn,factorial
from sklearn.linear_model import Ridge
import joblib
import os
import Spherical_Harmonic_InversionModel_Functions

path = 'Spherical_Harmonic_Model/2h_1sData'
# path = 'Spherical_Harmonic_Model/Ridged_Model'
# Make Dir
os.makedirs(path,exist_ok=True)


Model = 'jrm33'
# Orbits and Doy time
data = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_Data_1s_2h.csv')
B_Ex = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_Ex_1s_2h.csv')
B_In = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_In_1s_2h.csv')





B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data,B_Ex)

# Maximum degree of internal field
# Nmax = 10

# Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data,B_In_obs,path=path,NMIN=1,NMAX=21)

# alpha_list = [0.8,1]
# for alpha in alpha_list:
#     Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data,B_In_obs,path=path,NMIN=10,NMAX=10,Ridge_alpha=alpha,
#                                                                  LSTSQ_On=False,SVD_On=False)

# Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data,B_In_obs,path=path,NMAX=10,NMIN=10,SVD_On=True,SVD_rcond=1e-15,LSTSQ_On=False,Ridge_On=False)
