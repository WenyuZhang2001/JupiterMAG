import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import CoordinateTransform
import con2020
import JupiterMag as jm
import Juno_Mag_Data_Make
import MyPlot_Functions
from scipy.special import lpmn,factorial
from scipy.optimize import minimize
import csv


Nmax = 2
# Replace 'path_to_your_csv_file.csv' with the actual path to your CSV file
gnm_hnm_coeffi = np.genfromtxt(f'Spherical_Harmonic_Model/optimized_coefficients_gnm_hnm_Nmax{Nmax}.csv', delimiter=',')

# Assuming the first column contains 'gnm' and the second column contains 'hnm' coefficients
half_len = len(gnm_hnm_coeffi) // 2
gnm = gnm_hnm_coeffi[:half_len].reshape((Nmax + 1, Nmax + 1))
hnm = gnm_hnm_coeffi[half_len:].reshape((Nmax + 1, Nmax + 1))


year_doy_pj = {'2018':[[144,13]]}

# Model Compared to
Model = 'jrm33'

# read the data
def read_data(year_doy_pj):
    data = pd.DataFrame()

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year:[doy[0]]}
            date_list = Juno_Mag_Data_Make.dateList(year_doy)

            # read data
            Data = Juno_Mag_Data_Make.Read_Data_60s(year_doy)


            # 24 hours data
            Time_start = date_list['Time'].iloc[0]
            Time_end = Time_start+Juno_Mag_Data_Make.hour_1*24

            data_day = Data.loc[Time_start:Time_end]
            print(data_day)
            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])
    # data.index = data['Time']
    return data


data = read_data(year_doy_pj)
Time_start = data.index.min()
Time_end = Time_start+Juno_Mag_Data_Make.hour_1*24
# Time_start = data.index.min()+Juno_Mag_Data_Make.hour_1*3
# Time_end = Time_start+Juno_Mag_Data_Make.hour_1*9

data = data.loc[Time_start:Time_end]
print(data.describe())
# Calculate the Internal Field
B_Ex = Juno_Mag_Data_Make.MagneticField_External(data)
B_In = Juno_Mag_Data_Make.MagneticField_Internal(data,model=Model)

data['theta'] = data['theta']/360*2*np.pi
data['phi'] = data['phi']/360*2*np.pi

def B_In_obs_Calculate(data,B_Ex):
    B_In_obs = pd.DataFrame(columns=['Br','Btheta','Bphi','Btotal'])
    B_In_obs['Br'] = data['Br'] - B_Ex['Br']
    B_In_obs['Btheta'] = data['Btheta'] - B_Ex['Btheta']
    B_In_obs['Bphi'] = data['Bphi'] - B_Ex['Bphi']
    B_In_obs['Btotal'] = np.sqrt(B_In_obs['Br']**2 + B_In_obs['Btheta']**2 + B_In_obs['Bphi']**2)
    return B_In_obs

B_In_obs = B_In_obs_Calculate(data,B_Ex)

def compute_magnetic_field_components(gnm, hnm, r, theta, phi):
    a = 1.0  # Reference radius for Jupiter, normalized to 1
    Nmax = len(gnm) - 1
    Br = np.zeros(len(theta))
    Btheta = np.zeros(len(theta))
    Bphi = np.zeros(len(theta))

    for idx, (r_val, theta_val, phi_val) in enumerate(zip(r, theta, phi)):
        cos_theta_val = np.cos(theta_val)
        P, dP = lpmn(Nmax, Nmax, cos_theta_val)

        for n in range(1, Nmax + 1):
            for m in range(n + 1):
                # Schmidt semi-normalization factor
                normalization_factor = np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))
                # Adjust for the Condon-Shortley phase
                Condon_Shortley_phase = (-1) ** m

                # Apply the normalization and phase to Pnm and its derivative
                Pnm = P[m, n] * normalization_factor * Condon_Shortley_phase
                dPnm = dP[m, n] * normalization_factor * Condon_Shortley_phase

                g_nm = gnm[n, m]
                h_nm = hnm[n, m]

                # Compute magnetic field components
                Br[idx] += (n + 1) * (a / r_val) ** (n + 2) * (
                            g_nm * np.cos(m * phi_val) + h_nm * np.sin(m * phi_val)) * Pnm
                Btheta[idx] += (a / r_val) ** (n + 2) * (
                            g_nm * np.cos(m * phi_val) + h_nm * np.sin(m * phi_val)) * dPnm * (-np.sin(theta_val))
                Bphi[idx] += (a / r_val) ** (n + 2) * m * (
                            g_nm * np.sin(m * phi_val) - h_nm * np.cos(m * phi_val)) * Pnm / np.sin(theta_val)

    return Br, Btheta, Bphi

Br_pred, Btheta_pred, Bphi_pred = compute_magnetic_field_components(gnm,hnm,data['r'],data['theta'],data['phi'])
Btotal_pred = np.sqrt(Br_pred**2+Btheta_pred**2+Bphi_pred**2)

def calculate_rms_error(B_pred, B_obs):
    return np.sqrt(np.mean((B_pred - B_obs) ** 2))
plt.figure()
plt.subplot(5, 1, 1)
component = 'Br'
plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
plt.plot(data.index,B_In[component],label=f'{component}_{Model}',color='gray')
RMS = calculate_rms_error(Br_pred, B_In_obs[component].values)
plt.plot(data.index, Br_pred, label=f'{component}_model_SLSQP={RMS:.2f}', color='red')
plt.title(f'Nmax={Nmax}'
          f'\n{Time_start}-{Time_end}\n'
          f'{component}')
plt.ylabel(f'{component} (nT)')
plt.legend()

# Plot Btheta component
plt.subplot(5, 1, 2)
component = 'Bphi'
plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
plt.plot(data.index,B_In[component],label=f'{component}_{Model}',color='gray')
RMS = calculate_rms_error(Bphi_pred, B_In_obs[component].values)
plt.plot(data.index, Bphi_pred, label=f'{component}_model_SLSQP={RMS:.2f}', color='red')

plt.title(f'{component} ')
plt.ylabel(f'{component} (nT)')
plt.legend()

# Plot Bphi component
plt.subplot(5, 1, 3)
component = 'Btheta'
plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
plt.plot(data.index,B_In[component],label=f'{component}_{Model}',color='gray')
RMS = calculate_rms_error(Btheta_pred, B_In_obs[component].values)
plt.plot(data.index, Btheta_pred, label=f'{component}_model_SLSQP={RMS:.2f}', color='red')
plt.ylabel(f'{component} (nT)')
plt.legend()

# Plot Bphi component
plt.subplot(5, 1, 4)
component = 'Btotal'
plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
plt.plot(data.index,B_In[component],label=f'{component}_{Model}',color='gray')
RMS = calculate_rms_error(Btotal_pred, B_In_obs[component].values)
plt.plot(data.index, Btotal_pred, label=f'{component}_model_SLSQP={RMS:.2f}', color='red')
plt.ylabel(f'{component} (nT)')
plt.legend()

plt.subplot(5, 1, 5)
component = 'r'
plt.plot(data.index,data[component],label=f'{component}')
plt.ylabel(f'{component} (Rj)')
plt.xlabel('Time')
plt.legend()
# Adjust layout and show/save the figure
plt.tight_layout()
# plt.savefig(path+f'/Result_pic/{Nmax}'+f'/Model_Bfield_{Time_start}.jpg',dpi=300)
plt.show()
plt.close()


