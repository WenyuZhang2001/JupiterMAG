import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Juno_Mag_Data_Make
from scipy.special import lpmn,factorial
import joblib
import os
import Spherical_Harmonic_InversionModel_Functions






year_doy_pj = {'2021':[[52,32]]}

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

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])
    # data.index = data['Time']
    return data


data = read_data(year_doy_pj)
Time_start = data.index.min()
Time_end = Time_start+Juno_Mag_Data_Make.hour_1*24
# Time_start = data.index.min()+Juno_Mag_Data_Make.hour_1*4
# Time_end = Time_start+Juno_Mag_Data_Make.hour_1*3
# Time_end = Time_start+Juno_Mag_Data_Make.min_1*15

data = data.loc[Time_start:Time_end]
# Calculate the Internal Field
B_Ex = Juno_Mag_Data_Make.MagneticField_External(data)
B_In = Juno_Mag_Data_Make.MagneticField_Internal(data,model=Model)


B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data,B_Ex)



def PLot_Bfield_Model(data,Nmax=10,path = 'Spherical_Harmonic_Model/',Model_Ridge_On = True,Ridge_alpha=0,Model_JRM_on = True,Model_SVD_On = True,Model_LSTSQ_On = True):

    if Model_Ridge_On:
        B_Model_Ridge = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,Nmax=Nmax,path=path,method='Ridge',Ridge_alpha=Ridge_alpha)
        print('B Field by Ridged Model Calculated')

    if Model_SVD_On:

        B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,Nmax=Nmax,path=path,method='SVD')
        print('B Field by SVD Model Calculated')

    if Model_LSTSQ_On:
        B_Model_LSTSQ = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,Nmax=Nmax,path=path,method='LSTSQ')
        print('B Field by LSTSQ Model Calculated')



    # Define a function to calculate RMS error
    def calculate_rms_error(B_pred, B_obs):
        return np.sqrt(np.mean((B_pred - B_obs)**2))

    os.makedirs(path+f'/Result_pic/{Nmax}',exist_ok=True)
    # Plot the magnetic field components and RMS errors
    plt.figure(figsize=(15, 10))

    # Plot Br component
    plt.subplot(5, 1, 1)
    component = 'Br'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index,B_In[component],label=f'{component}_{Model} RMS={RMS:.2f}',color='gray')
    if Model_Ridge_On:
        alpha = B_Model_Ridge['alpha'].iloc[0]
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Ridge alpha = {alpha} RMS={RMS:.2f}', color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}',color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}',color='blue')
    plt.title(f'Nmax={Nmax}'
              f'\n{Time_start}-{Time_end}\n'
              f'{component}')
    plt.ylabel(f'{component} (nT)')
    plt.legend()

    # Plot Btheta component
    plt.subplot(5, 1, 2)
    component = 'Bphi'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_In[component], label=f'{component}_{Model} RMS={RMS:.2f}', color='gray')
    if Model_Ridge_On:
        alpha = B_Model_Ridge['alpha'].iloc[0]
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Ridge alpha = {alpha} RMS={RMS:.2f}',
                 color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}', color='blue')
    plt.title(f'{component} ')
    plt.ylabel(f'{component} (nT)')
    plt.legend()

    # Plot Bphi component
    plt.subplot(5, 1, 3)
    component = 'Btheta'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_In[component], label=f'{component}_{Model} RMS={RMS:.2f}', color='gray')
    if Model_Ridge_On:
        alpha = B_Model_Ridge['alpha'].iloc[0]
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Ridge alpha = {alpha} RMS={RMS:.2f}',
                 color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}', color='blue')
    plt.ylabel(f'{component} (nT)')
    plt.title(f'{component} ')
    plt.legend()

    # Plot Bphi component
    plt.subplot(5, 1, 4)
    component = 'Btotal'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_In[component], label=f'{component}_{Model} RMS={RMS:.2f}', color='gray')
    if Model_Ridge_On:
        alpha = B_Model_Ridge['alpha'].iloc[0]
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Ridge alpha = {alpha} RMS={RMS:.2f}',
                 color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}', color='blue')
    plt.ylabel(f'{component} (nT)')
    plt.title(f'{component} ')
    plt.legend()

    plt.subplot(5, 1, 5)
    component = 'r'
    plt.plot(data.index,data[component],label=f'{component}')
    plt.ylabel(f'{component} (Rj)')
    plt.xlabel('Time')
    plt.title(f'{component} ')
    plt.legend()
    # Adjust layout and show/save the figure
    plt.tight_layout()
    plt.savefig(path+f'/Result_pic/{Nmax}'+f'/Model_Bfield_{Time_start}.jpg',dpi=300)
    plt.close()
    # plt.show()
    print(f'Loop Ends Nmax = {Nmax}')
    print('-'*50)





# Nmax_list = [1,5,10,15,20]
Nmax_list = [10]
# path = 'Spherical_Harmonic_Model/Ridged_Model'
path = 'Spherical_Harmonic_Model/First50_Orbit_Model'
for Nmax in Nmax_list:
    # PLot_Bfield_Model(data,Nmax=Nmax,path=path,Ridge_alpha=1,
    #                   Model_SVD_On=False,Model_LSTSQ_On=False,Model_Ridge_On=False)
    PLot_Bfield_Model(data, Nmax=Nmax, path=path,
                      Model_SVD_On=True, Model_LSTSQ_On=False, Model_Ridge_On=False)