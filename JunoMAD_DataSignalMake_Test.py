# This file used to make the signal data set and identify use a GUI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Juno_Mag_Data_Make
from scipy.special import lpmn,factorial
import joblib
import os
import Spherical_Harmonic_InversionModel_Functions
import matplotlib.dates as mdates

year_doy_pj = {'2017':[[139,6]]}

data = Juno_Mag_Data_Make.read_24hData(year_doy_pj,freq=1)
Time_start = data.index.min()+Juno_Mag_Data_Make.hour_1*6 + Juno_Mag_Data_Make.min_1*45
# Time_end = data.index.max()
Time_end = Time_start+Juno_Mag_Data_Make.min_1*25

data = data.loc[Time_start:Time_end]


B_Ex = Juno_Mag_Data_Make.MagneticField_External(data)
B_In = Juno_Mag_Data_Make.MagneticField_Internal(data, model='jrm33',degree=20)
# B_In = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=20, path='Spherical_Harmonic_Model/2h_1sData',method='SVD')



B_residual = Juno_Mag_Data_Make.Caluclate_B_Residual(data,B_In,B_Ex)

# cancel the start time, make sure it begin at 00:00
start_time = B_residual.index.min()
offset = pd.DateOffset(minutes=start_time.minute, seconds=start_time.second,hours=start_time.hour)
B_residual.index = B_residual.index - offset

grouped_data = B_residual.groupby(pd.Grouper(freq='25T'))

# reset the time to the start time
B_residual.index  = B_residual.index +  offset

# Plotting each 25-minute slice
for time, group in grouped_data:

    time += offset
    group.index += offset

    plt.figure(figsize=(13,8))

    plt.subplot(4,2,1)
    plt.title(f'Residual B Field (nT) \n {time} - {time + Juno_Mag_Data_Make.min_1 * 25}')
    component = 'Bx'
    plt.plot(group.index,group[component])
    plt.ylabel(f'$\delta {component}$')
    plt.xticks([])

    plt.subplot(4, 2, 3)
    component = 'By'
    plt.plot(group.index, group[component])
    plt.ylabel(f'$\delta {component}$')
    plt.xticks([])

    plt.subplot(4, 2, 5)
    component = 'Bz'
    plt.plot(group.index, group[component])
    plt.ylabel(f'$\delta {component}$')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.subplot(4, 2, 2)
    plt.title(f'Residual B Field (nT) \n {time} - {time + Juno_Mag_Data_Make.min_1 * 25}')
    component = 'Br'
    plt.plot(group.index, group[component])
    plt.ylabel(f'$\delta {component}$')
    plt.xticks([])

    plt.subplot(4, 2, 4)
    component = 'Btheta'
    plt.plot(group.index, group[component])
    plt.ylabel(f'$\delta {component}$')
    plt.xticks([])

    plt.subplot(4, 2, 6)
    component = 'Bphi'
    plt.plot(group.index, group[component])
    plt.ylabel(f'$\delta {component}$')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.subplot(4, 2, (7,8))
    component = 'Btotal'
    plt.plot(group.index, group[component])
    plt.ylabel(f'$\delta {component}$')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel('Time')

    plt.tight_layout()
    plt.show()

