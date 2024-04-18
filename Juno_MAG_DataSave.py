#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Juno_Mag_Data_Make
import pandas as pd
import numpy as np


# In[2]:


year_doy_pj = {'2016':[[240,1],[346,3]],
              '2017':[[33,4],[86,5],[139,6],[191,7],[244,8],[297,9],[350,10]],
              '2018':[[38,11],[91,12],[144,13],[197,14],[249,15],[302,16],[355,17]],
              '2019':[[43,18],[96,19],[149,20]]}


# In[3]:


def Save_60sData(year_doy_pj):
    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year:[doy[0]]}
            # read data     
            data = Juno_Mag_Data_Make.Read_Data_60s(year_doy)
            # FLT
            Juno_MAG_FP = Juno_Mag_Data_Make.FootPrintCalculate(data)

            data = pd.concat([data,Juno_MAG_FP],axis=1)
            data['PJ'] = np.ones(len(data))*pj
            # save
            data.to_csv(f'JunoFGMData/Processed_Data/JunoMAG_FLT_60s_{year}{doy[0]:0>3d}_PJ{pj:0>2d}.csv')
            print(f'Data on {year}{doy[0]:0>3d}_PJ{pj:0>2d} saved')


# In[4]:


def Save_1sData(year_doy_pj):
    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year:[doy[0]]}
            # read data     
            data = Juno_Mag_Data_Make.Read_Data_1s(year_doy)
            # FLT
            Juno_MAG_FP = Juno_Mag_Data_Make.FootPrintCalculate(data)

            data = pd.concat([data,Juno_MAG_FP],axis=1)
            data['PJ'] = np.ones(len(data))*pj
            # save
            data.to_csv(f'JunoFGMData/Processed_Data/JunoMAG_FLT_1s_{year}{doy[0]:0>3d}_PJ{pj:0>2d}.csv')
            print(f'Data on {year}{doy[0]:0>3d}_PJ{pj:0>2d} saved')


# In[5]:


# Save_60sData(year_doy_pj)


# In[6]:


# Save_1sData(year_doy_pj)

# Save first 20 Orbits and B_Ex and B_In

# Model Compared to
Model = 'jrm33'

# read the data
def read_data(year_doy_pj):
    data = pd.DataFrame()
    Juno_MAG_FP = pd.DataFrame()

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year:[doy[0]]}
            date_list = Juno_Mag_Data_Make.dateList(year_doy)

            # read data
            Data = Juno_Mag_Data_Make.Read_Data_1s(year_doy)

            # 24 hours data
            # Time_start = date_list['Time'].iloc[0]
            # Time_end = Time_start+Juno_Mag_Data_Make.hour_1*24

            # Check the periJovian point time
            PeriJovian_time = Data['r'].idxmin()
            # 2 hour window data
            Time_start = PeriJovian_time - Juno_Mag_Data_Make.hour_1 * 1
            Time_end = Time_start + Juno_Mag_Data_Make.hour_1 * 3

            data_day = Data.loc[Time_start:Time_end]

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])

    return data

data = read_data(year_doy_pj)

# Calculate the Internal Field
B_Ex = Juno_Mag_Data_Make.MagneticField_External(data)
B_In = Juno_Mag_Data_Make.MagneticField_Internal(data,model=Model)


data.to_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_Data_1s_2h.csv')
B_In.to_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_In_1s_2h.csv')
B_Ex.to_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_Ex_1s_2h.csv')