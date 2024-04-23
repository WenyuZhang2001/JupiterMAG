import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import lpmn,factorial

# Nmax = 10
# theta_val = np.pi/4
#
#
# def schmidt_semi_normalization(n, m):
#     return ((-1) ** m) * np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))
#
#
# for n in range(1,Nmax + 1):
#     for m in range(n + 1):
#         P, dP = lpmn(m, n, np.cos(theta_val))
#         N_lm = schmidt_semi_normalization(n, m)
#         Snm = P[m,n]*N_lm
#         print(f'm = {m}  n = {n}')
#         print(f'P = {P[m,n]}')
#         print(f'N_ln = {N_lm}')
#         print(f'Snm = {Snm}')
#         print('='*50)

# print(True or True or True)

# a = np.array([1,2,3])
# b = np.array([4,5,6])
# c = np.array([7,8,9])
# df = pd.DataFrame()
# df['a']  = a
# df['b'] = b
# df['c'] = c
# print(np.vstack((a,b,c)))
# print(np.hstack((df['a'],df['b'],df['c'])).T.reshape(-1))

# theta_val = np.pi/2
# theta_val = 7/360*np.pi
# theta_val = 1e-10
# n = 1
# m = 1
# print(np.cos(theta_val))
# P1, dP1 = lpmn(m, n, np.cos(theta_val))
# P2, dP2 = lpmn(m, n, np.cos(theta_val)-1e-10)
# dP3 = (P1-P2)/2/1e-5
# print(f'P = {P1[m,n]}, dP = {dP1[m,n]}')
# print(dP3[m,n])

import Spherical_Harmonic_InversionModel_Functions

# gnm_hnm = Spherical_Harmonic_InversionModel_Functions.read_gnm_hnm_data(method='SVD', Nmax=2, path='Spherical_Harmonic_Model/2h_1sData')
#
# print(gnm_hnm)

if 0**2:
    print('0')