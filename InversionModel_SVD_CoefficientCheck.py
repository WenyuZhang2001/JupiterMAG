import os

import numpy as np
import matplotlib.pyplot as plt
import Spherical_Harmonic_InversionModel_Functions




def Check_SVD_SV(path,Nmax=20,Nmin=1):

    os.makedirs(path + f'/InversionTest_Picture', exist_ok=True)

    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.title('Each Nmax SVD Model Eigenvalue')
    for N in range(Nmin,Nmax):
        try:
            S = np.load(f'{path}/Inversion_SVD_coefficients_S_Nmax{N}.npy')
        except:
            continue
        plt.plot(S,marker='o',linestyle='-',alpha=0.5)
        plt.yscale('log')

        plt.text(len(S),S[-1],f'Nmax={N}',rotation=45)

    plt.ylabel('Eigenvalue log')
    plt.xlabel('Index')
    plt.grid()


    plt.subplot(2,1,2)
    plt.title('Each Nmax SVD Model Resolution Matrix Trace')
    for N in range(Nmin,Nmax):
        try:
            V = np.load(f'{path}/Inversion_SVD_coefficients_V_Nmax{N}.npy')
        except:
            continue
        Trace = np.trace(np.dot(V,V.T))
        plt.plot(N,Trace,marker='o',linestyle='-')

    plt.ylabel('Trace')
    plt.xlabel('Nmax')
    plt.grid()
    plt.savefig(path+f'/InversionTest_Picture/SVD_Model_Eg_Trace.jpg',dpi=300)
    plt.show()


def Check_SVD_Trace(path,Nmax=20,Nmin=1):
    plt.figure()
    for N in range(Nmin,Nmax):
        V = np.load(f'{path}/Inversion_SVD_coefficients_V_Nmax{N}.npy')
        R = np.dot(V,V.T)
        plt.imshow(R, cmap='viridis', interpolation='nearest',vmin=0.5,vmax=1)
        plt.colorbar()
        plt.title('Resolution Matrix R')
        plt.show()

def Check_Magnetic_Spectrum(path,Nmax=20,ParameterScaleOn=True,Rc=0.8):
    gnm_hnm_coeffi = Spherical_Harmonic_InversionModel_Functions.read_gnm_hnm_data(method='SVD', Nmax=Nmax,path=path)
    if ParameterScaleOn:
        Spherical_Harmonic_InversionModel_Functions.ParameterScale(gnm_hnm_coeffi,Rc=Rc,Nmax=Nmax)
    Rn = np.zeros(Nmax)
    for n in range(1, Nmax + 1):
        for m in range(n + 1):
            # gnm index
            # (n-1+3)*(n-1)/2 + m
            gnm_index = int((n + 2) * (n - 1) / 2 + m)
            # hnm index
            # (n-1+3)*(n-1)/2 - (n-1) + m-1 + gnm_num (= (n+3)*n/2
            hnm_index = int((n + 2) * (n - 1) / 2 - (n - 1) + m - 1 + (Nmax + 3) * Nmax / 2)

            Rn[n-1] += (n+1) * ( gnm_hnm_coeffi[gnm_index]**2 + gnm_hnm_coeffi[hnm_index]**2)

    plt.figure()
    plt.plot(range(1,Nmax+1),Rn,marker='o',linestyle='-')
    plt.ylabel(r'Rn $(nT)^(2)$')
    plt.yscale('log')
    plt.xlabel('Degree n')
    plt.title('Magnetic Spectrum of the SVD gnm and hnm')
    plt.grid()
    plt.show()



path = 'Spherical_Harmonic_Model/First50_Orbit_Model'

Check_Magnetic_Spectrum(path,Nmax=30,ParameterScaleOn=True,Rc=0.85)
# Check_SVD_SV(path=path,Nmax=21)