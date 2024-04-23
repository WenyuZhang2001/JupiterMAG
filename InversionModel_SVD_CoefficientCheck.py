import numpy as np
import matplotlib.pyplot as plt

path = 'Spherical_Harmonic_Model/First50_Orbit_Model'

plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.title('Each Nmax SVD Model Eigenvalue')
for Nmax in range(10,11):
    S = np.load(f'{path}/Inversion_SVD_coefficients_S_Nmax{Nmax}.npy')
    plt.plot(S,marker='o',linestyle='-',alpha=0.5)
    # plt.yscale('log')

    plt.text(len(S),S[-1],f'Nmax={Nmax}',rotation=45)

plt.ylabel('Eigenvalue log')
plt.xlabel('Index')
plt.grid()


plt.subplot(2,1,2)
plt.title('Each Nmax SVD Model Resolution Matrix Trace')
for Nmax in range(10,11):
    V = np.load(f'{path}/Inversion_SVD_coefficients_V_Nmax{Nmax}.npy')
    Trace = np.trace(np.dot(V,V.T))
    plt.plot(Nmax,Trace,marker='o',linestyle='-')

plt.ylabel('Trace')
plt.xlabel('Nmax')
plt.grid()
# plt.savefig(path+f'/Result_pic/SVD_Model_Eg_Trace.jpg',dpi=300)
plt.show()