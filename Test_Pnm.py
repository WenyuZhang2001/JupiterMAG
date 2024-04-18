import numpy as np
from scipy.special import lpmn

Nmax = 3
theta_val = np.pi/4
P, dP = lpmn(Nmax, Nmax, np.cos(theta_val))

print(P[0,3])
print((5/2)*np.cos(theta_val)*((np.cos(theta_val))**2-9/15))

print(P[2,3])
print((15)**(0.5)/2*np.cos(theta_val)*(np.sin(theta_val)**2))
