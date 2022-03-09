import numpy as np
import numpy.random as rd

def log_density(x, alpha, beta, tau, gamma, data):
  #Data must be an array
  n = len(data)
  dens = -n/2*np.log(tau) - n*alpha/2 - 1/(2*tau)*data[:,1].sum()
  
  for i in range(n):
    dens += beta*gamma**data[i,0]
  
  return dens
