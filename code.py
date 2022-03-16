import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def log_dens_gamma(gamma, alpha,beta, data):
    if gamma <= 0 or gamma >= 1:
        return 0
    
    dens  = data[:, 1].sum() - alpha + beta*np.power(gamma, data[:, 0]).sum()
    return dens

sigma=1/sqrt(tau)
param_defaut=[mu_alpha, sigma_alpha, mu_beta, sigma_beta, mu_tau, sigma_beta,sigma]

# Test 
    
def GibbsSampler(nchain, initialisation, data, param=param_defaut) :
   ## nchain: taille de la chaine
   
   # Initialisation 
   chain = np.zeros((nchain + 1, 4))
   chain[0,:] = initialisation
    
   for i in range(nchain):
    ## Mise a jour de alpha
    chain[i+1,0] = np.random.normal((1/(1/param[1]**2)+n/param[6]**2)*(param[0]/param[1]**2+sum(data[:,0])/param[6]),\
                                    1/(1/param[1]**2+n/param[6]))
    ## Mise a jour de  Beta
    chain[i+1,1] = np.random.normal((1/(1/param[3]**2)+n/param[6]**2)*(param[2]/param[3]**2+sum(data[:,0])/param[6]),\
                                    1/(1/param[3]**2+n/param[6]))
    ## Mise a jour de  Tau
    #scale = 1/beta
    chain[i+1,2] = rd.gamma(shape = params[2] + n/2, scale = 2/np.power(data[:, 1].sum() - alpha + beta*np.power(gamma, data[:, 0])[0], 2))
    
    ## Mise a jour de  Gamma
    prop = chain[i,3] + rd.uniform(-0.1, 0.1)
        
    top = log_dens_gamma(chain[i,3], chain[i,0], chain[i,1], chain[i,2])
    bottom =log_dens_gamma(chain[i-1,3], chain[i-1,0], chain[i-1,1], chain[i-1,2])
       
    acc_prob = np.exp(top - bottom)
        
    if np.random.uniform() < acc_prob:
        chain[i+1,3] = prop
    else:
        chain[i+1,3] = chain[i,3]
            
   return(chain)
        
        
        
        
        
    
       
        
