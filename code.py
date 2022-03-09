import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def log_dens_gamma(gamma, alpha, beta, data):
    dens  = data[:, 1].sum() - alpha + beta*np.power(gamma, data[:, 0])[0]
    return dens

  
def GibbsSampler(nchain, initialisation, data) :
   ## nchain: taille de la chaine
   
   # Initialisation 
    chain = np.zeros((nchain + 1, 4))
    chain[0,:] = 
    
   for i in range(nchain):
    ## Mise a jour de alpha
    chain[i+1,0] = np.random.normal(,)
    ## Mise a jour de  Beta
    chain[i+1,1] = np.random.normal(, )
    ## Mise a jour de  Tau
    chain[i+1,2] = np.random.normal(,)
    ## Mise a jour de  Gamma
 
        prop = chain[i,3] + 
        
        top = log_dens_gamma(chain[i,3], chain[i,0], chain[i,1], chain[i,2])
        bottom =log_dens_gamma(chain[i-1,3], chain[i-1,0], chain[i-1,1], chain[i-1,2])
        acc_prob = np.exp(top - bottom)
        
        if np.random.uniform() < acc_prob:
            chain[i+1,3] = prop
        else:
            chain[i+1,3] = chain[i,3]
            
        chain[i+1,3] = np.random.normal()
        
        
        
        
        
    
       
        
