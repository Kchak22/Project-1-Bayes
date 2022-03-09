import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

def log_dens_gamma(gamma, alpha, beta, data):
  #Data must be an array
  
    if tau <= 0 or alpha <= 1 or beta <= 1 or gamma < 0 or gamma > 1:
        return 0
  
    n = len(data)
    dens = 0
    
    for i in range(n):
        dens += np.log(data[i, 1] - alpha + beta*gamma**data[i, 0])
    
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
        
        top = 
        bottom = 
        acc_prob = np.exp(top - bottom)
        
        if np.random.uniform() < acc_prob:
            chain[i+1,3] = prop
        else:
            chain[i+1,3] = chain[i,3]
            
        chain[i+1,3] = np.random.normal()
        
        
        
        
        
    
       
        
