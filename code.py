import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def log_dens_gamma(gamma, alpha, beta, tau, data):
    if gamma <= 0 or gamma >= 1:
        return 0
    
    dens  = data[:, 1].sum() - alpha + beta*np.power(gamma, data[:, 0]).sum()
    return -tau*dens/2

#sigma=1/np.sqrt(tau)
#param_defaut=[mu_alpha, sigma_alpha, mu_beta, sigma_beta, mu_tau, sigma_tau,sigma]

# Test 
    
def GibbsSampler(nchain, initialisation, data, param=param_defaut) :
   ## nchain: taille de la chaine
   
   # Initialisation 
   chain = np.zeros((nchain + 1, 4))
   chain[0,:] = initialisation
    
   for i in range(nchain):
    ## Mise a jour de alpha
    chain[i+1,0] = np.random.normal((1/(1/param[1]**2)+n/param[6]**2)*(param[0]/param[1]**2+sum(data[:,1]+chain[i,1]*chain[i,3]**data[:,0])/param[6]),\
                                    1/(1/param[1]**2+n/param[6]))
    
    
    ## Mise a jour de  Beta
    numerateur_mu= 0
    denominateur= 0
    mu_bet = 0
    for k in range(n):
        numerateur_mu += ((data[k,1]+chain[i,0])*chain[i,3]**(-data[k,0]/2))/param[6]**2+param[3]**2
        denominateur += (param[3]**2)*chain[i,3]**(data[k,0]/2)
    denominateur +=  param[6]**2 
    mu_bet=numerateur_mu/denominateur
    sig_bet= (param[6]**2 + param[3]**2)/denominateur
    
    chain[i+1,1] = np.random.normal(mu_bet,sig_bet)
    
    
    
    
    ## Mise a jour de  Tau
    #scale = 1/beta
    chain[i+1,2] = rd.gamma(shape = param[2] + n/2, scale = 2/np.power(data[:, 1].sum() - chain[i,0] + chain[i,1]*np.power(chain[i,3], data[:, 0])[0], 2))
    
    ## Mise a jour de  Gamma
    prop = chain[i,3] + rd.uniform(-0.1, 0.1)
        
    top = log_dens_gamma(chain[i,3], chain[i,0], chain[i,1], chain[i,2],data)
    bottom =log_dens_gamma(chain[i-1,3], chain[i-1,0], chain[i-1,1], chain[i-1,2],data)
       
    acc_prob = np.exp(top - bottom)
        
    if np.random.uniform() < acc_prob:
        chain[i+1,3] = prop
    else:
        chain[i+1,3] = chain[i,3]
            
   return(chain)
        
#initilaisation des paramètres
initialisation = [1,1,1,0.9]
 
#importation des données
n = 27
data = np.transpose(np.array([[1, 1.5, 1.5, 1.5, 2.5, 4, 5, 5, 7, 8, 8.5, 9, 9.5, 9.5, 10, 12, 12, 13, 13, 14.5, 15.5, 15.5, 16.5, 17, 22.5, 29, 31.5],
                 [1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47, 2.19, 2.26, 2.4, 2.39, 2.41, 2.5, 2.32, 2.32, 2.43, 2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.7, 2.72, 2.57]]))


nchain=10000
param_defaut=[0.0, 10**6, 0.0, 10**6, 0.001, 0.001,1/np.sqrt(0.001)]


chain = GibbsSampler(nchain, initialisation, data, param_defaut)   
        
x=np.arange(nchain+1)
#plt.scatter(x, chain[:,1], c="r")
#plt.scatter(x, chain[:,2], c="b")
plt.scatter(x, chain[:,3], c="y")

      
