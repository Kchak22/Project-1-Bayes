import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt


def log_dens_gamma(gamma, alpha, beta, tau, data):
    if gamma <= 0 or gamma >= 1:
        return -10**15
    
    dens = 0
    for i in range(len(data)):
        dens  += (data[i, 1] - alpha + beta*gamma**data[i, 0])**2
        
    return tau*dens/2

#sigma=1/np.sqrt(tau)
#param_defaut=[mu_alpha, sigma_alpha, mu_beta, sigma_beta, alpha_tau, beta_tau]

# Test 
    
def GibbsSampler(nchain, initialisation, data, param) :
   ## nchain: taille de la chaine
   #chain[i,] = [alpha, beta, tau, gamma]
    accept_gam = 0
   
   # Initialisation 
    chain = np.zeros((nchain + 1, 4))
    chain[0,:] = initialisation
    alpha = initialisation[0]
    beta = initialisation[1]
    tau = initialisation[2]
    gamma = initialisation[3]
    n = data.shape[0]
    
    for i in range(nchain):
        
        ## Mise a jour de alpha
        var_alpha = 1.0 / (1 / param[1]**2 + n * tau)
        mu_alpha= tau * np.sum(data[:,1] + beta * gamma**data[:,0]) * var_alpha
       
        alpha = np.random.normal(loc = mu_alpha, scale = np.sqrt(var_alpha))
    
        ## Mise a jour de  Beta
        
        denom_beta = 1/(1/tau + param[3]**2 * np.sum(gamma ** (-data[:,0]/2)))
        mu_beta = np.sum(gamma ** (-data[:,0]/2) * (data[:,1] + alpha))/denom_beta
        var_beta = (1/tau + param[3]**2)/denom_beta
        
        chain[i+1,1] = np.random.normal(mu_beta, np.sqrt(var_beta))
    
    
    
    
        ## Mise a jour de  Tau
        #scale = 1/beta
        sum_scale=0
        for l in range(n):
                sum_scale+= (data[l,1]-alpha+beta*(gamma**data[l,0]) )**2  
        sum_scale=1/2*sum_scale+param[5]
        
        
        tau = rd.gamma(shape = param[4] + len(data)/2, scale = 1/sum_scale)
    
        ## Mise a jour de  Gamma
        prop = gamma + rd.uniform(-0.18, 0.18)
        
        bottom =log_dens_gamma(gamma, alpha, beta, tau,data)
        top = log_dens_gamma(prop, alpha, beta, tau,data)
        
       
        acc_prob = np.exp(top - bottom)
        
        if np.random.uniform() < acc_prob:
            gamma = prop
            accept_gam += 1
            
        chain[i+1] = [alpha, beta, tau, gamma]
            
    return(np.array(chain), accept_gam/nchain)
        
#initilaisation des paramètres
initialisation = [1,1,1,0.9]
 
#importation des données
n = 27
data = np.transpose(np.array([[1, 1.5, 1.5, 1.5, 2.5, 4, 5, 5, 7, 8, 8.5, 9, 9.5, 9.5, 10, 12, 12, 13, 13, 14.5, 15.5, 15.5, 16.5, 17, 22.5, 29, 31.5],
                 [1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47, 2.19, 2.26, 2.4, 2.39, 2.41, 2.5, 2.32, 2.32, 2.43, 2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.7, 2.72, 2.57]]))


nchain=10000
param_defaut=[0.0, 10**6, 0.0, 10**6, 1, 1]


chain, taux = GibbsSampler(nchain, initialisation, data, param_defaut)   

print(taux)
x=np.arange(nchain+1)
plt.plot(x[1000:], chain[1000:,0], c="r")
#plt.plot(x[1000:], chain[1000:,2], c="b")
#plt.plot(x[1000:], chain[1000:,3], c="y")

chainbis = chain[1000:,:]
chainbis[:, 2] = 1/np.sqrt(chainbis[:,2])
mean = np.mean(chainbis, axis = 0)
print(mean)
