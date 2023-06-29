#!/usr/bin/env python
# coding: utf-8

# # SDE integrator

# In[ ]:


#export

# Initialization
from DiffOperator import DifferentialOperator
# import derivativesTransferFunctions
import numpy as np
# import derivativesTransferFunctions


# ## Reward-driven regulatory mechanism

# In[ ]:


#export

def RegulatoryPsi(psi0, stimulusA, stimulusB, params):

    tF      = params[22] # final time of the trial
    dt      = params[23] # time step
    tauPsi  = params[25] # time scale of the regulatory mechanism
    sigma_r = params[26] # extrinsic noise level
    c0      = params[27] # extrinsic noise decay rate
    
    
    # Initialize the psi vector and the time for the trial
    
    psi = np.zeros(int(tF/dt)+1)
    psi[0] = psi0 # initial condition
    lambdaA = np.zeros(int(tF/dt)+1)
    lambdaB = np.zeros(int(tF/dt)+1)
    lambdaA[0] = psi[0]*stimulusA[0]+(1-psi[0])*stimulusB[0]
    lambdaB[0] = psi[0]*stimulusB[0]+(1-psi[0])*stimulusA[0]
    t = np.linspace(0, tF, int(tF/dt)+1)
    
    
    
    # Generate psi time trace for the whole trial
    for i in range(int(tF/dt)):
        psi[i+1] = psi[i]+(dt/tauPsi)*(-4*psi[i])*(psi[i]-0.5)*(psi[i]-1)+                            (1/(t[i+1]*c0)**2)*np.sqrt(dt)*sigma_r*np.random.normal(0, 1)/tauPsi
    
    lambdaA[i+1]= psi[i+1]*stimulusA[i+1]+(1-psi[i+1])*stimulusB[i+1]
    lambdaB[i+1] = psi[i+1]*stimulusB[i+1]+(1-psi[i+1])*stimulusA[i+1]
    
    return lambdaA, lambdaB, psi 


# ## Euler-Maruyama scheme with all modules

# In[ ]:


#export

def TimeStepping(V0, lambdaA, lambdaB, TF1, TF2, params): 
# V0: initial conditions for the state variables
# lambdaA, lambdaB: regulated stimuli
# TF1, TF2: transfer functions of RS and FS cells, respectively
# params: paramaters

# Glossary
    
# v_eA            : X[0]
# v_iA            : X[1]
# C_eAeA          : X[2]
# C_eAiA = C_iAeA : X[3]
# C_iAiA          : X[4]
# W_eA            : X[5]
# W_iA            : X[6]
# v_eB            : X[7]
# v_iB            : X[8]
# C_eBeB          : X[9]
# C_eBiB          : X[10]
# C_iBiB          : X[11]
# W_eB            : X[12]
# W_iB            : X[13]
# C_eAeB          : X[14]
# C_eAiB          : X[15]
# C_iAeB          : X[16]
# C_iAiB          : X[17]

# aRS = params[0]
# bRS = params[1]
# aFS = params[2]
# bFS = params[3]
# tauwRS = params[4]
# tauwFS = params[5]
# Ntot = params[6]
# pc = params[7]
# Ne = params[8]
# Ni = params[9]
# vAI = params[10]
# wce = params[11]
# wci = params[12]
# sigma = params[13]
# El = params[14]
# Qe = params[15]
# Qi = params[16]
# Te = params[17]
# Ti = params[18]
# Gl = params[19]
# Ee = params[20]
# Ei = params[21]
# tF = params[22]
# dt = params[23]

    # Set parameters and initialize the state variables    
    sigma = params[13]    
    tF    = params[22]
    dt    = params[23]
    T     = params[24] 
    
    X = np.zeros((int(tF/dt)+1, np.size(V0)))
    X[0,:] = V0       
    
    # Integrate in time via Euler-Maruyama scheme    
    intrinsicNoise = np.random.normal(0, 1, size=(int(tF/dt), 4)) # generate the intrinsic noise
    
    exc_aff_A = lambdaA
    exc_aff_B = lambdaB
    inh_aff_A = exc_aff_A # regulated stimuli are provided to both populations (TO BE CHECKED!!!, this might be also 0!!!!!)
    inh_aff_B = exc_aff_B # regulated stimuli are provided to both populations (TO BE CHECKED!!!, this might be also 0!!!!!)
    
    # Integrate in time!
    for i in range(int(tF/dt)):        
        
        X[i+1,:] = X[i,:] + dt*DifferentialOperator(X[i,:], TF1, TF2, params, exc_aff_A[i],                                                exc_aff_B[i], inh_aff_A[i], inh_aff_B[i])
        X[i+1,0:2] = X[i+1, 0:2] + (1/T)*np.sqrt(dt)*sigma*intrinsicNoise[i,0:2]
        X[i+1,7:9] = X[i+1, 7:9] + (1/T)*np.sqrt(dt)*sigma*intrinsicNoise[i,2:4]
    
    return X

