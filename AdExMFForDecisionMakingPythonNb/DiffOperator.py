#!/usr/bin/env python
# coding: utf-8

# # AdEx double pool SDE system

# In[4]:


#export

# Initialization
import numpy as np


# ## Derivatives of transfer functions with respect to firing rates

# In[5]:


#export

def diff_fe(TF, fe, fi ,XX, df=1e-5):
    return (TF(fe+df/2., fi,XX)-TF(fe-df/2.,fi,XX))/df                    # deltaTF/deltafe

def diff_fi(TF, fe, fi, XX, df=1e-5):
    return (TF(fe, fi+df/2.,XX)-TF(fe, fi-df/2.,XX))/df                   # delta2TF/deltafi2

def diff2_fe_fe(TF, fe, fi, XX, df=1e-5):
    return (diff_fe(TF, fe+df/2., fi,XX)-diff_fe(TF,fe-df/2.,fi,XX))/df   # delta/deltafe(deltaTF/deltafe)

def diff2_fi_fe(TF, fe, fi, XX, df=1e-5):
    return (diff_fi(TF, fe+df/2., fi,XX)-diff_fi(TF,fe-df/2.,fi,XX))/df   # delta/deltafi(deltaTF/deltafe)

def diff2_fe_fi(TF, fe, fi, XX, df=1e-5):
    return (diff_fe(TF, fe, fi+df/2.,XX)-diff_fe(TF,fe, fi-df/2.,XX))/df  # delta/deltafe(deltaTF/deltafi)

def diff2_fi_fi(TF, fe, fi, XX, df=1e-5):
    return (diff_fi(TF, fe, fi+df/2.,XX)-diff_fi(TF,fe, fi-df/2.,XX))/df  # delta/deltafi(deltaTF/deltafi)


# ## SDE system for time integration via Euler-Maruyama

# In[6]:


#export

# Building the SDE system

def DifferentialOperator(V, TF1, TF2, params, exc_aff_A, exc_aff_B, inh_aff_A, inh_aff_B):
       
    # exc_aff_A: stimulus related excitatory activity in eA
    # exc_aff_B: stimulus related excitatory activitiy in eB
    # inh_aff_A: stimulus related inhibitory activity in iA
    # inh_aff_B: stimulus related inhibitory activity in iB
    
    # Parameters -- Note: If parameters are changed, they should be 
    # changed also in transfer function parameter set!!! So better to keep them fixed, without any change!
    aRS = params[0]
    bRS = params[1]
    aFS = params[2]
    bFS = params[3]
    tauwRS = params[4]
    tauwFS = params[5]
    Ntot = params[6]
    pc = params[7]
    Ne = params[8]
    Ni = params[9]
    vAI = params[10]
    wce = params[11]
    wci = params[12]
    # sigma = params[13]
    El = params[14]
    Qe = params[15]
    Qi = params[16]
    Te = params[17]
    Ti = params[18]
    Gl = params[19]
    Ee = params[20]
    Ei = params[21]
    # tF = params[22]
    # dt = params[23]
    T  = params[24]

    
    # General definitions
    vAI_A = vAI         # base drive to keep Pool A in AI state (given only to excitatory population eA)
    vAI_B = vAI         # base drive to keep Pool B in AI state (given only to excitatory population eB)
    wCe = wce*pc*Ntot/2 # short notation for total coupling weight (/2 since one pool)
    wCi = wci*pc*Ntot/2 # short notation for total coupling weight (/2 since one pool)
    Ne = Ntot*pc/2      # number of excitatory neurons in one pool 
    Ni = Ntot*(1-pc)/2  # number of inhibitory neurons in one pool
    
    # General definitions for Pool A
    v_cross_on_exc_inputA = (V[7]+exc_aff_B+vAI_B)*wCe # excitatory coupling from eB to eA 
    v_cross_on_inh_inputA = (V[7]+exc_aff_B+vAI_B)*wCi # excitatory coupling from eB to iA 

    excinputTF1_A = V[0] + vAI_A + exc_aff_A + v_cross_on_exc_inputA # V[0]: recurrent excitatory coupling for RS-cells
    inhinputTF1_A = V[1] + inh_aff_A + v_cross_on_inh_inputA         # V[1]: in-column inhibitory coupling for RS-cells
    
    excinputTF2_A = V[0] + vAI_A + exc_aff_A + v_cross_on_exc_inputA # V[0]: recurrent excitatory coupling for FS-cells
    inhinputTF2_A = V[1] + inh_aff_A + v_cross_on_inh_inputA         # V[1]: in-column inhibitory coupling for FS-cells
    
    # General definitions for Pool B
    v_cross_on_exc_inputB = (V[0]+exc_aff_A+vAI_A)*wCe # excitatory coupling from eB to eA 
    v_cross_on_inh_inputB = (V[0]+exc_aff_A+vAI_A)*wCi # excitatory coupling from eB to iA 

    excinputTF1_B = V[7] + vAI_B + exc_aff_B + v_cross_on_exc_inputB # V[7]: recurrent excitatory coupling for RS-cells
    inhinputTF1_B = V[8] + inh_aff_B + v_cross_on_inh_inputB         # V[8]: in-column inhibitory coupling for RS-cells
    
    excinputTF2_B = V[7] + vAI_B + exc_aff_B + v_cross_on_exc_inputB # V[7]: recurrent excitatory coupling for FS-cells
    inhinputTF2_B = V[8] + inh_aff_B + v_cross_on_inh_inputB         # V[8]: in-column inhibitory coupling for FS-cells

    
    # POOL A state variables

    def A0(V):
        
        return 1/T*(            .5*V[2]*diff2_fe_fe(TF1, excinputTF1_A, inhinputTF1_A,V[5])+            .5*V[3]*diff2_fe_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])+            .5*V[3]*diff2_fi_fe(TF1, excinputTF1_A, inhinputTF1_A,V[5])+            .5*V[4]*diff2_fi_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])+            V[14]*(diff2_fe_fe(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCe+            diff2_fe_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCi)+            V[16]*(diff2_fe_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCe+            diff2_fi_fi(TF1, excinputTF1_A, inhinputTF1_A, V[5])*wCi)+             0.5*V[9]*(diff2_fe_fe(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCe**2+                       diff2_fi_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCi**2+            2*diff2_fe_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCi*wCe)+            TF1(excinputTF1_A, inhinputTF1_A,V[5])-V[0])



    def A1(V):

          return 1/T*(            .5*V[2]*diff2_fe_fe(TF2, excinputTF2_A,                              inhinputTF2_A,V[6])+            .5*V[3]*diff2_fe_fi(TF2, excinputTF2_A,                              inhinputTF2_A,V[6])+            .5*V[3]*diff2_fi_fe(TF2, excinputTF2_A,                              inhinputTF2_A,V[6])+            .5*V[4]*diff2_fi_fi(TF2, excinputTF2_A,                              inhinputTF2_A,V[6])+             V[14]*(diff2_fe_fe(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCe+                    diff2_fe_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCi)+             V[16]*(diff2_fe_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCe+                    diff2_fi_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCi)+             .5*V[9]*(diff2_fe_fe(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCe**2+                      diff2_fi_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCi**2+                      2*diff2_fe_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCi*wCe)+             TF2(excinputTF2_A, inhinputTF2_A,V[6])-V[1])            
        


    def A2(V):

        return 1/T*(             1./Ne*TF1(excinputTF1_A,inhinputTF1_A,V[5])*                 (1./T-TF1(excinputTF1_A,inhinputTF1_A,V[5]))+             (TF1(excinputTF1_A,inhinputTF1_A,V[5])-V[0])**2+             2.*V[2]*diff_fe(TF1,excinputTF1_A,inhinputTF1_A,V[5])+             2.*V[3]*diff_fi(TF1,excinputTF1_A,inhinputTF1_A,V[5])+             2.*V[14]*(diff_fe(TF1,excinputTF1_A,inhinputTF1_A,V[5])*wCe+             diff_fi(TF1,excinputTF1_A,inhinputTF1_A,V[5])*wCi)-2.*V[2])



    def A3(V): # mu, nu = e,i, then lbd = e then i

        return 1/T*(            (TF1(excinputTF1_A,inhinputTF1_A,V[5])-V[0])*             (TF2(excinputTF2_A,inhinputTF2_A,V[6])-V[1])+            V[2]*diff_fe(TF2, excinputTF2_A, inhinputTF2_A,V[6])+            V[3]*diff_fe(TF1, excinputTF1_A, inhinputTF1_A,V[5])+            V[3]*diff_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])+            V[4]*diff_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])+            V[16]*(diff_fe(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCe+            diff_fi(TF1, excinputTF1_A, inhinputTF1_A,V[5])*wCi)+            V[14]*(diff_fe(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCe+            diff_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCi) +            -2.*V[3])

  
    
    def A4(V): 

        return 1/T*(                 1./Ni*TF2(excinputTF2_A, inhinputTF2_A,V[6])*                     (1./T-TF2(excinputTF2_A, inhinputTF2_A,V[6]))+                 (TF2(excinputTF2_A, inhinputTF2_A,V[6])-V[1])**2+                 2.*V[3]*diff_fe(TF2, excinputTF2_A, inhinputTF2_A,V[6])+                 2.*V[4]*diff_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])+                 2.*V[16]*(diff_fe(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCe+                          diff_fi(TF2, excinputTF2_A, inhinputTF2_A,V[6])*wCi)
                     -2.*V[4])
    

    def A5(V):

        fe_A = 2*wce*Ne*(V[0]+vAI_A+exc_aff_A) + wce*Ne*(V[7]+vAI_B+exc_aff_B)        
        fi_A = 2*wci*Ni*V[1]+ wci*Ne*(V[7]+vAI_B+exc_aff_B)
        muGe_A, muGi_A = Qe*Te*fe_A, Qi*Ti*fi_A
        muG_A = Gl+muGe_A+muGi_A
        muV_A = (muGe_A*Ee+muGi_A*Ei+Gl*El-V[5])/muG_A

        return (-V[5]/tauwRS+(bRS)*V[0]+aRS*(muV_A-El)/tauwRS)


    def A6(V):
        return  (-V[6]/1.0+0.*V[1]) # inhibitory cells do not have any adaptation, therefore 0!


    
    # POOL B state variables    
      


    def A7(V):

        return 1/T*(            .5*V[9]*diff2_fe_fe(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])+            .5*V[10]*diff2_fe_fi(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])+            .5*V[10]*diff2_fi_fe(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])+            .5*V[11]*diff2_fi_fi(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])+            .5*V[2]*(diff2_fe_fe(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])*wCe**2+                      diff2_fi_fi(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])*wCi**2+                      2*diff2_fe_fi(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])*wCe*wCi)+            V[14]*(diff2_fe_fe(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])*wCe+                       diff2_fe_fi(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])*wCi)+            V[15]*(diff2_fe_fi(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])*wCe+                       diff2_fi_fi(TF1, excinputTF1_B,                              inhinputTF1_B,V[12])*wCi)+            TF1(excinputTF1_B, inhinputTF1_B,V[12])-V[7])
    


    def A8(V):     

        return 1/T*(                  .5*V[9]*diff2_fe_fe(TF2, excinputTF2_B,                                      inhinputTF2_B,V[13])+                  .5*V[10]*diff2_fe_fi(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])+                  .5*V[10]*diff2_fi_fe(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])+                  .5*V[11]*diff2_fi_fi(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])+                     .5*V[2]*(diff2_fe_fe(TF2, excinputTF2_B,                                          inhinputTF2_B,V[13])*wCe**2+                              diff2_fi_fi(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])*wCi**2+                              2*diff2_fe_fi(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])*wCe*wCi)+                   V[14]*(diff2_fe_fe(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])*wCe+                              diff2_fe_fi(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])*wCi)+                   V[15]*(diff2_fe_fi(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])*wCe+                              diff2_fi_fi(TF2, excinputTF2_B,                                       inhinputTF2_B,V[13])*wCi)+                   TF2(excinputTF2_B, inhinputTF2_B,V[13])-V[8])                    



    def A9(V):

        return 1/T*(                 1./Ne*TF1(excinputTF1_B, inhinputTF1_B,V[12])*                     (1./T-TF1(excinputTF1_B, inhinputTF1_B,V[12]))+                 (TF1(excinputTF1_B, inhinputTF1_B,V[12])-V[7])**2+                 2.*V[9]*diff_fe(TF1, excinputTF1_B,                                 inhinputTF1_B,V[12])+                 2.*V[10]*diff_fi(TF1, excinputTF1_B,                                  inhinputTF1_B,V[12])+                     2.*V[14]*(diff_fe(TF1, excinputTF1_B,                                  inhinputTF1_B,V[12])*wCe+                           diff_fi(TF1, excinputTF1_B,                                  inhinputTF1_B,V[12])*wCi)-2.*V[9])              



    def A10(V): # mu, nu = e,i, then lbd = e then i

        return 1/T*(                  (TF1(excinputTF1_B, inhinputTF1_B,V[12])-V[7])*                     (TF2(excinputTF2_B,                          inhinputTF2_B,V[13])-V[8])+                  V[9]*diff_fe(TF2, excinputTF2_B,                               inhinputTF2_B,V[13])+                  V[10]*diff_fe(TF1, excinputTF1_B,                                inhinputTF1_B,V[12])+                  V[10]*diff_fi(TF2, excinputTF2_B,                                inhinputTF2_B,V[13])+                  V[11]*diff_fi(TF1, excinputTF1_B, inhinputTF1_B,V[12])+
                  V[15]*(diff_fe(TF1, excinputTF1_B, inhinputTF1_B,V[12])*wCe+\
                         diff_fi(TF1, excinputTF1_B, inhinputTF1_B,V[12])*wCi)+\

                  V[14]*(diff_fe(TF2, excinputTF2_B, inhinputTF2_B,V[13])*wCe+\
                         diff_fi(TF2, excinputTF2_B, inhinputTF2_B,V[13])*wCi)+\
                     -2.*V[10])



    def A11(V):

        return 1/T*(                 1./Ni*TF2(excinputTF2_B,                           inhinputTF2_B,V[13])*(1./T-                   TF2(excinputTF2_B, inhinputTF2_B,V[13]))+                 (TF2(excinputTF2_B,                      inhinputTF2_B,V[13])-V[8])**2+                 2.*V[10]*diff_fe(TF2, excinputTF2_B,                                  inhinputTF2_B,V[13])+                 2.*V[11]*diff_fi(TF2, excinputTF2_B,                                  inhinputTF2_B,V[13])+                     2.*V[15]*(diff_fe(TF2, excinputTF2_B,                                  inhinputTF2_B,V[13])*wCe+                               diff_fi(TF2, excinputTF2_B,                                  inhinputTF2_B,V[13])*wCi)-2.*V[11])                     


    
    def A12(V):
       
        fe_B = 2*wce*Ne*(V[7]+vAI_B+exc_aff_B) + wce*Ne*(V[0]+vAI_A+exc_aff_A)
        fi_B = 2*wci*Ni*V[8]+ wci*Ne*(V[0]+vAI_A+exc_aff_A)
        
        muGe_B, muGi_B = Qe*Te*fe_B, Qi*Ti*fi_B
        muG_B = Gl+muGe_B+muGi_B
        muV_B = (muGe_B*Ee+muGi_B*Ei+Gl*El-V[12])/muG_B

        return (-V[12]/tauwRS+(bRS)*V[7]+aRS*(muV_B-El)/tauwRS)



    def A13(V):
        return  (-V[13]/1.0+0.*V[8])  # inhibitory population does not have any adaptation, therefore 0!

    
    # Cross-pool state variables (cross-pool covariance terms)  
    
    
    def A14(V):


        return 1/T*(                  (TF1(excinputTF1_A, inhinputTF1_A,V[5])-V[0])*                     (TF1(excinputTF1_B,                          inhinputTF1_B,V[12])-V[7])+                  V[14]*diff_fe(TF1, excinputTF1_A,                               inhinputTF1_A,V[5])+                  V[16]*diff_fi(TF1, excinputTF1_A,                                inhinputTF1_A, V[5])+                     V[9]*(diff_fe(TF1, excinputTF1_A,                                inhinputTF1_A, V[5])*wCe+                       diff_fi(TF1, excinputTF1_A,                                inhinputTF1_A, V[5])*wCi)+                     V[2]*(diff_fe(TF1, excinputTF1_B,                                inhinputTF1_B, V[12])*wCe+                       diff_fi(TF1, excinputTF1_B,                                inhinputTF1_B, V[12])*wCi)+                     V[14]*diff_fe(TF1, excinputTF1_B,                                inhinputTF1_B, V[12])+                  V[15]*diff_fi(TF1, excinputTF1_B,                                inhinputTF1_B, V[12])+-2.*V[14])             



    def A15(V):

        return 1/T*(                  (TF1(excinputTF1_A, inhinputTF1_A,V[5])-V[0])*                     (TF2(excinputTF2_B,                          inhinputTF2_B,V[13])-V[8])+                  V[15]*diff_fe(TF1, excinputTF1_A,                               inhinputTF1_A,V[5])+                  V[17]*diff_fi(TF1, excinputTF1_A,                                inhinputTF1_A, V[5])+                     V[10]*(diff_fe(TF1, excinputTF1_A,                                inhinputTF1_A, V[5])*wCe+                       diff_fi(TF1, excinputTF1_A,                                inhinputTF1_A, V[5])*wCi)+                     V[2]*(diff_fe(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])*wCe+                       diff_fi(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])*wCi)+                     V[14]*diff_fe(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])+                  V[15]*diff_fi(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])-2.*V[15])                     





    def A16(V):

        return 1/T*(                  (TF2(excinputTF2_A, inhinputTF2_A,V[6])-V[1])*                     (TF1(excinputTF1_B,                          inhinputTF1_B,V[12])-V[7])+                  V[14]*diff_fe(TF2, excinputTF2_A,                               inhinputTF2_A,V[6])+                  V[16]*diff_fi(TF2, excinputTF2_A,                                inhinputTF2_A, V[6])+                     V[9]*(diff_fe(TF2, excinputTF2_A,                                inhinputTF2_A, V[6])*wCe+                       diff_fi(TF2, excinputTF2_A,                                inhinputTF2_A, V[5])*wCi)+                     V[3]*(diff_fe(TF1, excinputTF1_B,                                inhinputTF1_B, V[12])*wCe+                       diff_fi(TF2, excinputTF2_B,                                inhinputTF1_B, V[13])*wCi)+                     V[16]*diff_fe(TF1, excinputTF1_B,                                inhinputTF1_B, V[12])+                  V[17]*diff_fi(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])+                     -2.*V[16])                     






    def A17(V):

        return 1/T*(                  (TF2(excinputTF2_A, inhinputTF2_A,V[6])-V[1])*                     (TF2(excinputTF2_B,                          inhinputTF2_B,V[13])-V[8])+                  V[15]*diff_fe(TF2, excinputTF2_A,                               inhinputTF2_A,V[6])+                  V[17]*diff_fi(TF2, excinputTF2_A,                                inhinputTF2_A, V[6])+                     V[10]*(diff_fe(TF2, excinputTF2_A,                                inhinputTF2_A, V[6])*wCe+                       diff_fi(TF2, excinputTF2_A,                                inhinputTF2_A, V[5])*wCi)+                     V[3]*(diff_fe(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])*wCe+                       diff_fi(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])*wCi)+                     V[16]*diff_fe(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])+                  V[17]*diff_fi(TF2, excinputTF2_B,                                inhinputTF2_B, V[13])-2.*V[17])                     

    # Return the SDE system state variables as a vector 
    
    res = np.array([A0(V), A1(V), A2(V), A3(V), A4(V), A5(V), A6(V), A7(V), A8(V), A9(V),                         A10(V), A11(V), A12(V), A13(V), A14(V), A15(V), A16(V), A17(V)])
    
    
    
    
    '''def Diff_OP(V, exc_aff=0, exc_aff_B=0, inh_aff=0, pure_exc_aff=0, pure_exc_aff_B=0, inh_fract=0):
                return     res = np.array([A0(V), A1(V), A2(V), A3(V), A4(V), A5(V), A6(V), A7(V), A8(V), A9(V),\
                         A10(V), A11(V), A12(V), A13(V), A14(V), A15(V), A16(V), A17(V)])

            return Diff_OP'''
    
                         
    return res

