#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:26:25 2024

@author: willy
Quentin Version created on Wed Aug 24 10:55:40 2022
"""



import numpy as np
import math
#import random as rdm
#import io
#import pickle
#from enum import Enum
import time
#
#import Instancegenerator as ig
#import Instancegeneratorversion2 as ig2
#import matplotlib.pyplot as plt
#import copy 


class configuration:
    
    # This class represent a configuration i-e a repartition of the prosumers between the two coalitions
    
    identifier = None # Contain id of configuration (binary number of size |actor| + 1, a 0 indicate this actor use LRI, 1 for NoSmart
    sumprod_LRI = None # Contain the sum of prodit for LRI
    sumprod_NoS = None # Contain the sum of prodit for NoSmart
    sumcons_LRI = None # Contain the sum of consit for LRI
    sumcons_NoS = None # Contain the sum of consit for NoSmart
    value = None # Value of the configuration i-e result of the caracteristic function of the game for this configuration
    size = None # Size of the big coalition
    
    def __init__(self,N):
        self.identifier = np.zeros(2**(N), dtype=object)
        self.sumprod_LRI = np.zeros(2**(N))
        self.sumprod_NoS = np.zeros(2**(N))
        self.sumcons_LRI = np.zeros(2**(N))
        self.sumcons_NoS = np.zeros(2**(N))
        self.value = np.zeros(2**(N))
        self.size = np.zeros(2**(N), dtype=np.int32)
    
                    

class redistribution:
    
    
    phiepominus_LRI = None # Parameter : describe benefit of selling energy to EPO using LRI
    phiepoplus_LRI = None # Parameter : describe cost of buying energy from EPO using LRI
    phiepoplus_NoS = None # Parameter : describe benefit of selling energy to EPO using NoSmart
    phiepominus_NoS = None # Parameter : describe cost of buying energy from EPO using NoSmart
    
    prod_LRI = None # Sum of prodit on all period for each actor using LRI
    prod_NoS = None # Sum of prodit on all period for each actor using NoSmart
    
    cons_LRI = None # Sum of consit on all period for each actor using LRI
    cons_NoS = None # Sum of consit on all period for each actor using NoSmart
    
    configuration = None
    
    value = None # Value to redistribute for each actor
    basevalue = None # Value of ER for the LRI 
    
    def __init__(self, phiepoplus_LRI, phiepominus_LRI, phiepoplus_NoS, phiepominus_NoS,
                 N, prod_LRI, prod_NoS, cons_LRI, cons_NoS, basevalue):
        
        self.basevalue = basevalue
        self.phiepominus_LRI = phiepominus_LRI
        self.phiepoplus_LRI = phiepoplus_LRI
    
        self.phiepoplus_NoS = phiepoplus_NoS
        self.phiepominus_NoS = phiepominus_NoS
        
        self.prod_LRI = prod_LRI
        self.prod_NoS = prod_NoS
        
        self.cons_LRI = cons_LRI
        self.cons_NoS = cons_NoS 
        
        self.configuration = configuration(N)
        
        for i in range(2**N,2**(N+1)):
            self.configuration.identifier[i-2**N] = format(i,"b")
            
        self.value = np.zeros(N)
        
        for i in range(2**N):
            # self.configuration.size[i] = len([ch for ch in str(self.configuration.identifier[i]) if ch=='0']) - 1
            self.configuration.size[i] = len([ch for ch in str(self.configuration.identifier[i]) if ch=='0'])
    
    def computeF(self, N):
        """
        Compute caracteristic function of the game
        """
        
        for i in range(2**N):
            
            for j in range(N):
                # Testing the value of the j-th bit of the configuration identifier 0 meaning using LRI and 1 usinf NoSmart
                if ((int(self.configuration.identifier[i])//(10**j)))%2 == 0:
                    self.configuration.sumprod_LRI[i] += self.prod_LRI[j]
                    self.configuration.sumcons_LRI[i] += self.cons_LRI[j]
                else:
                    self.configuration.sumprod_NoS[i] += self.prod_NoS[j]
                    self.configuration.sumcons_NoS[i] += self.cons_NoS[j]
        
            betaminus_LRI = self.phiepominus_LRI
            betaminus_NoS = self.phiepominus_NoS
            betaplus_LRI = self.phiepoplus_LRI
            betaplus_NoS = self.phiepoplus_NoS
            
            self.configuration.value[i] \
                = self.basevalue \
                    - (betaplus_LRI * self.configuration.sumprod_LRI[i] \
                       - betaminus_LRI * self.configuration.sumcons_LRI[i] \
                       + betaplus_NoS * self.configuration.sumprod_NoS[i] \
                       - betaminus_NoS * self.configuration.sumcons_NoS[i] )
                        
        
    def shapley(self, N):
        
        for i in range(2**N):
            for j in range(N):
                # Testing the value of the j-th bit of the configuration identifier 0 meaning using LRI 1 for NoSmart
                if (int(self.configuration.identifier[i])//(10**j))%2 == 0:
                    if N - self.configuration.size[i] - 1 > -1:
                        self.value[j] +=  max(((math.factorial(int(self.configuration.size[i]))*math.factorial((N - int(self.configuration.size[i]) - 1)))\
                            /math.factorial(N)),1/math.factorial(N))*self.configuration.value[i]
                    else :
                        self.value[j] += 1/math.factorial(N) * self.configuration.value[i] 
                    
                else:
                    if N - self.configuration.size[i] - 1 > -1:
                        self.value[j] -=  max(((math.factorial(int(self.configuration.size[i]))*math.factorial((N - int(self.configuration.size[i]) - 1)))\
                            /math.factorial(N)),1/math.factorial(N))*self.configuration.value[i]
                            
                    else :
                        self.value[j] -= 1/math.factorial(N) * self.configuration.value[i] 
      
    def computeShapleyValue(self, N):
        """
        compute shapley Values for all prosumers

        Parameters
        ----------
        N : int
            The number of prosumers.

        Returns
        -------
        value: numpy.array of shape (N).

        """
        
        redi = redistribution(phiepoplus_LRI=self.phiepoplus_LRI, phiepominus_LRI=self.phiepominus_LRI, 
                              phiepoplus_NoS=self.phiepoplus_NoS, phiepominus_NoS=self.phiepominus_NoS, 
                              N=N, 
                              prod_LRI=self.prod_LRI, prod_NoS=self.prod_NoS, 
                              cons_LRI=self.cons_LRI, cons_NoS=self.cons_NoS, 
                              basevalue=self.basevalue)
        # print("identifier = "+ str(redi.configuration.identifier))
        
        # print("size = "+ str(redi.configuration.size))
                
        redi.computeF(N)
        # print("config values = " + str(redi.configuration.value))
        
        redi.shapley(N)
        #print("prosumers shapley values = " + str(redi.value))
        
        return redi.value
        
        
      
if __name__ == '__main__':

    start = time.time()
    
    # Test code
    rng = np.random.default_rng()
    
    N = 20 #3, 20: PROBLEM
    prod_LRI = np.ones(N)
    cons_LRI = np.zeros(N)
    prod_NoS = np.zeros(N)
    cons_NoS = np.zeros(N)
    
    basevalue = 3  # Value of ER for the LRI
    phiepominus_LRI = 3 # rng.integers(low=0, high=5, size=1)[0] # Parameter : describe benefit of selling energy to EPO using LRI
    phiepoplus_LRI = 2 # rng.integers(low=0, high=5, size=1)[0] # Parameter : describe cost of buying energy from EPO using LRI
    phiepoplus_NoS = 1 #rng.integers(low=0, high=5, size=1)[0] # Parameter : describe benefit of selling energy to EPO using NoSmart
    phiepominus_NoS = 1 #rng.integers(low=0, high=5, size=1)[0] # Parameter : describe cost of buying energy from EPO using NoSmart
            
    
    redi = redistribution(phiepoplus_LRI=phiepoplus_LRI, phiepominus_LRI=phiepominus_LRI, 
                          phiepoplus_NoS=phiepoplus_NoS, phiepominus_NoS=phiepominus_NoS, 
                          N=N, 
                          prod_LRI=prod_LRI, prod_NoS=prod_NoS, 
                          cons_LRI=cons_LRI, cons_NoS=cons_NoS, 
                          basevalue=basevalue)

    shapleyValues = redi.computeShapleyValue(N)
    
    print(f"Runtime = {time.time() - start}")
    
    # print("identifier = "+ str(redi.configuration.identifier))
    
    # print("size = "+ str(redi.configuration.size))
            
    # redi.computeF(N)
    # print("config values = " + str(redi.configuration.value))
    
    # redi.shapley(N)
    # print("prosumers shapley values = " + str(redi.value))