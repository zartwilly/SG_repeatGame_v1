# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:16:11 2022

modified from 09/06/2024

@author: Quentin
"""

import numpy as np
import random as rdm

class Instancegenaratorv2:
    
    # This is the second version of the instance generator used to generate production and consumption values for prosumers
    
    production = None
    consumption = None
    situation = None
    laststate = None

    def __init__(self, N:int, T:int, rho:int):
        """
        
        We generate N actors and maxPeriod=T+h periods 

        Parameters
        ----------
        N : int
            number of prosumers
        T : int
            number of periods .
        rho : int
            the next periods to add at T periods. In finish, we generate (T + rho) periods.
            This parameter enables the prediction of values from T+1 to T + rho periods
            rho << T ie rho=3 < T=5

        Returns
        -------
        None.

        """
        
        self.production = np.zeros((N, T+rho))
        self.consumption = np.zeros((N,T+rho))
        self.situation = np.zeros((N, T+rho))
        self.laststate = np.ones((N,3))
        
    def generate(self, transitionprobabilities, repartition, values, probabilities):
        """
        transitionprobabilities = probabilities of transition from A to B1, B1 to A, B2 to C, C to B2
        repartition = repartition between the two groups of situation {A,B1} and {B2,C}
        values : matrix containing the ranges of values used in each situation dedicated generator
        values[0] = [m1a,M1a]
        values[1] = [m1b,M1b,m2b,M2b,cb]
        values[2] = [m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
        probabilities : matrix containing probabilities for changing from one state to another inside the two state Markov chains B,C1,C2
        probabilities[0] = [P1b,P2b]
        probabilities[1] = [P1c,P2c,P3c,P4c]
        
        
        """
        
        # Initial random repartition between situation A(1), B1(2), B2(3) and C(4)
        for i in range(repartition[0]):
            self.situation[i][0] = rdm.randint(1,2)
        for i in range(repartition[1]):
            self.situation[repartition[0] + i][0] = rdm.randint(3,4)
        
        for i in range(self.production.shape[0]):
            
            for j in range(self.production.shape[1]):
                
                if self.situation[i][j] == 1 :
                    
                    # Set production and consumption for the period j
                    self.production[i][j] = 0
                    self.consumption[i][j] = rdm.randint(values[0][0],values[0][1])
                    
                    # Define situation for next period
                    if j < self.production.shape[1]-1 :
                        roll = rdm.uniform(0,1)
                        if roll < 1 - transitionprobabilities[0] :
                            self.situation[i][j+1] = 1
                        
                        else :
                            self.situation[i][j+1] = 2
                            self.laststate[i][0] = 1
                    
                elif self.situation[i][j] == 2 :
                    
                    # Set consumption for period j
                    self.consumption[i][j] = values[1][4]
                    
                    # Set production for period j
                    if self.laststate[i][0] == 1:
                        if rdm.uniform(0,1) <= probabilities[0][0]:
                            self.laststate[i][0] = 2    
                        
                        self.production[i][j] = rdm.randint(values[1][0],values[1][1])
                    
                    else:
                        if rdm.uniform(0,1) <= probabilities[0][1]:
                            self.laststate[i][0] = 1  
                        
                        self.production[i][j] = rdm.randint(values[1][2],values[1][3])
                    
                    # Define situation for next period
                    if j < self.production.shape[1]-1 :
                        roll = rdm.uniform(0,1)
                        if roll < 1 - transitionprobabilities[1] :
                            self.situation[i][j+1] = 1
                        
                        else :
                            self.situation[i][j+1] = 2
                        
                elif self.situation[i][j] == 3 :
                    
                    # Set consumption for period j
                    self.consumption[i][j] = values[1][4]
                    
                    # Set production for period j
                    if self.laststate[i][0] == 1:
                        if rdm.uniform(0,1) <= probabilities[0][0]:
                            self.laststate[i][0] = 2  
                            
                        self.production[i][j] = rdm.randint(values[1][0],values[1][1])
                    
                    else:
                        if rdm.uniform(0,1) <= probabilities[0][1]:
                            self.laststate[i][0] = 1  
                        self.production[i][j] = rdm.randint(values[1][2],values[1][3])
                   
                    # Define situation for next period
                    if j < self.production.shape[1]-1 :
                        roll = rdm.uniform(0,1)
                        
                        if roll < 1 - transitionprobabilities[2]:
                            self.situation[i][j+1] = 3
                        
                        else :
                            self.situation[i][j+1] = 4
                            self.laststate[i][1] = 1
                            self.laststate[i][2] = 1
                
                else :
                    self.production[i][j] = rdm.randint(values[2][2],values[2][3])
                    self.consumption[i][j] = rdm.randint(values[2][4],values[2][5])
                                                         
                    # Define situation for next period
                    if j < self.production.shape[1]-1 :
                        roll = rdm.uniform(0,1)
                        if roll < 1 - transitionprobabilities[3] :
                            self.situation[i][j+1] = 3
                            self.laststate[i][0] = 2
                        else :
                            self.situation[i][j+1] = 4
                        

if __name__ == "__main__":
    
    # test code
    transitionprobabilities = [0.5,0.5,0.5,0.5]
    
    N_actors = 10
    T_periods = 10
    rho = 5 # the next periods to add at T_periods. In finish, we generate (T_periods + rho) periods
    g = Instancegenaratorv2(N=N_actors,T=T_periods, rho=rho)
    
    repartition = [5,5]
    #values = [m1a,M1a,m1b,M1b,m2b,M2b,cb,m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
    values = [[5,15],[5,10,25,27,24],[5,10,35,40,20,25,30,35]]
    
    #probabilities = [P1b,P2b,P1c,P2c,P3c,P4c]
    probabilities = [[0.5,0.5],[0.5,0.5,0.6,0.5]]
    
    g.generate(transitionprobabilities, repartition, values, probabilities)            
    
    # print(g.production)
                            
    # print(g.consumption)
                           
                        
                        
                        
                        
                        
                        
                        
                        