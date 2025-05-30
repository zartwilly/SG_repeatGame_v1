# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:16:11 2022

modified from 09/06/2024

@author: Quentin
"""

import numpy as np
import random as rdm
import json

NUM = 3

class Instancegenaratorv2:
    
    # This is the second version of the instance generator used to generate production and consumption values for prosumers
    
    production = None
    consumption = None
    storage = None
    storage_max = None
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
        
        self.production = np.zeros((N, T+NUM*rho))
        self.consumption = np.zeros((N,T+NUM*rho))
        self.storage = np.zeros((N,T+NUM*rho))
        self.storage_max = np.zeros((N,T+NUM*rho))
        self.situation = np.zeros((N, T+NUM*rho))
        self.laststate = np.ones((N,3))
        
        
    def insert_random_PCvalues_4_prosumers(self, i, t, transitionprobabilities, repartition, values, probabilities):
        """
        allocate random values of production and consumption for prosumers according to 
        transitionprobabilities, repartition, probabilities
        """
        if self.situation[i][t] == 1 :
            
            # Set production and consumption for the period j
            self.production[i][t] = 0
            self.consumption[i][t] = rdm.randint(values[0][0],values[0][1])
            
            # Define situation for next period
            if t < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                if roll < 1 - transitionprobabilities[0] :
                    self.situation[i][t+1] = 1
                
                else :
                    self.situation[i][t+1] = 2
                    self.laststate[i][0] = 1
            
        elif self.situation[i][t] == 2 :
            
            # Set consumption for period j
            self.consumption[i][t] = values[1][4]
            
            # Set production for period j
            if self.laststate[i][0] == 1:
                if rdm.uniform(0,1) <= probabilities[0][0]:
                    self.laststate[i][0] = 2    
                
                self.production[i][t] = rdm.randint(values[1][0],values[1][1])
            
            else:
                if rdm.uniform(0,1) <= probabilities[0][1]:
                    self.laststate[i][0] = 1  
                
                self.production[i][t] = rdm.randint(values[1][2],values[1][3])
            
            # Define situation for next period
            if t < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                if roll < 1 - transitionprobabilities[1] :
                    self.situation[i][t+1] = 1
                
                else :
                    self.situation[i][t+1] = 2
                
        elif self.situation[i][t] == 3 :
            
            # Set consumption for period j
            self.consumption[i][t] = values[1][4]
            
            # Set production for period j
            if self.laststate[i][0] == 1:
                if rdm.uniform(0,1) <= probabilities[0][0]:
                    self.laststate[i][0] = 2  
                    
                self.production[i][t] = rdm.randint(values[1][0],values[1][1])
            
            else:
                if rdm.uniform(0,1) <= probabilities[0][1]:
                    self.laststate[i][0] = 1  
                self.production[i][t] = rdm.randint(values[1][2],values[1][3])
           
            # Define situation for next period
            if t < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                
                if roll < 1 - transitionprobabilities[2]:
                    self.situation[i][t+1] = 3
                
                else :
                    self.situation[i][t+1] = 4
                    self.laststate[i][1] = 1
                    self.laststate[i][2] = 1
        
        else :
            self.production[i][t] = rdm.randint(values[2][2],values[2][3])
            self.consumption[i][t] = rdm.randint(values[2][4],values[2][5])
                                                 
            # Define situation for next period
            if t < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                if roll < 1 - transitionprobabilities[3] :
                    self.situation[i][t+1] = 3
                    self.laststate[i][0] = 2
                else :
                    self.situation[i][t+1] = 4
        pass
    
    def generate(self, transitionprobabilities, repartition, values, probabilities, scenario):
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
                
                self.storage_max[i][j] = scenario.get('instance').get('smax')
                
                self.insert_random_PCvalues_4_prosumers( i, j, transitionprobabilities, 
                                                        repartition, values, probabilities)
                # if self.situation[i][j] == 1 :
                    
                #     # Set production and consumption for the period j
                #     self.production[i][j] = 0
                #     self.consumption[i][j] = rdm.randint(values[0][0],values[0][1])
                    
                #     # Define situation for next period
                #     if j < self.production.shape[1]-1 :
                #         roll = rdm.uniform(0,1)
                #         if roll < 1 - transitionprobabilities[0] :
                #             self.situation[i][j+1] = 1
                        
                #         else :
                #             self.situation[i][j+1] = 2
                #             self.laststate[i][0] = 1
                    
                # elif self.situation[i][j] == 2 :
                    
                #     # Set consumption for period j
                #     self.consumption[i][j] = values[1][4]
                    
                #     # Set production for period j
                #     if self.laststate[i][0] == 1:
                #         if rdm.uniform(0,1) <= probabilities[0][0]:
                #             self.laststate[i][0] = 2    
                        
                #         self.production[i][j] = rdm.randint(values[1][0],values[1][1])
                    
                #     else:
                #         if rdm.uniform(0,1) <= probabilities[0][1]:
                #             self.laststate[i][0] = 1  
                        
                #         self.production[i][j] = rdm.randint(values[1][2],values[1][3])
                    
                #     # Define situation for next period
                #     if j < self.production.shape[1]-1 :
                #         roll = rdm.uniform(0,1)
                #         if roll < 1 - transitionprobabilities[1] :
                #             self.situation[i][j+1] = 1
                        
                #         else :
                #             self.situation[i][j+1] = 2
                        
                # elif self.situation[i][j] == 3 :
                    
                #     # Set consumption for period j
                #     self.consumption[i][j] = values[1][4]
                    
                #     # Set production for period j
                #     if self.laststate[i][0] == 1:
                #         if rdm.uniform(0,1) <= probabilities[0][0]:
                #             self.laststate[i][0] = 2  
                            
                #         self.production[i][j] = rdm.randint(values[1][0],values[1][1])
                    
                #     else:
                #         if rdm.uniform(0,1) <= probabilities[0][1]:
                #             self.laststate[i][0] = 1  
                #         self.production[i][j] = rdm.randint(values[1][2],values[1][3])
                   
                #     # Define situation for next period
                #     if j < self.production.shape[1]-1 :
                #         roll = rdm.uniform(0,1)
                        
                #         if roll < 1 - transitionprobabilities[2]:
                #             self.situation[i][j+1] = 3
                        
                #         else :
                #             self.situation[i][j+1] = 4
                #             self.laststate[i][1] = 1
                #             self.laststate[i][2] = 1
                
                # else :
                #     self.production[i][j] = rdm.randint(values[2][2],values[2][3])
                #     self.consumption[i][j] = rdm.randint(values[2][4],values[2][5])
                                                         
                #     # Define situation for next period
                #     if j < self.production.shape[1]-1 :
                #         roll = rdm.uniform(0,1)
                #         if roll < 1 - transitionprobabilities[3] :
                #             self.situation[i][j+1] = 3
                #             self.laststate[i][0] = 2
                #         else :
                #             self.situation[i][j+1] = 4
                  
      
    def generate_TESTDBG(self, transitionprobabilities, repartition, values, probabilities, scenario):
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
            
            for t in range(self.production.shape[1]):
                
                self.storage_max[i][t] = scenario.get('instance').get('smax')
                
                if i < 10 :
                    self.consumption[i][t] = 10
                    if t % 15 < 9 :
                        self.production[i][t] = 11
                    else:
                        self.production[i][t] = 9

                else:
                    self.consumption[i][t] = 3
                    if t % 15 < 5 :
                        self.production[i][t] = 3
                    else:
                        self.production[i][t] = 2
                
    
    def generate_data(self, transitionprobabilities, repartition, values, probabilities, scenario):
        """
        generate data from random values and specific values of Smax
        
        N=8, T=20, rho=5
        each player has data of T+rho periods
        
        we have 2 groups of actors : GA and PA
        Ga have production = 10, consumption = 4 and storage_max = 6
        PA have production = 0, consumption = 6 and storage_max = 2

        Returns
        -------
        None.

        """
        
        # Initial random repartition between situation A(1), B1(2), B2(3) and C(4)
        for i in range(repartition[0]):
            self.situation[i][0] = rdm.randint(1,2)
        for i in range(repartition[1]):
            self.situation[repartition[0] + i][0] = rdm.randint(3,4)
            
        # generate consumption and production for T+rho periods
        for i in range(self.production.shape[0]):
            
            for t in range(self.production.shape[1]):
                
                if i < 10:
                    self.storage_max[i][t] = 5
                else:
                    self.storage_max[i][t] = 2
                
                self.insert_random_PCvalues_4_prosumers( i, t, transitionprobabilities, 
                                                        repartition, values, probabilities)
                

    def generate_dataset_version20092024(self, transitionprobabilities, repartition, values, probabilities):
        """
        generate data from overleaf version of 20/09/2024
        this version contains new a version of stock value prediction with 
        variables SP, cal_G, Help.
        
        N=8, T=20, rho=5
        each player has data of T+rho periods
        
        we have 2 groups of actors : GA and PA
        Ga have production = 10, consumption = 4 and storage_max = 6
        PA have production = 0, consumption = 6 and storage_max = 2

        Returns
        -------
        None.

        """
        
        # Initial random repartition between situation A(1), B1(2), B2(3) and C(4)
        for i in range(repartition[0]):
            self.situation[i][0] = rdm.randint(1,2)
        for i in range(repartition[1]):
            self.situation[repartition[0] + i][0] = rdm.randint(3,4)
            
        # generate consumption and production for T+rho periods
        for i in range(self.production.shape[0]):
            
            for t in range(self.production.shape[1]):
                
                if i<4 and t<10:
                    self.consumption[i][t] = 4
                    self.production[i][t] = 10
                    self.storage_max[i][t] = 6
                elif i<4 and t>=10:
                    self.consumption[i][t] = 4
                    self.production[i][t] = 1
                    self.storage_max[i][t] = 6
                elif i>=4 and t<5:
                    self.consumption[i][t] = 6
                    self.production[i][t] = 0
                    self.storage_max[i][t] = 2
                elif i>=4 and t>=5:
                    self.consumption[i][t] = 4
                    self.production[i][t] = 0
                    self.storage_max[i][t] = 2
                    
          
    def generate_data_GivenStrategies(self, scenario):
        """
         generate data from a given strategies coming to a scenario file

        Parameters
        ----------
        scenario : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # generate consumption and production for T+rho periods
        if scenario.get("simul").get("debug_data") is None:
            pass
        else:
            for i in range(self.production.shape[0]):
                for t in range(self.production.shape[1]):
                    self.consumption[i][t] = scenario.get("simul")\
                                                .get("debug_data")\
                                                .get("t_"+str(t))\
                                                .get("a_"+str(i))\
                                                .get("C")
                    
                    self.production[i][t] = scenario.get("simul")\
                                                .get("debug_data")\
                                                .get("t_"+str(t))\
                                                .get("a_"+str(i))\
                                                .get("P")
                                                
                    self.storage[i][t] = scenario.get("simul")\
                                                .get("debug_data")\
                                                .get("t_"+str(t))\
                                                .get("a_"+str(i))\
                                                .get("S")
                    self.storage_max[i][t] = scenario.get("simul")\
                                                .get("debug_data")\
                                                .get("t_"+str(t))\
                                                .get("a_"+str(i))\
                                                .get("Smax")
                    pass
                
                pass
        

if __name__ == "__main__":
    
    # test code
    transitionprobabilities = [0.5,0.5,0.5,0.5]
    
    N_actors = 10
    T_periods = 10
    rho = 5 # the next periods to add at T_periods. In finish, we generate (T_periods + rho) periods
    # g = Instancegenaratorv2(N=N_actors,T=T_periods, rho=rho)
    
    repartition = [5,5]
    #values = [m1a,M1a,m1b,M1b,m2b,M2b,cb,m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
    values = [[5,15],[5,10,25,27,24],[5,10,35,40,20,25,30,35]]
    
    #probabilities = [P1b,P2b,P1c,P2c,P3c,P4c]
    probabilities = [[0.5,0.5],[0.5,0.5,0.6,0.5]]
    
    # g.generate(transitionprobabilities, repartition, values, probabilities)            
    
    # print(g.production)
                            
    # print(g.consumption)
    
    
    # test with scenariofile
    scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho5.json"
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    g = Instancegenaratorv2(N=scenario["instance"]["N_actors"],
                            T=scenario["simul"]["nbPeriod"], 
                            rho=scenario["simul"]["rho"])
    g.generate_data_GivenStrategies(scenario=scenario)  
                           
                        
                        
                        
                        
                        
                        
                        
                        