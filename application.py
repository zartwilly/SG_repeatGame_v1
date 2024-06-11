#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:48:57 2024

@author: willy

application is the environment of the repeted game
"""
import numpy as np
import agents as ag
import smartgrid as sg


class App:
    
    """
    this class is used to call the various algorithms
    """
    
    SG = None # Smartgrid
    N_actors = None  # number of actors
    maxstep = None # Number of learning steps
    maxstep_init = None # Number of learning steps for initialisation Max, Min Prices. in this step, no strategies' policies are updated
    threshold = None
    mu = None # To define
    b = None # learning rate / Slowdown factor LRI
    h = None # value to indicate how many periods we use to predict futures values of P or C
    rho = None
    ObjSG = None # Objective value for a SG over all periods for LRI
    ObjValai = None # Objective value for each actor over all periods for LRI
    valNoSG_A = None # sum of prices payed by all actors during all periods by running algo A without SG
    valSG_A = None # sum of prices payed by all actors during all periods by running algo A with SG
    valNoSGCost_A = None # 
    
    def __init__(self, N_actors, maxstep, mu, b, rho, h, maxstep_init, threshold):
        self.maxstep = maxstep
        self.N_actors = N_actors
        self.mu = mu
        self.b = b
        self.h = h
        self.rho = rho
        self.maxstep_init = maxstep_init
        self.threshold = threshold
        self.ObjSG = 0
        self.ObjValai = np.zeros(N_actors)
        self.valNoSG_A = 0
        self.valSG_A = 0
        
        
    def computeObjValai(self):
        """
        Compute the fitness value of actors at the end of game
        
        Parameters
        ----------
        application : APP
            an instance of an application
        """
                
        for i in range(self.SG.prosumers.size):
            sumObjai = 0
            sumObjai = np.sum(self.SG.prosumers[i].price)
            self.ObjValai[i] = sumObjai
            
    def computeObjSG(self):
        """
        Compute the fitness value of the SG grid at the end of game
        
        Parameters
        ----------
        N_actors : int
            an instance of time t
        """
        
        sumValSG = np.sum(self.SG.ValSG)
        self.ObjSG = sumValSG
        
    def computeValSG(self):
        """
        

        Parameters
        ----------
        application : APP
            an instance of an application
        """
        self.valSG_A = np.sum(self.SG.ValSG)
        
    def computeValNoSG(self):
        """
        

        Parameters
        ----------
        application : APP
            an instance of an application
        """
        self.valNoSG_A = np.sum(self.SG.ValNoSG)
        
    def computeValNoSGCost_A(self):
        """
        

        Returns
        -------
        None.

        """
        self.valNoSGCost_A = np.sum(self.SG.ValNoSGCost)
        
    def runSyA(self, plot, file): 
        """
        Run SyA algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
        """
        T_periods = self.SG.maxperiod
        
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeSyA(period=t)
            
            # Update prodit,consit and period + 1 storage values
            self.SG.updateSmartgrid(period=t)
            
            ## compute what each actor has to paid/gain at period t 
            ## (ValEgo, ValNoSG, ValSG, reduct, repart, price, ) 
            ## ------ start -------
            # Calculate inSG and outSG
            self.SG.computeSumInput(period=t)
            self.SG.computeSumOutput(period=t)
            
            # calculate valNoSGCost_t
            self.SG.computeValNoSGCost(period=t)
            
            # calculate valEgoc_t
            self.SG.computeValEgoc(period=t)
            
            # calculate valNoSG_t
            self.SG.computeValNoSG(period=t)
            
            # calculate ValSG_t
            self.SG.computeValSG(period=t)
            
            # calculate Reduct_t
            self.SG.computeReduct(period=t)
            
            # calculate repart_t
            self.SG.computeRepart(period=t, mu=self.mu)
            
            # calculate price_t
            self.SG.computePrice(period=t)
            
            
            ## ------ end -------
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        
        # plot variables ValNoSG, ValSG
            
    
    def runSSA(self, plot, file): 
        """
        Run SSA (selfish Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
        """
        T_periods = self.SG.maxperiod
        
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeSSA(period=t, maxperiod=self.SG.maxperiod, rho=self.rho)
            
            # Update prodit,consit and period + 1 storage values
            self.SG.updateSmartgrid(period=t)
            
            ## compute what each actor has to paid/gain at period t 
            ## (ValEgo, ValNoSG, ValSG, reduct, repart, price, ) 
            ## ------ start -------
            # Calculate inSG and outSG
            self.SG.computeSumInput(period=t)
            self.SG.computeSumOutput(period=t)
            
            # calculate valNoSGCost_t
            self.SG.computeValNoSGCost(period=t)
            
            # calculate valEgoc_t
            self.SG.computeValEgoc(period=t)
            
            # calculate valNoSG_t
            self.SG.computeValNoSG(period=t)
            
            # calculate ValSG_t
            self.SG.computeValSG(period=t)
            
            # calculate Reduct_t
            self.SG.computeReduct(period=t)
            
            # calculate repart_t
            self.SG.computeRepart(period=t, mu=self.mu)
            
            # calculate price_t
            self.SG.computePrice(period=t)
            
            ## ------ end -------
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        # plot variables ValNoSG, ValSG
        
    def runCSA(self, plot, file): 
        """
        Run CSA (centralised Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
        """
        T_periods = self.SG.maxperiod
        
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeCSA(period=t)
            
            # Update prodit,consit and period + 1 storage values
            self.SG.updateSmartgrid(period=t)
            
            ## compute what each actor has to paid/gain at period t 
            ## (ValEgo, ValNoSG, ValSG, reduct, repart, price, ) 
            ## ------ start -------
            # Calculate inSG and outSG
            self.SG.computeSumInput(period=t)
            self.SG.computeSumOutput(period=t)
            
            # calculate valNoSGCost_t
            self.SG.computeValNoSGCost(period=t)
            
            # calculate valEgoc_t
            self.SG.computeValEgoc(period=t)
            
            # calculate valNoSG_t
            self.SG.computeValNoSG(period=t)
            
            # calculate ValSG_t
            self.SG.computeValSG(period=t)
            
            # calculate Reduct_t
            self.SG.computeReduct(period=t)
            
            # calculate repart_t
            self.SG.computeRepart(period=t, mu=self.mu)
            
            # calculate price_t
            self.SG.computePrice(period=t)
            
            ## ------ end -------
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        # plot variables ValNoSG, ValSG
    
        
    def run_LRI_4_onePeriodT_oneStepK(self, period, boolInitMinMax):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t.
        boolInitMinMax : Bool
            require the game whether LRI probabilities strategies are updated or not.
            if True, LRI probabilities are not updated otherwise.
            This part is used for initializing the min and max values

        Returns
        -------
        None.

        """
        # Update prosumers' modes following LRI mode selection
        self.SG.updateModeLRI(period, self.threshold)
        
        # Update prodit, consit and period + 1 storage values
        self.SG.updateSmartgrid(period)
        
        # Calculate inSG and outSG
        self.SG.computeSumInput(period)
        self.SG.computeSumOutput(period)
    
        ## compute what each actor has to paid/gain at period t 
        ## Calculate ValNoSGCost, ValEgo, ValNoSG, ValSG, Reduct, Repart
        ## ------ start ------
        
        # calculate valNoSGCost_t
        self.SG.computeValNoSGCost(period)
        
        # calculate valEgoc_t
        self.SG.computeValEgoc(period)
        
        # calculate valNoSG_t
        self.SG.computeValNoSG(period)
        
        # calculate ValSG_t
        self.SG.computeValSG(period)
        
        # calculate Reduct_t
        self.SG.computeReduct(period)
        
        # calculate repart_t
        self.SG.computeRepart(period, mu=self.mu)
        
        # calculate price_t
        self.SG.computePrice(period)
        
        ## ------ end ------
        
        # Compute(Update) min/max Learning cost (LearningCost) for prosumers
        self.SG.computeLCost_LCostMinMax(period)
        
        # boolInitMinMax == False, we update probabilities (prmod) of prosumers strategies
        if not boolInitMinMax:
            # Calculate utility
            self.SG.computeUtility(period)
            
            # Update probabilities for choosing modes
            self.SG.updateProbaLRI(period, self.b)
        
        pass
    
    def runLRI_REPART(self, plot, file):
        """
        Run LRI algorithm with the repeated game
        
        Parameters
        ----------
        file : TextIO
            file to save some informations of runtime

        Returns
        -------
        None.

        """
        K = self.maxstep
        T = self.SG.maxperiod
        L = self.maxstep_init
        
        for t in range(T):
                        
            # Update the state of each prosumer
            self.SG.updateState(t)
            
            # Initialization game of min/max Learning cost (LearningCost) for prosumers
            for l in range(L):
                self.run_LRI_4_onePeriodT_oneStepK(t, boolInitMinMax=True)
                
            # Game with learning steps
            for k in range(K):
                self.run_LRI_4_onePeriodT_oneStepK(t, boolInitMinMax=False)
                
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        file.write("___Threshold___ \n")
        # Determines if the threshold has been reached
        N = self.SG.prosumers.size
        for t in range(T):
            for i in range(N):
                if (self.SG.prosumers[i].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[i].prmode[t][1]) < self.threshold):
                    file.write("Threshold not reached for period "+ str(i+1) +"\n") 
                    for Ni in range(N):
                        file.write("Prosumer " + str(Ni) + " : "+ str(self.SG.prosumers[Ni].prmode[i][0]) + "\n")
                    break
                
        
        # Determines for each period if it attained a Nash equilibrium and if not if one exist
        file.write("___Nash___ : NOT DEFINE \n")
                
    
            
                
                
                
                
                
        
        

