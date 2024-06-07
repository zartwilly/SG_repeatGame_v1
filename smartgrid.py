#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:07:22 2024

@author: willy

smartgrid_rg is the smartgrid in the repeat game that centralizes the parameters of the environment
"""
import math
import random as rdm
import numpy as np
import agents as ag
import auxiliary_functions as aux

class Smartgrid :
    
    # This class represent a smartgrid
    maxperiod = None # Number of periods
    prosumers = None # All prosumers inside the smartgrid
    LCostmax = None # Maximum benefit for each prosumer for each period
    LCostmin = None # Minimum benefit for each prosumer for each period
    insg = None # Sum of electricity input from all prosumers
    outsg = None # Sum of electricity outputs from all prosumers
    ValEgoc = None # sum of all valOne 
    ValNoSG = None
    ValSG = None
    Reduct = None
    strategy_profile = None 
    Cost = None
    DispSG = None
    rho = None
    TauS = None # contains the tau array for all players at period t 
    
    
    # TODO to delete
    # piepoplus = None # Unitary price of electricity purchased by EPO
    # piepominus = None # Unitary price of electricity sold by EPO
    # piplus = None # Unitary benefit of electricity sold to SG (independently from EPO)
    # piminus = None # Unitary cost of electricity bought from SG (independently from EPO)
    # unitaryben = None # Unitary benefit of electricity sold to SG (possibly partially to EPO)
    # unitarycost = None # Unitary cost of electricity bought from SG (possibly partially from EPO)
    # betaplus = None # Intermediate values for computing piplus and real benefit 
    # betaminus = None # Intermediate values for computing piminus and real cost
    # czerom = None # Upper bound a prosumer could have to pay
    # realprod = None # Real value of production for each prosumers (different from the predicted production)
    # realstate = None # Real state of each prosumers when using real production value (can be the same as the one determined with predicted production)
    
    
    def __init__(self, N, maxperiod, initialprob, rho):
        """
        N = number of prosumers, 
        maxperiod = max numbers of periods
        initialprob : initial value of probabilities for LRI, 
        rho = steps to be taken into account for stock prediction
        
        """
        self.rho = rho
        self.TauS = np.array(shape=(N, rho+1))
        self.prosumers = np.ndarray(shape=(N),dtype=ag.Prosumer)
        self.maxperiod = maxperiod
        for i in range(N):
            self.prosumers[i] = ag.Prosumer(maxperiod, initialprob)   
        #self.bgmax = np.zeros((N,maxperiod))
        
        self.LCostmax = np.zeros(maxperiod)
        self.LCostmin = np.zeros(maxperiod)
        
        self.insg = np.zeros(maxperiod)       
        self.outsg = np.zeros(maxperiod)
        
        self.ValEgoc = np.zeros(maxperiod)
        self.ValNoSG = np.zeros(maxperiod)
        self.ValSG = np.zeros(maxperiod)
        self.Reduct = np.zeros(maxperiod)
        dt = np.dtype([('agent', np.int), ('strategy', ag.Mode)])
        self.strategy_profile = np.ndarray(shape=(N, maxperiod), dtype=dt)
        self.Cost = np.zeros(maxperiod)
        self.DispSG = np.zeros(rho+1)
        
    ###########################################################################
    #                   compute smartgrid variables :: start
    ###########################################################################
    def computeSumInput(self, period): 
        """
        Calculate the sum of the production of all prosumers during a period
        
        Parameters
        ----------
        period: int 
            an instance of time t
        """
        tmpsum = 0
        for i in range(self.prosumers.size):
            tmpsum = tmpsum + self.prosumers[i].prodit[period]
        self.insg[period] = tmpsum
    
    def computeSumOutput(self, period): 
        """
        Calculate sum of the consumption of all prosumers during a period
        
        Parameters
        ----------
        period: int 
            an instance of time t
        """
        tmpsum = 0
        for i in range(self.prosumers.size):
            tmpsum = tmpsum + self.prosumers[i].consit[period]
        self.outsg[period] = tmpsum
        
    def computeValEgoc(self, period):
        """
        compute ValEgoc ie the sum of all actors valOne

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        sumValEgoc = 0
        for i in range(self.prosumers.size):
            sumValEgoc += self.prosumers[i].valOne[period]
            
        self.valEgoc[period] = sumValEgoc
        
    def computeValNoSG(self, period):
        """
        compute valNoSG for all actors

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        sumValNoSG = 0
        for i in range(self.prosumers.size):
            sumValNoSG += self.prosumers[i].valNoSG[period]
            
        self.ValNoSG[period] = sumValNoSG
        
    def computeValSG(self, period):
        """
        compute the gain of the grid ie what a grid have to pay to EPO minus what a grid receive from EPO

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        outinsg = aux.phiepominus(self.outsg[period] - self.insg[period])
        inoutsg = aux.phiepoplus(self.insg[period] - self.outsg[period])
        self.ValSG[period] = outinsg - inoutsg
    
    def computeReduct(self, period):
        """
        Compute Reduct ie ValNoSG_t - ValSG_t

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        self.Reduct[period] = self.valNoSG[period] - self.ValSG[period]
        
    def computePrice(self, period):
        """
        compute the price by which each actor have to pay or sell an electricity 

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        np.array(N,1) : float

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].price[period] \
                = self.prosumers[i].valNoSG[period] \
                    - self.prosumers[i].Repart[period]
            
        
    def computeDispSG(self, period, h):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t.
        h : int
            the next h periods to predict the stock at the period "period" .
            1 <= h <= rho

        Returns
        -------
        None.

        """
        """
        nextperiod = period if period == self.maxperiod-1 else period+1
        sumDisp_th = 0
        for i in range(self.prosumers.size):
            Stplus1 = self.prosumers[i].storage[nextperiod]
            sumPC_th = 0
            for j in range(1, h):
                Citj, Pitj = None, None
                if period+j <= self.maxperiod:
                    Citj = self.prosumers[i].consumption[period+j]
                    Pitj = self.prosumers[i].production[period+j]
                else:
                    Citj = self.prosumers[i].consumption[self.maxperiod]
                    Pitj = self.prosumers[i].production[self.maxperiod]
                sumPC_th += Pitj - Citj
            sumDisp_th += Stplus1 + sumPC_th
            
        self.DispSG[period] = sumDisp_th
        """
        
        nextperiod = period if period == self.maxperiod-1 else period+1
        sumDisp_th = 0
        for i in range(self.prosumers.size):
            Stplus1 = self.prosumers[i].storage[nextperiod]
            sumDisp_th += Stplus1 + np.sum(self.prosumers[i].PC_th[:h+1])
            
        self.DispSG[h] = sumDisp_th
     
    def computeTau_actors(self, period, rho):
        """
        compute tau_i for all actors

        Parameters
        ----------
        period : int
            an instance of time t
        
        maxperiod: int
            max period
            
        rho: int
            number of steps to select for predition 

        Returns
        -------
        TauS array of shape (N, rho+1).

        """
        
        for i in range(self.prosumers.size):
            self.prosumers[i].computeTau(period, self.maxperiod, rho)
            self.TauS[i] = self.prosumers[i].tau
            self.TauS[i][self.TauS[i] < 0] = 0
            # select ai
            a_ = self.prosumers[i].tau
            if a_[a_<0].size == 0:
                self.prosumer[i].ai = rho +1
            else:
                self.prosumer[i].ai = min(a_[a_<0])
            
        
    def computeHighLow(self, period, rho):
        """
        compute High, Low variables at period t for each actor
        """
        high_itj, low_itj = 0, 0
        self.computeTau_actors(period, rho)
        for i in range(self.prosumers.size):
            for j in range(1, rho+1):
                sumTauSAis = np.sum(self.TauS[:,j])
                if self.DispSG[j] < sumTauSAis:
                    high_itj += aux.apv(self.prosumers[i].tau[j])
                else:
                    low_itj += aux.apv(self.prosumers[i].tau[j])
                    
            self.prosumers[i].High[period] = high_itj
            self.prosumers[i].Low[period] = low_itj
            
    def compute_RS_highPlus(self, period):
        """
        compute rs_{i, High}^{plus} for all actors

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        nextperiod = period if period == self.maxperiod-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_high_plus[period] = \
                min(aux.apv(self.prosumers[i].storage[nextperiod] - self.prosumers[i].storage[period]), 
                    aux.apv( self.prosumers[i].High[period] - self.prosumers[i].storage[period])
                    )
        
    def compute_RS_highMinus(self, period):
        """
        compute rs_{i, High}^{minus} for all actors

        Parameters
        ----------
        Period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        nextperiod = period if period == self.maxperiod-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_high_minus[period] = \
                min(aux.apv(self.prosumers[i].storage[period] - self.prosumers[i].storage[nextperiod]),
                    aux.apv(self.prosumers[i].High[period] - self.prosumers[i].storage[nextperiod])
                    )
        
    def compute_RS_lowPlus(self, period):
        """
        compute rs_{i, Low}^{plus} for all actors

        Parameters
        ----------
        period : int
            an instance of time t.
            
        Returns
        -------
        None.

        """
        nextperiod = period if period == self.maxperiod-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_low_plus[period] = \
                min(aux.apv(self.prosumers[i].storage[nextperiod] - self.prosumers[i].storage[period] - self.prosumers[i].rs_high_plus[period]),
                    aux.apv(self.prosumers[i].High[period] + self.prosumers[i].Low[period] - self.prosumers[i].storage[period]),
                    self.prosumers[i].Low[period]
                    )
        
    def compute_RS_lowMinus(self, period):
        """
        compute rs_{i, Low}^{minus} for all actors

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        nextperiod = period if period == self.maxperiod-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_low_plus[period] = \
                min(aux.apv(self.prosumers[i].storage[period] - self.prosumers[i].storage[nextperiod] - self.prosumers[i].rs_high_plus[period]),
                    aux.apv(self.prosumers[i].High[period] + self.prosumers[i].Low[period] - self.prosumers[i].storage[nextperiod]),
                    self.prosumers[i].Low[period]
                    )
        
    def computeValStock(self, period, rho):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        for i in range(self.prosumers.size):
            ai = self.prosumers[i].ai
            part3 = (rho+1-ai)/rho
            part2 = self.ValEgoc[period]/self.ValNoSG[period]
            part1 = aux.phiepominus(self.prosumers[i].rs_high_plus[period]) \
                    + aux.phiepoplus(self.prosumers[i].rs_low_plus[period]) \
                    - (aux.phiepominus(self.prosumers[i].rs_high_minus[period]) \
                       + aux.phiepoplus(self.prosumers[i].rs_low_minus))
                        
            self.prosumers[i].valStock[period] = part1 * part2 * part3
            
    def computeLCost_LCostMinMax(self, period):
        """
        Compute the learning Cost of all players, 
        learning (max, min) Cost for all players over the learning steps

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].Lcost[period] \
                = self.prosumers[i].price[period] - self.prosumers[i].valStock[period]
                
            if self.prosumers[i].LCostmin["Lcost"] == None \
                or self.prosumers[i].LCostmin["Lcost"] > self.prosumers[i].Lcost[period] :
                self.prosumers[i].LCostmin["Lcost"] = self.prosumers[i].Lcost[period]
                self.prosumers[i].LCostmin["price"] = self.prosumers[i].price[period]
                self.prosumers[i].LCostmin["valStock"] = self.prosumers[i].valStock[period]
                self.prosumers[i].LCostmin["mode"] = self.prosumers[i].mode[period]
                self.prosumers[i].LCostmin["state"] = self.prosumers[i].state[period]
                
            if self.prosumers[i].LCostmax["Lcost"] == None \
                or self.prosumers[i].LCostmax["Lcost"] < self.prosumers[i].Lcost[period] :
                self.prosumers[i].LCostmax["Lcost"] = self.prosumers[i].Lcost[period]
                self.prosumers[i].LCostmax["price"] = self.prosumers[i].price[period]
                self.prosumers[i].LCostmax["valStock"] = self.prosumers[i].valStock[period]
                self.prosumers[i].LCostmax["mode"] = self.prosumers[i].mode[period]
                self.prosumers[i].LCostmax["state"] = self.prosumers[i].state[period]
            
                
        
    ###########################################################################
    #                   compute smartgrid variables :: end
    ###########################################################################    
        
    ###########################################################################
    #                   compute actors' repartition gains :: start
    #                       repart, shapley, UCB
    ########################################################################### 
    def computeRepart(self, period, mu):
        """
        compute the part of the gain of each actor
        
        Parameters
        ----------
        period : int
            an instance of time t
            
        mu: float (mu in [0,1])
            a input parameter of the game

        Returns
        -------
        np.array(N,1) : float
        
        """
        N = self.prosumers.size
        
        part1 = mu * (self.Reduct[period] / N )
        
        
        for i in range(N):
            frac = (self.Reduct[period] * self.prosumers[i].prodit[period]) / self.insg[period]
             
            self.prosumers[i].Repart[period] = part1 + (1-mu) * frac
        

    ###########################################################################
    #                   compute actors' repartition gains :: end
    ########################################################################### 
    
    
    ###########################################################################
    #                       update prosumers variables:: start
    ###########################################################################
    def updateState(self, period): 
        """
        Change prosumer's state based on its production, comsumption and available storage
        
        """
        N = self.prosumers.size
        
        for i in range(N):    
            if self.prosumers[i].production[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] = ag.State.SURPLUS
            
            elif self.prosumers[i].production[period] + self.prosumers[i].storage[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] =ag. State.SELF
            
            else :
                self.prosumers[i].state[period] = ag.State.DEFICIT
                
    def updateSmartgrid(self, period, maxperiod): 
        """
        Update storage , consit, prodit based on mode and state
        
        """
        N = self.prosumers.size
        
        nextperiod = period if period == maxperiod-1 else period+1
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT:
                self.prosumers[i].prodit[period] = 0
                if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS:
                    self.prosumers[i].storage[nextperiod] = 0
                    self.prosumers[i].consit[period] = self.prosumers[i].consumption[period] - (self.prosumers[i].production[period] + self.prosumers[i].storage[period])
                
                else :
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period]
                    self.prosumers[i].consit[period] = self.prosumers[i].consumption[period] - self.prosumers[i].production[period]
            
            elif self.prosumers[i].state[period] == ag.State.SELF:
                self.prosumers[i].prodit[period] = 0
                
                if self.prosumers[i].mode[period] == ag.Mode.CONSMINUS:
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period]
                    self.prosumers[i].consit[period] = self.prosumers[i].consumption[period] - self.prosumers[i].production[period]
                
                else :
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period] - (self.prosumers[i].consumption[period] - self.prosumers[i].production[period])
                    self.prosumers[i].consit[period] = 0
            else :
                self.prosumers[i].consit[period] = 0
                
                if self.prosumers[i].mode[period] == ag.Mode.DIS:
                    self.prosumers[i].storage[nextperiod] = min(self.prosumers[i].smax,self.prosumers[i].storage[period] +\
                                                                   (self.prosumers[i].production[period] - self.prosumers[i].consumption[period]))
                    self.prosumers[i].prodit[period] = aux.apv(self.prosumers[i].production[period] - self.prosumers[i].consumption[period] -\
                                                                (self.prosumers[i].smax - self.prosumers[i].storage[period] ))
                else:
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period]
                    self.prosumers[i].prodit[period] = self.prosumers[i].production[period] - self.prosumers[i].consumption[period]
    
    
    def updateModeSyA(self, period): 
        """
        Update mode using rules from SyA algortihm
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
            elif self.prosumers[i].state[period] == ag.State.SELF :
                self.prosumers[i].mode[period] = ag.Mode.DIS
                
            else :
                self.prosumers[i].mode[period] = ag.Mode.DIS
                
    def updateModeSSA(self, period, maxperiod, rho):
        """
        Update Mode using the self stock algorithm (SSA)
        
        before executing this function, running computeXi from agents.py
        
        """
        for i in range(self.prosumers.size):
            
            self.prosumers[i].computeX(period, maxperiod, rho)
            Xi = self.prosumers[i].Xi[period]
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
            elif self.prosumers[i].state[period] == ag.State.SELF :
                self.prosumers[i].mode[period] = ag.Mode.DIS
                
            elif self.prosumers[i].storage[period] >= Xi:
                self.prosumers[i].mode[period] = ag.Mode.PROD
            else:
                self.prosumers[i].mode[period] = ag.Mode.DIS
    
    ###########################################################################
    #                       update prosumers variables:: end
    ###########################################################################
    