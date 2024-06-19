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
    nbperiod = None # Number of periods
    rho = None      # the next periods to add at nbperiod periods. This parameter enables the prediction of values from nbperiod+1 to nbperiod+rho periods with rho << nbperiod
    prosumers = None # All prosumers inside the smartgrid
    LCostmax = None # Maximum benefit for each prosumer for each period
    LCostmin = None # Minimum benefit for each prosumer for each period
    insg = None # Sum of electricity input from all prosumers
    outsg = None # Sum of electricity outputs from all prosumers
    ValEgoc = None # sum of all valOne 
    ValNoSG = None
    ValSG = None
    ValNoSGCost = None
    Reduct = None
    strategy_profile = None 
    Cost = None
    DispSG = None
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
    
    
    def __init__(self, N: int, nbperiod:int, initialprob: float, rho:int):
        """
        
        Parameters
        ----------
        N : int
            number of prosumers
            
        nbperiod : int
            explicit max number of periods for a game.
            
        initialprob : float 
                initial value of probabilities for LRI 
                
        rho : int
            steps to be taken into account for stock prediction
            the next periods to add at nbperiod periods. 
            This parameter enables the prediction of values from nbperiod+1 to nbperiod+rho periods 
            with rho << nbperiod
            
        
        """
        self.rho = rho
        self.TauS = np.ndarray(shape=(N, rho+1))
        self.prosumers = np.ndarray(shape=(N),dtype=ag.Prosumer)
        self.nbperiod = nbperiod
        for i in range(N):
            self.prosumers[i] = ag.Prosumer(nbperiod=nbperiod, initialprob=initialprob, rho=rho)   
        #self.bgmax = np.zeros((N,maxperiod))
        
        self.LCostmax = np.zeros(nbperiod)
        self.LCostmin = np.zeros(nbperiod)
        
        self.insg = np.zeros(nbperiod)       
        self.outsg = np.zeros(nbperiod)
        
        self.ValEgoc = np.zeros(nbperiod)
        self.ValNoSG = np.zeros(nbperiod)
        self.ValSG = np.zeros(nbperiod)
        self.ValNoSGCost = np.zeros(nbperiod)
        self.Reduct = np.zeros(nbperiod)
        dt = np.dtype([('agent', int), ('strategy', ag.Mode)])
        self.strategy_profile = np.ndarray(shape=(N, nbperiod), dtype=dt)
        self.Cost = np.zeros(nbperiod)
        self.DispSG = np.zeros(rho+1)
        
    ###########################################################################
    #                   compute smartgrid variables :: start
    ###########################################################################
    def computeSumInput(self, period:int) -> float: 
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
    
    def computeSumOutput(self, period:int) -> float: 
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
        
    def computeValEgoc(self, period:int) -> float:
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
            self.prosumers[i].computeValOne(period=period, nbperiod=self.nbperiod, rho=self.rho)
            sumValEgoc += self.prosumers[i].valOne[period]
            
        self.ValEgoc[period] = sumValEgoc
        
    def computeValNoSG(self, period:int) -> float:
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
            self.prosumers[i].computeValNoSG(period=period)
            sumValNoSG += self.prosumers[i].valNoSG[period]
            
        self.ValNoSG[period] = sumValNoSG
        
    def computeValSG(self, period:int) -> float:
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
        outinsg = aux.phiepominus( aux.apv(self.outsg[period] - self.insg[period] ))
        inoutsg = aux.phiepoplus( aux.apv(self.insg[period] - self.outsg[period] ))
        self.ValSG[period] = outinsg - inoutsg
    
    def computeValNoSGCost(self, period:int) -> float:
        """
        compute the gain when we have no smart grid.

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        phiPlusInsg = aux.phiepoplus(self.insg[period])
        phiMinusOutsg = aux.phiepominus(self.outsg[period])
        self.ValNoSGCost[period] = phiPlusInsg - phiMinusOutsg
        
    def computeReduct(self, period:int) -> float:
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
        self.Reduct[period] = self.ValNoSG[period] - self.ValSG[period]
        
    def computePrice(self, period:int) -> float:
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
            
            
    def computeDispSG(self, period:int, h:int) -> float:
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
        
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1
        sumDisp_th = 0
        for i in range(self.prosumers.size):
            Stplus1 = self.prosumers[i].storage[nextperiod]
            sumDisp_th += Stplus1 + np.sum(self.prosumers[i].PC_th[:h+1])
            
        self.DispSG[h] = sumDisp_th
    
    def computeTau_actors(self, period:int, rho:int):
        """
        compute tau_i for all actors

        Parameters
        ----------
        period : int
            an instance of time t
            
        rho : int
            number of periods to select for predition 
            rho << nbperiod ie rho=3 < nbperiod=5

        Returns
        -------
        TauS array of shape (N, rho+1).

        """
        
        for i in range(self.prosumers.size):
            self.prosumers[i].computePC_CP_th(period=period, nbperiod=self.nbperiod, rho=rho)
            self.prosumers[i].computeTau(period=period, nbperiod=self.nbperiod, rho=rho)
            self.TauS[i] = self.prosumers[i].tau
            self.TauS[i][self.TauS[i] < 0] = 0
            # select ai
            a_ = self.prosumers[i].tau
            if a_[a_<0].size == 0:
                self.prosumers[i].alphai = rho +1
            else:
                self.prosumers[i].alphai = min(a_[a_<0])
            
        
    # def computeHighLow_OLD(self, period:int) -> float:
    #     """
    #     compute High, Low variables at period t for each actor
        
    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t

    #     Returns
    #     -------
    #     float
        
    #     """
    #     high_itj, low_itj = 0, 0
    #     self.computeTau_actors(period, self.rho)
    #     for i in range(self.prosumers.size):
    #         for j in range(1, self.rho+1):
    #             sumTauSAis = np.sum(self.TauS[:,j])
    #             # if self.DispSG[j] < sumTauSAis:
    #             if self.DispSG[j] < sumTauSAis:
    #                 high_itj += aux.apv(self.prosumers[i].tau[j])
    #             else:
    #                 low_itj += aux.apv(self.prosumers[i].tau[j])
                    
    #         self.prosumers[i].High[period] = high_itj
    #         self.prosumers[i].Low[period] = low_itj
            
    def computeHighLow(self, period:int) -> float:
        """
        compute High, Low variables at period t for each actor
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float
        
        """
        high_itj, low_itj = 0, 0
        # self.computeTau_actors(period, self.rho)
        for i in range(self.prosumers.size):
            for j in range(1, self.rho+1):
                # TODO DEBUG                                                    =====> TODELETE
                sumTauSForRhos = np.sum(self.TauS, axis=0)
                # print(f"sumTauSAis: shape = {sumTauSForRhos.shape}, TauS: shape = {self.TauS.shape}") =====> TODELETE
                # if self.DispSG[j] < sumTauSAis:
                if self.DispSG[j] < sumTauSForRhos[j]:
                    high_itj += aux.apv(self.prosumers[i].tau[j])
                else:
                    low_itj += aux.apv(self.prosumers[i].tau[j])
                    
            self.prosumers[i].High[period] = high_itj
            self.prosumers[i].Low[period] = low_itj
            
    def compute_RS_highPlus(self, period:int) -> float:
        """
        compute rs_{i, High}^{plus} for all actors

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float.

        """
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_high_plus[period] = \
                min(aux.apv(self.prosumers[i].storage[nextperiod] - self.prosumers[i].storage[period]), 
                    aux.apv( self.prosumers[i].High[period] - self.prosumers[i].storage[period])
                    )
        
    def compute_RS_highMinus(self, period:int) -> float:
        """
        compute rs_{i, High}^{minus} for all actors

        Parameters
        ----------
        Period : int
            an instance of time t.

        Returns
        -------
        float.

        """
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_high_minus[period] = \
                min(aux.apv(self.prosumers[i].storage[period] - self.prosumers[i].storage[nextperiod]),
                    aux.apv(self.prosumers[i].High[period] - self.prosumers[i].storage[nextperiod])
                    )
        
    def compute_RS_lowPlus(self, period:int) -> float:
        """
        compute rs_{i, Low}^{plus} for all actors

        Parameters
        ----------
        period : int
            an instance of time t.
            
        Returns
        -------
        float.

        """
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_low_plus[period] = \
                min(aux.apv(self.prosumers[i].storage[nextperiod] - self.prosumers[i].storage[period] - self.prosumers[i].rs_high_plus[period]),
                    aux.apv(self.prosumers[i].High[period] + self.prosumers[i].Low[period] - self.prosumers[i].storage[period]),
                    self.prosumers[i].Low[period]
                    )
        
    def compute_RS_lowMinus(self, period:int) -> float:
        """
        compute rs_{i, Low}^{minus} for all actors

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float.

        """
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1
        
        for i in range(self.prosumers.size):
            self.prosumers[i].rs_low_plus[period] = \
                min(aux.apv(self.prosumers[i].storage[period] - self.prosumers[i].storage[nextperiod] - self.prosumers[i].rs_high_plus[period]),
                    aux.apv(self.prosumers[i].High[period] + self.prosumers[i].Low[period] - self.prosumers[i].storage[nextperiod]),
                    self.prosumers[i].Low[period]
                    )
        
    def computeValStock(self, period:int) -> float:
        """
        calculate the prosumer stock impact during a strategy profile SP^t=strat_{1,s}^t,...,strat_{N,s}^t

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        for i in range(self.prosumers.size):
            alphai = self.prosumers[i].alphai
            part3 = (self.rho+1-alphai) / self.rho
            part2 = self.ValEgoc[period] / self.ValNoSG[period]
            part1 = aux.phiepominus(self.prosumers[i].rs_high_plus[period]) \
                    + aux.phiepoplus(self.prosumers[i].rs_low_plus[period]) \
                    - (aux.phiepominus(self.prosumers[i].rs_high_minus[period]) \
                       + aux.phiepoplus(self.prosumers[i].rs_low_minus[period]))
                        
            # TODO =====> TODELETE
            # print(f"t={period},Ai={i} => part1 : {round(part1, 2)}, part2 : {round(part2, 5)}, part3 : {round(part3, 2)} ")
            # print(f"        =>  rs_high+: {self.prosumers[i].rs_high_plus[period]},")  
            # print(f"        =>  rs_low+: {self.prosumers[i].rs_low_plus[period]},") 
            # print(f"        =>  rs_high-: {self.prosumers[i].rs_high_minus[period]},") 
            # print(f"        =>  rs_low-: {self.prosumers[i].rs_low_minus[period]} ")

            self.prosumers[i].valStock[period] = part1 * part2 * part3
            
    def computeLCost_LCostMinMax(self, period:int):
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
                
            # TODO =====> TODELETE
            # if self.prosumers[i].Lcostmin[period] == 0 \
            #     or self.prosumers[i].Lcostmin[period] > self.prosumers[i].Lcost[period]:
            #         #self.prosumers[i].Lcostmin[period] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmin["Lcost"] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmin["price"] = self.prosumers[i].price[period]
            #         self.prosumers[i].LCostmin["valStock"] = self.prosumers[i].valStock[period]
            #         self.prosumers[i].LCostmin["mode"] = self.prosumers[i].mode[period]
            #         self.prosumers[i].LCostmin["state"] = self.prosumers[i].state[period]
                    
            # if self.prosumers[i].Lcostmax[period] == 0 \
            #     or self.prosumers[i].Lcostmax[period] < self.prosumers[i].Lcost[period] :
            #         #self.prosumers[i].Lcostmax[period] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmax["Lcost"] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmax["price"] = self.prosumers[i].price[period]
            #         self.prosumers[i].LCostmax["valStock"] = self.prosumers[i].valStock[period]
            #         self.prosumers[i].LCostmax["mode"] = self.prosumers[i].mode[period]
            #         self.prosumers[i].LCostmax["state"] = self.prosumers[i].state[period]
            
    def computeUtility(self, period:int): 
        """
        Calculate utility function using min, max and last prosumer's Learning cost (LCost)
        
        Parameters
        ----------
        period : int
            an instance of time t.
            
        Returns
        -------
        None.
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            # TODO =====> TODELETE
            # print(f"i={i}, Lcost={round(self.prosumers[i].Lcost[period],2)}, LCostmax={round(self.prosumers[i].LCostmax['Lcost'],2)}, LCostmin={round(self.prosumers[i].LCostmin['Lcost'], 2)}")
            # if (self.prosumers[i].LCostmax !=0 or self.prosumers[i].LCostmin != 0):
            #     self.prosumers[i].utility[period] \
            #         = (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].Lcost[period]) \
            #             / (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].LCostmin["Lcost"])
            # else:
            #     self.prosumers[i].utility[period] = 0
                
            # print(f"i={i}, LCostmax={self.prosumers[i].LCostmax != 0 }")
            
            if self.prosumers[i].LCostmax["Lcost"] != self.prosumers[i].LCostmin["Lcost"]:
                self.prosumers[i].utility[period] \
                    = (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].Lcost[period]) \
                        / (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].LCostmin["Lcost"])
            else:
                self.prosumers[i].utility[period] = 0
        
    ###########################################################################
    #                   compute smartgrid variables :: end
    ###########################################################################    
        
    ###########################################################################
    #                   compute actors' repartition gains :: start
    #                       repart, shapley, UCB
    ########################################################################### 
    def computeRepart(self, period:int, mu:float):
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
            frac = (self.Reduct[period] * self.prosumers[i].prodit[period]) / max(1, self.insg[period])
             
            self.prosumers[i].Repart[period] = part1 + (1-mu) * frac
        

    ###########################################################################
    #                   compute actors' repartition gains :: end
    ########################################################################### 
    
    
    ###########################################################################
    #                       update prosumers variables:: start
    ###########################################################################
    def updateState(self, period:int): 
        """
        Change prosumer's state based on its production, comsumption and available storage
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        
        
        """
        N = self.prosumers.size
        
        for i in range(N):    
            if self.prosumers[i].production[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] = ag.State.SURPLUS
            
            elif self.prosumers[i].production[period] + self.prosumers[i].storage[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] = ag. State.SELF
            
            else :
                self.prosumers[i].state[period] = ag.State.DEFICIT
                
    def updateSmartgrid(self, period:int): 
        """
        Update storage , consit, prodit based on mode and state
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        
        """
        N = self.prosumers.size
        
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1
        
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
    
    def updateProbaLRI(self, period:int, slowdown:float): 
        """
        Update probability for LRI based mode choice
        
        Parameters
        ----------
        period : int
            an instance of time t.
            
        slowdown : float
            a learning parameter called slowdown factor 0 <= b <= 1
            
        Returns
        -------
        None.
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                if self.prosumers[i].mode[period] == ag.Mode.DIS :
                    self.prosumers[i].prmode[period][0] = min(1, self.prosumers[i].prmode[period][0] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] = min(1, self.prosumers[i].prmode[period][1] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] = 1 - self.prosumers[i].prmode[period][1]
                    
            elif self.prosumers[i].state[period] == ag.State.SELF:
                if self.prosumers[i].mode[period] == ag.Mode.DIS :
                    self.prosumers[i].prmode[period][0] = min(1,self.prosumers[i].prmode[period][0] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] = min(1,self.prosumers[i].prmode[period][1] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] = 1 - self.prosumers[i].prmode[period][1]
            else :
                if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS :
                    self.prosumers[i].prmode[period][0] = min(1,self.prosumers[i].prmode[period][0] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] = min(1,self.prosumers[i].prmode[period][1] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] = 1 - self.prosumers[i].prmode[period][1]
    
    def updateModeLRI(self, period:int, threshold:float): 
        """
        Update mode using rules from LRI
        
        Parameters
        ----------
        period : int
            an instance of time t.
            
        threshold : float
            a parameter for which a we stop learning when prabability mode is greater than threshold
            threshold in [0,1]
            
        Returns
        -------
        None.
        """
        N = self.prosumers.size
        
        for i in range(N):
            rand = rdm.uniform(0,1)
            
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                if (rand <= self.prosumers[i].prmode[period][0] and self.prosumers[i].prmode[period][1] < threshold) or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.PROD
            
            elif self.prosumers[i].state[period] == ag.State.SELF :
                if (rand <= self.prosumers[i].prmode[period][0] and self.prosumers[i].prmode[period][1] < threshold) or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
            
            else :
                if (rand <= self.prosumers[i].prmode[period][0] and self.prosumers[i].prmode[period][1] < threshold) or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                    
    
    def updateModeSyA(self, period:int): 
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
    
    def updateModeCSA(self, period:int): 
        """
        Update mode using rules from CSA algortihm
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                
            elif self.prosumers[i].state[period] == ag.State.SELF :
                self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
            else :
                self.prosumers[i].mode[period] = ag.Mode.PROD
                
    
    def updateModeSSA(self, period:int):
        """
        Update Mode using the self stock algorithm (SSA)
        
        before executing this function, running computeXi from agents.py
        
        """
        for i in range(self.prosumers.size):
            
            self.prosumers[i].computeX(period=period, nbperiod=self.nbperiod, rho=self.rho)
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
    