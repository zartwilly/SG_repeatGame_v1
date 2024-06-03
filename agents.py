#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:08:11 2024

@author: willy

agents file contains functions and classes for items that are independant actions 
in the game
"""
import numpy as np
from enum import Enum
import fonctions_auxiliaires as fct_aux

#%% state
class State(Enum): # Define possible states for a prosumer
    DEFICIT = "DEFICIT" 
    SELF = "SELF"
    SURPLUS = "SURPLUS"
    
#%% mode
class Mode(Enum):  # Define possible modes for a prosumer
    CONSPLUS = "CONS+"
    CONSMINUS = "CONS-"
    DIS = "DIS"
    PROD = "PROD"
    
#%% prosumer
class Prosumer:
    
    # This class represent a prosumer or an actor of the smartgrid

    state = None  # State of the prosumer for each period , possible values = {DEFICIT,SELF,SURPLUS}
    production = None # Electricity production during each period
    consumption = None # Electricity consumption during each period
    prodit = None # Electricity inserted to the SG
    consit = None # Electricity consumed from the SG
    storage = None # Electricity storage at the beginning of each period
    smax = None # Electricity storage capacity
    gamma = None # Incentive to store or preserve electricity
    mode = None # Mode of the prosumer for each period, possible values = {CONSPLUS,CONSMINUS,DIS,PROD}
    prmode = None # Probability to choose between the two possible mode for each state at each period shaped like prmode[period][mode] (LRI only)
    utility = None # Result of utility function for each period
    #minbg = None # Minimum benefit obtained during all periods
    #maxbg = None # Maximum benefit obtained during all periods
    #benefit = None # Benefits for each period
    
    ##### new parameters variables for Repeated game ########
    # prediction stock
    rho_cons = None # a prediction capacity of an actor for consumption
    rho_prod = None # a prediction capacity of an actor for production
    tau = None # the stock demand of each actor for h next periods
    High = None # a MAX needed stock for each actor at step t
    Low = None  # a MIN needed stock for each actor at step t
    rs_high_plus = None  # a required stock value for max needed stock
    rs_low_plus = None   # a required stock value for min needed stock
    rs_high_minus = None # a missing stock value for max needed stock
    rs_low_minus = None  # a missing stock value for min needed stock
    # what each actor will gain or lose
    ObjValue = None # Value of Objective function of each actor ObjAi
    price = None # price by by each prosumer during all periods
    valOne = None # value 
    valNoSG = None # 
    valStock = None # 
    Repart = None # a repartition function based on shapley value
    cost = None # cost of each actor
    Lcost = None # a learning cost
    benefit = None # reward for each prosumer at each period
    

    def __init__(self, maxperiod, initialprob):
        """
        maxperiod : explicit ; 
        initialprob : initial value of prmode[0]
        """
        
        self.state = np.zeros(maxperiod, dtype=State)
        self.production = np.zeros(maxperiod) 
        self.consumption = np.zeros(maxperiod)
        self.prodit = np.zeros(maxperiod)
        self.consit = np.zeros(maxperiod)
        self.storage = np.zeros(maxperiod) 
        self.smax = 0
        self.gamma = np.zeros(maxperiod) 
        self.mode = np.zeros(maxperiod, dtype=Mode)
        self.prmode = np.zeros((maxperiod,2))
        for i in range(maxperiod):
            self.prmode[i][0] = initialprob
            self.prmode[i][1] = 1 - initialprob
        self.utility = np.zeros(maxperiod)
        self.benefit = np.zeros(maxperiod)
        
        ##### new parameters variables for Repeated game ########
        self.rho_cons = 0
        self.rho_prod = 0
        self.tau = np.zeros(maxperiod)
        self.High = np.zeros(maxperiod)
        self.Low = np.zeros(maxperiod)
        self.rs_high_plus = np.zeros(maxperiod)
        self.rs_low_plus = np.zeros(maxperiod)
        self.rs_high_minus = np.zeros(maxperiod)
        self.rs_low_minus = np.zeros(maxperiod)
        
        self.ObjValue = 0 #np.zeros(maxperiod)
        self.price = np.zeros(maxperiod)
        self.valOne = np.zeros(maxperiod)
        self.valNoSG = np.zeros(maxperiod)
        self.valStock = np.zeros(maxperiod)
        self.Repart = np.zeros(maxperiod)
        self.cost = np.zeros(maxperiod)
        self.Lcost = np.zeros(maxperiod)
        self.benefit = 0

    def computeValOne(self, period, maxperiod):
        """
        compute the value of each actor at one period

        Parameters
        ----------
        period : int
            an instance of time t
            
        maxperiod : int
            explicit max number of periods for a game

        Returns
        -------
        float.

        """
        nextperiod = period if period == maxperiod-1 else period+1
        
        
        phiminus = fct_aux.apv(self.consumption[period] 
                               - self.production[period] 
                               - self.storage[period])
        phiplus = fct_aux.apv(self.production[period] 
                              - self.consumption[period] 
                              - (self.storage[nextperiod] - self.storage[period]))
        
        self.valOne[period] = phiminus - phiplus
        
    def computeValNoSG(self, period):
        """
        compute the value of one actor a_i at one period

        Parameters
        ----------
        period : TYPE
            an instance of time t

        Returns
        -------
        float.

        """
        self.valNoSG[period] = fct_aux.phiminus(self.consit[period]) \
                                - fct_aux.phiplus(self.prodit[period])
        
        