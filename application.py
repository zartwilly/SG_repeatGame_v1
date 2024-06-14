#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:48:57 2024

@author: willy

application is the environment of the repeted game
"""
import numpy as np
import pandas as pd
import agents as ag
import smartgrid as sg
import itertools as it
import json, io, os


class App:
    
    """
    this class is used to call the various algorithms
    """
    
    SG = None # Smartgrid
    N_actors = None  # number of actors
    nbperiod = None # max number of periods for a game.
    maxstep = None # Number of learning steps
    maxstep_init = None # Number of learning steps for initialisation Max, Min Prices. in this step, no strategies' policies are updated
    threshold = None
    mu = None # To define
    b = None # learning rate / Slowdown factor LRI
    h = None # value to indicate how many periods we use to predict futures values of P or C
    rho = None # the next periods to add at Nb_periods
    ObjSG = None # Objective value for a SG over all periods for LRI
    ObjValai = None # Objective value for each actor over all periods for LRI
    valNoSG_A = None # sum of prices payed by all actors during all periods by running algo A without SG
    valSG_A = None # sum of prices payed by all actors during all periods by running algo A with SG
    valNoSGCost_A = None # 
    dicoLRI_onePeriod_oneStep = None # a dictionnary to save a running for one period and one step
    
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
        self.dicoLRI_onePeriod_oneStep = dict()
        
        
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
        
    def runSyA(self, plot:bool, file:bool): 
        """
        Run SyA algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
        """
        T_periods = self.SG.nbperiod
        
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
            
    
    def runSSA(self, plot:bool, file:bool): 
        """
        Run SSA (selfish Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
        """
        T_periods = self.SG.nbperiod
        
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeSSA(period=t)
            
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
        
    def runCSA(self, plot:bool, file:bool): 
        """
        Run CSA (centralised Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
        """
        T_periods = self.SG.nbperiod
        
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
    
        
    def run_LRI_4_onePeriodT_oneStepK(self, period:int, boolInitMinMax:bool):
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
        
        # calculate ValStock
        self.SG.computeValStock(period)
        
        
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
        T = self.SG.nbperiod
        L = self.maxstep_init
        
        for t in range(T):
                        
            # Update the state of each prosumer
            self.SG.updateState(t)
            
            # Initialization game of min/max Learning cost (LearningCost) for prosumers
            for l in range(L):
                self.run_LRI_4_onePeriodT_oneStepK(period=t, boolInitMinMax=True)
                
            # Game with learning steps
            for k in range(K):
                self.run_LRI_4_onePeriodT_oneStepK(period=t, boolInitMinMax=False)
                
            pass
                
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        file.write("___Threshold___ \n")

        ## show prmode for each prosumer at all period and determined when the threshold has been reached
        N = self.SG.prosumers.size
        for Ni in range(N):
            file.write(f"Prosumer = {Ni} \n")
            for t in range(T):
                if (self.SG.prosumers[Ni].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) < self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][0])+" < threshold="+ str(self.threshold) + "\n")
                elif (self.SG.prosumers[Ni].prmode[t][0] >= self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) < self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][0])+" >= threshold="+ str(self.threshold) + " ==>  prob[0] #### \n")
                elif (self.SG.prosumers[Ni].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) >= self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][1])+" >= threshold="+ str(self.threshold) + " ==> prob[1] #### \n")
                else:
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][1])+" >= threshold="+ str(self.threshold) + " ==> prob[1] #### \n")
                
                    
        
        # Determines for each period if it attained a Nash equilibrium and if not if one exist
        file.write("___Nash___ : NOT DEFINE \n")
                
    ######### ----------------   debut : TEST SAVE running  ------------------------------------
    def save_LRI_2_json_onePeriod_oneStep(self, period, step):
        """
        save data from LRI execution for one period 

        Parameters
        ----------
        period : int, 
            DESCRIPTION. The default is t.
        step: int
            one epoch/step for learning

        Returns
        -------
        None.

        """
        N = self.N_actors
        insg = self.SG.insg[period]
        outsg = self.SG.outsg[period]
        ValEgoc = self.SG.ValEgoc[period]
        ValNoSG = self.SG.ValNoSG[period]
        ValSG = self.SG.ValSG[period]
        LCostmax = self.SG.LCostmax[period]
        LCostmin = self.SG.LCostmin[period]
        Cost = self.SG.Cost[period]
        
        #dicoLRI_onePeriod_oneStep = dict()
        for i in range(N):
            production = self.SG.prosumers[i].production[period]
            consumption = self.SG.prosumers[i].consumption[period]
            storage = self.SG.prosumers[i].storage[period]
            rs_highplus = self.SG.prosumers[i].rs_high_plus[period]
            rs_highminus = self.SG.prosumers[i].rs_high_minus[period]
            rs_lowplus = self.SG.prosumers[i].rs_low_plus[period]
            rs_lowminus = self.SG.prosumers[i].rs_low_minus[period]
            prodit = self.SG.prosumers[i].prodit[period]
            consit = self.SG.prosumers[i].consit[period]
            mode = self.SG.prosumers[i].mode[period]
            state = self.SG.prosumers[i].state[period]
            prmode0 = self.SG.prosumers[i].prmode[period][0]
            prmode1 = self.SG.prosumers[i].prmode[period][1]
            utility = self.SG.prosumers[i].utility[period]
            price = self.SG.prosumers[i].price[period]
            valOne_i = self.SG.prosumers[i].valOne[period]
            valNoSG_i = self.SG.prosumers[i].valNoSG[period]
            valStock_i = self.SG.prosumers[i].valStock[period]
            Repart_i = self.SG.prosumers[i].Repart[period]
            cost = self.SG.prosumers[i].cost[period]
            Lcost = self.SG.prosumers[i].Lcost[period]
            
            storage_t_plus_1 = self.SG.prosumers[i].storage[period+1]
            
            self.dicoLRI_onePeriod_oneStep["prosumer"+str(i)] = {
                "period": period,
                "step": step,
                "production":production,
                "consumption": consumption,
                "storage": storage,
                "storaget+1": storage_t_plus_1,
                "rs_high+": rs_highplus,
                "rs_high-": rs_highminus,
                "rs_low+": rs_lowplus,
                "rs_low-": rs_lowminus,
                "prodit": prodit,
                "consit": consit,
                "mode": str(mode),
                "state": str(state),
                "prmode0": prmode0,
                "prmode1": prmode1,
                "utility": utility,
                "price": price,
                "valOne_i": valOne_i,
                "valNoSG_i":valNoSG_i,
                "valStock_i":valStock_i,
                "Repart_i": Repart_i,
                "cost": cost,
                "Lcost": Lcost,
                "insg": insg,
                "outsg": outsg,
                "ValEgoc": ValEgoc,
                "ValNoSG": ValNoSG,
                "ValSG": ValSG,
                "LCostmax": LCostmax,
                "LCostmin": LCostmin,
                "Cost": Cost,
                }
        pass
    
    def runLRI_REPART_SAVERunning(self, plot, file, scenario):
        """
        Run LRI algorithm with the repeated game
        
        Parameters
        ----------
        plot: bool
            yes if you want a figure of some variables
        file : TextIO
            file to save some informations of runtime
        scenario: dict
            DESCRIPTION

        Returns
        -------
        None.

        """
        K = self.maxstep
        T = self.SG.nbperiod
        L = self.maxstep_init
        
        df_ts = []
        for t in range(T):
                        
            # Update the state of each prosumer
            self.SG.updateState(t)
            
            # Initialization game of min/max Learning cost (LearningCost) for prosumers
            for l in range(L):
                print(f"t={t} learning LCostMin_max l={l}")
                self.run_LRI_4_onePeriodT_oneStepK(period=t, boolInitMinMax=True)
                
            # # Game with learning steps
            # for k in range(K):
            #     self.run_LRI_4_onePeriodT_oneStepK(period=t, boolInitMinMax=False)
                
            # pass
        
            # --- DEBUG: START Game with learning steps
            dicoLRI_onePeriod_KStep = dict()
            df_t = []
            for k in range(K):
                self.dicoLRI_onePeriod_oneStep = dict()
                self.run_LRI_4_onePeriodT_oneStepK(period=t, boolInitMinMax=False)
                
                self.save_LRI_2_json_onePeriod_oneStep(period=t, step=k)
                dicoLRI_onePeriod_KStep["step_"+str(k)] = self.dicoLRI_onePeriod_oneStep
                df_tk = pd.DataFrame.from_dict(self.dicoLRI_onePeriod_oneStep, orient="index")
                df_t.append(df_tk)
            
            df_ts.append(df_t)
            
            #####  start : save execution to json file
            try:
                to_unicode = unicode
            except NameError:
                to_unicode = str
            #jsonLRI_onePeriod = json.dumps(dicoLRI_onePeriod_KStep)
            # Write JSON file
            # os.path.join(scenario['scenarioPath'], 'data', f'runLRI_t={t}.json    ====> TODELETE
            # with io.open(os.path.join(scenario['scenarioCorePath'],'data', f'runLRI_t={t}.json'), 'w', encoding='utf8') as outfile:   ====> TODELETE
            with io.open(os.path.join(scenario["scenarioCorePathData"], f'runLRI_t={t}.json'), 'w', encoding='utf8') as outfile:
            #with io.open(f'./data/runLRI_t={t}.json', 'w', encoding='utf8') as outfile:  ====> TODELETE
                str_ = json.dumps(dicoLRI_onePeriod_KStep,
                                  indent=4, sort_keys=True,
                                  #separators=(',', ': '), 
                                  ensure_ascii=False
                                  )
                outfile.write(to_unicode(str_))
            #####  end : save execution to json file
            
            pass
            
            
        
            # --- DEBUG: END Game with learning steps
                
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        file.write("___Threshold___ \n")

        ## show prmode for each prosumer at all period and determined when the threshold has been reached
        N = self.SG.prosumers.size
        for Ni in range(N):
            file.write(f"Prosumer = {Ni} \n")
            for t in range(T):
                if (self.SG.prosumers[Ni].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) < self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][0])+" < threshold="+ str(self.threshold) + "\n")
                elif (self.SG.prosumers[Ni].prmode[t][0] >= self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) < self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][0])+" >= threshold="+ str(self.threshold) + " ==>  prob[0] #### \n")
                elif (self.SG.prosumers[Ni].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) >= self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][1])+" >= threshold="+ str(self.threshold) + " ==> prob[1] #### \n")
                else:
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][1])+" >= threshold="+ str(self.threshold) + " ==> prob[1] #### \n")
                
                    
        
        # Determines for each period if it attained a Nash equilibrium and if not if one exist
        file.write("___Nash___ : NOT DEFINE \n")
        
        
        # merge list of dataframes to one dataframe
        df_ts_ = list(it.chain.from_iterable(df_ts))
        df = pd.concat(df_ts_, axis=0)
        runLRI_SumUp_txt = "runLRI_MergeDF.csv"
        # df.to_csv(os.path.join(scenario['scenarioCorePath'], 'data', runLRI_SumUp_txt))  ====> TODELETE
        df.to_csv(os.path.join(scenario["scenarioCorePathData"], runLRI_SumUp_txt))
        
        
        
        
    ######### -------------------  END : TEST SAVE running  ------------------------------------
            
                
    ######### -------------------  SyA START : TEST SAVE running  ------------------------------------
    def create_dico_for_onePeriod(self, period:int):
        """
        

        Parameters
        ----------
        period : int
            DESCRIPTION.

        Returns
        -------
        dico_onePeriod : dict
            DESCRIPTION.

        """
        dico_onePeriod = dict()
        for i in range(self.N_actors):
            dico_onePeriod["prosumer"+str(i)] = {
                "period": period,
                "production": self.SG.prosumers[i].production[period],
                "consumption": self.SG.prosumers[i].consumption[period],
                "storage": self.SG.prosumers[i].storage[period],
                "storaget+1": self.SG.prosumers[i].storage[period+1],
                "prodit": self.SG.prosumers[i].prodit[period],
                "consit": self.SG.prosumers[i].consit[period],
                "mode": str(self.SG.prosumers[i].mode[period]),
                "state": str(self.SG.prosumers[i].state[period]),
                "prmode0": self.SG.prosumers[i].prmode[period][0],
                "prmode1": self.SG.prosumers[i].prmode[period][1],
                "utility": self.SG.prosumers[i].utility[period],
                "price": self.SG.prosumers[i].price[period],
                "Cost": self.SG.Cost[period],
                "valOne_i": self.SG.prosumers[i].valOne[period],
                "valNoSG_i": self.SG.prosumers[i].valNoSG[period],
                "Repart_i": self.SG.prosumers[i].Repart[period],
                "cost": self.SG.prosumers[i].cost[period],
                "Lcost": self.SG.prosumers[i].Lcost[period],
                "insg": self.SG.insg[period],
                "outsg": self.SG.outsg[period],
                "ValEgoc": self.SG.ValEgoc[period],
                "ValNoSG": self.SG.ValNoSG[period],
                "ValSG": self.SG.ValSG[period],
                }
            
        return dico_onePeriod
    
    def runSyA_SAVERunning(self, plot:bool, file:bool, scenario:dict): 
        """
        Run SyA algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
            
        scenario: dict
            dictionnary of all parameters for a game
            
        """
        T_periods = self.SG.nbperiod
        
        df_ts = []
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
            
            dico_onePeriod = dict()
            dico_onePeriod = self.create_dico_for_onePeriod(period=t)
                
            df_t = pd.DataFrame.from_dict(dico_onePeriod, orient="index")
            df_ts.append(df_t)
            
            
            ## ------ end -------
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        
        # plot variables ValNoSG, ValSG
        
        # merge list of dataframes to one dataframe
        df = pd.concat(df_ts, axis=0)
        runAlgo_SumUp_txt = "runSyA_MergeDF.csv"
        df.to_csv(os.path.join(scenario["scenarioCorePathData"], runAlgo_SumUp_txt))
                
                
    ######### -------------------  SyA END : TEST SAVE running  ------------------------------------
            
    
    ######### -------------------  SSA START : TEST SAVE running  ------------------------------------
    def runSSA_SAVERunning(self, plot:bool, file:bool, scenario:dict): 
        """
        Run SSA (selfish Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
            
        scenario: dict
            dictionnary of all parameters for a game
        
        """
        T_periods = self.SG.nbperiod
        
        df_ts = []
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeSSA(period=t)
            
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
            
            dico_onePeriod = dict()
            dico_onePeriod = self.create_dico_for_onePeriod(period=t)
                
            df_t = pd.DataFrame.from_dict(dico_onePeriod, orient="index")
            df_ts.append(df_t)
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        # plot variables ValNoSG, ValSG
        
        # merge list of dataframes to one dataframe
        df = pd.concat(df_ts, axis=0)
        runAlgo_SumUp_txt = "runSSA_MergeDF.csv"
        df.to_csv(os.path.join(scenario["scenarioCorePathData"], runAlgo_SumUp_txt))
        
    ######### -------------------  SSA END : TEST SAVE running  ------------------------------------

    ######### -------------------  CSA START : TEST SAVE running  ------------------------------------
    def runCSA_SAVERunning(self, plot:bool, file:bool, scenario:dict): 
        """
        Run CSA (centralised Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
            
        scenario: dict
            dictionnary of all parameters for a game
            
        """
        T_periods = self.SG.nbperiod
        
        df_ts = []
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
            
            dico_onePeriod = dict()
            dico_onePeriod = self.create_dico_for_onePeriod(period=t)
                
            df_t = pd.DataFrame.from_dict(dico_onePeriod, orient="index")
            df_ts.append(df_t)
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObjValai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        
        # plot variables ValNoSG, ValSG
    
        # merge list of dataframes to one dataframe
        df = pd.concat(df_ts, axis=0)
        runAlgo_SumUp_txt = "runCSA_MergeDF.csv"
        df.to_csv(os.path.join(scenario["scenarioCorePathData"], runAlgo_SumUp_txt))
        
    ######### -------------------  CSA END : TEST SAVE running  ------------------------------------
    