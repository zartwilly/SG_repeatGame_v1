#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:04:24 2024

@author: willy

run the developped algortihms for evaluation some variables
"""
import io
import json
import pickle
import os.path
import application as apps
import Instancegeneratorversion2 as ig2
import smartgrid as sg
import auxiliary_functions as aux
import agents as ag

from pathlib import Path



###############################################################################
#                DEBUT : ALGO LRI_REPART_DBG
###############################################################################

#------------------------------------------------------------------------------
#                FIN : Save data in the log file
#------------------------------------------------------------------------------
def monitoring_after_algorithm(algoName, file, application):
    """
    
    monitoring some variables after running some algorithms
    

    Returns
    -------
    None.

    """ 
    file.write("\n___Storage___ \n")
    for i in range(application.SG.prosumers.size):
        file.write("__Prosumer " + str(i + 1) + "___\n")
        for t in range(application.SG.nbperiod):
            file.write("Period " + str(t + 1))
            file.write(" : Storage : " + str(application.SG.prosumers[i].storage[t])+ "\n")
            
    file.write("\n___InSG, OutSG___ \n")
    for t in range(application.SG.nbperiod):
        file.write(" *** Period " + str(t + 1))
        file.write(" InSG : " + str(application.SG.insg[t]))
        file.write(" OutSG: "+ str(application.SG.outsg[t]))
        file.write(" valNoSGCost: " + str(application.SG.ValNoSGCost[t]) +"*** \n")
        for i in range(application.SG.prosumers.size):
            file.write("__Prosumer " + str(i + 1) +":")
            file.write(" Cons = "+ str(application.SG.prosumers[i].consit[t]))
            file.write(", Prod = "+ str(application.SG.prosumers[i].prodit[t]))
            file.write(", mode = "+ str(application.SG.prosumers[i].mode[t]))
            file.write(", state = "+ str(application.SG.prosumers[i].state[t]))
            file.write("\n")
            
    file.write("\n___Metrics___"+ "\n")
    file.write("ValSG : "+ str(application.valSG_A)+ "\n")
    file.write("valNoSG_A    : "+ str(application.valNoSG_A)+ "\n")
    file.write("valNoSGCost_A    : "+ str(application.valNoSGCost_A)+ "\n")
    file.write("ValObjAi    : "+"\n")
    for i in range(application.SG.prosumers.size):
        file.write("__Prosumer " + str(i + 1) + "___ :" +str(round(application.ObjValai[i], 2)) + "\n")
        
    file.write(f"________RUN END {algoName} " + str(1) +"_________" + "\n\n")
    
def monitoring_before_algorithm(file, application):
    """
    monitoring some variables BEFORE running some algorithms

    Returns
    -------
    None.

    """
    print("________RUN ",1,"_________")
    file.write("________RUN " + str(1) +"_________" + "\n")
    
    file.write("\n___Configuration___ \n")
    for i in range(application.SG.prosumers.size):
        file.write("__Prosumer " + str(i + 1) + "___\n")
        for t in range(application.SG.nbperiod):
            file.write("Period " + str(t + 1))
            file.write(" : Production : " + str(application.SG.prosumers[i].production[t]))
            file.write(" Consumption : " + str(application.SG.prosumers[i].consumption[t]))
            file.write(" Storage : " + str(application.SG.prosumers[i].storage[t])+ "\n")
            
#------------------------------------------------------------------------------
#                FIN : Save data in the log file
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
#                DEBUT : Generer des donnees selon scenarios
#------------------------------------------------------------------------------
def generer_data_from_scenario(scenario:dict,
                                   N_actors:int, nbperiod:int, rho:int,
                                    transitionprobabilities:list,
                                   repartition:list,
                                   values:list, 
                                   probabilities:list, 
                                   dbg_generateData:bool=True):
    # scenarioPath : "data_scenario"
    # scenarioName : "data_NAME_DAY-MM-YY-HH-MM.pkl"
    
    # path_name = os.path.join(scenarioPath, scenarioName+".pkl") ===> TODELETE
    path_name = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], scenario["scenarioName"]+".pkl")
    
    g = None
    # Is there the data file on repository?
    checkfile = os.path.isfile(path_name)
    if (checkfile == True and dbg_generateData == True) or (checkfile == True and dbg_generateData == False):
        # file exists in which form?
        # return g which contains data load
        
        print("**** Load pickle data: START ****")
        # with open(os.path.join(scenarioPath, scenarioName+'.pkl'), 'rb') as f:  # open a text file ===> TODELETE
        with open(os.path.join(scenario["scenarioCorePath"], scenario["scenarioName"]+'.pkl'), 'rb') as f:  # open a text file
            g = pickle.load(f)
        f.close()
        
        print("**** Load pickle data : END ****")
        
    else:
        # checkfile == False and dbg_generateData == False or  =(checkfile == False and dbg_generateData == True) :
        print("**** Create pickle data : START ****")
        # file not exists
        g = ig2.Instancegenaratorv2(N=N_actors, T=nbperiod, rho=rho)
        
        if dbg_generateData:
            g.generate_TESTDBG(transitionprobabilities,repartition,values,probabilities)
        else:
            g.generate(transitionprobabilities,repartition,values,probabilities)
        
        # with open(os.path.join(scenarioPath, scenarioName+'.pkl'), 'wb') as f:  # open a text file ===> TODELETE
        with open(os.path.join(scenario["scenarioCorePath"], scenario["scenarioName"]+'.pkl'), 'wb') as f:  # open a text file
            pickle.dump(g, f)
        f.close()
        
        print("**** Create pickle data : END ****")
        
    return g


def Initialization_game(scenario):
    """
    initialization of variables of an object application  for DEBUGGING
    
    Returns
    -----
    App
    """
    # Load all scenario parameters
    scenarioPath = scenario["scenarioPath"]
    scenarioName = scenario["scenarioName"]
    N_actors = scenario["instance"]["N_actors"]
    nbPeriod = scenario["simul"]["nbPeriod"]
    maxstep = scenario["algo"]["LRI_REPART"]["maxstep"]
    maxstep_init = scenario["algo"]["LRI_REPART"]["maxstep_init"]
    mu = scenario["algo"]["LRI_REPART"]["mu"]
    threshold = scenario["algo"]["LRI_REPART"]["threshold"]
    slowdownfactor = scenario["algo"]["LRI_REPART"]["slowdownfactor"]
    rho = scenario["simul"]["rho"]
    h = scenario["algo"]["LRI_REPART"]["h"]
    initialprob = scenario["algo"]["LRI_REPART"]["initialprob"]
    transitionprobabilities = scenario["simul"]["transitionprobabilities"]
    repartition = scenario["simul"]["repartition"]
    values = scenario["simul"]["values"]
    probabilities = scenario["simul"]["probabilities"]
    dbg_generateData = scenario["dbg_generateData"]
    # = scenario[""]
    
    scenario["scenarioCorePath"] = os.path.join(scenario["scenarioPath"], scenario["scenarioName"])
    
    
    # Initialisation of the apps
    application = apps.App(N_actors=N_actors, maxstep=maxstep, mu=mu, 
                           b=slowdownfactor, rho=rho, h=h, 
                           maxstep_init=maxstep_init, threshold=threshold)
    application.SG = sg.Smartgrid(N=N_actors, nbperiod=nbPeriod, 
                                  initialprob=initialprob, rho=rho)
    
    # Configuration of the instance generator
    g = generer_data_from_scenario(scenario=scenario,
                                       N_actors=N_actors, nbperiod=nbPeriod, 
                                       rho=rho,
                                       transitionprobabilities=transitionprobabilities,
                                       repartition=repartition,
                                       values=values, 
                                       probabilities=probabilities, 
                                       dbg_generateData=dbg_generateData)
    
    # Initialisation of production, consumption and storage using the instance generator
    N = application.SG.prosumers.size
    T = application.SG.nbperiod
    
    for i in range(N):
        for t in range(T):
            application.SG.prosumers[i].production[t] = g.production[i][t]
            application.SG.prosumers[i].consumption[t] = g.consumption[i][t]
            
            if dbg_generateData:
                application.SG.prosumers[i].smax = 5 if i < 10 else 2
                # if i < 10 :
                #     application.SG.prosumers[i].smax = 5
                # else:
                #     application.SG.prosumers[i].smax = 2
            else:
                # put initial storage variable 
                application.SG.prosumers[i].storage[0] = 0
                application.SG.prosumers[i].smax = 15 #20
            
        # put initial storage variable 
        application.SG.prosumers[i].storage[0] = 0
        
    return application

def create_repo_for_save_jobs(scenario:dict):
    scenarioCorePath = os.path.join(scenario["scenarioPath"], scenario["scenarioName"])
    scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas")
    scenarioCorePathDataAlgoName = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", scenario["algoName"])
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
    scenario["scenarioCorePath"] = scenarioCorePath
    scenario["scenarioCorePathData"] = scenarioCorePathData
    scenario["scenarioCorePathDataAlgoName"] = scenarioCorePathDataAlgoName
    scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
    
    
    # create a scenarioPath if not exists
    Path(scenarioCorePathData).mkdir(parents=True, exist_ok=True)
    Path(scenarioCorePathDataAlgoName).mkdir(parents=True, exist_ok=True)
    Path(scenarioCorePathDataViz).mkdir(parents=True, exist_ok=True)
    
    
    return scenario

#------------------------------------------------------------------------------
#                FIN : Generer des donnees selon scenarios
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : algo CSA avec instance charge
#------------------------------------------------------------------------------
def run_CSA(scenario, logfiletxt):
    """
    run CSA algorithm

    Parameters
    ----------
    logfile : txt
        path Logs file 
    Returns
    -------
    None.

    """
    algoName = "CSA"
    scenario["algoName"] = algoName
    scenario = create_repo_for_save_jobs(scenario)
    
    # Initialisation of the apps
    # application = Initialization_game(scenario)
    application = Initialization_game(scenario)

    # Display for the run beginning 
    logfile = os.path.join(scenario["scenarioCorePathDataAlgoName"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                                # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # Execute CSA
    file.write("\n_______CSA_______"+ "\n")
    application.runCSA(plot=False, file=file, scenario=scenario)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    
    # End execute CSA
    print("________RUN END CSA ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathDataAlgoName"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    return application
#------------------------------------------------------------------------------
#                FIN : algo CSA avec instance charge
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : algo SSA avec instance charge
#------------------------------------------------------------------------------
def run_SSA(scenario, logfiletxt):
    """
    run SSA (selfish stock algorithm) algorithm

    Parameters
    ----------
    logfile : txt
        path Logs file 
    Returns
    -------
    None.

    """
    algoName = "SSA"
    scenario["algoName"] = algoName
    scenario = create_repo_for_save_jobs(scenario)
    
    # Initialisation of the apps
    # application = Initialization_game(scenario)
    application = Initialization_game(scenario)
    
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning 
    logfile = os.path.join(scenario["scenarioCorePathDataAlgoName"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                                # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # Execute SSA
    file.write(f"\n_______{algoName}_______"+ "\n")
    application.runSSA(plot=False,file=file, scenario=scenario)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    
    # End execute syA
    print("________RUN END SSA ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathDataAlgoName"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    return application
#------------------------------------------------------------------------------
#                FIN : algo SSA avec instance charge
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#                DEBUT : algo SyA avec instance charge
#------------------------------------------------------------------------------
def run_syA(scenario, logfiletxt):
    """
    run syA algorithm

    Parameters
    ----------
    logfile : txt
        path Logs file 
    Returns
    -------
    None.

    """
    algoName = "SyA"
    scenario["algoName"] = algoName
    scenario = create_repo_for_save_jobs(scenario)
    

    # Initialisation of the apps
    # application = Initialization_game(scenario)
    application = Initialization_game(scenario)
    
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning
    logfile = os.path.join(scenario["scenarioCorePathDataAlgoName"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    # Execute SyA
    file.write(f"\n_______{algoName}_______"+ "\n")
    # application.runSyA(plot=False,file=file)
    application.runSyA(plot=False, file=file, scenario=scenario)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    # End execute syA
    print("________RUN END SyA ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathDataAlgoName"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    
    return application
#------------------------------------------------------------------------------
#                FIN : algo SyA avec instance charge
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : algo LRI avec instance charge
#------------------------------------------------------------------------------
def run_LRI_REPART(scenario, logfiletxt):
    """
    run LRI REPART algorithm for debug

    Parameters
    ----------
    scenario: dict
        contains list of variables for generating/loading data, etc...
    logfile : txt
        path Logs file 
    Returns
    -------
    None.

    """
    algoName = "LRI_REPART"
    scenario["algoName"] = algoName
    scenario = create_repo_for_save_jobs(scenario)
    
    
    # Initialisation of the apps
    # application = Initialization_game(scenario)
    application = Initialization_game(scenario)

    # Display for the run beginning
    logfile = os.path.join(scenario["scenarioCorePathDataAlgoName"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                                # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # # Execute LRI_REPART
    file.write(f"\n_______{algoName}_______"+ "\n")
    #application.runLRI_REPART(plot=False, file=file)
    application.runLRI_REPART(plot=False, file=file, scenario=scenario, algoName=algoName)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    
    
    # End execute LRI REPART
    print("________RUN END LRI_REPART ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathDataAlgoName"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    
    return application
#------------------------------------------------------------------------------
#                FIN : algo LRI avec instance charge
#------------------------------------------------------------------------------

###############################################################################
#                END : ALGO LRI_REPART_DBG
###############################################################################


if __name__ == '__main__':

    logfiletxt = "traceApplication.txt"
    scenarioFile = "./scenario1.json"
    scenarioFile = "./data_scenario/scenario_test_LRI.json"
    scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N4_T3.json"
    scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N10_T5.json"
    
    
    import time
    start = time.time()
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
        if "SyA" in scenario["algo"]:
            run_syA(scenario, logfiletxt)
        if "SSA" in scenario["algo"]:
            run_SSA(scenario, logfiletxt)
        if "CSA" in scenario["algo"]:
            run_CSA(scenario, logfiletxt)
        if "LRI_REPART" in scenario["algo"]:
            run_LRI_REPART(scenario, logfiletxt )
        pass

    print(f"Running time = {time.time() - start}")