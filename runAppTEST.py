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



def config_instance(N_actors, nbperiod, rho):
    """
    generate data for a game with 
    N_actors prosumers
    nbperiod periods
    rho next periods to predict values 
    the shape data is N*(nbperiod+rho)

    Parameters
    ----------
    N_actors : int
        number of prosumers
    nbperiod : int
        number of periods .
    rho : int
        the next periods to add at T periods. In finish, we generate (T + rho) periods.
        This parameter enables the prediction of values from T+1 to T + rho periods
        rho << T ie rho=3 < T=5

    Returns
    -------
    g : TYPE
        DESCRIPTION.

    """
    g = ig2.Instancegenaratorv2(N=N_actors, T=nbperiod, rho=rho)
    transitionprobabilities = [0.4, 0.6, 0.5, 0.4]
    repartition = [8, 7]
    
    # values = [m1a,M1a,m1b,M1b,m2b,M2b,cb,m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
    values = [[5,20],[5,15,25,35,24],[20,30,35,50,20,35,40,55]]
    
    # probabilities = [P1b,P2b,P1c,P2c,P3c,P4c]
    probabilities = [[0.5,0.5],[0.7,0.6,0.6,0.5]]
    

    g.generate(transitionprobabilities,repartition,values,probabilities)
    
    return g

def Initialization_game(scenario):
    """
    initialization of variables of an object application 
    
    Returns
    -----
    App
    """
    # Load all scenario parameters
    name = scenario["name"]
    for var, val in scenario["algo"]["LRI_REPART"].items():
        globals()[var] = val
    
    for var, val in scenario["instance"].items():
        globals()[var] = val
        
    for var, val in scenario["simul"].items():
        globals()[var] = val
        """
    maxstep = 5 * pow(10, 1)            #  5 * pow(10, 4)
    maxstep_init = 5
    slowdownfactor = pow(10, -3)        # 0.001
    threshold = 0.8
    N_actors = 15
    nbPeriod = 10                      # 101
    initialprob = 0.5
    mu = pow(10, -1)                    # 0.1
    rho = 5                             # 0.1
    h = 5
"""

    # Initialisation of the apps
    application = apps.App(N_actors=N_actors, maxstep=maxstep, mu=mu, 
                           b=slowdownfactor, rho=rho, h=h, 
                           maxstep_init=maxstep_init, threshold=threshold)
    application.SG = sg.Smartgrid(N=N_actors, nbperiod=nbPeriod, 
                                  initialprob=initialprob, rho=rho)
    
    # Configuration of the instance generator
    g = config_instance(N_actors=N_actors, nbperiod=nbPeriod, rho=rho)
    
    # Initialisation of production, consumption and storage using the instance generator
    N = application.SG.prosumers.size
    T = application.SG.nbperiod
    
    for i in range(N):
        for t in range(T):
            application.SG.prosumers[i].production[t] = g.production[i][t]
            application.SG.prosumers[i].consumption[t] = g.consumption[i][t]
        application.SG.prosumers[i].storage[0] = 0
        application.SG.prosumers[i].smax = 20
        
    return application

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
    
    # Initialisation of the apps
    application = Initialization_game(scenario)
    
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning 
    file = io.open(logfiletxt,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    # Execute SyA
    file.write("\n_______SyA_______"+ "\n")
    application.runSyA(plot=False,file=file)
    
    monitoring_after_algorithm(algoName='syA', file=file, application=application)
    
    # End execute syA
    print("________RUN END syA ",1,"_________ \n")
    
    return application

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
    # Initialisation of the apps
    application = Initialization_game(scenario)

    # Display for the run beginning 
    file = io.open(logfiletxt,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # Execute CSA
    file.write("\n_______CSA_______"+ "\n")
    application.runCSA(plot=False, file=file)
    
    monitoring_after_algorithm(algoName='CSA', file=file, application=application)
    
    
    
    # End execute CSA
    print("________RUN END CSA ",1,"_________ \n")
    
    return application


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
    # Initialisation of the apps
    application = Initialization_game(scenario)
    
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning 
    file = io.open(logfiletxt,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # Execute SSA
    file.write("\n_______SSA_______"+ "\n")
    application.runSSA(plot=False,file=file)
    
    monitoring_after_algorithm(algoName='SSA', file=file, application=application)
    
    
    # End execute syA
    print("________RUN END SSA ",1,"_________ \n")
    
    return application

def run_LRI_REPART(scenario, logfiletxt):
    """
    run LRI REPART algorithm

    Parameters
    ----------
    logfile : txt
        path Logs file 
    Returns
    -------
    None.

    """
        
    # Initialisation of the apps
    application = Initialization_game(scenario)

    # Display for the run beginning 
    file = io.open(logfiletxt,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # # Execute LRI_REPART
    algoName = "LRI_REPART"
    file.write(f"\n_______{algoName}_______"+ "\n")
    #application.runLRI_REPART(plot=False, file=file)
    application.runLRI_REPART_SAVERunning(plot=False, file=file)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    
    
    # End execute LRI REPART
    print("________RUN END LRI_REPART ",1,"_________ \n")
    
    return application


###############################################################################
#                DEBUT : ALGO LRI_REPART_DBG
###############################################################################

#------------------------------------------------------------------------------
#                DEBUT : Generer des donnees selon scenarios
#------------------------------------------------------------------------------

def creation_instance(transitionprobabilities, repartition, values, probabilities):
    """
    creation instance de generation de donnees
    creation of instances generating data
    
    transitionprobabilities: [w, x, y, z],
    repartition: [xx, yy]
    # values = [m1a,M1a,m1b,M1b,m2b,M2b,cb,m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
    values = [[5,20],[5,15,25,35,24],[20,30,35,50,20,35,40,55]]
    
    # probabilities = [P1b,P2b,P1c,P2c,P3c,P4c]
    probabilities = [[0.5,0.5],[0.7,0.6,0.6,0.5]]

    Returns
    -------
    None.

    """
    
    scenario = {"transitionprobabilities": transitionprobabilities, 
                      "repartition": repartition,
                      "values": values,
                      "probabilities": probabilities,
                      "scenarioName": "scenarioSelfish"
                      }
        
    pass

def generer_data_from_scenario(newInstance:bool, scenario:dict, N_actors:int, 
                               nbperiod:int, rho:int):
    """
    generate data from new/existing scenario

    Parameters
    ----------
    newInstance : bool
        DESCRIPTION.
    scenario : dict
        DESCRIPTION.
        exple: {"transitionprobabilities": transitionprobabilities, 
                "repartition": repartition,
                "values": values,
                "probabilities": probabilities,
                "scenarioName": "scenarioSelfish"
                }
    N_actors : int
        DESCRIPTION.
    nbperiod : int
        DESCRIPTION.
    rho : int
        DESCRIPTION.

    Returns
    -------
    None.

    """
    transitionprobabilities = [0.4, 0.6, 0.5, 0.4]
    repartition = [8, 7]
    # values = [m1a,M1a,m1b,M1b,m2b,M2b,cb,m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
    values = [[5,20],[5,15,25,35,24],[20,30,35,50,20,35,40,55]]
    # probabilities = [P1b,P2b,P1c,P2c,P3c,P4c]
    probabilities = [[0.5,0.5],[0.7,0.6,0.6,0.5]]
    
    scenario = {"transitionprobabilities": transitionprobabilities, 
                "repartition": repartition,
                "values": values,
                "probabilities": probabilities,
                "scenarioName": "scenarioSelfish"
                }
    
    if newInstance:
        # creation instance
        # generation de donnees
        # serialisation
        # sauvegarde pickle
        
        # creation instance
        g = ig2.Instancegenaratorv2(N=N_actors, T=nbperiod, rho=rho)
        
        # generation de donnees
        g.generate(scenario["transitionprobabilities"],
                   scenario["repartition"], scenario["values"],
                   scenario["probabilities"])
        
        
    else:
        # recuperation du pickle
        # deserialisation 
        # generation data
        # sauvegarde
    
        pass
    

def generer_data_from_scenario_DBG(scenario:dict,
                                   N_actors:int, nbperiod:int, rho:int,
                                    transitionprobabilities:list,
                                   repartition:list,
                                   values:list, 
                                   probabilities:list):
    # scenarioPath : "data_scenario"
    # scenarioName : "data_NAME_DAY-MM-YY-HH-MM.pkl"
    
    # path_name = os.path.join(scenarioPath, scenarioName+".pkl") ===> TODELETE
    path_name = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], scenario["scenarioName"]+".pkl")
    
    g = None
    # Is there the data file on repository?
    checkfile = os.path.isfile(path_name)
    if checkfile:
        # file exists in which form?
        # return g which contains data load
        
        print("**** Load pickle data: START ****")
        # with open(os.path.join(scenarioPath, scenarioName+'.pkl'), 'rb') as f:  # open a text file ===> TODELETE
        with open(os.path.join(scenario["scenarioCorePath"], scenario["scenarioName"]+'.pkl'), 'rb') as f:  # open a text file
            g = pickle.load(f)
        f.close()
        
        print("**** Load pickle data : END ****")
        
    else:
        print("**** Create pickle data : START ****")
        # file not exists
        g = ig2.Instancegenaratorv2(N=N_actors, T=nbperiod, rho=rho)
        
        g.generate(transitionprobabilities,repartition,values,probabilities)
        
        # with open(os.path.join(scenarioPath, scenarioName+'.pkl'), 'wb') as f:  # open a text file ===> TODELETE
        with open(os.path.join(scenario["scenarioCorePath"], scenario["scenarioName"]+'.pkl'), 'wb') as f:  # open a text file
            pickle.dump(g, f)
        f.close()
        
        print("**** Create pickle data : END ****")
        
    return g
    
def Initialization_game_DBG(scenario):
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
    # = scenario[""]
    # = scenario[""]
    
    scenario["scenarioCorePath"] = os.path.join(scenario["scenarioPath"], scenario["scenarioName"])
    
    
    # for var, val in scenario["algo"]["LRIRepart"].items():
    #     globals()[var] = val
    
    # for var, val in scenario["instance"].items():
    #     globals()[var] = val
        
    # for var, val in scenario["simul"].items():
    #     globals()[var] = val

    
    # Initialisation of the apps
    application = apps.App(N_actors=N_actors, maxstep=maxstep, mu=mu, 
                           b=slowdownfactor, rho=rho, h=h, 
                           maxstep_init=maxstep_init, threshold=threshold)
    application.SG = sg.Smartgrid(N=N_actors, nbperiod=nbPeriod, 
                                  initialprob=initialprob, rho=rho)
    
    # Configuration of the instance generator
    # g = config_instance(N_actors=N_actors, nbperiod=nbPeriod, rho=rho        ====> TODELETE
    g = generer_data_from_scenario_DBG(scenario=scenario,
                                       N_actors=N_actors, nbperiod=N_actors, 
                                       rho=rho,
                                       transitionprobabilities=transitionprobabilities,
                                       repartition=repartition,
                                       values=values, 
                                       probabilities=probabilities)
    
    # Initialisation of production, consumption and storage using the instance generator
    N = application.SG.prosumers.size
    T = application.SG.nbperiod
    
    for i in range(N):
        for t in range(T):
            application.SG.prosumers[i].production[t] = g.production[i][t]
            application.SG.prosumers[i].consumption[t] = g.consumption[i][t]
        application.SG.prosumers[i].storage[0] = 0
        application.SG.prosumers[i].smax = 20
        
    return application

def create_repo_for_save_jobs(scenario:dict):
    scenarioCorePath = os.path.join(scenario["scenarioPath"], scenario["scenarioName"])
    scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "data")
    scenario["scenarioCorePath"] = scenarioCorePath
    scenario["scenarioCorePathData"] = scenarioCorePathData
    
    # create a scenarioPath if not exists
    Path(scenarioCorePathData).mkdir(parents=True, exist_ok=True)
    
    
    return scenario
#------------------------------------------------------------------------------
#                FIN : Generer des donnees selon scenarios
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : algo CSA avec instance charge
#------------------------------------------------------------------------------
def run_CSA_DBG(scenario, logfiletxt):
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
    scenario = create_repo_for_save_jobs(scenario)
    
    # Initialisation of the apps
    application = Initialization_game(scenario)

    # Display for the run beginning 
    logfile = os.path.join(scenario["scenarioCorePathData"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # Execute CSA
    file.write("\n_______CSA_______"+ "\n")
    application.runCSA_SAVERunning(plot=False, file=file, scenario=scenario)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    
    # End execute CSA
    print("________RUN END CSA ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathData"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    return application
#------------------------------------------------------------------------------
#                FIN : algo CSA avec instance charge
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : algo SSA avec instance charge
#------------------------------------------------------------------------------
def run_SSA_DBG(scenario, logfiletxt):
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
    scenario = create_repo_for_save_jobs(scenario)
    
    # Initialisation of the apps
    application = Initialization_game(scenario)
    
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning 
    logfile = os.path.join(scenario["scenarioCorePathData"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # Execute SSA
    file.write(f"\n_______{algoName}_______"+ "\n")
    application.runSSA_SAVERunning(plot=False,file=file, scenario=scenario)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    
    # End execute syA
    print("________RUN END SSA ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathData"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    return application
#------------------------------------------------------------------------------
#                FIN : algo SSA avec instance charge
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#                DEBUT : algo SyA avec instance charge
#------------------------------------------------------------------------------
def run_syA_DBG(scenario, logfiletxt):
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
    scenario = create_repo_for_save_jobs(scenario)
    

    # Initialisation of the apps
    application = Initialization_game_DBG(scenario)
    
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning
    logfile = os.path.join(scenario["scenarioCorePathData"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    # Execute SyA
    file.write(f"\n_______{algoName}_______"+ "\n")
    # application.runSyA(plot=False,file=file)
    application.runSyA_SAVERunning(plot=False, file=file, scenario=scenario)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    # End execute syA
    print("________RUN END SyA ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathData"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    
    return application
#------------------------------------------------------------------------------
#                FIN : algo SyA avec instance charge
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : algo LRI avec instance charge
#------------------------------------------------------------------------------
def run_LRI_REPART_DBG(scenario, logfiletxt):
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
    scenario = create_repo_for_save_jobs(scenario)
    
    
    # Initialisation of the apps
    application = Initialization_game_DBG(scenario)

    # Display for the run beginning
    logfile = os.path.join(scenario["scenarioCorePathData"], algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                              # Logs file
    
    monitoring_before_algorithm(file, application)
    
    
    # # Execute LRI_REPART
    file.write(f"\n_______{algoName}_______"+ "\n")
    #application.runLRI_REPART(plot=False, file=file)
    application.runLRI_REPART_SAVERunning(plot=False, file=file, 
                                          scenario=scenario)
    
    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    
    
    # End execute LRI REPART
    print("________RUN END LRI_REPART ",1,"_________ \n")
    
    # Save application to Pickle format
    with open(os.path.join(scenario["scenarioCorePathData"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
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
    scenarioPath = "./scenario1.json"
    scenarioPath = "./data_scenario/scenario_test_LRI.json"
    
    
    import time
    start = time.time()
    with open(scenarioPath) as file:
        scenario = json.load(file)
        
        # g = Initialization_game_DBG(scenario)

        if "SyA" in scenario["algo"]:
            # run_syA(scenario, logfiletxt)
            run_syA_DBG(scenario, logfiletxt)
        if "SSA" in scenario["algo"]:
            #run_SSA(scenario, logfiletxt)
            run_SSA_DBG(scenario, logfiletxt)
        if "CSA" in scenario["algo"]:
            #run_CSA(scenario, logfiletxt)
            run_CSA_DBG(scenario, logfiletxt)
        if "LRI_REPART" in scenario["algo"]:
            # run_LRI_REPART(scenario, logfiletxt)
            run_LRI_REPART_DBG(scenario, logfiletxt )
        pass

    print(f"Running time = {time.time() - start}")