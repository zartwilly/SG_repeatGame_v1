#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:04:24 2024

@author: willy

run the developped algortihms for evaluation some variables
"""
import io
import application as apps
import Instancegeneratorversion2 as ig2
import smartgrid as sg
import auxiliary_functions as aux
import agents as ag



def config_instance(N_actors, maxperiod):
    g = ig2.Instancegenaratorv2(N=N_actors, T=maxperiod)
    transitionprobabilities = [0.4, 0.6, 0.5, 0.4]
    repartition = [8, 7]
    
    # values = [m1a,M1a,m1b,M1b,m2b,M2b,cb,m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
    values = [[5,20],[5,15,25,35,24],[20,30,35,50,20,35,40,55]]
    
    # probabilities = [P1b,P2b,P1c,P2c,P3c,P4c]
    probabilities = [[0.5,0.5],[0.7,0.6,0.6,0.5]]
    

    g.generate(transitionprobabilities,repartition,values,probabilities)
    
    return g
    
def run_syA(logfiletxt):
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
    maxstep = 5 * pow(10, 1)            #  5 * pow(10, 4)
    maxstep_init = 5
    slowdownfactor = pow(10, -3)        # 0.001
    threshold = 0.8
    N_actors = 15
    maxperiod = 10                      # 101
    initialprob = 0.5
    mu = pow(10, -1)                    # 0.1
    rho = 5                             # 0.1
    h = 5
    
        
    # Initialisation of the apps
    application = apps.App(N_actors=N_actors, maxstep=maxstep, mu=mu, 
                           b=slowdownfactor, rho=rho, h=h, maxstep_init=maxstep_init)
    application.SG = sg.Smartgrid(N=N_actors, maxperiod=maxperiod, 
                                  initialprob=initialprob, rho=rho)
    
    # Configuration of the instance generator
    g = config_instance(N_actors=N_actors, maxperiod=maxperiod)
    
    # Initialisation of production, consumption and storage using the instance generator
    N = application.SG.prosumers.size
    T = application.SG.maxperiod
    
    for i in range(N):
        for t in range(T):
            application.SG.prosumers[i].production[t] = g.production[i][t]
            application.SG.prosumers[i].consumption[t] = g.consumption[i][t]
        application.SG.prosumers[i].storage[0] = 0
        application.SG.prosumers[i].smax = 20
        
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning 
    file = io.open(logfiletxt,"w")                                              # Logs file
    
    print("________RUN ",1,"_________")
    file.write("________RUN " + str(1) +"_________" + "\n")
    
    file.write("\n___Configuration___ \n")
    for i in range(N):
        file.write("__Prosumer " + str(i + 1) + "___\n")
        for t in range(T):
            file.write("Period " + str(t + 1))
            file.write(" : Production : " + str(application.SG.prosumers[i].production[t]))
            file.write(" Consumption : " + str(application.SG.prosumers[i].consumption[t]))
            file.write(" Storage : " + str(application.SG.prosumers[i].storage[t])+ "\n")
            
    
    # Execute SyA
    file.write("\n_______SyA_______"+ "\n")
    application.runSyA(plot=False,file=file)
    
    file.write("\n___Storage___ \n")
    for i in range(N):
        file.write("__Prosumer " + str(i + 1) + "___\n")
        for t in range(T):
            file.write("Period " + str(t + 1))
            file.write(" : Storage : " + str(application.SG.prosumers[i].storage[t])+ "\n")
            
    file.write("\n___InSG, OutSG___ \n")
    for t in range(T):
        file.write(" *** Period " + str(t + 1))
        file.write(" InSG : " + str(application.SG.insg[t])+ " OutSG: "+ str(application.SG.outsg[t]) +"*** \n")
        for i in range(N):
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
    for i in range(N):
        file.write("__Prosumer " + str(i + 1) + "___ :" +str(round(application.ObjValai[i], 2)) + "\n")
    
    # End execute syA
    print("________RUN END syA ",1,"_________ \n")
    file.write("________RUN END syA " + str(1) +"_________" + "\n\n")

def run_SSA(logfiletxt):
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
    maxstep = 5 * pow(10, 1)            #  5 * pow(10, 4)
    maxstep_init = 5
    slowdownfactor = pow(10, -3)        # 0.001
    threshold = 0.8
    N_actors = 15
    maxperiod = 10                      # 101
    initialprob = 0.5
    mu = pow(10, -1)                    # 0.1
    rho = 5                             # 0.1
    h = 5
    
        
    # Initialisation of the apps
    application = apps.App(N_actors=N_actors, maxstep=maxstep, mu=mu, 
                           b=slowdownfactor, rho=rho, h=h, maxstep_init=maxstep_init)
    application.SG = sg.Smartgrid(N=N_actors, maxperiod=maxperiod, 
                                  initialprob=initialprob, rho=rho)
    
    # Configuration of the instance generator
    g = config_instance(N_actors=N_actors, maxperiod=maxperiod)
    
    # Initialisation of production, consumption and storage using the instance generator
    N = application.SG.prosumers.size
    T = application.SG.maxperiod
    
    for i in range(N):
        for t in range(T):
            application.SG.prosumers[i].production[t] = g.production[i][t]
            application.SG.prosumers[i].consumption[t] = g.consumption[i][t]
        application.SG.prosumers[i].storage[0] = 0
        application.SG.prosumers[i].smax = 20
        
    # ignore last period to exclude overflow: I do not know the importance to exclude last period
    # application.SG.maxperiod = application.SG.maxperiod - 1


    # Display for the run beginning 
    file = io.open(logfiletxt,"w")                                              # Logs file
    
    print("________RUN ",1,"_________")
    file.write("________RUN " + str(1) +"_________" + "\n")
    
    file.write("\n___Configuration___ \n")
    for i in range(N):
        file.write("__Prosumer " + str(i + 1) + "___\n")
        for t in range(T):
            file.write("Period " + str(t + 1))
            file.write(" : Production : " + str(application.SG.prosumers[i].production[t]))
            file.write(" Consumption : " + str(application.SG.prosumers[i].consumption[t]))
            file.write(" Storage : " + str(application.SG.prosumers[i].storage[t])+ "\n")
            
    
    # Execute SSA
    file.write("\n_______SSA_______"+ "\n")
    application.runSSA(plot=False,file=file)
    
    file.write("\n___Storage___ \n")
    for i in range(N):
        file.write("__Prosumer " + str(i + 1) + "___\n")
        for t in range(T):
            file.write("Period " + str(t + 1))
            file.write(" : Storage : " + str(application.SG.prosumers[i].storage[t])+ "\n")
            
    file.write("\n___InSG, OutSG___ \n")
    for t in range(T):
        file.write(" *** Period " + str(t + 1))
        file.write(" InSG : " + str(application.SG.insg[t]))
        file.write(" OutSG: "+ str(application.SG.outsg[t]))
        file.write(" valNoSGCost: " + str(application.SG.ValNoSGCost[t]) +"*** \n")
        for i in range(N):
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
    for i in range(N):
        file.write("__Prosumer " + str(i + 1) + "___ :" +str(round(application.ObjValai[i], 2)) + "\n")
    
    # End execute syA
    print("________RUN END SSA ",1,"_________ \n")
    file.write("________RUN END SSA " + str(1) +"_________" + "\n\n")


if __name__ == '__main__':

    logfiletxt = "traceApplication.txt"
    #run_syA(logfiletxt)
    run_SSA(logfiletxt)
    pass
