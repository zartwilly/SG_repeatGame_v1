#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:10:34 2024

@author: willy

Execution program runs the all algorithms and the visualization of KPI 

"""
import os
import json
import time
import numpy as np
import pandas as pd
import visuData as visu
import runApp as ra
import redistribution as redis

logfiletxt = "traceApplication.txt"
scenarioFile = "./scenario1.json"
scenarioFile = "./data_scenario/scenario_test_LRI.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N4_T3.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N10_T5.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N10_T100_K5000.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N10_T100_K5000_B3.json"

scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K5000_B3_NEWDATADBG.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K5000_B3_Rho2_NEWDATADBG.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K5000_B3_Rho10_NEWDATADBG.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K10000_B2_Rho5_NEWDATADBG.json"

# scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K100_B2_Rho5_NEWDATADBG.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K50_B2_Rho5_scalarDivide_DBG.json"

scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N10_T5.json"

scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K5000_B2_Rho5.json"


scenarioFile = "./data_scenario/scenario_SelfishDB_LRI_N20_T100_K5000_B2_Rho5.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K5000_B2_Rho5_newFormula_QSTOCK.json"

#scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T5_SHAPLEY_DBG.json"
scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N10_T5_SHAPLEY_DBG.json"
# dataset instance particuliere
scenarioFile = "./data_scenario/scenario_SelfishVersion20092024_LRI_N8_T20_RHO5_SHAPLEY.json"
# dataset aleatoire
scenarioFile = "./data_scenario/scenario_SelfishVersion20092024_datasetAleatoire_LRI_N10_T100_RHO5_SHAPLEY.json"

is_generateData = False #True #False
is_generateData_version20092024 = not is_generateData
is_shapleyValueCalculate = False #True #False
PlotDataVStockQTsock = True



def redistribution_bwt_lri_nosmart(app_LRI, app_NoS):
    """
    Compute shapley values for prosumers between LRI and no smart algo like CSA, SyA, SSA

    Parameters
    ----------
    app_LRI : App
        DESCRIPTION.
    app_NoS : App
        DESCRIPTION.

    Returns
    -------
    shapleyValues : TYPE
        DESCRIPTION.

    """
    # Retrieving prodit and consit for LRI hard and NoSmart
    N = app_LRI.N_actors
    prod_LRI = np.zeros(N)
    cons_LRI = np.zeros(N)
    prod_NoS = np.zeros(N)
    cons_NoS = np.zeros(N)

    basevalue = np.sum(app_LRI.SG.Reduct)  # Value of ER for the LRI
    phiepominus_LRI = np.sum(app_LRI.SG.ValNoSGCost) # Parameter : describe benefit of selling energy to EPO using LRI
    phiepoplus_LRI = np.sum(app_LRI.SG.ValSG) # Parameter : describe cost of buying energy from EPO using LRI
    phiepominus_NoS = np.sum(app_NoS.SG.ValNoSGCost) # Parameter : describe benefit of selling energy to EPO using NoSmart
    phiepoplus_NoS = np.sum(app_NoS.SG.ValSG) # Parameter : describe cost of buying energy from EPO using NoSmart

    for i in range(N):    
        prod_LRI[i]  = sum(app_LRI.SG.prosumers[i].prodit)
        cons_LRI[i]  = sum(app_LRI.SG.prosumers[i].consit)
        
        prod_NoS[i]  = sum(app_NoS.SG.prosumers[i].prodit)
        cons_NoS[i]  = sum(app_NoS.SG.prosumers[i].consit)
        
    redi = redis.redistribution(phiepoplus_LRI=phiepoplus_LRI, phiepominus_LRI=phiepominus_LRI, 
                          phiepoplus_NoS=phiepoplus_NoS, phiepominus_NoS=phiepominus_NoS, 
                          N=N, 
                          prod_LRI=prod_LRI, prod_NoS=prod_NoS, 
                          cons_LRI=cons_LRI, cons_NoS=cons_NoS, 
                          basevalue=basevalue)

    shapleyValues = redi.computeShapleyValue(N)
    
    return shapleyValues


app_syA = None
app_SSA = None
app_CSA = None
app_LRI = None

start = time.time()
with open(scenarioFile) as file:
    scenario = json.load(file)
    scenario["is_generateData"] = is_generateData
    scenario["is_generateData_version20092024"] = is_generateData_version20092024
    
    if "SyA" in scenario["algo"]:
        # ra.run_syA(scenario, logfiletxt)
        app_syA = ra.run_syA(scenario, logfiletxt)
        pass
    if "SSA" in scenario["algo"]:
        # ra.run_SSA(scenario, logfiletxt)
        app_SSA = ra.run_SSA(scenario, logfiletxt)
        pass
    if "CSA" in scenario["algo"]:
        # ra.run_CSA(scenario, logfiletxt)
        app_CSA = ra.run_CSA(scenario, logfiletxt)
        pass
    if "LRI_REPART" in scenario["algo"]:
        # ra.run_LRI_REPART(scenario, logfiletxt)
        app_LRI = ra.run_LRI_REPART(scenario, logfiletxt)
        pass
    pass

### ------- START : redistribution with shapley values between LRI and sysA -------
df_shapleys = pd.DataFrame()
if is_shapleyValueCalculate:
    print("__________ Compute Shapley Values _______________")
    shapleyValuesLRISyA = redistribution_bwt_lri_nosmart(app_LRI=app_LRI, app_NoS=app_syA)
    
    shapleyValuesLRISSA = redistribution_bwt_lri_nosmart(app_LRI=app_LRI, app_NoS=app_SSA)
    
    shapleyValuesLRICSA = redistribution_bwt_lri_nosmart(app_LRI=app_LRI, app_NoS=app_CSA)
    
    prosumers = [f"prosumer{i}" for i in range(shapleyValuesLRISyA.shape[0])]
    
    tuples = (prosumers, shapleyValuesLRISyA, shapleyValuesLRISSA, shapleyValuesLRICSA)
    columns = ['Prosumers','LRISyA','LRISSA','LRICSA']
    
    df_shapleys = pd.DataFrame(np.column_stack(tuples), columns=columns)
    print(f"*LRISyA = {shapleyValuesLRISyA} \n *LRISSA = {shapleyValuesLRISSA}, \n *LRICSA = {shapleyValuesLRICSA}")
    
### ------- END : redistribution with shapley values -------


# -------- START : first run ---------------

print("# --- VISU --- ")

scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
scenario["PlotDataVStockQTsock"] = PlotDataVStockQTsock

scenarioVizFile = os.path.join(scenario["scenarioCorePathDataViz"], scenario["scenarioName"]+'_VIZ.json')
checkfile = os.path.isfile(scenarioVizFile)
scenarioViz = dict()
if checkfile:
    with open(scenarioVizFile) as file:
        scenarioViz = json.load(file)
    pass
else:
    scenarioViz = {"algoName": list(scenario["algo"].keys()), 
                 "graphs":[["ValSG_ts", "ValNoSG_ts", "Bar"], ["QttEPO", "line"], ["MaxPrMode", "line]"] ]
                 }
    pass

apps_pkls = visu.load_all_algos_V1(scenario, scenarioViz)


initial_period = 0

df_SG, df_APP, df_PROSUMERS, dfs_VStock, dfs_QTStock_R \
    = visu.create_df_SG_V2_SelectPeriod(apps_pkls_algos=apps_pkls, initial_period=initial_period)

app_PerfMeas = visu.plot_ManyApp_perfMeasure_V2(df_APP, 
                                                df_SG, 
                                                df_PROSUMERS, 
                                                dfs_VStock, 
                                                dfs_QTStock_R, 
                                                df_shapleys)
app_PerfMeas.run_server(debug=True)

# -------- END : first run ---------------




print(f"Running time = {time.time() - start}")

