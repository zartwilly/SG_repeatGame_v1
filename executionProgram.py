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

# scenario aleatoire DBG
scenarioFile = "./data_scenario/DBG_version20092024_dataAleatoire_N10_T5_K5_B1_rho5_NotSHAPLEY.json"

# 4) scenario data donnee version20092024 avec Si!=0 pour t=0 pour tous prosumers
#scenarioFile = "./data_scenario/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho5_StockDiffT0_NotSHAPLEY.json"

# 3) scenario data donnee version20092024 avec Si=0 pour t=0 pour tous prosumers
scenarioFile = "./data_scenario/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho5_StockZeroT0_NotSHAPLEY.json"

# 2) scenario data aleatoire avec Si!=0 pour t=0 pour tous prosumers
#scenarioFile = "./data_scenario/scenario_version20092024_dataAleatoire_N10_T100_K5000_B1_rho5_StockDiffT0_NotSHAPLEY.json"

# 1) scenario data aleatoire avec Si=0 pour t=0 pour tous prosumers
#scenarioFile = "./data_scenario/scenario_version20092024_dataAleatoire_N10_T100_K5000_B1_rho5_StockZeroT0_NotSHAPLEY.json"

# 0) scenario data donnee version20092024 DBG avec Si!=0 pour t=0 pour tous prosumers
#scenarioFile = "./data_scenario/scenario_version20092024_DBG_dataDonnee_N8_T20_K50_B1_rho5_StockDiffT0_NotSHAPLEY.json"

# # test rho=1) scenario data donnee version20092024 avec Si=0 pour t=0 pour tous prosumers
# scenarioFile = "./data_scenario/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho1_StockZeroT0_NotSHAPLEY.json"

######## test various rho values from Dominique Game Scenario #################
# 0) rho=1
# scenarioFile = "./data_scenario_JeuDominique/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho1_StockZeroT0_NotSHAPLEY.json"
# 1) rho=2
# scenarioFile = "./data_scenario_JeuDominique/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho2_StockZeroT0_NotSHAPLEY.json"
# 2) rho=3
# scenarioFile = "./data_scenario_JeuDominique/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho3_StockZeroT0_NotSHAPLEY.json"
# 3) rho=5
# scenarioFile = "./data_scenario_JeuDominique/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho5_StockZeroT0_NotSHAPLEY.json"
# 4) rho=7
# scenarioFile = "./data_scenario_JeuDominique/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho7_StockZeroT0_NotSHAPLEY.json"
# 5) rho=10
# scenarioFile = "./data_scenario_JeuDominique/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho10_StockZeroT0_NotSHAPLEY.json"
# 0 bis) rho=1
# scenarioFile = "./data_scenario_JeuDominique/scenario_version20092024BENSmax18_dataDonnee_N8_T20_K5000_B1_rho1_StockZeroT0_NotSHAPLEY.json"

# DEBUG: GIVEN STRATEGIES 
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho1.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho2.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho3.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho4.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho5.json"
#scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_rho5.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho08.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho10.json"

scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho05_Smax18_bestieT6.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho05_Smax18_bestieT7.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho05_Smax24_bestieT6.json"
# scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho05_Smax30_bestieT5.json"




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
    # scenario["simul"]["is_storage_zero"] = True if scenario["simul"]["is_storage_zero"] == "True" else False
    # scenario["simul"]["is_generateData"] = True if scenario["simul"]["is_generateData"] == "True" else False
    # scenario["simul"]["is_generateData_version20092024"] = not scenario["simul"]["is_generateData"]
    # scenario["simul"]["is_shapleyValueCalculate"] = True if scenario["simul"]["is_shapleyValueCalculate"] == "True" else False
    # scenario["simul"]["is_plotDataVStockQTsock"] = True if scenario["simul"]["is_plotDataVStockQTsock"] == "True" else False
    
    
    if "SyA" in scenario["algo"]:
        # ra.run_syA(scenario, logfiletxt)
        scenario["algo_name"] = "SyA"
        app_syA = ra.run_syA(scenario, logfiletxt)
        pass
    if "Bestie" in scenario["algo"]:
        scenario["algo_name"] = "Bestie"
        app_Bestie = ra.run_Bestie(scenario, logfiletxt)
        pass
    if "SSA" in scenario["algo"]:
        # ra.run_SSA(scenario, logfiletxt)
        scenario["algo_name"] = "SSA"
        app_SSA = ra.run_SSA(scenario, logfiletxt)
        pass
    if "CSA" in scenario["algo"]:
        # ra.run_CSA(scenario, logfiletxt)
        scenario["algo_name"] = "CSA"
        app_CSA = ra.run_CSA(scenario, logfiletxt)
        pass
    if "LRI_REPART" in scenario["algo"]:
        # ra.run_LRI_REPART(scenario, logfiletxt)
        scenario["algo_name"] = "LRI"
        app_LRI = ra.run_LRI_REPART(scenario, logfiletxt)
        pass
    pass

scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas")


# data = {"SyA_X_ai": list(app_syA.X_ai), "SyA_Y_ai": list(app_syA.Y_ai),
#         "CSA_X_ai": list(app_CSA.X_ai), "CSA_Y_ai": list(app_CSA.Y_ai),
#         "SSA_X_ai": list(app_SSA.X_ai), "SSA_Y_ai": list(app_SSA.Y_ai),
#         "LRI_X_ai": list(app_LRI.X_ai), "LRI_Y_ai": list(app_LRI.Y_ai),
#         "Bestie_X_ai": list(app_Bestie.X_ai), "Bestie_Y_ai": list(app_Bestie.Y_ai)
#         }
# pd.DataFrame(data).to_csv( os.path.join(scenarioCorePathDataViz, "df_X_Y_ai.csv") )

# ps = []
# algoNames = ['CSA','SSA','LRI', 'SyA', 'Bestie']
# for algoName in algoNames:
#     data_algo = dict()
#     for k, v in data.items():
#         if algoName in k:
#            data_algo[k] = v
           
#     data_algo["prosumers"]=[f"prosumer_{i}" for i in range(app_LRI.Y_ai.shape[0])]
#     df = pd.DataFrame(data_algo)
#     df_sort = df.sort_values(by='SyA_X_ai', ascending=True)
    
#     # Créer un index numérique pour les prosumers
#     df_sort['prosumer_index'] = range(len(df_sort))
    
#     # Création du graphique
#     p = figure(title=f"Nuage de points {algoName}_X_ai et {algoName}_Y_ai par prosumer",
#                x_axis_label='Prosumer Index', y_axis_label=f'Valeurs {algoName}')
    
#     # Ajouter les points pour SyA_X_ai
#     p.circle(x=df_sort['prosumer_index'], y=df_sort[f'{algoName}_X_ai'], 
#              size=10, color="blue", legend_label=f'{algoName}_X_ai')
    
#     # Ajouter les points pour SyA_Y_ai
#     p.circle(x=df_sort['prosumer_index'], y=df_sort[f'{algoName}_Y_ai'], 
#              size=10, color="red", legend_label='f{algoName}_Y_ai')
    
#     ps.append([p])
    


### ------- START : redistribution with shapley values between LRI and sysA -------
is_shapleyValueCalculate = scenario["simul"]["is_shapleyValueCalculate"]
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


# # -------- START : first run ---------------

# print("# --- VISU --- ")

# scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
# scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
# #scenario["PlotDataVStockQTsock"] = PlotDataVStockQTsock

# scenarioVizFile = os.path.join(scenario["scenarioCorePathDataViz"], scenario["scenarioName"]+'_VIZ.json')
# checkfile = os.path.isfile(scenarioVizFile)
# scenarioViz = dict()
# if checkfile:
#     with open(scenarioVizFile) as file:
#         scenarioViz = json.load(file)
#     pass
# else:
#     scenarioViz = {"algoName": list(scenario["algo"].keys()), 
#                  "graphs":[["ValSG_ts", "ValNoSG_ts", "Bar"], ["QttEPO", "line"], ["MaxPrMode", "line]"] ]
#                  }
#     pass

# apps_pkls = visu.load_all_algos_V1(scenario, scenarioViz)


# initial_period = 0

# # df_SG, df_APP, df_PROSUMERS, dfs_VStock, dfs_QTStock_R \
# #     = visu.create_df_SG_V2_SelectPeriod(apps_pkls_algos=apps_pkls, initial_period=initial_period)
    
# df_SG = None; df_APP = None; df_PROSUMERS = None, 
# dfs_VStock = None; dfs_QTStock_R = None; dfs_QTStock_GA_PA = None
    
# split_GA_PA = eval(scenario["simul"]["split_GA_PA"])
# if split_GA_PA:
#     df_SG, df_APP, df_PROSUMERS, dfs_VStock, dfs_QTStock_R, dfs_QTStock_GA_PA \
#         = visu.create_df_SG_V2_SelectPeriod_4_GA_PA(apps_pkls_algos=apps_pkls, 
#                                              initial_period=initial_period, 
#                                              split_GA_PA=split_GA_PA)
# else:
#     df_SG, df_APP, df_PROSUMERS, dfs_VStock, dfs_QTStock_R \
#         = visu.create_df_SG_V2_SelectPeriod(apps_pkls_algos=apps_pkls, initial_period=initial_period)

# app_PerfMeas = visu.plot_ManyApp_perfMeasure_V2(df_APP, 
#                                                 df_SG, 
#                                                 df_PROSUMERS, 
#                                                 dfs_VStock, 
#                                                 dfs_QTStock_R, 
#                                                 df_shapleys, 
#                                                 dfs_QTStock_GA_PA,
#                                                 scenario["scenarioCorePathDataViz"] )
# app_PerfMeas.run_server(debug=True)

# # -------- END : first run ---------------

# --------------- visu Courbe 0-5 : First ---------------
import visuDataTEST as visutest

with open(scenarioFile) as file:
    scenario = json.load(file)
    
scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas")

scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
scenario["scenarioCorePathData"] = scenarioCorePathData


apps_pkls = visutest.load_all_algos_apps(scenario)
df_prosumers = visutest.create_df_SG(apps_pkls=apps_pkls, index_GA_PA=0)

 
app_PerfMeas = visutest.plot_all_figures(df_prosumers, scenarioCorePathDataViz)

app_PerfMeas.run_server(debug=True)
# --------------- --------------- visu Courbe 0-5 : First ---------------


print(f"Running time = {time.time() - start}")

