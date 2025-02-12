##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:42:03 2024

@author: willy

Remake DataViz with
    2) bokeh  
    1) dash 
"""
import os
import json
import pickle
import pandas as pd
import auxiliary_functions as aux

import itertools as it


import plotly.graph_objects as go

import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import dcc, html
from dash import callback
from dash.dependencies import Input, Output
from dash import callback_context
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from pathlib import Path


###############################################################################
#                   CONSTANTES: debut
###############################################################################
PREFIX_DF = "runLRI_df_"
DF_LRI_NAME = "run_LRI_REPART_DF_T_Kmax.csv"
LRI_ALGO_NAME = "LRI_REPART"

COLORS = {"SyA":"gray", "Bestie":"red", "CSA":"yellow", "SSA":"green", "LRI_REPART":"blue"}

###############################################################################
#                   CONSTANTES: fin
###############################################################################


###############################################################################
#                   load all algo pickle apps: debut
###############################################################################
def load_all_algos_apps(scenario: dict) -> list:
    """
    load all algo pickle apps: each app links to each algo

    Parameters
    ----------
    scenario : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    apps_pkls = []
    
    algoNames = list(scenario.get("algo").keys())
    for algoName in algoNames:
        # load pickle file
        try:
            scenarioPathAlgoName = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", algoName)
            
            with open(os.path.join(scenarioPathAlgoName, scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'rb') as f:
                app_pkl = pickle.load(f)                                        # deserialize using load()
                
                df = None
                if algoName == LRI_ALGO_NAME:
                    if DF_LRI_NAME in os.listdir(scenarioPathAlgoName):
                        df = pd.read_csv(os.path.join(scenarioPathAlgoName, DF_LRI_NAME), skiprows=0, index_col=0)
                else:
                    df_algo_name = f"run{algoName}_MergeDF.csv"
                    df_algo_name
                    df = pd.read_csv(os.path.join(scenarioPathAlgoName, df_algo_name), skiprows=0, index_col=0)
                    df["algoName"] = algoName
                    
                apps_pkls.append((app_pkl, algoName, scenario["scenarioName"], df))      
        
        except FileNotFoundError:
            print(f" {scenario['scenarioName']+'_'+algoName+'_APP.pkl'}  NOT EXIST")
            
    return apps_pkls
###############################################################################
#                   load all algo pickle apps: Fin 
###############################################################################


###############################################################################
#                   Merge alls dataframe : debut
###############################################################################
def create_df_SG(apps_pkls: list, index_GA_PA=0) -> pd.DataFrame:
    """
    create a dataframe from all algos sumup dataframes

    Parameters
    ----------
    app_pkls : list
        list of algorithm applications in the format 
        (app_pkl, algoName, scenario["scenarioName"], df)
    index_GA_PA : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    df_prosumers = list()
    
    for app_pkl in apps_pkls:
        df_tmp = app_pkl[-1]
        
        df_prosumers.append(df_tmp)
        
    df_prosumers = pd.concat(df_prosumers)
    df_prosumers["coef_phiepominus"].bfill(inplace=True)
    df_prosumers["coef_phiepoplus"].bfill(inplace=True)
    df_prosumers["scenarioName"].bfill(inplace=True)
    df_prosumers.drop(columns=['prosumers'], axis=1, inplace=True)
    
    df_prosumers.reset_index(names='prosumers', inplace=True)
    
    return df_prosumers
###############################################################################
#                   Merge alls dataframe : fin
###############################################################################


###############################################################################
#                   plot valSG and valNoSG : debut
###############################################################################
def plot_curve_valSGNoSG(df_prosumers: pd.DataFrame, scenarioCorePathDataViz:str):
    """
    curve plot of valSG and valNoSG

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # create dataframe valnosg et valsg
    df_valNoSG = df_prosumers[["period", "algoName", "valNoSG_i"]] \
                    .groupby(["algoName","period"]).sum().reset_index()
    df_valSG = df_prosumers[["period", "algoName", "ValSG"]]\
                    .groupby(["algoName","period"]).mean().reset_index()
                    
    df_valSGNoSG = df_valSG.merge(df_valNoSG, on=["period", "algoName"])
    
    
    # plot a curbe plot 
    fig_SG_col = go.Figure()
    fig_NoSG_col = go.Figure()
    
    for name_col in ["ValSG", "valNoSG_i"]:
        
        for num_algo, algoName in enumerate(df_valSG.algoName.unique()):
            if name_col == "ValSG":
                fig_SG_col.add_trace(
                    go.Scatter(x=df_valSGNoSG['period'], 
                               y=df_valSGNoSG[df_valSGNoSG.algoName == algoName][name_col], 
                               name= algoName,
                               mode='lines+markers', 
                               marker = dict(color = COLORS[algoName])
                            )
                    )
            else:
                fig_NoSG_col.add_trace(
                    go.Scatter(x=df_valSGNoSG['period'], 
                               y=df_valSGNoSG[df_valSG.algoName == algoName][name_col], 
                               name= algoName,
                               mode='lines+markers', 
                               marker = dict(color = COLORS[algoName])
                            )
                    )
            
    nameScenario = df_prosumers.scenarioName.unique()[0]
    fig_SG_col.update_layout(xaxis_title='periods', yaxis_title='values', 
                             title={'text':f''' {nameScenario}: show ValSG KPI for all algorithms ''',
                                     #'xanchor': 'center',
                                     'yanchor': 'bottom', 
                                     }, 
                             legend_title_text='left'
                            )
    fig_NoSG_col.update_layout(xaxis_title='periods', yaxis_title='values', 
                             title={'text':f''' {nameScenario}: show ValNoSG KPI for all algorithms ''',
                                     #'xanchor': 'center',
                                     'yanchor': 'bottom', 
                                     }, 
                             legend_title_text='left'
                            )
    
    # save Image
    fig_NoSG_col.write_image( os.path.join(scenarioCorePathDataViz, f"Courbe0_ValNoSG.png" ) ) 
    fig_SG_col.write_image( os.path.join(scenarioCorePathDataViz, f"Courbe0_ValSG.png" ) ) 
    
    htmlDivVal_SG = html.Div([html.H1(children="ValSG"), 
                        html.Div(children=f''' {nameScenario}: show ValSG KPI for all algorithms '''), 
                        dcc.Graph(id='graphValSG', figure=fig_SG_col),
                        ])
    
    htmlDivVal_NoSG = html.Div([html.H1(children="ValNoSG"), 
                        html.Div(children=f''' {nameScenario}: show ValNoSG KPI for all algorithms '''), 
                        dcc.Graph(id='graphValNoSG', figure=fig_NoSG_col),
                        ])
    
    
    return htmlDivVal_SG, htmlDivVal_NoSG
###############################################################################
#                   plot valSG and valNoSG : fin
###############################################################################

###############################################################################
#                   plot som(LCOST/Price) for LRI : debut
###############################################################################
def plot_LcostPrice(df_prosumers: pd.DataFrame, scenarioCorePathDataViz:str):
    """
    plot sum of LCOST/Price

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    algoName = "LRI_REPART"
    
    df_pros_algo = df_prosumers[df_prosumers.algoName == algoName]
    
    df_pros_algo['lcost_price'] = df_pros_algo['price'] - df_pros_algo['valStock_i']
    df_pros_algo['lcost_price'][df_pros_algo['lcost_price'] < 0] = 0
    df_pros_algo['lcost_price'] = df_pros_algo['lcost_price'] / df_pros_algo['price']
    
    df_lcost_price = df_pros_algo[["period", "lcost_price"]].groupby("period").sum()
    df_lcost_price.reset_index(inplace=True)
    
    fig_ratio = go.Figure()
    
    fig_ratio.add_trace(
        go.Scatter(x=df_lcost_price['period'], 
                   y=df_lcost_price['lcost_price'], 
                   name= "ratio_Lcost_over_Price",
                   mode='lines+markers', 
                   marker = dict(color = COLORS[algoName])
                )
        )
    
    fig_ratio.update_layout(xaxis_title='periods', yaxis_title='values', 
                             title={'text':''' ratio Lcost by time over time for LRI algorithms ''',
                                     #'xanchor': 'center',
                                     'yanchor': 'bottom', 
                                     }, 
                             legend_title_text='left'
                            )
    
    # save image 
    fig_ratio.write_image( os.path.join(scenarioCorePathDataViz, f"Courbe1_LcostByPricebyPeriod4LRI.png" ) ) 
    
    
    htmlDivRatioLcost = html.Div([html.H1(children="sum(Lcost/Price)"), 
                        html.Div(children=''' ratio Lcost by time over time for LRI algorithms '''), 
                        dcc.Graph(id='graphRatio', figure=fig_ratio),
                        ])
    
    return htmlDivRatioLcost
    
###############################################################################
#                   plot som(LCOST/Price) for LRI : fin
###############################################################################

###############################################################################
#                   plot QTStock all LRI, SSA, Bestie : debut
###############################################################################
def plot_sumQTStock(df_prosumers: pd.DataFrame, scenarioCorePathDataViz:str):
    """
        
    plot the sum of QTStock over time for algorithms SSA, LRI, et Bestie

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ALGOS = ["LRI_REPART", "SSA", "Bestie"]
    df_pros_algos = df_prosumers[df_prosumers['algoName'].isin(ALGOS)]
    
    df_QTStock = df_pros_algos[['period', 'algoName','QTStock', 'scenarioName']].groupby(["algoName","period"]).sum().reset_index()
    
    
    fig_qtstock = go.Figure()
    for algoName in df_pros_algos.algoName.unique().tolist():
        fig_qtstock.add_trace(
            go.Scatter(x=df_QTStock['period'], 
                       y=df_QTStock[df_QTStock.algoName == algoName]["QTStock"], 
                       name= algoName,
                       mode='lines+markers', 
                       marker = dict(color = COLORS[algoName])
                    )
            )
        
    nameScenario = df_QTStock.scenarioName.unique()[0]
    fig_qtstock.update_layout(xaxis_title='periods', yaxis_title='values', 
                              title={'text':''' show QTStock KPI for all algorithms ''',
                                       #'xanchor': 'center',
                                       'yanchor': 'bottom', 
                                       }, 
                               legend_title_text='left'
                              )
    
    # save image 
    fig_qtstock.write_image( os.path.join(scenarioCorePathDataViz, f"Courbe2_QTStockSumByProsumersbyPeriodByAlgo.png" ) ) 
    
    htmlDivQTStock = html.Div([html.H1(children="QTStock"), 
                        html.Div(children=''' show QTStock KPI for all algorithms '''), 
                        dcc.Graph(id='graphQtstock', figure=fig_qtstock),
                        ])
    
    return htmlDivQTStock
    
###############################################################################
#                   plot QTStock all LRI, SSA, Bestie : FIN
###############################################################################

###############################################################################
#                   plot sum storage all LRI, SSA, Bestie : debut
###############################################################################
def plot_sumStorage(df_prosumers:pd.DataFrame, scenarioCorePathDataViz:str):
    """
    plot storage evolution over the time

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    ALGOS = ["LRI_REPART", "SSA", "Bestie"]
    df_pros_algos = df_prosumers[df_prosumers['algoName'].isin(ALGOS)]
    
    df_Sis = df_pros_algos[['period', 'algoName','storage', 'scenarioName']].groupby(["algoName","period"]).sum().reset_index()
    
    
    fig_Si = go.Figure()
    for algoName in df_pros_algos.algoName.unique().tolist():
        fig_Si.add_trace(
            go.Scatter(x=df_Sis['period'], 
                       y=df_Sis[df_Sis.algoName == algoName]["storage"], 
                       name= algoName,
                       mode='lines+markers', 
                       marker = dict(color = COLORS[algoName])
                    )
            )
        
    nameScenario = df_Sis.scenarioName.unique()[0]
    fig_Si.update_layout(xaxis_title='periods', yaxis_title='values', 
                            title={'text':''' show Storage KPI for all algorithms ''',
                                     #'xanchor': 'center',
                                     'yanchor': 'bottom', 
                                     }, 
                             legend_title_text='left'
                        )
    
    # save image 
    fig_Si.write_image( os.path.join(scenarioCorePathDataViz, f"Courbe3_storageByPeriod.png" ) ) 
    
    htmlDivSis = html.Div([html.H1(children="Storage"), 
                        html.Div(children=''' show Storage KPI for all algorithms '''), 
                        dcc.Graph(id='graphSis', figure=fig_Si),
                        ])
    
    return htmlDivSis
    
    
###############################################################################
#                   plot sum storage all LRI, SSA, Bestie : FIN
###############################################################################

###############################################################################
#                   visu bar plot of actions(modes) : debut
###############################################################################
def plot_barModes(df_prosumers: pd.DataFrame, scenarioCorePathDataViz: str):
    """
    

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # resultat = df_prosumers.groupby('algoName')[['period','mode']].value_counts().unstack(fill_value=0)

    # # Transformation des comptages en valeurs stochastiques (probabilités)
    # resultat_stochastique = resultat.div(resultat.sum(axis=1), axis=0)

    # # Affichage du résultat stochastique
    # resultat_stochastique.reset_index(inplace=True)
    
    # #resultat_stochastique.set_index(['period', 'algoName']).plot(kind='bar', stacked=True, figsize=(10, 6))
    
    # # Réorganiser les données en format long pour Plotly
    # resultat_long = resultat_stochastique.melt(
    #                     id_vars=["period", "algoName"], 
    #                     value_vars=resultat_stochastique.columns,
    #                     var_name="mode", value_name="probabilite")
    
    # # Création du graphique avec Plotly
    # figMode = px.bar(resultat_long, x="period", y="probabilite", color="algoName", barmode="stack",
    #              title="Distribution stochastique des modes par Période et Algorithme",
    #              labels={"probabilite": "Probabilité", "period": "Période", "algoName": "Algorithme"},
    #              facet_col="mode", facet_col_wrap=2)
    
    
    
    # htmlDivModes = html.Div([html.H1(children="Distribution des strategies Modes"), 
    #                     html.Div(children=''' show distribution of strategies KPI for all algorithms '''), 
    #                     dcc.Graph(id='graphModes', figure=figMode),
    #                     ])

    # return htmlDivModes
    
    # value_counts
    df_res = df_prosumers.groupby('algoName')[['period','mode']].value_counts().unstack(fill_value=0)
    
    #Affichage du résultat stochastique
    df_stoc = df_res.div(df_res.sum(axis=1), axis=0).reset_index()


    modes = df_prosumers["mode"].unique().tolist()
    algoNames = df_prosumers["algoName"].unique().tolist()
    periods = df_prosumers["period"].unique().tolist()

    liste_4uplets = []
    for algoName, period, mode in it.product(algoNames, periods, modes):
        df_tmp = df_stoc[(df_stoc['algoName']==algoName) & 
                         (df_stoc['period']==period) ]
        #print(f"{algoName}, {period}, {mode}")
        value = df_tmp[mode].unique()[0]
        
        liste_4uplets.append((algoName, period, mode, value))
        
    df_alPeMoVal = pd.DataFrame(liste_4uplets, columns=["algoName", "period", "mode", "value"])
    
    # # Création du graphique avec Plotly
    figMode = px.bar(df_alPeMoVal[df_alPeMoVal['period'].isin([i for i in range(1,max(df_alPeMoVal['period'])+1)])], 
                 x="period", y="value", color="mode", facet_col="algoName")
    
    # save image
    figMode.write_image( os.path.join(scenarioCorePathDataViz, f"Courbe4_barplotOfStrategies.png" ) ) 
    
    htmlDivModes = html.Div([html.H1(children="Distribution des strategies Modes"), 
                        html.Div(children=''' show distribution of strategies KPI for all algorithms '''), 
                        dcc.Graph(id='graphModes', figure=figMode),
                        ])

    return htmlDivModes

    
###############################################################################
#                   visu bar plot of actions(modes) : FIN
###############################################################################

###############################################################################
#                 visu all prosumers with LCost(strat)==0 : debut
###############################################################################
def plot_numberProsumerLcostEqalZero(df_prosumers: pd.DataFrame, scenarioCorePathDataViz:str):
    """
    

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    algoName = [mot for mot in df_prosumers.algoName.unique() if 'LRI' in mot][0]
    
    df_lri = df_prosumers[df_prosumers.algoName == algoName]

    cols_2_select = ['prosumers','period','Lcost']
    
    df_lcost = df_lri[cols_2_select].groupby('period')\
                .apply(lambda x: (x['Lcost'] == 0).sum())\
                    .reset_index(name='count')
    
    # # Création du graphique avec Plotly
    figLcost = px.scatter( df_lcost, x="period", y="count")
    
    figLcost.write_image( os.path.join(scenarioCorePathDataViz, f"Courbe5_LRI_#prosumersByPeriodWithLcost=0.png" ) ) 
    
    # visualisation
    htmlDivLriLcost = html.Div([html.H1(children="LRI: #prosumers by period with Lcost=0"), 
                        html.Div(children=''' LRI: #prosumers by period with Lcost=0 '''), 
                        dcc.Graph(id='graphLriLcost', figure=figLcost),
                        ])
    
    return htmlDivLriLcost
    
###############################################################################
#                 visu all prosumers with LCost(strat)==0 : fin
###############################################################################

###############################################################################
#                   visu all plots : debut
###############################################################################
def plot_all_figures(df_prosumers: pd.DataFrame, scenarioCorePathDataViz: str): 
    """
    plot all figures from requests of latex document

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    htmlDivs = list()
    
    # Plot bar plot of valNoSG and valSG
    htmlDivVal_SG, htmlDivVal_NoSG = plot_curve_valSGNoSG(df_prosumers, scenarioCorePathDataViz)
    
    htmlDivs.append(htmlDivVal_SG)
    htmlDivs.append(htmlDivVal_NoSG)
    
    # courbe 1 : plot curve over the time the part of Lcost by Price: 
    htmlDivRatioLcost = plot_LcostPrice(df_prosumers, scenarioCorePathDataViz)
    htmlDivs.append(htmlDivRatioLcost)
    
    # courbe2 : plot QTSTock over the time
    htmlDivQTStock = plot_sumQTStock(df_prosumers, scenarioCorePathDataViz)
    htmlDivs.append(htmlDivQTStock)
    
    # courbe3: plot Storage Si
    htmlDivSis = plot_sumStorage(df_prosumers, scenarioCorePathDataViz)
    htmlDivs.append(htmlDivSis)
    
    # courbe4: plot bar of actions
    htmlDivModes = plot_barModes(df_prosumers, scenarioCorePathDataViz)
    htmlDivs.append(htmlDivModes)
    
    # courbe5: Number of prosumer by period with lcost == 0
    htmlDivLriLcost = plot_numberProsumerLcostEqalZero(df_prosumers, scenarioCorePathDataViz)
    htmlDivs.append(htmlDivLriLcost)
    
    
    # run app 
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app_PerfMeas = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    
    # app_PerfMeas.layout = html.Div(children=[ 
    #                         html.H1(children='Performance Measures',
    #                                 style={'textAlign': 'center'}
    #                                 ),
    #                         html.Div(children='Dash: plot measures for all algorithms.', 
    #                                  style={'textAlign': 'center'}),
    #                         dcc.Graph(id='perfMeas-graph', figure=fig),
    #                         dcc.Graph(id='perfMeas-graph_SG', figure=fig_SG),
    #                         ])
    app_PerfMeas.layout = html.Div(children=htmlDivs)
    
    return app_PerfMeas
    
###############################################################################
#                   visu all plots : fin
###############################################################################



if __name__ == '__main__':
    
    scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho5.json"
    
    
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
    scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas")
    
    scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
    scenario["scenarioCorePathData"] = scenarioCorePathData
    
    
    apps_pkls = load_all_algos_apps(scenario)
    df_prosumers = create_df_SG(apps_pkls=apps_pkls, index_GA_PA=0)
    
    
    app_PerfMeas = plot_all_figures(df_prosumers, scenarioCorePathDataViz)
    
    app_PerfMeas.run_server(debug=True)
    
    
    
    
    
