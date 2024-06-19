#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:42:03 2024

@author: willy

test for visualization with
    1) bokeh  
    2) dash 
"""
import os

import json
import pickle
import runApp
import application
import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from pathlib import Path


COLORS = ['gray', 'red', 'yellow', 'green']

def run_all_algos(scenarioPath):
    """
    execute all algos in dictionnary dico

    Parameters
    ----------
    dico : dict
        contains name all algos for example
        {"SSA":{"name":"SSA","logfile":"traceApplication_SSA.txt"}, 
         "SyA":{"name":"SyA","logfile":"traceApplication_SyA.txt"},
         "CSA":{"name":"SSA","logfile":"traceApplication_CSA.txt"},
         "LRI_REPART":{"name":"LRI_REPART","logfile":"traceApplication_LRI_REPART.txt"},
         }

    Returns
    -------
    None.

    """
    with open(scenarioPath) as file:
        scenario = json.load(file)
        apps = []
        
        if "SyA" in scenario["algo"]:
            app_SyA = runApp.run_syA(scenario, logfiletxt="traceApplication_SyA.txt")
            apps.append((app_SyA, "SyA", scenario["name"]))
        if "SSA" in scenario["algo"]:
            app_SSA = runApp.run_SSA(scenario, logfiletxt="traceApplication_SSA.txt")
            apps.append((app_SSA, "SSA", scenario["name"]))
        if "CSA" in scenario["algo"]:
            app_CSA = runApp.run_CSA(scenario, logfiletxt="traceApplication_CSA.txt")
            apps.append((app_CSA, "CSA", scenario["name"]))
        if "LRIRepart" in scenario["algo"]:
            app_LRI_REPART = runApp.run_LRI_REPART(scenario, logfiletxt="traceApplication_LRI_REPART.txt")
            apps.append((app_LRI_REPART, "LRI_REPART", scenario["name"]))
       
        return apps
    
    
    
    # app_SyA = runApp.run_syA(logfiletxt="traceApplication_SyA.txt")
    # app_SSA = runApp.run_SSA(logfiletxt="traceApplication_SSA.txt")
    # app_CSA = runApp.run_CSA(logfiletxt="traceApplication_CSA.txt")
    # app_LRI_REPART = runApp.run_LRI_REPART(logfiletxt="traceApplication_LRI_REPART.txt")
    
    # return (app_SyA, "SyA"), (app_SSA, "SSA"), (app_CSA, "CSA"), (app_LRI_REPART,"LRI_REPART")


def create_df_of_prosumers(app_al):
    """
    create dataframe for algorithm application  

    Parameters
    ----------
    app_al : App
        DESCRIPTION.

    Returns
    -------
    a dataframe.

    """
    N = app_al.SG.prosumers.size
    T = app_al.SG.nbperiod
    Ts = np.arange(T)
    prosumers = dict()
    for i in range(N):
        Pis = app_al.SG.prosumers[i].production
        Cis = app_al.SG.prosumers[i].consumption
        Sis = app_al.SG.prosumers[i].storage
        prodits = app_al.SG.prosumers[i].prodit
        consits = app_al.SG.prosumers[i].consit
        utilities = app_al.SG.prosumers[i].utility
        modes = app_al.SG.prosumers[i].mode
        ValNoSGis = app_al.SG.prosumers[i].valNoSG
        ValStocks = app_al.SG.prosumers[i].valStock
        ValRepart = app_al.SG.prosumers[i].Repart
        
        dico = {"T":Ts, "Pis": Pis, "Cis":Cis, "Sis":Sis, "Prodits":prodits, 
                "Consits":consits, "utility": utilities, "modes":modes, 
                "ValNoSGis":ValNoSGis, "ValStocks":ValStocks, 
                "ValRepart":ValRepart}
        df_prosumeri = pd.DataFrame(dico)
        prosumers["prosumer"+str(i)] = df_prosumeri
        
    return prosumers

def create_df_of_prosumers_MERGEALGOS(tu_algos:tuple) -> pd.DataFrame:
    """
    merge of algorithm values in one dataframe containing values by prosumers

    Parameters
    ----------
    tu_algos : tuple
        DESCRIPTION.

    Returns
    -------
    pd.DataFrame

    """
    df_algos = list()
    for tu_algo in tu_algos:
        app_al, algoName = tu_algo
        N = app_al.SG.prosumers.size
        T = app_al.SG.nbperiod
        Ts = np.arange(T)
        
        for i in range(N):
            Pis = app_al.SG.prosumers[i].production
            Cis = app_al.SG.prosumers[i].consumption
            Sis = app_al.SG.prosumers[i].storage
            prodits = app_al.SG.prosumers[i].prodit
            consits = app_al.SG.prosumers[i].consit
            utilities = app_al.SG.prosumers[i].utility
            modes = app_al.SG.prosumers[i].mode
            ValNoSGis = app_al.SG.prosumers[i].valNoSG
            ValStocks = app_al.SG.prosumers[i].valStock
            ValRepart = app_al.SG.prosumers[i].Repart
            
            
            algos = np.repeat(algoName, repeats=app_al.SG.nbperiod)
            prosumers = np.repeat("prosumer"+str(i), repeats=app_al.SG.nbperiod)
            dico = {"algoName": algos, "T":Ts, "Prosumers": prosumers,
                    "Pis": Pis, "Cis":Cis, "Sis":Sis, "Prodits":prodits, 
                    "Consits":consits, "utility": utilities, "modes":modes, 
                    "ValNoSGis":ValNoSGis, "ValStocks":ValStocks, 
                    "ValRepart":ValRepart}
            df_prosumeri = pd.DataFrame(dico)
            
            df_algos.append(df_prosumeri)
            
    df_algos = pd.concat(df_algos, axis=0, ignore_index=True)
        
    return df_algos
   
def create_df_SG(tu_algos:tuple) -> pd.DataFrame:
    """
    merge of algorithm values in one dataframe
    
    Parameters
    ----------
    tu_algo: tuple
        list of application containing values
        example:
           tu_algos =  (app_SyA, "SyA"), (app_SSA, "SSA"), (app_CSA, "CSA"), (app_LRI_REPART,"LRI_REPART")
        
    Returns
    -------
    df_SG:pd.DataFrame, df_APP:pd.DataFrame
    """
    df_algos = list()
    df_APPs = list()
    for tu_algo in tu_algos:
        app_al, algoName = tu_algo
        valEgoc_ts = app_al.SG.ValEgoc
        ValNoSG_ts = app_al.SG.ValNoSG
        ValSG_ts = app_al.SG.ValSG
        Reduct_ts = app_al.SG.Reduct
        Cost_ts = app_al.SG.Cost
        insg_ts = app_al.SG.insg
        outsg_ts = app_al.SG.outsg
        T = app_al.SG.nbperiod
        Ts = np.arange(T)
        algos = np.repeat(algoName, repeats=app_al.SG.nbperiod)
        
        df_algo = pd.DataFrame({"algoName": algos, "Cost_ts": Cost_ts, "T":Ts,
                              "ValEgoc_ts": valEgoc_ts, "ValSG_ts": ValSG_ts,
                              "ValNoSG_ts": ValNoSG_ts, "Reduct_ts": Reduct_ts, 
                              "In_SG_ts": insg_ts, "Out_SG_ts": outsg_ts,
                               })
        df_APP = pd.DataFrame({"algoName": [algoName],
                               "valNoSG_A": [app_al.valNoSG_A], 
                               "valSG_A": [app_al.valSG_A]})
        df_algos.append(df_algo)
        df_APPs.append(df_APP)
    
    df_SG = pd.concat(df_algos, axis=0, ignore_index=True)
    
    df_APP = pd.concat(df_APPs, axis=0, ignore_index=True)
        
    return df_SG, df_APP

def create_df_SG_NEW(apps_algos:list) -> pd.DataFrame:
    """
    merge of algorithm values in one dataframe
    
    Parameters
    ----------
    apps_algos: tuple
        list of application containing values
        example:
           apps_algos =  (app_SyA, "SyA", "LRI versus egoistes"), (app_SSA, "SSA", "LRI versus egoistes"), 
                       (app_CSA, "CSA", "LRI versus egoistes"), (app_LRI_REPART,"LRI_REPART", "LRI versus egoistes")
        
    Returns
    -------
    df_SG:pd.DataFrame, df_APP:pd.DataFrame
    """
    df_algos = list()
    df_APPs = list()
    for tu_app_algo in apps_algos:
        app_al, algoName, nameScenario = tu_app_algo
        valEgoc_ts = app_al.SG.ValEgoc
        ValNoSG_ts = app_al.SG.ValNoSG
        ValSG_ts = app_al.SG.ValSG
        Reduct_ts = app_al.SG.Reduct
        Cost_ts = app_al.SG.Cost
        insg_ts = app_al.SG.insg
        outsg_ts = app_al.SG.outsg
        T = app_al.SG.nbperiod
        Ts = np.arange(T)
        algos = np.repeat(algoName, repeats=app_al.SG.nbperiod)
        scenarios = np.repeat(nameScenario, repeats=app_al.SG.nbperiod)
        
        df_algo = pd.DataFrame({"nameScenario": scenarios, "algoName": algos, 
                                "Cost_ts": Cost_ts, "T":Ts,
                              "ValEgoc_ts": valEgoc_ts, "ValSG_ts": ValSG_ts,
                              "ValNoSG_ts": ValNoSG_ts, "Reduct_ts": Reduct_ts, 
                              "In_SG_ts": insg_ts, "Out_SG_ts": outsg_ts,
                               })
        df_APP = pd.DataFrame({"algoName": [algoName],
                               "valNoSG_A": [app_al.valNoSG_A], 
                               "valSG_A": [app_al.valSG_A],
                               "nameScenario": [nameScenario]})
        df_algos.append(df_algo)
        df_APPs.append(df_APP)
    
    df_SG = pd.concat(df_algos, axis=0, ignore_index=True)
    
    df_APP = pd.concat(df_APPs, axis=0, ignore_index=True)
        
    return df_SG, df_APP

def plot_oneApp_prosumers(prosumers:dict, app_al:application.App, algoName:str):
    """
    plot with dash prosumers kPI for one algorithm

    Parameters
    ----------
    prosumers : dict
        dict of prosumers of game.
        
    app_al : application
        application of one algorithm
    
    algoName: str
        name of algorithm

    Returns
    -------
    None.

    """
    cpt = 0
    prosumers = create_df_of_prosumers(app_al)
    htmlDivs = list()
    for key_namePro, df_prosumeri in prosumers.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['Pis'], 
                                 name='Production', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['Cis'], 
                                 name='consumption', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['Sis'], 
                                 name='Storage', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['Prodits'], 
                                 name='prod', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['Consits'], 
                                 name='cons', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['utility'], 
                                 name='utility', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['ValNoSGis'], 
                                 name='ValNoSGi', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['ValStocks'], 
                                 name='ValStock', mode='lines+markers'
                                 )
                      )
        fig.add_trace(go.Scatter(x=df_prosumeri['T'], y=df_prosumeri['ValRepart'], 
                                 name='ValRepart', mode='lines+markers'
                                 )
                      )
        
        cpt += 1
        htmlDiv = html.Div([html.H1(children=key_namePro), 
                            html.Div(children=f''' Dash: show prosumers KPI for {algoName} algorithm '''), 
                            dcc.Graph(id='graph'+str(cpt), figure=fig),
                            ])
        fig.update_layout(xaxis_title='periods', yaxis_title='values')
        htmlDivs.append(htmlDiv)
    
    # run app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children = htmlDivs)
    
    
    return app

def plot_ManyApp_perfMeasure(df_APP: pd.DataFrame):
    """
    plot measure performances (ValNoSG_A, ValSG_A ) for all run algorithms

    Parameters
    ----------
    df_APP : pd.DataFrame
        a dataframe that the columns are : algoName, ValNoSG_A, ValSG_A 

    Returns
    -------
    None.

    """
    fig = px.bar(df_APP, x="algoName", y="valNoSG_A", #["valNoSG_A", "valSG_A"], 
                 barmode="group", title="Performance Measures of Algorithms")
    fig1 = px.bar(df_APP, x="algoName", y="valSG_A", #["valNoSG_A", "valSG_A"], 
                 barmode="group", title="Performance Measures of Algorithms")
    
    
    # run app
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app_PerfMeas = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    
    app_PerfMeas.layout = html.Div(children=[ 
                            html.H1(children='Hello Dash',
                                    style={'textAlign': 'center'}
                                    ),
                            html.Div(children='Dash: A web application framework for your data.', 
                                     style={'textAlign': 'center'}),
                            dcc.Graph(id='perfMeas-graph', figure=fig),
                            dcc.Graph(id='perfMeas1-graph', figure=fig1)
                            ])
    
    ####################### new version: debut#################################
    # x=['a','b','c','d']
    # fig = go.Figure(go.Bar(x =x, y=[2,5,1,9], name='Montreal',
    #                    base = 0, width = 0.2, offset = 0.0,
    #                    marker = dict(color = 'rgb(0,120,255)')))
    
    # fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa',
    #                 width = 0.2, offset = -0.4, base=0,
    #                 marker = dict(color = 'rgb(250,60,0)')))
    # fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto',
    #                  width = 0.2, offset = -0.2,
    #                  marker = dict(color = 'rgb(250,130,0)')))
    
    # fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':['d','a','c','b']})
    # fig.show()
    
    # fig = go.Figure(go.Bar(x=df_APP.algoName, y=df_APP.valNoSG_A, name='valNoSG_A',
    #                    base = 0, width = 0.2, offset = 0.0,
    #                    marker = dict(color = 'rgb(0,120,255)')))
    
    # fig.add_trace(go.Bar(x=df_APP.algoName, y=df_APP.valSG_A, name='valSG_A',
    #                 width = 0.2, offset = -0.4, base=0,
    #                 marker = dict(color = 'rgb(250,60,0)')))
    # fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto',
    #                  width = 0.2, offset = -0.2,
    #                  marker = dict(color = 'rgb(250,130,0)')))
    
    # fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':['d','a','c','b']})
    # fig.show()
    
    
    ####################### new version: end  #################################
        
    return app_PerfMeas

def plot_ManyApp_perfMeasure_DBG(df_APP: pd.DataFrame, df_SG: pd.DataFrame):
    """
    plot measure performances (ValNoSG_A, ValSG_A ) for all run algorithms

    Parameters
    ----------
    df_APP : pd.DataFrame
        a dataframe that the columns are : algoName, ValNoSG_A, ValSG_A 
        
    df_SG : pd.DataFrame
        a dataframe that the columns are : 'algoName', 'Cost_ts', 'T', '
        valEgoc_ts', 'ValSG_ts', 'ValNoSG_ts', 'Reduct_ts'

    Returns
    -------
    None.

    """
    ####################### 1er version DF_APP: first  ###############################
    ## show in xaxis the algorithms and yaxis the values of val{NoSG_A, SG_A}

    
    # fig = go.Figure(go.Bar(x=df_APP.algoName, y=df_APP.valNoSG_A, name='valNoSG_A',
    #                    base = 0, width = 0.2, offset = 0.0,
    #                    marker = dict(color = 'rgb(0,120,255)')))
    
    # fig.add_trace(go.Bar(x=df_APP.algoName, y=df_APP.valSG_A, name='valSG_A',
    #                 width = 0.2, offset = -0.4, base=0,
    #                 marker = dict(color = 'rgb(250,60,0)')))
    
    # #fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':['d','a','c','b']})
    # #fig.show()
    # fig.update_layout(barmode='group', # "stack" | "group" | "overlay" | "relative"
    #                   xaxis={'categoryorder':'array'}, 
    #                   xaxis_title="algorithms", yaxis_title="values", 
    #                   title_text="Performance Measures")

    ####################### 1er version DF_APP: end  #################################
    
    # ####################### 2er version: first  ###############################
    # ## show in xaxis the algorithms and yaxis the values of val{NoSG_A, SG_A} 
    # ## group by group ie the bars of val{NoSG_A, SG_A} values by algorithms
    # ## values of valNoSG_A for algorithms (SSA, CSA, SyA, LRI) and 
    # ## values of valSG_A for algorithms (SSA, CSA, SyA, LRI)
    # fig = go.Figure()
    # fig.add_trace(go.Bar(x=df_APP.algoName.tolist(), 
    #                       y=df_APP.valNoSG_A.tolist(), name='valNoSG_A',
    #                       base = 0.0, width = 0.2, offset = 0.0,
    #                       marker = dict(color = 'rgb(0,120,255)')
    #                       )
    #               )
    # fig.add_trace(go.Bar(x=df_APP.algoName.tolist(), 
    #                       y=df_APP.valSG_A.tolist(), name='valSG_A',
    #                       width = 0.2, offset = -0.4, base=0,
    #                       marker = dict(color = 'rgb(250,60,0)')
    #                       )
    #               )
    
    # fig.update_layout(barmode='group', # "stack" | "group" | "overlay" | "relative"
    #                   boxmode='group', 
    #                   xaxis={'categoryorder':'array', 'categoryarray':df_APP.algoName.tolist()},  
    #                   xaxis_title="algorithms", yaxis_title="values", 
    #                   title_text="Performance Measures")
    # fig.update_traces(marker_color="red", selector={"name": df_APP.algoName.tolist()[0]})
    # fig.update_traces(marker_color="yellow", selector={"name": df_APP.algoName.tolist()[1]})
    # fig.update_traces(marker_color="green", selector={"name": df_APP.algoName.tolist()[2]})
    # fig.update_traces(marker_color="pink", selector={"name": df_APP.algoName.tolist()[3]})
    
    
    # ####################### 2er version DF_APP: end  #################################
    
    ####################### 2er version DF_APP: first  ###############################
    ## use subplots
    ## show in xaxis the algorithms and yaxis the values of val{NoSG_A, SG_A} 
    ## group by group ie the bars of val{NoSG_A, SG_A} values by algorithms
    ## values of valNoSG_A for algorithms (SSA, CSA, SyA, LRI) and 
    ## values of valSG_A for algorithms (SSA, CSA, SyA, LRI)
    # fig = make_subplots(rows=1, cols=2)
    # fig.add_trace(go.Bar(x=df_APP.algoName, 
    #                       y=df_APP.valNoSG_A, name='valNoSG_A',
    #                       #base = 0.0, width = 0.2, offset = 0.0,
    #                       #marker = dict(color = 'rgb(0,120,255)')
    #                       ),
    #               row=1, col=1,
    #               )
    # fig.add_trace(go.Bar(x=df_APP.algoName, 
    #                       y=df_APP.valSG_A, name='valSG_A',
    #                       #width = 0.2, offset = -0.4, base=0,
    #                       #marker = dict(color = 'rgb(250,60,0)')
    #                       ), 
    #               row=1, col=2,
    #               )
    
    # fig.update_layout(barmode='group', # "stack" | "group" | "overlay" | "relative"
    #                   boxmode='group', 
    #                   xaxis={'categoryorder':'array', 'categoryarray':df_APP.algoName.tolist()}, 
    #                   xaxis_title="algorithms", yaxis_title="values", 
    #                   title_text="Performance Measures")
    # fig.update_traces(marker_color="red", selector={"name": df_APP.algoName.tolist()[0]})
    # fig.update_traces(marker_color="yellow", selector={"name": df_APP.algoName.tolist()[1]})
    # fig.update_traces(marker_color="green", selector={"name": df_APP.algoName.tolist()[2]})
    # fig.update_traces(marker_color="pink", selector={"name": df_APP.algoName.tolist()[3]})
    
    ####################### 2er version DF_APP: end  #################################
    
    ####################### 3er version DF_APP: first  ###############################
    htmlDivs = list()
    
    #------- debug start
    fig = go.Figure()
    df_APP_T = df_APP.T
    new_header = df_APP_T.iloc[0]; 
    df_APP_T = df_APP_T[1:]; 
    df_APP_T.columns = new_header
    index = df_APP_T.index.tolist(); index.remove("nameScenario")
    for num_app, algoName in enumerate(df_APP_T.columns):
        fig.add_trace(go.Bar(x=index, 
                              y=df_APP_T.iloc[:,num_app].tolist(), name=df_APP_T.columns.tolist()[num_app],
                              base = 0.0, width = 0.2, offset = 0.2*num_app,
                              marker = dict(color = COLORS[num_app])
                              )
                      )
    
    fig.update_layout(barmode='stack', # "stack" | "group" | "overlay" | "relative"
                      #boxmode='group', 
                      xaxis={'categoryorder':'array', 'categoryarray':df_APP.algoName.tolist()},  
                      xaxis_title="algorithms", yaxis_title="values", 
                      title_text="Performance Measures")
    
    htmlDiv_df_APP = html.Div([
                html.H1(children='Performance Measures',
                        style={'textAlign': 'center'}
                        ),
                html.Div(children=f"scenario <{df_APP_T.loc['nameScenario',:].unique()[0]}>: plot measures for all algorithms.", 
                         style={'textAlign': 'center'}),
                dcc.Graph(id='perfMeas-graph', figure=fig),
            ])
    htmlDivs.append(htmlDiv_df_APP)
    
    #------- debug end
    
    
    # fig = go.Figure()
    # df_APP_T = df_APP.T
    # new_header = df_APP_T.iloc[0]; 
    # df_APP_T = df_APP_T[1:]; 
    # df_APP_T.columns = new_header
    # fig.add_trace(go.Bar(x=df_APP_T.index.tolist(), 
    #                       y=df_APP_T.iloc[:,0].tolist(), name=df_APP_T.columns.tolist()[0],
    #                       base = 0.0, width = 0.2, offset = 0.0,
    #                       marker = dict(color = 'gray')
    #                       )
    #               )
    # fig.add_trace(go.Bar(x=df_APP_T.index.tolist(), 
    #                       y=df_APP_T.iloc[:,1].tolist(), name=df_APP_T.columns.tolist()[1],
    #                       base=0.0, width = 0.2, offset = 0.2,
    #                       marker = dict(color = 'red')
    #                       )
    #               )
    # fig.add_trace(go.Bar(x=df_APP_T.index.tolist(), 
    #                       y=df_APP_T.iloc[:,2].tolist(), name=df_APP_T.columns.tolist()[2],
    #                       base=0.0, width = 0.2, offset = 0.4,
    #                       marker = dict(color = 'yellow')
    #                       )
    #               )
    # fig.add_trace(go.Bar(x=df_APP_T.index.tolist(), 
    #                       y=df_APP_T.iloc[:,3].tolist(), name=df_APP_T.columns.tolist()[3],
    #                       base=0.0, width = 0.2, offset = 0.6,
    #                       marker = dict(color = 'green')
    #                       )
    #               )
    
    # fig.update_layout(barmode='stack', # "stack" | "group" | "overlay" | "relative"
    #                   #boxmode='group', 
    #                   xaxis={'categoryorder':'array', 'categoryarray':df_APP.algoName.tolist()},  
    #                   xaxis_title="algorithms", yaxis_title="values", 
    #                   title_text="Performance Measures")
    
    # htmlDiv_df_APP = html.Div([
    #             html.H1(children='Performance Measures',
    #                     style={'textAlign': 'center'}
    #                     ),
    #             html.Div(children='Dash: plot measures for all algorithms.', 
    #                      style={'textAlign': 'center'}),
    #             dcc.Graph(id='perfMeas-graph', figure=fig),
    #         ])
    # htmlDivs.append(htmlDiv_df_APP)
    
    ####################### 3er version DF_APP: end  ###############################
    
    ####################### 1er version DF_SG: first  ###############################
    ## show in xaxis the algorithms and yaxis the values of val{NoSG_A, SG_A}
    
    # compute Out_SG_ts - In_SG_ts
    df_SG["Out_SG_ts-In_SG_ts"] = df_SG.Out_SG_ts - df_SG.In_SG_ts
    
    cpt = 0
    #htmlDivs = list()
    #fig_SG = go.Figure()
    nameScenarios = df_SG.nameScenario.unique()
    name_cols = ['ValSG_ts', 'ValNoSG_ts', "Out_SG_ts-In_SG_ts"]
    for nameScenario in nameScenarios:
        for name_col in name_cols:
            fig_SG_col = go.Figure()
            cpt += 1
            if name_col != "Out_SG_ts-In_SG_ts":
                for num_app, algoName in enumerate(df_SG.algoName.unique()):
                    #cpt += 1
                    df_SG_algo = df_SG[df_SG.algoName == algoName]
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo[name_col], 
                                              name= algoName,
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
            else:
                for num_app, algoName in enumerate(df_SG.algoName.unique()):
                    #cpt += 1
                    df_SG_algo = df_SG[df_SG.algoName == algoName]
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo[name_col], 
                                              name= algoName,
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo["In_SG_ts"], 
                                              name= algoName+'_In_SG_ts',
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo["Out_SG_ts"], 
                                              name= algoName+'_Out_SG_ts',
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
            fig_SG_col.update_layout(xaxis_title='periods', yaxis_title='values', 
                                     title={'text':f''' {nameScenario}: show {name_col} KPI for all algorithms ''',
                                             #'xanchor': 'center',
                                             'yanchor': 'bottom', 
                                             }, 
                                     legend_title_text='left'
                                    )
            htmlDiv = html.Div([html.H1(children=name_col), 
                                html.Div(children=f''' {nameScenario}: show {name_col} KPI for all algorithms '''), 
                                dcc.Graph(id='graph'+str(cpt), figure=fig_SG_col),
                                ])
            
            htmlDivs.append(htmlDiv)
    
    ####################### 1er version DF_SG: end  #################################
    
    
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
#                START : Plot with Pickle backup
###############################################################################
def create_repo_for_save_jobs(scenario:dict):
    scenarioCorePath = os.path.join(scenario["scenarioPath"], scenario["scenarioName"])
    scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "data")
    scenarioCorePathVisu = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "visu")
    scenario["scenarioCorePath"] = scenarioCorePath
    scenario["scenarioCorePathData"] = scenarioCorePathData
    scenario["scenarioCorePathVisu"] = scenarioCorePathVisu
    
    # create a scenarioPath if not exists
    Path(scenarioCorePathData).mkdir(parents=True, exist_ok=True)  
    
    return scenario

    
def load_all_algos(scenario:dict):
    """
    load all algorithms saved in the pickle format

    Parameters
    ----------
    scenario : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    app_pkls = []
    
    for algoName in scenario["algo"].keys():
        # load pickle file
        with open(os.path.join(scenario["scenarioCorePathData"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'rb') as f:
            app_pkl = pickle.load(f) # deserialize using load()
            app_pkls.append((app_pkl, algoName, scenario["scenarioName"]))
        pass

    return app_pkls
    pass

def create_df_SG_V1(apps_pkls_algos:list) -> pd.DataFrame:
    """
    merge of algorithm values in one dataframe
    
    Parameters
    ----------
    apps_algos: tuple
        list of application containing values
        example:
           apps_algos =  (app_SyA, "SyA", "LRI versus egoistes"), (app_SSA, "SSA", "LRI versus egoistes"), 
                       (app_CSA, "CSA", "LRI versus egoistes"), (app_LRI_REPART,"LRI_REPART", "LRI versus egoistes")
        
    Returns
    -------
    df_SG:pd.DataFrame, df_APP:pd.DataFrame
    """
    df_algos = list()
    df_APPs = list()
    df_PROSUMERS = list()
    df_prosumers_algos =list() 
    for tu_app_algo in apps_pkls_algos:
        app_al, algoName, nameScenario = tu_app_algo
        valEgoc_ts = app_al.SG.ValEgoc
        ValNoSG_ts = app_al.SG.ValNoSG
        ValSG_ts = app_al.SG.ValSG
        Reduct_ts = app_al.SG.Reduct
        Cost_ts = app_al.SG.Cost
        insg_ts = app_al.SG.insg
        outsg_ts = app_al.SG.outsg
        T = app_al.SG.nbperiod
        Ts = np.arange(T)
        algos = np.repeat(algoName, repeats=app_al.SG.nbperiod)
        scenarios = np.repeat(nameScenario, repeats=app_al.SG.nbperiod)
        
        df_algo = pd.DataFrame({"nameScenario": scenarios, "algoName": algos, 
                                "Cost_ts": Cost_ts, "T":Ts,
                              "ValEgoc_ts": valEgoc_ts, "ValSG_ts": ValSG_ts,
                              "ValNoSG_ts": ValNoSG_ts, "Reduct_ts": Reduct_ts, 
                              "In_SG_ts": insg_ts, "Out_SG_ts": outsg_ts,
                               })
        df_APP = pd.DataFrame({"algoName": [algoName],
                               "valNoSG_A": [app_al.valNoSG_A], 
                               "valSG_A": [app_al.valSG_A],
                               "nameScenario": [nameScenario]})
        
        for i in range(app_al.SG.prosumers.size):
            Pis = app_al.SG.prosumers[i].production[:T]
            Cis = app_al.SG.prosumers[i].consumption[:T]
            Sis = app_al.SG.prosumers[i].storage[:T]
            prodits = app_al.SG.prosumers[i].prodit
            consits = app_al.SG.prosumers[i].consit
            utilities = app_al.SG.prosumers[i].utility
            modes = app_al.SG.prosumers[i].mode
            ValNoSGis = app_al.SG.prosumers[i].valNoSG
            ValStocks = app_al.SG.prosumers[i].valStock
            ValRepartis = app_al.SG.prosumers[i].Repart
            
            algos = np.repeat(algoName, repeats=T)
            prosumers = np.repeat("prosumer"+str(i), repeats=T)
            dico = {"algoName": algos, "T":Ts, "Prosumers": prosumers,
                    "Pis": Pis, "Cis":Cis, "Sis":Sis, "Prodits":prodits, 
                    "Consits":consits, "utility": utilities, "modes":modes, 
                    "ValNoSGis":ValNoSGis, "ValStocks":ValStocks, 
                    "ValRepartis":ValRepartis}
            df_prosumeri = pd.DataFrame(dico)
            
            df_prosumers_algos.append(df_prosumeri)
            
        df_algos.append(df_algo)
        df_APPs.append(df_APP)
    
    df_SG = pd.concat(df_algos, axis=0, ignore_index=True)
    
    df_APP = pd.concat(df_APPs, axis=0, ignore_index=True)
    
    df_PROSUMERS = pd.concat(df_prosumers_algos, axis=0, ignore_index=True)
        
    return df_SG, df_APP, df_PROSUMERS


def plot_ManyApp_perfMeasure_V1(df_APP: pd.DataFrame, df_SG: pd.DataFrame, df_PROSUMERS: pd.DataFrame):
    """
    plot measure performances (ValNoSG_A, ValSG_A ) for all run algorithms

    Parameters
    ----------
    df_APP : pd.DataFrame
        a dataframe that the columns are : algoName, ValNoSG_A, ValSG_A 
        
    df_SG : pd.DataFrame
        a dataframe that the columns are : 'algoName', 'Cost_ts', 'T', '
        valEgoc_ts', 'ValSG_ts', 'ValNoSG_ts', 'Reduct_ts'

    Returns
    -------
    None.

    """
    
    ####################### 3er version DF_APP: first  ###############################
    htmlDivs = list()
    
    fig = go.Figure()
    df_APP_T = df_APP.T
    new_header = df_APP_T.iloc[0]; 
    df_APP_T = df_APP_T[1:]; 
    df_APP_T.columns = new_header
    index = df_APP_T.index.tolist(); index.remove("nameScenario")
    for num_app, algoName in enumerate(df_APP_T.columns):
        fig.add_trace(go.Bar(x=index, 
                              y=df_APP_T.iloc[:,num_app].tolist(), name=df_APP_T.columns.tolist()[num_app],
                              base = 0.0, width = 0.2, offset = 0.2*num_app,
                              marker = dict(color = COLORS[num_app])
                              )
                      )
    
    fig.update_layout(barmode='stack', # "stack" | "group" | "overlay" | "relative"
                      #boxmode='group', 
                      xaxis={'categoryorder':'array', 'categoryarray':df_APP.algoName.tolist()},  
                      xaxis_title="algorithms", yaxis_title="values", 
                      title_text="Performance Measures")
    
    htmlDiv_df_APP = html.Div([
                html.H1(children='Performance Measures',
                        style={'textAlign': 'center'}
                        ),
                html.Div(children=f"scenario <{df_APP_T.loc['nameScenario',:].unique()[0]}>: plot measures for all algorithms.", 
                         style={'textAlign': 'center'}),
                dcc.Graph(id='perfMeas-graph', figure=fig),
            ])
    htmlDivs.append(htmlDiv_df_APP)
    
    ####################### 3er version DF_APP: end  ###############################
    
    ####################### 1er version DF_SG: first  ###############################
    ## show in xaxis the algorithms and yaxis the values of val{NoSG_A, SG_A}
    
    # compute Out_SG_ts - In_SG_ts
    df_SG["Out_SG_ts-In_SG_ts"] = df_SG.Out_SG_ts - df_SG.In_SG_ts
    
    cpt = 0
    #htmlDivs = list()
    #fig_SG = go.Figure()
    nameScenarios = df_SG.nameScenario.unique()
    name_cols = ['ValSG_ts', 'ValNoSG_ts', "Out_SG_ts-In_SG_ts"]
    for nameScenario in nameScenarios:
        for name_col in name_cols:
            fig_SG_col = go.Figure()
            cpt += 1
            if name_col != "Out_SG_ts-In_SG_ts":
                for num_app, algoName in enumerate(df_SG.algoName.unique()):
                    #cpt += 1
                    df_SG_algo = df_SG[df_SG.algoName == algoName]
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo[name_col], 
                                              name= algoName,
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
            else:
                for num_app, algoName in enumerate(df_SG.algoName.unique()):
                    #cpt += 1
                    df_SG_algo = df_SG[df_SG.algoName == algoName]
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo[name_col], 
                                              name= algoName,
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo["In_SG_ts"], 
                                              name= algoName+'_In_SG_ts',
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo["Out_SG_ts"], 
                                              name= algoName+'_Out_SG_ts',
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
            fig_SG_col.update_layout(xaxis_title='periods', yaxis_title='values', 
                                     title={'text':f''' {nameScenario}: show {name_col} KPI for all algorithms ''',
                                             #'xanchor': 'center',
                                             'yanchor': 'bottom', 
                                             }, 
                                     legend_title_text='left'
                                    )
            htmlDiv = html.Div([html.H1(children=name_col), 
                                html.Div(children=f''' {nameScenario}: show {name_col} KPI for all algorithms '''), 
                                dcc.Graph(id='graph'+str(cpt), figure=fig_SG_col),
                                ])
            
            htmlDivs.append(htmlDiv)
    
    ####################### 1er version DF_SG: end  #################################
    
    #####################  1er version DF_PROSUMERS: START   ###################
    # TODO 
    DICO_COLORS = {'LRI_REPART':'gray', 'CSA':'red', 'SSA':'yellow', 'SyA':'green'}
    for nameScenario in nameScenarios:
        for algoName in df_PROSUMERS.algoName.unique().tolist():
            
            df_pro_algo = df_PROSUMERS[df_PROSUMERS.algoName == algoName]
            
            df_pro_algo_PCS = df_pro_algo.groupby("T")[['Pis', 'Cis', 'Sis']].aggregate('sum')
            df_pro_algo_PCS["T"] = np.arange(df_pro_algo_PCS.shape[0])
            
            fig_PCS = go.Figure()
            fig_PCS.add_trace(go.Scatter(x=df_pro_algo_PCS["T"], y=df_pro_algo_PCS["Pis"], 
                                         name= "Pis",
                                         mode='lines+markers', 
                                         marker = dict(color = COLORS[0])
                                         )
                              )
            fig_PCS.add_trace(go.Scatter(x=df_pro_algo_PCS["T"], y=df_pro_algo_PCS["Cis"], 
                                         name= "Cis",
                                         mode='lines+markers', 
                                         marker = dict(color = COLORS[1])
                                         )
                              )
            fig_PCS.add_trace(go.Scatter(x=df_pro_algo_PCS["T"], y=df_pro_algo_PCS["Sis"], 
                                         name= "Sis",
                                         mode='lines+markers', 
                                         marker = dict(color = COLORS[2])
                                         )
                              )
            
            fig_PCS.update_layout(xaxis_title='periods', yaxis_title='values', 
                                     title={'text':f''' {nameScenario}: show {algoName} sum of prosumers Production, Consumption and Storage ''',
                                             #'xanchor': 'center',
                                             'yanchor': 'bottom', 
                                             }, 
                                     legend_title_text='left'
                                    )
            htmlDiv = html.Div([html.H1(children=algoName+" Pis, Cis, Sis"), 
                                html.Div(children=f''' {nameScenario}: show {algoName} sum of prosumers Production, Consumption and Storage  '''), 
                                dcc.Graph(id='graph_PCS_'+algoName, figure=fig_PCS),
                                ])
            
            htmlDivs.append(htmlDiv)
            
            
    #####################  1er version DF_PROSUMERS: END    ###################
    
    
    
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
#                END : Plot with Pickle backup
###############################################################################


if __name__ == '__main__':
    scenarioPath = "./scenario1.json"
    scenarioPath = "./data_scenario/scenario_test_LRI.json"
    scenarioPath = "./data_scenario/scenario_SelfishDebug_LRI_N10_T5.json"
    
    
    with open(scenarioPath) as file:
        scenario = json.load(file)
    
    scenario = create_repo_for_save_jobs(scenario)
    
    apps_pkls = load_all_algos(scenario)
    
    df_SG, df_APP, df_PROSUMERS = create_df_SG_V1(apps_pkls_algos=apps_pkls)
    
    app_PerfMeas = plot_ManyApp_perfMeasure_V1(df_APP, df_SG, df_PROSUMERS)
    app_PerfMeas.run_server(debug=True)
    
    
    
    # apps = run_all_algos(scenarioPath=scenarioPath)
    # df_SG, df_APP = create_df_SG_NEW(apps_algos=apps)
    
    # app_PerfMeas = plot_ManyApp_perfMeasure_DBG(df_APP, df_SG)
    # app_PerfMeas.run_server(debug=True)
    
    
    
    ## (app_SyA,nameSyA), (app_SSA,nameSSA), (app_CSA,nameCSA), (app_LRI_REPART,nameLRIREPART) = run_all_algos()
    ## prosumers_SyA = create_df_of_prosumers(app_SyA)
    ## tu_algos = ((app_SyA,nameSyA), (app_SSA,nameSSA), (app_CSA,nameCSA), (app_LRI_REPART,nameLRIREPART) )
    ## df_SG, df_APP = create_df_SG(tu_algos)
    ## #app = plot_oneApp_prosumers(prosumers=prosumers_SyA, app_al=app_SyA, algoName=nameSyA)
    ## #app.run_server(debug=True)
    
    ## #app_PerfMeas = plot_ManyApp_perfMeasure(df_APP)
    ## app_PerfMeas = plot_ManyApp_perfMeasure_DBG(df_APP, df_SG)
    ## app_PerfMeas.run_server(debug=True)
    ##pass

