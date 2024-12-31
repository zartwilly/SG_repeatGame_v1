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
#import dash_core_components as dcc
#import dash_html_components as html
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from pathlib import Path


COLORS = ['gray', 'red', 'yellow', 'green', "blue"]

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

    
def load_all_algos(scenario:dict, scenarioViz:dict):
    """
    load all algorithms saved in the pickle format

    Parameters
    ----------
    scenario : dict
        DESCRIPTION.
        
    scenarioViz : dict
        DESCRIPTION

    Returns
    -------
    None.

    """
    app_pkls = []
    
    for algoName in scenario["algo"].keys():
        # load pickle file
        try:
            with open(os.path.join(scenario["scenarioCorePathData"], scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'rb') as f:
                app_pkl = pickle.load(f) # deserialize using load()
                app_pkls.append((app_pkl, algoName, scenario["scenarioName"]))
        except FileNotFoundError:
            print(f" {scenario['scenarioName']+'_'+algoName+'_APP.pkl'}  NOT EXIST")
        pass

    return app_pkls
    pass

def load_all_algos_V1(scenario:dict, scenarioViz: dict):
    """
    load all algorithms saved in the pickle format

    Parameters
    ----------
    scenario : dict
        DESCRIPTION.

    scenarioViz: dict)
        DESCRIPTION.
        exple : {"algoName": list(scenario["algo"].keys()), 
                 "graphs":[["ValSG_ts", "ValNoSG_ts", "Bar"], 
                           ["QttEPO", "line"], ["MaxPrMode", "line]"] ]
                }
    Returns
    -------
    None.

    """
    app_pkls = []
    
    for algoName in scenarioViz["algoName"]:
        # load pickle file
        try:
            scenarioCorePathDataAlgoName = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", algoName)
            with open(os.path.join(scenarioCorePathDataAlgoName, scenario["scenarioName"]+"_"+algoName+"_APP"+'.pkl'), 'rb') as f:
                app_pkl = pickle.load(f) # deserialize using load()
                #app_pkls.append((app_pkl, algoName, scenario["scenarioName"]))
                dfs = list()
                if algoName == "LRI_REPART" and scenario["simul"]["is_plotDataVStockQTsock"]:
                    # load .csv into list of dataframe
                    ## look for dataframes beginning to 'runLRI_df_'
                    prefix = "runLRI_df_"
                    prefixed = [filename for filename in os.listdir(scenarioCorePathDataAlgoName) if filename.startswith(prefix)]
                    for pref in prefixed:
                        df_tmp = pd.read_csv(os.path.join(scenarioCorePathDataAlgoName, pref), skiprows=0)
                        print(f" *** {pref } terminé")
                        dfs.append(df_tmp)
                
                app_pkls.append((app_pkl, algoName, scenario["scenarioName"], dfs))       
                
        except FileNotFoundError:
            print(f" {scenario['scenarioName']+'_'+algoName+'_APP.pkl'}  NOT EXIST")
        pass

    return app_pkls
    pass

def create_df_SG_V1(apps_pkls_algos:list, initial_period:int) -> pd.DataFrame:
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
        Ts = np.arange(initial_period, T)
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


def create_df_SG_V1_SelectPeriod(apps_pkls_algos:list, initial_period:int) -> pd.DataFrame:
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
        valEgoc_ts = app_al.SG.ValEgoc[initial_period:]
        ValNoSG_ts = app_al.SG.ValNoSG[initial_period:]
        ValSG_ts = app_al.SG.ValSG[initial_period:]
        Reduct_ts = app_al.SG.Reduct[initial_period:]
        Cost_ts = app_al.SG.Cost[initial_period:]
        insg_ts = app_al.SG.insg[initial_period:]
        outsg_ts = app_al.SG.outsg[initial_period:]
        T = app_al.SG.nbperiod
        Ts = np.arange(initial_period, T)
        algos = np.repeat(algoName, repeats=len(Ts))
        scenarios = np.repeat(nameScenario, repeats=len(Ts))
        
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
            Pis = app_al.SG.prosumers[i].production[initial_period:T]
            Cis = app_al.SG.prosumers[i].consumption[initial_period:T]
            Sis = app_al.SG.prosumers[i].storage[initial_period:T]
            prodits = app_al.SG.prosumers[i].prodit[initial_period:T]
            consits = app_al.SG.prosumers[i].consit[initial_period:T]
            utilities = app_al.SG.prosumers[i].utility[initial_period:T]
            modes = app_al.SG.prosumers[i].mode[initial_period:T]
            ValNoSGis = app_al.SG.prosumers[i].valNoSG[initial_period:T]
            ValStocks = app_al.SG.prosumers[i].valStock[initial_period:T]
            ValRepartis = app_al.SG.prosumers[i].Repart[initial_period:T]
            
            algos = np.repeat(algoName, repeats= len(Ts))
            prosumers = np.repeat("prosumer"+str(i), repeats=len(Ts))
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


def create_df_SG_V2_SelectPeriod(apps_pkls_algos:list, initial_period:int) -> pd.DataFrame:
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
    dfs_VStock = list()
    dfs_QTStock = list()
    for tu_app_algo in apps_pkls_algos:
        app_al, algoName, nameScenario, dfs = tu_app_algo
        valEgoc_ts = app_al.SG.ValEgoc[initial_period:]
        ValNoSG_ts = app_al.SG.ValNoSG[initial_period:]
        ValSG_ts = app_al.SG.ValSG[initial_period:]
        Reduct_ts = app_al.SG.Reduct[initial_period:]
        Cost_ts = app_al.SG.Cost[initial_period:]
        insg_ts = app_al.SG.insg[initial_period:]
        outsg_ts = app_al.SG.outsg[initial_period:]
        T = app_al.SG.nbperiod
        Ts = np.arange(initial_period, T)
        algos = np.repeat(algoName, repeats=len(Ts))
        scenarios = np.repeat(nameScenario, repeats=len(Ts))
        
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
            Pis = app_al.SG.prosumers[i].production[initial_period:T]
            Cis = app_al.SG.prosumers[i].consumption[initial_period:T]
            Sis = app_al.SG.prosumers[i].storage[initial_period:T]
            prodits = app_al.SG.prosumers[i].prodit[initial_period:T]
            consits = app_al.SG.prosumers[i].consit[initial_period:T]
            utilities = app_al.SG.prosumers[i].utility[initial_period:T]
            modes = app_al.SG.prosumers[i].mode[initial_period:T]
            ValNoSGis = app_al.SG.prosumers[i].valNoSG[initial_period:T]
            ValStocks = app_al.SG.prosumers[i].valStock[initial_period:T]
            ValRepartis = app_al.SG.prosumers[i].Repart[initial_period:T]
            PrMode0_is = app_al.SG.prosumers[i].prmode[initial_period:T][:,0]
            PrMode1_is = app_al.SG.prosumers[i].prmode[initial_period:T][:,1]
            #PrMode_is = app_al.SG.prosumers[i].prmode[initial_period:T]
            
            algos = np.repeat(algoName, repeats= len(Ts))
            prosumers = np.repeat("prosumer"+str(i), repeats=len(Ts))
            dico = {"algoName": algos, "T":Ts, "Prosumers": prosumers,
                    "Pis": Pis, "Cis":Cis, "Sis":Sis, "Prodits":prodits, 
                    "Consits":consits, "utility": utilities, "modes":modes, 
                    "ValNoSGis":ValNoSGis, "ValStocks":ValStocks, 
                    "ValRepartis":ValRepartis, 
                    "PrMode0_is": PrMode0_is, "PrMode1_is":PrMode1_is}
            df_prosumeri = pd.DataFrame(dico)
            
            df_prosumers_algos.append(df_prosumeri)
            
        df_algos.append(df_algo)
        df_APPs.append(df_APP)
        
        # dataset for LRI containing VStock, QTstock by period
        dfs_VStock, dfs_QTStock = [], []
        if algoName == "LRI_REPART" and len(dfs) != 0:
            for df_tmp in dfs:
                df_tmp_ts = df_tmp[['valStock_i','QTStock', 'period']].aggregate('mean')
                dfs_VStock.append(df_tmp_ts)
                
                # looking for the lastest step for each period
                df_QTStock_t = df_tmp[df_tmp.step == df_tmp.step.max()][["period","prosumers","QTStock"]]
                dfs_QTStock.append(df_QTStock_t)
                
        
            dfs_VStock = pd.concat(dfs_VStock, axis=1).T
            dfs_QTStock = pd.concat(dfs_QTStock, axis=0)
            
    
    df_SG = pd.concat(df_algos, axis=0, ignore_index=True)
    
    df_APP = pd.concat(df_APPs, axis=0, ignore_index=True)
    
    df_PROSUMERS = pd.concat(df_prosumers_algos, axis=0, ignore_index=True)
    
    df_PROSUMERS['maxPrMode'] = df_PROSUMERS[['PrMode0_is', 'PrMode1_is']].values.max(axis=1)
    
    # df_dbg = df_PROSUMERS.groupby('T').agg({'maxPrMode':'mean'})
    # df_SG = pd.merge(df_SG, df_dbg, on='T', how='outer')
    
    df_dbg = df_PROSUMERS.groupby(['algoName','T']).agg({'maxPrMode':'mean'}).reset_index()
    df_SG = pd.merge(df_SG, df_dbg, on=['algoName','T'])
    
    dfs_QTStock = dfs_QTStock.reset_index(drop=True)
    dfs_QTStock_R = dfs_QTStock.groupby('period').QTStock.apply(lambda x: pd.Series([(x<=0).sum(), (x>0).sum()])).unstack().reset_index()
    dfs_QTStock_R.rename(columns={0: "R_t_minus", 1: "R_t_plus"}, inplace=True)
    dfs_QTStock_som = dfs_QTStock.groupby("period")["QTStock"].agg(sum)
    
    dfs_QTStock_R = pd.merge(dfs_QTStock_R, dfs_QTStock_som, on="period")
    
    dfs_QTStock_R["MoyQTStock"] = (dfs_QTStock_R["QTStock"] / dfs_QTStock_R["R_t_plus"]).fillna(0)
    
    return df_SG, df_APP, df_PROSUMERS, dfs_VStock, dfs_QTStock_R


###############################################################################
#                   Create df_SG_V3_SelectPeriod : Debut
###############################################################################
def create_df_SG_V3_SelectPeriod(apps_pkls_algos:list, initial_period:int) -> pd.DataFrame:
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
    dfs_VStock = list()
    dfs_QTStock = list()
    for tu_app_algo in apps_pkls_algos:
        app_al, algoName, nameScenario, dfs = tu_app_algo
        valEgoc_ts = app_al.SG.ValEgoc[initial_period:]
        ValNoSG_ts = app_al.SG.ValNoSG[initial_period:]
        ValSG_ts = app_al.SG.ValSG[initial_period:]
        Reduct_ts = app_al.SG.Reduct[initial_period:]
        Cost_ts = app_al.SG.Cost[initial_period:]
        insg_ts = app_al.SG.insg[initial_period:]
        outsg_ts = app_al.SG.outsg[initial_period:]
        T = app_al.SG.nbperiod
        Ts = np.arange(initial_period, T)
        algos = np.repeat(algoName, repeats=len(Ts))
        scenarios = np.repeat(nameScenario, repeats=len(Ts))
        
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
            Pis = app_al.SG.prosumers[i].production[initial_period:T]
            Cis = app_al.SG.prosumers[i].consumption[initial_period:T]
            Sis = app_al.SG.prosumers[i].storage[initial_period:T]
            prodits = app_al.SG.prosumers[i].prodit[initial_period:T]
            consits = app_al.SG.prosumers[i].consit[initial_period:T]
            utilities = app_al.SG.prosumers[i].utility[initial_period:T]
            modes = app_al.SG.prosumers[i].mode[initial_period:T]
            ValNoSGis = app_al.SG.prosumers[i].valNoSG[initial_period:T]
            ValStocks = app_al.SG.prosumers[i].valStock[initial_period:T]
            ValRepartis = app_al.SG.prosumers[i].Repart[initial_period:T]
            PrMode0_is = app_al.SG.prosumers[i].prmode[initial_period:T][:,0]
            PrMode1_is = app_al.SG.prosumers[i].prmode[initial_period:T][:,1]
            #PrMode_is = app_al.SG.prosumers[i].prmode[initial_period:T]
            
            algos = np.repeat(algoName, repeats= len(Ts))
            prosumers = np.repeat("prosumer"+str(i), repeats=len(Ts))
            dico = {"algoName": algos, "T":Ts, "Prosumers": prosumers,
                    "Pis": Pis, "Cis":Cis, "Sis":Sis, "Prodits":prodits, 
                    "Consits":consits, "utility": utilities, "modes":modes, 
                    "ValNoSGis":ValNoSGis, "ValStocks":ValStocks, 
                    "ValRepartis":ValRepartis, 
                    "PrMode0_is": PrMode0_is, "PrMode1_is":PrMode1_is}
            df_prosumeri = pd.DataFrame(dico)
            
            df_prosumers_algos.append(df_prosumeri)
            
        df_algos.append(df_algo)
        df_APPs.append(df_APP)
        
        # dataset for LRI containing VStock, QTstock by period
        dfs_VStock, dfs_QTStock = [], []
        if algoName == "LRI_REPART" and len(dfs) != 0:
            for df_tmp in dfs:
                df_tmp_ts = df_tmp[['valStock_i','QTStock', 'period']].aggregate('mean')
                dfs_VStock.append(df_tmp_ts)
                
                # looking for the lastest step for each period
                df_QTStock_t = df_tmp[df_tmp.step == df_tmp.step.max()][["period","prosumers","QTStock"]]
                dfs_QTStock.append(df_QTStock_t)
                
        
            dfs_VStock = pd.concat(dfs_VStock, axis=1).T
            dfs_QTStock = pd.concat(dfs_QTStock, axis=0)
            
    
    df_SG = pd.concat(df_algos, axis=0, ignore_index=True)
    
    df_APP = pd.concat(df_APPs, axis=0, ignore_index=True)
    
    df_PROSUMERS = pd.concat(df_prosumers_algos, axis=0, ignore_index=True)
    
    df_PROSUMERS['maxPrMode'] = df_PROSUMERS[['PrMode0_is', 'PrMode1_is']].values.max(axis=1)
    
    # df_dbg = df_PROSUMERS.groupby('T').agg({'maxPrMode':'mean'})
    # df_SG = pd.merge(df_SG, df_dbg, on='T', how='outer')
    
    df_dbg = df_PROSUMERS.groupby(['algoName','T']).agg({'maxPrMode':'mean'}).reset_index()
    df_SG = pd.merge(df_SG, df_dbg, on=['algoName','T'])
    
    dfs_QTStock = dfs_QTStock.reset_index(drop=True)
    dfs_QTStock_R = dfs_QTStock.groupby('period').QTStock.apply(lambda x: pd.Series([(x<=0).sum(), (x>0).sum()])).unstack().reset_index()
    dfs_QTStock_R.rename(columns={0: "R_t_minus", 1: "R_t_plus"}, inplace=True)
    dfs_QTStock_som = dfs_QTStock.groupby("period")["QTStock"].agg(sum)
    
    dfs_QTStock_R = pd.merge(dfs_QTStock_R, dfs_QTStock_som, on="period")
    
    dfs_QTStock_R["MoyQTStock"] = (dfs_QTStock_R["QTStock"] / dfs_QTStock_R["R_t_plus"]).fillna(0)
    
    return df_SG, df_APP, df_PROSUMERS, dfs_VStock, dfs_QTStock_R

###############################################################################
#               Create df_SG_V3_SelectPeriod : Fin
###############################################################################

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
    

def plot_ManyApp_perfMeasure_V2(df_APP: pd.DataFrame, df_SG: pd.DataFrame, df_PROSUMERS: pd.DataFrame, dfs_VStock: pd.DataFrame, dfs_QTStock_R: pd.DataFrame, df_shapleys: pd.DataFrame, scenarioCorePathDataViz:str):
    """
    plot measure performances (ValNoSG_A, ValSG_A ) for all run algorithms

    Parameters
    ----------
    df_APP : pd.DataFrame
        a dataframe that the columns are : algoName, ValNoSG_A, ValSG_A 
        
    df_SG : pd.DataFrame
        a dataframe that the columns are : 'algoName', 'Cost_ts', 'T', '
        valEgoc_ts', 'ValSG_ts', 'ValNoSG_ts', 'Reduct_ts'
        
    dfs_VStock: pd.DataFrame
        a dataframe that the columns are : 'valStock_i', 'QTStock', 'period'
        
    dfs_QTStock_R: pd.DataFrame
        a dataframe that the columns are : period, R_t_minus, R_t_plus, QTStock, MoyQTStock

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
    
    nameScenario = df_APP_T.loc['nameScenario',:].unique()[0]
    fig.update_layout(barmode='stack', # "stack" | "group" | "overlay" | "relative"
                      #boxmode='group', 
                      xaxis={'categoryorder':'array', 'categoryarray':df_APP.algoName.tolist()},  
                      xaxis_title="algorithms", yaxis_title="values", 
                      title_text="Performance Measures")
    
    fig.write_image( os.path.join(scenarioCorePathDataViz, f"PerformanceMeasures__{nameScenario}.png" ) )
    
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
    df_SG["QttEPO_ts"] = df_SG.Out_SG_ts - df_SG.In_SG_ts
    
    cpt = 0
    #htmlDivs = list()
    #fig_SG = go.Figure()
    nameScenarios = df_SG.nameScenario.unique()
    name_cols = ['ValSG_ts', 'ValNoSG_ts', "QttEPO_ts", "maxPrMode"]
    for nameScenario in nameScenarios:
        for name_col in name_cols:
            fig_SG_col = go.Figure()
            cpt += 1
            if name_col == "maxPrMode":
                for num_app, algoName in enumerate(df_SG.algoName.unique()):
                    df_SG_algo = df_SG[df_SG.algoName == algoName]
                    fig_SG_col.add_trace(go.Scatter(x=df_SG_algo['T'], y=df_SG_algo[name_col], 
                                              name= algoName,
                                              mode='lines+markers', 
                                              marker = dict(color = COLORS[num_app])
                                              )
                                         )
            elif name_col == "QttEPO_ts":
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
            fig_SG_col.update_layout(xaxis_title='periods', yaxis_title='values', 
                                     title={#'text':f''' {nameScenario}: show {name_col} KPI for all algorithms ''',
                                            'text':f''' show {name_col} KPI for all algorithms ''',
                                             #'xanchor': 'center',
                                             'yanchor': 'bottom', 
                                             }, 
                                     legend_title_text='left'
                                    )
            fig_SG_col.write_image( os.path.join(scenarioCorePathDataViz, f"{name_col}_{nameScenario}.png" ) )
            
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
                                     #title={'text':f''' {nameScenario}: show {algoName} sum of prosumers Production, Consumption and Storage ''',
                                     title={'text':f''' show {algoName} sum of prosumers Production, Consumption and Storage ''',
                                      
                                            #'xanchor': 'center',
                                             'yanchor': 'bottom', 
                                             }, 
                                     legend_title_text='left'
                                    )
            
            fig_PCS.write_image( os.path.join(scenarioCorePathDataViz, f"{algoName}_PiCiSi_{nameScenario}.png" ) )
            
            htmlDiv = html.Div([html.H1(children=algoName+" Pis, Cis, Sis"), 
                                html.Div(children=f''' {nameScenario}: show {algoName} sum of prosumers Production, Consumption and Storage  '''), 
                                dcc.Graph(id='graph_PCS_'+algoName, figure=fig_PCS),
                                ])
            
            htmlDivs.append(htmlDiv)
            
            
    #####################  1er version DF_PROSUMERS: END    ###################
    
    #####################  1er version dfs_VStock: START    ###################
    name_cols = dfs_VStock.columns.tolist()
    dfs_VStock = dfs_VStock.sort_values(by=['period'], ascending=True)
    for num_col, name_col in enumerate(name_cols):
        fig_VStock_col = go.Figure()
        cpt += 1
        
        fig_VStock_col.add_trace(go.Scatter(x=dfs_VStock['period'], 
                                            y=dfs_VStock[name_col], 
                                            name= "LRI_REPART_VSTOCK",
                                            mode='lines+markers', 
                                            marker = dict(color = COLORS[num_col])
                                      )
                                 )
        
        fig_VStock_col.update_layout(xaxis_title='period', yaxis_title='values', 
                                 title={#'text':f''' {nameScenario}: show {name_col} ''',
                                        'text':f''' LRI_REPART: show {name_col} ''',
                                         #'xanchor': 'center',
                                         'yanchor': 'bottom', 
                                         }, 
                                 legend_title_text='left'
                                )
        
        fig_VStock_col.write_image( os.path.join(scenarioCorePathDataViz, f"LRI_REPART_{name_col}_{nameScenario}.png" ) )
        
        htmlDiv = html.Div([html.H1(children=name_col), 
                            html.Div(children=f''' {nameScenario}: show {name_col} VSTOCK '''), 
                            dcc.Graph(id='graph'+str(cpt), figure=fig_VStock_col),
                            ])
        
        htmlDivs.append(htmlDiv)
    
    #####################  1er version dfs_VStock: END      ###################
    
    #####################  1er version dfs_QTStock: START    ##################
    name_cols = dfs_QTStock_R.columns[-3:]
    fig_QTStock_col = go.Figure()
    for num_col, name_col in enumerate(name_cols):
        cpt += 1
        fig_QTStock_col.add_trace(go.Bar(x=dfs_QTStock_R['period'], 
                                            y=dfs_QTStock_R[name_col],
                                            name=name_col,
                                            base = 0.0, width = 0.2, offset = 0.2*num_col, 
                                            marker = dict(color = COLORS[num_col])
                                      )
                                 )
        
    fig_QTStock_col.update_layout(barmode='stack', # "stack" | "group" | "overlay" | "relative"
                          #boxmode='group', 
                        xaxis={'categoryorder':'array', 'categoryarray':df_APP.algoName.tolist()},  
                        xaxis_title="Understanding Variables QTStock, R and MoyQTStock", yaxis_title="values", 
                        title_text="Understanding Variables QTStock, R and MoyQTStock")
        
    fig_QTStock_col.write_image( os.path.join(scenarioCorePathDataViz, f"LRI_REPART_QTStock_Rt_MoyQTStock_{nameScenario}.png" ) )
    
    htmlDiv = html.Div([html.H1(children=name_col), 
                        html.Div(children=f''' {nameScenario}: show {name_col} QTSTOCK, R_t and MoyQTStock'''), 
                        dcc.Graph(id='graphQTStock_Rt', figure=fig_QTStock_col),
                        ])
    
    htmlDivs.append(htmlDiv)
        
    #  Version With lines containing many variables ###########################
    fig_PCS = go.Figure()
    fig_PCS.add_trace(go.Scatter(x=dfs_QTStock_R["period"], y=dfs_QTStock_R[name_cols[0]], 
                                 name= name_cols[0],
                                 mode='lines+markers', 
                                 marker = dict(color = COLORS[0])
                                 )
                      )
    fig_PCS.add_trace(go.Scatter(x=dfs_QTStock_R["period"], y=dfs_QTStock_R[name_cols[1]], 
                                 name= name_cols[1],
                                 mode='lines+markers', 
                                 marker = dict(color = COLORS[1])
                                 )
                      )
    fig_PCS.add_trace(go.Scatter(x=dfs_QTStock_R["period"], y=dfs_QTStock_R[name_cols[2]], 
                                 name= name_cols[2],
                                 mode='lines+markers', 
                                 marker = dict(color = COLORS[2])
                                 )
                      )
    
    htmlDiv = html.Div([html.H1(children="Show Variables"+" QTstock, R_t, MoyQTStock"), 
                        html.Div(children=f''' {nameScenario}: show {algoName} sum of prosumers QTStock, R_t '''), 
                        dcc.Graph(id='graph_QTSRt'+algoName, figure=fig_PCS),
                        ])
    
    htmlDivs.append(htmlDiv)
    
    
    #####################  1er version dfs_QTStock: END      ##################
    
    #####################  1er version df_shapleys: START      ##################
    if not df_shapleys.empty:
        fig_shapley = go.Figure()
        df_shapleys = df_shapleys.set_index("Prosumers")
        index = df_shapleys.index.tolist();
        for num_app, algoName in enumerate(df_shapleys.columns):
            fig_shapley.add_trace(go.Bar(x=index, 
                                  y=df_shapleys.iloc[:,num_app].tolist(), name=df_shapleys.columns.tolist()[num_app],
                                  base = 0.0, width = 0.2, offset = 0.2*num_app,
                                  marker = dict(color = COLORS[num_app])
                                  )
                          )
        
        fig_shapley.update_layout(barmode='stack', # "stack" | "group" | "overlay" | "relative"
                          #boxmode='group', 
                          xaxis={'categoryorder':'array', 'categoryarray':df_shapleys.columns.tolist()},  
                          xaxis_title="Prosumers", yaxis_title="Shapley Values", 
                          title_text="Shapley Values between LRI and others algorithms")
        
        htmlDiv_df_shapley = html.Div([
                    html.H1(children="Shapley Values between LRI and others algorithms",
                            style={'textAlign': 'center'}
                            ),
                    html.Div(children=f"scenario <{df_APP_T.loc['nameScenario',:].unique()[0]}>: plot shapley values for all algorithms.", 
                             style={'textAlign': 'center'}),
                    dcc.Graph(id='shapleyValue-graph', figure=fig_shapley),
                ])
        htmlDivs.append(htmlDiv_df_shapley)
    #####################  1er version df_shapleys: END      ##################
    
    
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

###############################################################################
#                       START : Plot with Bokeh
###############################################################################
from bokeh.plotting import figure, save
# Create Bokeh-Table with DataFrame:
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.core.properties import value
from bokeh.io import show, output_file
from bokeh.transform import dodge

from bokeh.models.tools import HoverTool, PanTool, BoxZoomTool, WheelZoomTool 
from bokeh.models.tools import RedoTool, ResetTool, SaveTool, UndoTool
TOOLS = [PanTool(), BoxZoomTool(), WheelZoomTool(), UndoTool(),
         RedoTool(), ResetTool(), SaveTool(),
         HoverTool(tooltips=[("Price", "$y"), ("Time", "$x")]) ]


def plot_BOKEH_ManyApps(df_APP: pd.DataFrame, 
                        df_SG: pd.DataFrame, 
                        df_PROSUMERS: pd.DataFrame, 
                        dfs_VStock: pd.DataFrame, 
                        dfs_QTStock_R: pd.DataFrame, 
                        df_shapleys: pd.DataFrame, 
                        scenarioCorePathDataViz: str):
    """
    plot measure performances (ValNoSG_A, ValSG_A ) for all run algorithms

    Parameters
    ----------
    df_APP : pd.DataFrame
        a dataframe that the columns are : algoName, ValNoSG_A, ValSG_A
        
    df_SG : pd.DataFrame
        a dataframe that the columns are : 'algoName', 'Cost_ts', 'T', '
        valEgoc_ts', 'ValSG_ts', 'ValNoSG_ts', 'Reduct_ts'
        
    dfs_VStock: pd.DataFrame
        a dataframe that the columns are : 'valStock_i', 'QTStock', 'period'
        
    dfs_QTStock_R: pd.DataFrame
        a dataframe that the columns are : period, R_t_minus, R_t_plus, QTStock, MoyQTStock

    Returns
    -------
    None.

    """
    # set output to static HTML file
    htmlfile = os.path.join(scenarioCorePathDataViz, "plot_game_variables.html")
    output_file(filename=htmlfile, title="Static HTML file")
    
    
    df_valSGNoSG = df_APP.copy()
    df_valSGNoSG = df_valSGNoSG.set_index("algoName")
    df_valSGNoSG = df_valSGNoSG.drop(columns=['nameScenario'], axis=1)
    cols = df_valSGNoSG.columns.tolist()
    
    df_valSGNoSG = df_valSGNoSG.reset_index()
    
    data = df_valSGNoSG.to_dict(orient='list')
    source = ColumnDataSource(data=data)
    
    
    idx = df_valSGNoSG["algoName"].tolist()
    p_bar = figure(x_range=idx, 
                   y_range=(df_valSGNoSG[cols].values.min() - 10, df_valSGNoSG[cols].values.max() ), 
                   height=350, title="Performance Measures",
                   toolbar_location=None, tools=TOOLS)
    
    p_bar.vbar(x=dodge("algoName", -0.1, range=p_bar.x_range), 
               top=cols[0], width=0.2, source=source,
               color="#c9d9d3", legend_label=cols[0])
    
    p_bar.vbar(x=dodge("algoName", 0.1, range=p_bar.x_range), 
               top=cols[1], width=0.2, source=source,
               color="#e84d60", legend_label=cols[1])
    
    p_bar.x_range.range_padding = 0.2
    p_bar.y_range.start = df_valSGNoSG[cols].values.min() -1 if df_valSGNoSG[cols].values.min() < 0 else 0
    #p_bar.y_range.start = 0 if df_valSGNoSG[cols].values.min() < 0 else 0
    p_bar.xgrid.grid_line_color = None
    #p_bar.legend.location = "top_left"
    #p_bar.legend.orientation = "horizontal"
    #p_bar.add_layout(Legend(), 'right')
    p_bar.add_layout(p_bar.legend[0], 'right')

    show(p_bar)
    
    save(p_bar)
    
    
def plot_bokeh_test_pivot(df_APP: pd.DataFrame, 
                            df_SG: pd.DataFrame, 
                            df_PROSUMERS: pd.DataFrame, 
                            dfs_VStock: pd.DataFrame, 
                            dfs_QTStock_R: pd.DataFrame, 
                            df_shapleys: pd.DataFrame, 
                            scenarioCorePathDataViz: str):
    
    # set output to static HTML file
    htmlfile = os.path.join(scenarioCorePathDataViz, "plot_game_variables.html")
    output_file(filename=htmlfile, title="Static HTML file")
    
    
    df_valSGNoSG = df_APP.copy()
    df_valSGNoSG = df_valSGNoSG.set_index("algoName")
    df_valSGNoSG = df_valSGNoSG.drop(columns=['nameScenario'], axis=1)
    cols = df_valSGNoSG.columns.tolist()
    
    df_valSGNoSG = df_valSGNoSG.reset_index()
    df_valSGNoSG = df_valSGNoSG.melt(id_vars=["algoName"])
    df_valSGNoSG = df_valSGNoSG.set_index(['variable', 'algoName'])
    
    factors = df_valSGNoSG.index.tolist()
    data = df_valSGNoSG.to_dict(orient='list')
    source = ColumnDataSource(data = dict(x=factors,
                                          data=data['value']) )
    
    p_bar = figure(x_range=FactorRange(*factors), 
                   height=250,
                   toolbar_location=None, tools=TOOLS)
    
    p_bar.vbar(x="x", width=0.9, alpha=0.5, 
               color=["blue", "red"], source=source,
               )
    
    p_bar.x_range.range_padding = 0.2
    p_bar.y_range.start = df_valSGNoSG[cols].values.min() -1 if df_valSGNoSG[cols].values.min() < 0 else 0
    p_bar.xgrid.grid_line_color = None
    p_bar.add_layout(p_bar.legend[0], 'right')

    show(p_bar)
    
    save(p_bar)
   
###############################################################################
#                       END : Plot with Bokeh
###############################################################################

if __name__ == '__main__':
    scenarioFile = "./scenario1.json"
    scenarioFile = "./data_scenario/scenario_test_LRI.json"
    scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N10_T5.json"
    scenarioFile = "./data_scenario/scenario_SelfishDB_LRI_N20_T100_K5000_B2_Rho5.json"
    scenarioFile = "./data_scenario/scenario_SelfishDebug_LRI_N20_T100_K5000_B2_Rho5_newFormula_QSTOCK.json"
    
    # TODO delete after test 
    scenarioFile = "./data_scenario/scenario_version20092024_dataDonnee_N8_T20_K5000_B1_rho5_StockDiffT0_NotSHAPLEY.json"
    
    # is_plotDataVStockQTsock = True
    
    with open(scenarioFile) as file:
        scenario = json.load(file)
    
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
    scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas")
    scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
    scenario["scenarioCorePathData"] = scenarioCorePathData
    
    scenario["simul"]["is_plotDataVStockQTsock"] = True if scenario["simul"]["is_plotDataVStockQTsock"] == "True" else False
    
    
    ## test if scenarioName+"Viz".json exists ---> start
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

    apps_pkls = load_all_algos_V1(scenario, scenarioViz)
    
    initial_period = 0 # 2, 10 
    
    df_SG, df_APP, df_PROSUMERS, dfs_VStock, dfs_QTStock_R \
        = create_df_SG_V2_SelectPeriod(apps_pkls_algos=apps_pkls, initial_period=initial_period)

    
    is_plot_BOKEH = False #True

    if not is_plot_BOKEH :
        df_shapleys = pd.DataFrame()
    
        app_PerfMeas = plot_ManyApp_perfMeasure_V2(df_APP, df_SG, df_PROSUMERS, dfs_VStock, dfs_QTStock_R, df_shapleys)
        app_PerfMeas.run_server(debug=True)
    else:
        df_shapleys = pd.DataFrame()
        # plot_BOKEH_ManyApps(df_APP, df_SG, df_PROSUMERS, 
        #                     dfs_VStock, dfs_QTStock_R, df_shapleys,
        #                     scenarioCorePathDataViz)
        
        plot_bokeh_test_pivot(df_APP, df_SG, df_PROSUMERS, 
                              dfs_VStock, dfs_QTStock_R, df_shapleys,
                              scenarioCorePathDataViz)
    
    # -------------------------------------------------------------------------
    # with open(scenarioPath) as file:
    #     scenario = json.load(file)
    
    # scenario = create_repo_for_save_jobs(scenario)
    
    # apps_pkls = load_all_algos(scenario)
    
    # initial_period = 2 #10
    # df_SG, df_APP, df_PROSUMERS = create_df_SG_V1_SelectPeriod(apps_pkls, initial_period)
    
    # app_PerfMeas = plot_ManyApp_perfMeasure_V1(df_APP, df_SG, df_PROSUMERS)
    # app_PerfMeas.run_server(debug=True)
    # -------------------------------------------------------------------------
    
    
    
    
    # df_SG, df_APP, df_PROSUMERS = create_df_SG_V1(apps_pkls_algos=apps_pkls)
    
    # app_PerfMeas = plot_ManyApp_perfMeasure_V1(df_APP, df_SG, df_PROSUMERS)
    # app_PerfMeas.run_server(debug=True)
    
    
    # new curves  #### ======> TODELETE
    # initial_period = 2 #10
    # if initial_period > 0:
    #     df_SG, df_APP, df_PROSUMERS = create_df_SG_V1_SelectPeriod(apps_pkls, initial_period)
        
    # app_PerfMeas = plot_ManyApp_perfMeasure_V1(df_APP, df_SG, df_PROSUMERS)
    # app_PerfMeas.run_server(debug=True)
    
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


