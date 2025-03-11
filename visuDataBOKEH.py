#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:57:32 2025

@author: willy

visualization with bokeh
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

from bokeh.layouts import layout
from bokeh.plotting import figure, show, output_file, save
from bokeh.transform import factor_cmap
from bokeh.transform import dodge
from bokeh.palettes import Spectral5
from bokeh.models import ColumnDataSource
from bokeh.models import FactorRange
from bokeh.models import Legend
from bokeh.models import HoverTool


###############################################################################
#                   CONSTANTES: debut
###############################################################################
PREFIX_DF = "runLRI_df_"
DF_LRI_NAME = "run_LRI_REPART_DF_T_Kmax.csv"
LRI_ALGO_NAME = "LRI_REPART"

COLORS = {"SyA":"gray", "Bestie":"red", "CSA":"yellow", "SSA":"green", "LRI_REPART":"blue"}

# define a list of markers to use for the scatter plot
MARKERS = ["circle", "square", "triangle"]

# set up the tooltips
TOOLTIPS_Val_SG_NoSG = [
    ("value", "$y{(0,0)}"),
]
TOOLTIPS_LCOST = [("value", "$y{(0,0)}")]
TOOLTIPS_MODES = [("value", "$y{.5,3}")]
TOOLTIPS_XY_ai = [("value", "$y{(0.1,1)}")]

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
    
    
    # set up the figure
    plotValSG = figure(
        title="{nameScenario}: show ValSG Value KPI for all algorithms",
        height=300,
        sizing_mode="stretch_width",  # use the full width of the parent element
        tooltips=TOOLTIPS_Val_SG_NoSG,
        output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",  # enable box zoom by default
    )
    
    plotValNoSG = figure(
        title="{nameScenario}: show ValNoSG Value KPI for all algorithms",
        height=300,
        sizing_mode="stretch_width",  # use the full width of the parent element
        tooltips=TOOLTIPS_Val_SG_NoSG,
        output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",  # enable box zoom by default
    )
    
    # source_valSG = ColumnDataSource(data=df_valSG)
    # source_valNoSG = ColumnDataSource(data=df_valNoSG)
    
    for algoName in df_valSG.algoName.unique().tolist():
        df_valSG_algo = df_valSG[df_valSG.algoName == algoName]
        plotValSG.line(x=df_valSG_algo["period"], y=df_valSG_algo["ValSG"], 
                       line_width=2, color=COLORS[algoName], alpha=0.8, 
                       legend_label=algoName)
        
        df_valNoSG_algo = df_valNoSG[df_valNoSG.algoName == algoName]
        plotValNoSG.line(x=df_valNoSG_algo["period"], y=df_valNoSG_algo["valNoSG_i"], 
                       line_width=2, color=COLORS[algoName], alpha=0.8, 
                       legend_label=algoName)
        
        
        
    plotValSG.legend.location = "top_left"
    plotValSG.legend.click_policy = "hide"
    plotValNoSG.legend.location = "top_left"
    plotValNoSG.legend.click_policy = "hide"
    
    return plotValSG, plotValNoSG

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
    
    plotLcost = figure(
        title="ratio Lcost by time over time for LRI algorithms",
        height=300,
        sizing_mode="stretch_width",  # use the full width of the parent element
        tooltips=TOOLTIPS_LCOST,
        output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",  # enable box zoom by default
    )
    
    plotLcost.line(x=df_lcost_price["period"], y=df_lcost_price['lcost_price'], 
                   line_width=2, color=COLORS[algoName], alpha=0.8, 
                   legend_label=algoName)
    
    plotLcost.scatter(x=df_lcost_price["period"], y=df_lcost_price['lcost_price'],
                      size=10, color="red", alpha=0.5)
    
    plotLcost.legend.location = "top_left"
    
    return plotLcost

###############################################################################
#                   plot som(LCOST/Price) for LRI : FIN
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
    
    df_QTStock = df_pros_algos[['period', 'algoName','QTStock', 'scenarioName']]\
                    .groupby(["algoName","period"]).sum().reset_index()
    
    # set up the figure
    plotQTstock = figure(
        title=" show QTStock KPI for all algorithms ",
        height=300,
        sizing_mode="stretch_width",  # use the full width of the parent element
        tooltips=TOOLTIPS_LCOST,
        output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",  # enable box zoom by default
    )
    
    for algoName in df_pros_algos.algoName.unique().tolist():
        df_qtstock_algo = df_QTStock[df_QTStock.algoName == algoName]
        plotQTstock.line(x=df_qtstock_algo["period"], y=df_qtstock_algo["QTStock"], 
                         line_width=2, color=COLORS[algoName], alpha=0.8, 
                         legend_label=algoName)
        plotQTstock.scatter(x=df_qtstock_algo["period"], y=df_qtstock_algo["QTStock"],
                            size=2, color="red", alpha=0.5)
    
    plotQTstock.legend.location = "top_left"
    plotQTstock.legend.click_policy = "hide"
    
    return plotQTstock
    
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
    
    df_Sis = df_pros_algos[['period', 'algoName','storage', 'scenarioName']]\
                .groupby(["algoName","period"]).sum().reset_index()
    
    # set up the figure
    plotSis = figure(
        title=" show Storage KPI for all algorithms ",
        height=300,
        sizing_mode="stretch_width",  # use the full width of the parent element
        tooltips=TOOLTIPS_LCOST,
        output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",  # enable box zoom by default
    )
    
    for algoName in df_pros_algos.algoName.unique().tolist():
        df_Sis_algo = df_Sis[df_Sis.algoName == algoName]
        plotSis.line(x=df_Sis_algo["period"], y=df_Sis_algo["storage"], 
                         line_width=2, color=COLORS[algoName], alpha=0.8, 
                         legend_label=algoName)
        plotSis.scatter(x=df_Sis_algo["period"], y=df_Sis_algo["storage"],
                            size=2, color="red", alpha=0.5)
        
    plotSis.legend.location = "top_left"
    plotSis.legend.click_policy = "hide"
    
    return plotSis
    
###############################################################################
#                   plot sum storage all LRI, SSA, Bestie : FIN
###############################################################################

###############################################################################
#                   plot QTTepo all LRI, SSA, Bestie : Debut
###############################################################################
def plotQTTepo(df_prosumers: pd.DataFrame, scenarioCorePathDataViz: str):
    """
    plot QttEpo

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ALGOS = ["LRI_REPART", "SSA", "Bestie"]
    df_pros_algos = df_prosumers[df_prosumers['algoName'].isin(ALGOS)]
    
    df_Qttepo = df_pros_algos[['period', 'algoName','prodit', 'consit', 'scenarioName']]\
                    .groupby(["algoName","period"]).sum().reset_index()
    df_Qttepo.rename(columns={"prodit":"insg", "consit":"outsg"}, inplace=True)
    
    # df_Qttepo["Qttepo"] = df_Qttepo["outsg"] - df_Qttepo["insg"]
    # df_Qttepo['Qttepo'] = df_Qttepo['Qttepo'].apply(lambda x: x if x>=0 else 0)
    
    df_Qttepo["Qttepo"] = df_Qttepo["insg"] - df_Qttepo["outsg"]
    # df_Qttepo['Qttepo'] = df_Qttepo['Qttepo'].apply(lambda x: x if x>=0 else 0)
    
    # set up the figure
    plotQttepo = figure(
        title=" show QttEpo = sum_{i in N}(prod_i^t - cons_i^t) KPI for all algorithms ",
        height=300,
        sizing_mode="stretch_width",  # use the full width of the parent element
        tooltips=TOOLTIPS_LCOST,
        output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",  # enable box zoom by default
    )
    
    for algoName in df_pros_algos.algoName.unique().tolist():
        df_Qttepo_algo = df_Qttepo[df_Qttepo.algoName == algoName]
        plotQttepo.line(x=df_Qttepo_algo["period"], y=df_Qttepo_algo["Qttepo"], 
                         line_width=2, color=COLORS[algoName], alpha=0.8, 
                         legend_label=algoName)
        plotQttepo.scatter(x=df_Qttepo_algo["period"], y=df_Qttepo_algo["Qttepo"],
                            size=2, color="red", alpha=0.5)
        
    plotQttepo.legend.location = "top_left"
    plotQttepo.legend.click_policy = "hide"
    
    return plotQttepo

###############################################################################
#                   plot QTTepo all LRI, SSA, Bestie : FIN
###############################################################################

###############################################################################
#                   plot QTTepo_t_{plus,minus} all LRI, SSA, Bestie : DEBUT
###############################################################################
def plotQTTepo_t_minus_plus(df_prosumers: pd.DataFrame, scenarioCorePathDataViz: str):
    """
    plot QttEpo_t_{minus, plus}

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ALGOS = ["LRI_REPART", "SSA", "Bestie"]
    df_pros_algos = df_prosumers[df_prosumers['algoName'].isin(ALGOS)]
    
    df_Qttepo = df_pros_algos[['period', 'algoName','prodit', 'consit', 'scenarioName']]\
                    .groupby(["algoName","period"]).sum().reset_index()
    df_Qttepo.rename(columns={"prodit":"insg", "consit":"outsg"}, inplace=True)
    
    df_Qttepo["Qttepo_t_minus"] = df_Qttepo["outsg"] - df_Qttepo["insg"]
    df_Qttepo['Qttepo_t_minus'] = df_Qttepo['Qttepo_t_minus'].apply(lambda x: x if x>=0 else 0)
    
    df_Qttepo["Qttepo_t_plus"] = df_Qttepo["insg"] - df_Qttepo["outsg"]
    df_Qttepo['Qttepo_t_plus'] = df_Qttepo['Qttepo_t_plus'].apply(lambda x: x if x>=0 else 0)
    
    plots_list = []
    for algoName in df_pros_algos.algoName.unique().tolist():
        df_Qttepo_t_algo = df_Qttepo[df_Qttepo.algoName == algoName]
        
        plotQttepo_t_algo = figure(
            title=f"{algoName} show QttEpo_t^[+,-] KPI for all algorithms ",
            height=300,
            sizing_mode="stretch_width",  # use the full width of the parent element
            tooltips=TOOLTIPS_LCOST,
            output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
            tools="pan,box_zoom,reset,save",
            active_drag="box_zoom",  # enable box zoom by default
        )
        
        plotQttepo_t_algo.line(x=df_Qttepo_t_algo["period"], 
                               y=df_Qttepo_t_algo["Qttepo_t_minus"], 
                               line_width=2, color="#00a933", alpha=0.8, 
                               legend_label="Qttepo_t_minus")
        plotQttepo_t_algo.scatter(x=df_Qttepo_t_algo["period"], 
                                  y=df_Qttepo_t_algo["Qttepo_t_minus"],
                                  size=5, color="#00a933", alpha=0.5)
        
        plotQttepo_t_algo.line(x=df_Qttepo_t_algo["period"], 
                               y=df_Qttepo_t_algo["Qttepo_t_plus"], 
                               line_width=2, color="#800080", alpha=0.8, 
                               legend_label="Qttepo_t_plus")
        plotQttepo_t_algo.scatter(x=df_Qttepo_t_algo["period"], 
                                  y=df_Qttepo_t_algo["Qttepo_t_plus"],
                                  size=5, color="#800080", alpha=0.5)
        
        
        
        plotQttepo_t_algo.legend.location = "top_left"
        plotQttepo_t_algo.legend.click_policy = "hide"
        
        plots_list.append(plotQttepo_t_algo)
        
    return plots_list
    
###############################################################################
#                   plot QTTepo_t_{plus,minus} all LRI, SSA, Bestie : FIN
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
    
    x = list(zip(df_alPeMoVal['algoName'],df_alPeMoVal['period'].astype(str), df_alPeMoVal['mode']))
    counts = list(df_alPeMoVal['value'])
    source = ColumnDataSource(data=dict(x=x, counts=counts))
    
    # set up the figure
    plotBarMode = figure(
        x_range=FactorRange(*x),
        title=" Distribution des strategies Modes ",
        height=300,
        sizing_mode="stretch_width",  # use the full width of the parent element
        tooltips=TOOLTIPS_LCOST,
        output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        tools="pan,box_zoom,reset,save",
        active_drag="box_zoom",  # enable box zoom by default
    )
    
    # # Création du graphique avec Bokeh
    plotBarMode.vbar(x='x', top='counts', width=0.9, source=source, 
                      fill_color=factor_cmap('x', palette=Spectral5, 
                                            factors=df_alPeMoVal['mode'].unique().tolist(), 
                                            start=1, end=2)
                      )
    
    
    # save image
    plotBarMode.x_range.range_padding = 0.1
    plotBarMode.xaxis.major_label_orientation = 1
    

    return plotBarMode


def plot_barModesBis(df_prosumers: pd.DataFrame, scenarioCorePathDataViz: str):
    """
    

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # value_counts
    df_res = df_prosumers.groupby('algoName')[['period','mode']].value_counts().unstack(fill_value=0)
    
    #Affichage du résultat stochastique
    df_stoc = df_res.div(df_res.sum(axis=1), axis=0).reset_index()

    
    # rename columns 
    df_stoc = df_stoc.rename(columns={"Mode.CONSMINUS":"CONSMINUS", 
                                      "Mode.CONSPLUS":"CONSPLUS", 
                                      "Mode.DIS":"DIS", 
                                      "Mode.PROD":"PROD"})
    
    modes = ["CONSMINUS", "CONSPLUS", "DIS", "PROD"]
    df_stoc["period"] = df_stoc["period"].astype(str)
    factors = list(zip(df_stoc["algoName"], df_stoc["period"]))
    
    source = ColumnDataSource(data=dict(
                x=factors,
                CONSMINUS=df_stoc["CONSMINUS"].tolist(),
                CONSPLUS=df_stoc["CONSPLUS"].tolist(),
                DIS=df_stoc["DIS"].tolist(),
                PROD=df_stoc["PROD"].tolist()
                ))
    
    plotBarMode = figure(x_range=FactorRange(*factors), height=450,
                         toolbar_location=None, tools="", 
                         tooltips=TOOLTIPS_MODES)

    plotBarMode.vbar_stack(modes, x='x', width=0.9, alpha=0.5, 
                           color=["blue", "red", "yellow", "cyan"], 
                           source=source,
                           legend_label=modes)
    
    plotBarMode.y_range.start = 0
    plotBarMode.y_range.end = 1
    plotBarMode.x_range.range_padding = 0.1
    plotBarMode.xaxis.major_label_orientation = 1
    plotBarMode.xgrid.grid_line_color = None
    plotBarMode.legend.location = "top_center"
    plotBarMode.legend.orientation = "horizontal"
    plotBarMode.title = " Distribution des strategies Modes "
    
    plotBarMode.add_layout(Legend(), 'right')
    
    return plotBarMode
    
    
###############################################################################
#                   visu bar plot of actions(modes) : FIN
###############################################################################

###############################################################################
#                   visu bar plot ValSG and ValNoSG : debut
###############################################################################
from bokeh.palettes import Category10

def plot_performanceAlgo(df_prosumers: pd.DataFrame, scenarioCorePathDataViz: str):
    """
    

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # df_valNoSG = df_prosumers[["period", "algoName", "valNoSG_i"]] \
    #                 .groupby(["algoName","period"]).sum().reset_index()
    # df_valSG = df_prosumers[["period", "algoName", "ValSG"]]\
    #                 .groupby(["algoName","period"]).mean().reset_index()
                    
    # df_valSGNoSG = df_valSG.merge(df_valNoSG, on=["period", "algoName"])
    
    # df_Perf = df_valSGNoSG[['algoName', 'period', 'ValSG', 'valNoSG_i']]\
    #                 .groupby(['algoName']).sum().reset_index()
                    
    # df_Perf = df_Perf.drop('period', axis=1)
    
    # df_Perf.rename(columns={"valNoSG_i":"valNoSG"}, inplace=True)
    
    
    # # Transformation des données pour affichage côte à côte en groupant par algorithme
    # categories = df_Perf['algoName'].tolist()
    # metrics = ['ValSG', 'valNoSG']
    # x_labels = [(cat, metric) for cat in categories for metric in metrics]
    # y_values = [df_Perf.loc[df_Perf['algoName'] == cat, metric].values[0] for cat, metric in x_labels]
    
    # # Création du ColumnDataSource avec des valeurs associées aux barres
    # source = ColumnDataSource(data=dict(
    #             x=[f"{cat}_{metric}" for cat, metric in x_labels],
    #             y=y_values,
    #             algo=[cat for cat, metric in x_labels],
    #             metric=[metric for cat, metric in x_labels]
    # ))
    
    # # Création de la figure avec un regroupement par algorithme
    # plot_Perf = figure(x_range=FactorRange(*[f"{cat}_{metric}" for cat, metric in x_labels]), 
    #             title="Comparaison ValSG et valNoSG",
    #             toolbar_location=None, tools="")
    
    # # Ajout des barres côte à côte avec la bonne référence aux colonnes du source
    # types = ['ValSG', 'valNoSG']
    # colors = ["blue", "red"]
    # plot_Perf.vbar(x='x', top='y', width=0.4, source=source, fill_color=factor_cmap('x', palette=colors, factors=metrics, start=1))

    # # for metric, color in zip(types, colors):
    # #     plot_Perf.vbar(x='x', top='y', width=0.4, source=source, legend_label=metric, color=color)
    
    # # Ajout du HoverTool pour afficher les valeurs
    # hover = HoverTool()
    # hover.tooltips = [
    #     ("Algorithme", "@algo"),
    #     ("Type", "@metric"),
    #     ("Valeur", "@y")
    # ]
    # plot_Perf.add_tools(hover)
    
    # # Personnalisation du graphique
    # plot_Perf.xgrid.grid_line_color = None
    # plot_Perf.y_range.start = 0
    # plot_Perf.xaxis.major_label_orientation = 1.2
    # plot_Perf.xaxis.axis_label = "AlgoName et Type"
    # plot_Perf.yaxis.axis_label = "Valeurs"
    # plot_Perf.title.align = "center"
    # #plot_Perf.legend.click_policy = "hide"
    
    ###########################################################################
    #           new version of various colors
    ###########################################################################
    df_valNoSG = df_prosumers[["period", "algoName", "valNoSG_i"]] \
                    .groupby(["algoName","period"]).sum().reset_index()
    df_valSG = df_prosumers[["period", "algoName", "ValSG"]]\
                    .groupby(["algoName","period"]).mean().reset_index()
                    
    df_valSGNoSG = df_valSG.merge(df_valNoSG, on=["period", "algoName"])
    
    df_Perf = df_valSGNoSG[['algoName', 'period', 'ValSG', 'valNoSG_i']]\
                    .groupby(['algoName']).sum().reset_index()
                    
    df_Perf = df_Perf.drop('period', axis=1)
    
    df_Perf.rename(columns={"valNoSG_i":"valNoSG"}, inplace=True)
    
    # Transformation des données pour affichage côte à côte en groupant par algorithme
    categories = df_Perf['algoName'].tolist()
    metrics = ['ValSG', 'valNoSG']
    x_labels = [(cat, metric) for cat in categories for metric in metrics]
    y_values = [df_Perf.loc[df_Perf['algoName'] == cat, metric].values[0] for cat, metric in x_labels]
    
    # Création du ColumnDataSource avec des valeurs associées aux barres
    source = ColumnDataSource(data=dict(
                    x=[f"{cat}_{metric}" for cat, metric in x_labels],
                    y=y_values,
                    algo=[cat for cat, metric in x_labels],
                    metric=[metric for cat, metric in x_labels]
        ))
    
    # Palette de couleurs pour chaque algorithme
    palette = Category10[len(categories)]
    colors = {cat: color for cat, color in zip(categories, palette)}
    
    # Création de la figure avec un regroupement par algorithme
    plot_Perf = figure(x_range=FactorRange(*[f"{cat}_{metric}" for cat, metric in x_labels]), 
                    title="Comparaison ValSG et valNoSG",
                    toolbar_location=None, tools="")
    
    # Ajout des barres côte à côte avec la bonne référence aux colonnes du source
    types = ['ValSG', 'valNoSG']
    plot_Perf.vbar(x='x', top='y', width=0.4, source=source, 
                   fill_color=factor_cmap('algo', palette=list(colors.values()), factors=categories, start=1))
    
    # Ajout du HoverTool pour afficher les valeurs
    hover = HoverTool()
    hover.tooltips = [
        ("Algorithme", "@algo"),
        ("Type", "@metric"),
        ("Valeur", "@y")
    ]
    plot_Perf.add_tools(hover)
    
    # Personnalisation du graphique
    plot_Perf.xgrid.grid_line_color = None
    plot_Perf.y_range.start = 0
    plot_Perf.xaxis.major_label_orientation = 1.2
    plot_Perf.xaxis.axis_label = "AlgoName et Type"
    plot_Perf.yaxis.axis_label = "Valeurs"
    plot_Perf.title.align = "center"

    ###########################################################################
    #           new version of various colors
    ###########################################################################
    
    return plot_Perf
    
###############################################################################
#                   visu bar plot ValSG and ValNoSG : FIN
###############################################################################

###############################################################################
#                visu bar plot ValSG and ValNoSG with meanLRI: debut
###############################################################################
def plot_performanceAlgo_meanLRI(scenarioCorePathDataViz: str):
    """
    

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    df_exec = pd.read_csv( os.path.join(scenarioCorePathDataViz, "df_exec.csv") )
    df_Perf = df_exec[["algoName","ValSG","ValNoSG"]].groupby("algoName").mean().reset_index()
    
    
    # Définition des couleurs spécifiques pour chaque algo
    colors = {
        "Bestie": "yellow",
        "CSA": "green",
        "LRI": "blue",
        "SSA": "red",
        "SyA": "purple"
    }
    
    data = { 
        'algoName': list(df_Perf['algoName']),
        'ValSG': list(df_Perf['ValSG']),
        'ValNoSG': list(df_Perf['ValNoSG'])
        }
    data['ValSG_Rounded'] = [round(val, 0) for val in data['ValSG']]
    data['ValNoSG_Rounded'] = [round(val, 0) for val in data['ValNoSG']]
    
    source = ColumnDataSource(data=data)

    plot_Perf_MeanLri = figure(x_range=data["algoName"], title="Performance Measures",
                height=450, toolbar_location=None, tools="hover",
                tooltips="$name @algoName: @$name")

    plot_Perf_MeanLri.vbar(x=dodge('algoName', -0.25, 
                                    range=plot_Perf_MeanLri.x_range), 
                            top='ValSG', source=source,
                            width=0.2, color="green", legend_label="ValSG")

    plot_Perf_MeanLri.vbar(x=dodge('algoName',  0.0,  
                                    range=plot_Perf_MeanLri.x_range), 
                            top='ValNoSG', source=source,
                            width=0.2, color="blue", legend_label="ValNoSG")


    plot_Perf_MeanLri.x_range.range_padding = 0.1
    plot_Perf_MeanLri.xgrid.grid_line_color = None
    plot_Perf_MeanLri.legend.location = "top_left"
    plot_Perf_MeanLri.legend.orientation = "horizontal"
    plot_Perf_MeanLri.legend.click_policy="mute"

    hover = HoverTool()
    hover.tooltips = [("Algorithm", "@algoName"), 
                      # ("ValSG", "@ValSG"), ("ValNoSG", "@ValNoSG"), 
                      ('ValSG_round', "@ValSG_Rounded"), ("ValNoSG_round", "@ValNoSG_Rounded")
                      ]
    plot_Perf_MeanLri.add_tools(hover)
    plot_Perf_MeanLri.add_layout(plot_Perf_MeanLri.legend[0], 'right')
    
    
    # ###########################################################################
    # #           new version of various colors
    # ###########################################################################
    
    # df_exec = pd.read_csv( os.path.join(scenario["scenarioCorePathDataViz"], "df_exec.csv") )
    
    # df_exec = pd.read_csv( os.path.join(scenarioCorePathDataViz, "df_exec.csv") )
    # df_Perf = df_exec[["algoName","ValSG","ValNoSG"]].groupby("algoName").mean().reset_index()
    
    # colors = {
    # "Bestie": "yellow",
    # "CSA": "green",
    # "LRI": "blue",
    # "SSA": "red",
    # "SyA": "purple"
    # }
    
    # # Transformation des données pour affichage côte à côte
    # x_labels = [(cat, metric) for cat in df_Perf['algoName'].tolist() for metric in ['ValSG', 'ValNoSG']]
    # y_values = [df_Perf.loc[df_Perf['algoName'] == cat, metric].values[0] for cat, metric in x_labels]
    
    # source = ColumnDataSource(data=dict(
    #     x=[f"{cat}_{metric}" for cat, metric in x_labels],
    #     y=y_values,
    #     algo=[cat for cat, metric in x_labels],
    #     metric=[metric for cat, metric in x_labels]
    # ))
    
    # # Création de la figure avec un regroupement par algorithme
    # plot_Perf_MeanLri = figure(x_range=[f"{cat}_{metric}" for cat, metric in x_labels], 
    #                   title="Comparaison ValSG et ValNoSG",
    #                   height=450, toolbar_location=None, tools="hover")
    
    # # Ajout des barres côte à côte avec des couleurs par algorithme
    # plot_Perf_MeanLri.vbar(x='x', top='y', width=0.4, source=source, 
    #                fill_color=factor_cmap('algo', palette=list(colors.values()), factors=df_Perf['algoName'].tolist()))
    
    # # Personnalisation du graphique
    # plot_Perf_MeanLri.xgrid.grid_line_color = None
    # plot_Perf_MeanLri.xaxis.major_label_orientation = 1.2
    
    # # Ajout du HoverTool pour afficher les valeurs
    # hover = HoverTool()
    # hover.tooltips = [("Algorithm", "@algo"), ("Type", "@metric"), ("Value", "@y")]
    # plot_Perf_MeanLri.add_tools(hover)
        
    # ###########################################################################
    # #           new version of various colors
    # ###########################################################################

    return plot_Perf_MeanLri
    
###############################################################################
#               visu bar plot ValSG and ValNoSG with meanLRI : FIN
###############################################################################

###############################################################################
#                visu bar plot ValSG and ValNoSG with meanLRI: debut
###############################################################################
def plot_X_Y_ai_OLD(scenarioCorePathDataViz: str):
    """
    

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df_X_Y_ai = pd.read_csv( os.path.join(scenario["scenarioCorePathDataViz"], "df_X_Y_ai.csv"), index_col=0)
    data_X_Y_ai = df_X_Y_ai.to_dict()
    
    # Définition des couleurs spécifiques pour chaque algo
    colors = {
        "Bestie": "yellow",
        "CSA": "green",
        "LRI": "blue",
        "SSA": "red",
        "SyA": "purple"
    }
    
    ps = []
    algoNames = ['CSA','SSA','LRI', 'SyA', 'Bestie']
    for algoName in algoNames:
        data_algo = dict()
        for k, v in data_X_Y_ai.items():
            if algoName in k:
               data_algo[k] = v
               
        #data_algo["prosumers"]=[f"prosumer_{i}" for i in range(df_X_Y_ai.shape[0])]
        dico_prosumers = dict()
        for i in range(df_X_Y_ai.shape[0]):
            dico_prosumers[i] = f"prosumer_{i}"
        data_algo["prosumers"] = dico_prosumers
        df = pd.DataFrame(data_algo)
        df_sort = df.sort_values(by=f'{algoName}_X_ai', ascending=True)
        
        # Créer un index numérique pour les prosumers
        df_sort['prosumer_index'] = range(len(df_sort))
        
        # Création du graphique
        p = figure(title=f"Nuage de points {algoName}_X_ai et {algoName}_Y_ai par prosumer",
                   x_axis_label='Prosumer Index', y_axis_label=f'Valeurs {algoName}', 
                   x_range=df_sort['prosumers'], tooltips=TOOLTIPS_XY_ai)
        
        # Rotation des étiquettes de l'axe des x
        p.xaxis.major_label_orientation = 3.14159 / 4  # Rotation à 90 degrés (π/2 radians)
        
        # Ajouter les points pour SyA_X_ai
        name_col = f'{algoName}_X_ai'
        p.circle(x=df_sort['prosumers'], y=df_sort[name_col], 
                 size=10, color="blue", legend_label=name_col)
        
        # Ajouter les points pour SyA_Y_ai
        name_col = f'{algoName}_Y_ai'
        p.circle(x=df_sort['prosumers'], y=df_sort[name_col], 
                 size=10, color="red", legend_label=name_col)
        
        p.legend.click_policy="mute"
        # hover = HoverTool()
        # list_hover = []
        # for k in data_algo.keys():
        #     col = k
        #     list_hover.append( (k, '@k'))
        # hover.tooltips = list_hover
        # hover.tooltips = [("Algorithm", "@algoName"), ("ValSG", "@ValSG"), 
        #                   ("ValNoSG", "@ValNoSG")]
        #p.add_tools(hover)
        
        ps.append([p])
    
    
    
    
    return ps
    

from bokeh.models import (CustomJS, LinearAxis, Range1d, Select,
                          WheelZoomTool, ZoomInTool, ZoomOutTool)
def plot_X_Y_ai(scenarioCorePathDataViz: str):
    """
    

    Parameters
    ----------
    df_prosumers : pd.DataFrame
        DESCRIPTION.
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # df_X_Y_ai = pd.read_csv( os.path.join(scenario["scenarioCorePathDataViz"], "df_X_Y_ai.csv"), index_col=0)
    df_X_Y_ai = pd.read_csv( os.path.join(scenarioCorePathDataViz, "df_X_Y_ai.csv"), index_col=0)
    
    data_X_Y_ai = df_X_Y_ai.to_dict()
    
    # Définition des couleurs spécifiques pour chaque algo
    colors = {
        "Bestie": "yellow",
        "CSA": "green",
        "LRI": "blue",
        "SSA": "red",
        "SyA": "purple"
    }
    
    ps = []
    algoNames = ['CSA','SSA','LRI', 'SyA', 'Bestie']
    for algoName in algoNames:
        data_algo = dict()
        for k, v in data_X_Y_ai.items():
            if algoName in k:
               data_algo[k] = v
               
        #data_algo["prosumers"]=[f"prosumer_{i}" for i in range(df_X_Y_ai.shape[0])]
        dico_prosumers = dict()
        for i in range(df_X_Y_ai.shape[0]):
            dico_prosumers[i] = f"prosumer_{i}"
        data_algo["prosumers"] = dico_prosumers
        df = pd.DataFrame(data_algo)
        df_sort = df.sort_values(by=f'{algoName}_X_ai', ascending=True)
        
        # Créer un index numérique pour les prosumers
        df_sort['prosumer_index'] = range(len(df_sort))
        
        # Création du graphique
        p = figure(title=f"Nuage de points {algoName}_X_ai et {algoName}_Y_ai par prosumer",
                   x_axis_label='Prosumer Index', y_axis_label=f'Valeurs Xi {algoName}', 
                   x_range=df_sort['prosumers'],  y_range=(0, df_sort[f'{algoName}_X_ai'].max()),
                   tooltips=TOOLTIPS_XY_ai, tools="pan,box_zoom,save,reset")
        
        # Rotation des étiquettes de l'axe des x
        p.xaxis.major_label_orientation = 3.14159 / 4  # Rotation à 90 degrés (π/2 radians)
        
        # Ajouter les points pour SyA_X_ai
        name_col = f'{algoName}_X_ai'
        blue_X_ai = p.circle(x=df_sort['prosumers'], y=df_sort[name_col], 
                             size=10, color="blue", legend_label=name_col)
        p.axis.axis_label_text_color = 'blue'
        
        # Ajouter les points pour SyA_Y_ai
        name_col = f'{algoName}_Y_ai'
        p.extra_y_ranges['foo'] = Range1d(df_sort[name_col].min()-3, df_sort[name_col].max())
        blue_Y_ai = p.circle(x=df_sort['prosumers'], y=df_sort[name_col], 
                             size=10, color="red", legend_label=name_col, 
                             y_range_name="foo")
        
        # second xy-axis
        ax2 = LinearAxis(
            axis_label=f'Valeurs Y_ai {algoName}',
            y_range_name="foo",
        )
        ax2.axis_label_text_color = 'red'
        p.add_layout(ax2, 'left')
        # ax3 = LinearAxis(
        #     axis_label="red circles",
        #     y_range_name="foo",
        # )
        # ax3.axis_label_text_color = 'red'
        # p.add_layout(ax3, 'below')
        
        
        wheel_zoom = WheelZoomTool()
        p.add_tools(wheel_zoom)
        
        p.legend.click_policy="mute"
        # hover = HoverTool()
        # list_hover = []
        # for k in data_algo.keys():
        #     col = k
        #     list_hover.append( (k, '@k'))
        # hover.tooltips = list_hover
        # hover.tooltips = [("Algorithm", "@algoName"), ("ValSG", "@ValSG"), 
        #                   ("ValNoSG", "@ValNoSG")]
        #p.add_tools(hover)
        
        zoom_in_blue = ZoomInTool(renderers=[blue_X_ai], description="Zoom in blue circles")
        zoom_out_blue = ZoomOutTool(renderers=[blue_X_ai], description="Zoom out blue circles")
        p.add_tools(zoom_in_blue, zoom_out_blue)
        
        zoom_in_red = ZoomInTool(renderers=[blue_X_ai], description="Zoom in red circles")
        zoom_out_red = ZoomOutTool(renderers=[blue_Y_ai], description="Zoom out red circles")
        p.add_tools(zoom_in_red, zoom_out_red)
        
        ps.append([p])
    
    
    
    
    return ps
    


###############################################################################
#               visu bar plot ValSG and ValNoSG with meanLRI : FIN
###############################################################################

###############################################################################
#               visu bar plot array nash equilibrium : DEBUT
###############################################################################
import numpy as np
def plot_nashEquilibrium_byPeriod(scenarioCorePathDataViz, M_execution_LRI):
    """
    

    Parameters
    ----------
    scenarioCorePathDataViz : TYPE
        DESCRIPTION.

    Returns
    -------
    plot_NE_brute : TYPE
        DESCRIPTION.

    """
    # df_X_Y_ai = pd.read_csv( os.path.join(scenario["scenarioCorePathDataViz"], "df_X_Y_ai.csv"), index_col=0)
    df_X_Y_ai = pd.read_csv( os.path.join(scenarioCorePathDataViz, "df_X_Y_ai.csv"), index_col=0)
    N = df_X_Y_ai.shape[0]
    
    # arr_NE_brute = np.load(os.path.join(scenario["scenarioCorePathDataViz"], "arr_NE_brute.npy"))
    arr_NE_brute = np.load(os.path.join(scenarioCorePathDataViz, "arr_NE_brute.npy"))

    
    prosumers = [f'prosumer{i}' for i in range(N)]
    prosumers.insert(0, "period")
    arr_NE_brute_N = np.array_split(arr_NE_brute, N, axis=1)
    
    counts_N = list()
    for arr_NE_brute_n in arr_NE_brute_N:
        count_1_per_period = np.sum(arr_NE_brute_n == 1, axis=1)
        count_1_per_period = count_1_per_period / M_execution_LRI
        counts_N.append(count_1_per_period)
        
    arr_N8 = np.vstack(counts_N)
    
    arr_T = np.sum(arr_N8, axis=0) / N
    
    df_T = pd.DataFrame(arr_T).reset_index()
    df_T.columns = ["periods", "Percent_NE_pure"]
    
    source = ColumnDataSource(df_T)
    
    
    plot_NE_brute = figure(title=f"LRI: Percent of Pure Nash Equilibrium per period",
                           x_axis_label='periods', y_axis_label=f'Percent', 
                           tooltips=TOOLTIPS_XY_ai)
    
    plot_NE_brute.vbar(x="periods", top="Percent_NE_pure", source=source, width=0.70)
    
    
    return plot_NE_brute
###############################################################################
#               visu bar plot array nash equilibrium : FIN
###############################################################################

###############################################################################
#               visu bar plot disribution proba min : DEBUT
###############################################################################
def plot_min_proba_distribution(df_prosumers: pd.DataFrame, scenarioCorePathDataViz:str):
    """
    for each period, get the min proba like that :
        prosumer_1 : 0.7 0.3  => 0.7
        prosumer_2 : 0.1 0.9  => 0.9     ---> z_i = 0.6
        ........
        prosumer_n : 0.6 0.4  => 0.6

    Parameters
    ----------
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    algoName = "LRI_REPART"
    cols_2_select = ["prosumers","period", "prmode0", "prmode1"]
    df_lri = df_prosumers[df_prosumers.algoName == algoName][cols_2_select]
    
    
    df_lri['pr_max'] = df_lri[['prmode0', 'prmode1']].max(axis=1)
    df_t_z = df_lri[["period",'pr_max']].groupby('period')['pr_max'].min().reset_index()
    
    df_t_z['pr_max'] = df_t_z['pr_max'].round(2)
    df_t_z['pr_max_deci'] = np.floor(df_t_z['pr_max'] * 10) / 10
    
    bins = np.linspace(0,1, num=21)
    
    # Générer des bins avec np.linspace
    bins = np.linspace(0, 1, num=21)
    
    # Calculer l'histogramme
    hist, edges = np.histogram(df_t_z['pr_max_deci'], bins=bins)
    
    # Créer un plot Bokeh
    p_distr = figure(width=800, height=400, title="Distribution de pr_max_deci", 
               x_axis_label='Valeurs', y_axis_label='Fréquence')
    
    # Créer un ColumnDataSource
    source = ColumnDataSource(data=dict(
        left=edges[:-1],
        right=edges[1:],
        top=hist,
    ))
    
    # Ajouter des rectangles pour représenter l'histogramme
    p_distr.quad(top='top', bottom=0, left='left', right='right', 
                 fill_color="skyblue", line_color="white", source=source)
    
    # Personnaliser les étiquettes de l'axe des x
    ticks = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
    tick_labels = [f"{edges[i]:.2f} - {edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    
    p_distr.xaxis.ticker = ticks
    p_distr.xaxis.major_label_overrides = {tick: label for tick, label in zip(ticks, tick_labels)}
    
    # Ajouter un outil de survol (HoverTool)
    hover = HoverTool(tooltips=[
        ("Valeur", "@left{0.2f} - @right{0.2f}"),
        ("Fréquence", "@top"),
    ])
    
    # Ajouter l'outil de survol au plot
    p_distr.add_tools(hover)
    
    return p_distr

###############################################################################
#               visu bar plot disribution proba min : FIN
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
    
    plotValSG, plotValNoSG = plot_curve_valSGNoSG(df_prosumers, scenarioCorePathDataViz)
    
    plotLcost = plot_LcostPrice(df_prosumers, scenarioCorePathDataViz)
    
    plotQTstock = plot_sumQTStock(df_prosumers, scenarioCorePathDataViz)
    
    plotSis = plot_sumStorage(df_prosumers, scenarioCorePathDataViz)
    
    #plotBarMode = plot_barModes(df_prosumers, scenarioCorePathDataViz)
    
    plotBarModeBis = plot_barModesBis(df_prosumers, scenarioCorePathDataViz)
    
    plotQttepo = plotQTTepo(df_prosumers, scenarioCorePathDataViz)
    
    plot_Perf = plot_performanceAlgo(df_prosumers, scenarioCorePathDataViz)
    
    plots_list = plotQTTepo_t_minus_plus(df_prosumers, scenarioCorePathDataViz)
    
    # create a layout
    lyt = layout(
        [
            [plotValSG], 
            [plotValNoSG], 
            [plotLcost],
            [plotQTstock],
            [plotSis], 
            # [plotBarMode], 
            [plotBarModeBis], 
            [plotQttepo], 
            [plot_Perf], 
            plots_list
            # [p1, p2],  # the first row contains two plots, spaced evenly across the width of notebook
            # [p3],  # the second row contains only one plot, spanning the width of notebook
        ],
        sizing_mode="stretch_width",  # the layout itself stretches to the width of notebook
    )
    
    # set output to static HTML file
    filename = os.path.join(scenarioCorePathDataViz, "plotCourbes.html")
    output_file(filename=filename, title="Static HTML file")
    
    save(lyt)
    
###############################################################################
#                   visu all plots : FIN
###############################################################################


###############################################################################
#                   visu all plots with mean LRI : debut
###############################################################################
def plot_all_figures_withMeanLRI(df_prosumers: pd.DataFrame, 
                                 scenarioCorePathDataViz: str, 
                                 M_execution_LRI: int, rho: int): 
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
    
    plotValSG, plotValNoSG = plot_curve_valSGNoSG(df_prosumers, scenarioCorePathDataViz)
    
    plotLcost = plot_LcostPrice(df_prosumers, scenarioCorePathDataViz)
    
    plotQTstock = plot_sumQTStock(df_prosumers, scenarioCorePathDataViz)
    
    plotSis = plot_sumStorage(df_prosumers, scenarioCorePathDataViz)
    
    #plotBarMode = plot_barModes(df_prosumers, scenarioCorePathDataViz)
    
    plotBarModeBis = plot_barModesBis(df_prosumers, scenarioCorePathDataViz)
    
    plotQttepo = plotQTTepo(df_prosumers, scenarioCorePathDataViz)
    
    plot_Perf = plot_performanceAlgo(df_prosumers, scenarioCorePathDataViz)
    
    plots_list = plotQTTepo_t_minus_plus(df_prosumers, scenarioCorePathDataViz)
    
    plot_Perf_MeanLri = plot_performanceAlgo_meanLRI(scenarioCorePathDataViz)
    
    ps_X_Y_ai = plot_X_Y_ai(scenarioCorePathDataViz)
    
    plot_NE_brute = plot_nashEquilibrium_byPeriod(scenarioCorePathDataViz, M_execution_LRI)
    
    p_distr = plot_min_proba_distribution(df_prosumers, scenarioCorePathDataViz)
    
    # create a layout
    lyt = layout(
        [
            [plotValSG], 
            [plotValNoSG], 
            [plotLcost],
            [plotQTstock],
            [plotSis], 
            # [plotBarMode], 
            [plotBarModeBis], 
            [plotQttepo], 
            # [plot_Perf], 
            [plot_Perf_MeanLri],
            plots_list, 
            ps_X_Y_ai, 
            plot_NE_brute, 
            p_distr
            # [p1, p2],  # the first row contains two plots, spaced evenly across the width of notebook
            # [p3],  # the second row contains only one plot, spanning the width of notebook
        ],
        sizing_mode="stretch_width",  # the layout itself stretches to the width of notebook
    )
    
    # set output to static HTML file
    filename = os.path.join(scenarioCorePathDataViz, f"plotCourbes_rho{rho}.html")
    output_file(filename=filename, title="Static HTML file")
    
    save(lyt)
    
###############################################################################
#                   visu all plots with mean LRI : FIN
###############################################################################

if __name__ == '__main__':
    
    scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho5.json"
    scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho05_Smax18_bestieT6.json"
    scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho05_Smax18_bestieT7.json"
    scenarioFile = "./data_scenario_JeuDominique/data_debug_GivenStrategies_rho05_Smax24_bestieT6.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_rho5_b01.json"

    
    
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas", "dataViz")
    scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["scenarioName"], "datas")
    
    scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
    scenario["scenarioCorePathData"] = scenarioCorePathData
    M_execution_LRI = scenario["simul"]["M_execution_LRI"]
    
    
    apps_pkls = load_all_algos_apps(scenario)
    df_prosumers = create_df_SG(apps_pkls=apps_pkls, index_GA_PA=0)
    
    # plot_all_figures(df_prosumers=df_prosumers, 
    #                  scenarioCorePathDataViz=scenarioCorePathDataViz)
    
    plot_all_figures_withMeanLRI(df_prosumers=df_prosumers, 
                                 scenarioCorePathDataViz=scenarioCorePathDataViz, 
                                 M_execution_LRI=M_execution_LRI, 
                                 rho=scenario['simul']['rho'])