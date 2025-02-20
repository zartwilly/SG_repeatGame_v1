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
    
    df_Qttepo["Qttepo"] = df_Qttepo["outsg"] - df_Qttepo["insg"]
    
    df_Qttepo['Qttepo'] = df_Qttepo['Qttepo'].apply(lambda x: x if x>=0 else 0)
    
    # set up the figure
    plotQttepo = figure(
        title=" show QttEpo KPI for all algorithms ",
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
    
    # Création de la figure avec un regroupement par algorithme
    plot_Perf = figure(x_range=FactorRange(*[f"{cat}_{metric}" for cat, metric in x_labels]), 
                title="Comparaison ValSG et valNoSG",
                toolbar_location=None, tools="")
    
    # Ajout des barres côte à côte avec la bonne référence aux colonnes du source
    types = ['ValSG', 'valNoSG']
    colors = ["blue", "red"]
    plot_Perf.vbar(x='x', top='y', width=0.4, source=source, fill_color=factor_cmap('x', palette=colors, factors=metrics, start=1))

    # for metric, color in zip(types, colors):
    #     plot_Perf.vbar(x='x', top='y', width=0.4, source=source, legend_label=metric, color=color)
    
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
    #plot_Perf.legend.click_policy = "hide"
    
    
    
    return plot_Perf
    
###############################################################################
#                   visu bar plot ValSG and ValNoSG : FIN
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
            [plot_Perf]
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
    
    plot_all_figures(df_prosumers=df_prosumers, 
                     scenarioCorePathDataViz=scenarioCorePathDataViz)