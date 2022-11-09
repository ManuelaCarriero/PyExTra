# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:59:04 2022

@author: asus
"""

import pandas as pd

import numpy as np

from statsmodels.tsa import stattools
import scipy.interpolate as interp
from scipy.stats.mstats import mquantiles

import matplotlib.pylab as plt

import seaborn as sns

N=16
#warmup_time = 20
dt = 100


def CreateDataframesList(simulation):
    """This function calculates the RNA and protein autocorrelation of N
    simulations and returns their values as lists."""

    dataframes_list = []
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('{}results_seed{}.csv'.format(simulation, n), sep=" ")
        #removedwarmup_results = simulation_results[simulation_results['Time'] > warmup_time]
        dataframes_list.append(simulation_results)
    
    return dataframes_list

dataframes_list_ar = CreateDataframesList(simulation='gillespie_autorepressor_')

dataframes_list_ts = CreateDataframesList(simulation='gillespie_toggleswitch_')




def acf_list(dataframes_list, molecule):
    """
    This function divides time signal of multiple simulations into parts
    given by dt and calculates autocorrelation for each of these

    Parameters
    ----------
    dataframes_list : list of dataframes
        list of dataframes where each dataframes contains simulations
        results for each random seed of the multiple simulations.
    molecule : str
        molecules of which you want to calculate the acf. The string to
        put in is not arbitrary since you have to choose the name 
        of the dataframe column title refered to molecule measurements.  

    Returns
    -------
    autocorrs_RNAs : list
        list of acfs calculated for each time interval of all simulations.

    """
    autocorrs_RNAs = []
    
    for df in dataframes_list:
        
        xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
        time = df['Time'] 
        RNAs = df[molecule]
        f_RNAs = interp.interp1d(time, RNAs, kind='previous')
        yinterp_RNAs = f_RNAs(xvals)
        nlags = len(yinterp_RNAs)
        
        #Tipo
        #yinterp_RNAs = f_RNAs(xvals[0])
        #yinterp_RNAs = f_RNAs(xvals[0])
        
        autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) 

        autocorrs_RNAs.append(autocorr_RNAs)
    
    return autocorrs_RNAs
    

autocorrs_RNAs_ar = acf_list(dataframes_list = dataframes_list_ar, molecule = 'Number of RNA molecules' )
autocorrs_RNAs_ts = acf_list(dataframes_list = dataframes_list_ts, molecule = 'Number of RNAs gene1' )

autocorrs_RNAs_ar

def compute_quantiles(autocorrs):
    

    quantiles_list = []
    
    autocorrs_RNAs_list_of_lists = []
    
    autocorrs_RNAs_list = []
    
    #for i in np.arange(0,len(autocorrs),1):
    
    for j in np.arange(0,len(autocorrs[0]),1):
        
        for i in np.arange(0,len(autocorrs),1):
            
            autocorrs_RNAs_list.append(autocorrs[i][j])
            
        a = autocorrs_RNAs_list.copy()
        
        autocorrs_RNAs_list_of_lists.append(a)
        
        del autocorrs_RNAs_list[:]
        
        
            
            
    autocorrs_RNAs_list_of_lists    
    
    for autocorrs in autocorrs_RNAs_list_of_lists:
        
        quantile = mquantiles(np.array(autocorrs), prob=[0.25,0.50,0.75])
        
        quantiles_list.append(quantile)
    
    return quantiles_list     



quantiles_list_ar = compute_quantiles(autocorrs = autocorrs_RNAs_ar)
quantiles_list_ts = compute_quantiles(autocorrs = autocorrs_RNAs_ts)




def Plot_Quantiles(title, quantiles_list):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    
    n_series = np.arange(0,len(quantiles_list)).tolist()
    
    for quantile, n in zip(quantiles_list, n_series):
    
        lower_quartile = quantile[0]
        median = quantile[1]
        upper_quartile = quantile[2]
        
        x = np.array([n])
        y = np.array([median])
        
        plt.axhline(y=0, linewidth=1, color='gray')
        
        plt.vlines(x, lower_quartile, upper_quartile, color = 'black')
        
        plt.plot(x, y, '.', color = 'black')
        
        plt.ylabel('Autocorrelation', size = 14)
        
        plt.xlabel('$t_{tot}/dt$', size = 14)
        
        
        ax.xaxis.set_ticks(n_series)
        ax.set_xticklabels(n_series)
        
        plt.title(title, size = 15)
        
        sns.despine(fig, bottom=False, left=False)
        
    plt.show()

Plot_Quantiles(title = 'Toggle Switch (RNAs gene1)', quantiles_list = quantiles_list_ts)

Plot_Quantiles(title = 'Autorepressor', quantiles_list = quantiles_list_ar)
