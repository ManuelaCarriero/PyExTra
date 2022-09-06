# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:45:47 2022

@author: asus
"""

import argparse
import configparser
import ast 
#import sys
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from statsmodels.tsa import stattools
import scipy.interpolate as interp



config = configparser.ConfigParser()

parser = argparse.ArgumentParser()

parser.add_argument("filename", help="read configuration file.")

parser.add_argument('-plot_acf', help='plot acf of RNAs and Proteins in separated graphs', action = "store_true")
parser.add_argument('-plot_acfs', help='plot acf of RNAs and Proteins in the same graph', action = "store_true")
parser.add_argument('-multiplesimulationsacfs_tauleap', help='plot acf of RNAs and Proteins in the same graph of multiple Tau-leap simulations', action = "store_true")
parser.add_argument('-multiplesimulationsacfs_gillespie', help='plot acf of RNAs and Proteins in the same graph of multiple Gillespie simulations', action = "store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

args = parser.parse_args()



config.read(args.filename)



def read_population():
    """This function reads population parameters from configuration file
    """
    
    state = config.get('POPULATION', 'state')
    
    def apply_pipe(func_list, obj):
        for function in func_list:
            obj = function(obj)
        return obj 
    
    starting_state = apply_pipe([ast.literal_eval, np.array], state)
    
    index = dict(config["INDEX"])
    
    for key,value in index.items():
        index[key] = int(value)
    
    active_genes = index['active_genes']
    inactive_genes = index['inactive_genes']
    RNAs = index['rnas']
    proteins = index['proteins']
    
    return starting_state, active_genes, inactive_genes, RNAs, proteins



starting_state, active_genes, inactive_genes, RNAs, proteins = read_population()



def read_k_values():
    """This function reads k parameters from configuration file
    """
    
    k_value = dict(config["RATES"])
    
    for key,value in k_value.items():
        k_value[key] = float(value)
    
    rates = namedtuple("Rates",['ka', 'ki', 'k1', 'k2', 'k3', 'k4'])
    rate = rates(ka = k_value['ka'],
                 ki = k_value['ki'],
                 k1 = k_value['k1'], 
                 k2 = k_value['k2'], 
                 k3 = k_value['k3'], 
                 k4 = k_value['k4'])
    
    return rate



rate = read_k_values()



def read_simulation_parameters():
    """This function reads simulation parameters from configuration file
    """
    
    simulation = dict(config["SIMULATION"])
    
    for key,value in simulation.items():
        if key == 'dt':
            simulation[key] = float(value)
        else:
            simulation[key] = int(value)
      
    time_limit = simulation['time_limit'] 
    N = simulation['n_simulations']
    warmup_time = simulation['warmup_time']
    seed_number = simulation['seed_number']
    dt = simulation['dt']
    
    return time_limit, N, warmup_time, seed_number, dt



time_limit, N, warmup_time, seed_number, dt = read_simulation_parameters()



file_path = r'C:\Users\asus\Desktop\{}.csv'
 
multiplesimulations_filepath = r'C:\\Users\asus\Desktop\{}.csv' 



nlags = config["ACF"].getint('n_lags')



#%%
 
def plot_acf(autocorr, title = ""):
    """This function plots autocorrelation
    as function of the number of lags."""
    
    plt.plot(np.arange(0,len(autocorr)), autocorr, linestyle = 'dotted', color='darkgrey')
    #plt.ylim(0,1.05)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title("{}".format(title))
    sns.despine(bottom=False, left=False)
    plt.show()



def autocorr(simulation):
    """This function returns autocorrelation values for RNAs and proteins 
     given the type of simulation."""
    
    results = pd.read_csv('{}simulation_results.csv'.format(simulation), sep=" ")
    
    results = results[results['Time'] > warmup_time]

    RNAs = results['Number of RNA molecules']
    
    proteins = results['Number of proteins']
    
    if simulation == 'tauleap':
        
        autocorr_RNAs = stattools.acf(RNAs, nlags = nlags, fft=False) #1500, len(RNAs) - 1
        autocorr_proteins = stattools.acf(proteins, nlags = nlags, fft=False)
        
    elif simulation == 'gillespie':
        
        time = results['Time']
        xvals = np.arange(results['Time'].iloc[0], results['Time'].iloc[-1], dt)
        
        f_RNAs = interp.interp1d(time, RNAs, kind='previous')
        yinterp_RNAs = f_RNAs(xvals)
        #plt.plot(time, RNAs, 'o', xvals, yinterp_RNAs, '-')
        #plt.show()
        
        f_proteins = interp.interp1d(time, proteins, kind='previous')
        yinterp_proteins = f_proteins(xvals)
        #plt.plot(time, proteins, 'o', xvals, yinterp_proteins, '-')
        #plt.show()
        
        autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
        autocorr_proteins = stattools.acf(yinterp_proteins, nlags = nlags, fft=False)
        
    
    return autocorr_RNAs, autocorr_proteins    


if args.plot_acf:   
    
    autocorr_RNAs, autocorr_proteins = autocorr(simulation = 'tauleap')
    
    
    
    plot_acf(autocorr=autocorr_RNAs, title = 'RNA autocorrelation Tau-leap simulation')
    
    plot_acf(autocorr=autocorr_proteins, title = 'Protein autocorrelation Tau-leap simulation')
    
    
    
    autocorr_RNAs, autocorr_proteins = autocorr(simulation = 'gillespie')
    
    
    
    plot_acf(autocorr = autocorr_RNAs, title = 'RNA autocorrelation Gillespie simulation')
    
    plot_acf(autocorr = autocorr_proteins, title = 'Protein autocorrelation Gillespie simulation')



# RNAs and Proteins

def plot_multiacf(autocorr1, label1, autocorr2, label2, title = ""):
    """This function plots autocorrelation of two chemical species
    in the same graph. By default the title is empty."""
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.arange(0,len(autocorr1)), autocorr1, label = '{}'.format(label1), linestyle = 'dotted', color = 'black')
    ax.plot(np.arange(0,len(autocorr2)), autocorr2, label = '{}'.format(label2), linestyle = 'dotted', color = 'grey')
    #plt.ylim(0,1.05)
    ax.set_xlabel('# of lags')
    ax.set_ylabel('Autocorrelation')
    ax.set_title("{}".format(title))
    ax.legend()
    sns.despine(fig, bottom=False, left=False)
    plt.show()


if args.plot_acfs:
    
    autocorr_RNAs, autocorr_proteins = autocorr(simulation = 'tauleap')
        
    plot_multiacf(autocorr1 = autocorr_RNAs, label1 = "RNAs", autocorr2 = autocorr_proteins, label2 = "Proteins", title = "Tau-leap simulation")
    
    autocorr_RNAs, autocorr_proteins = autocorr(simulation = 'gillespie')
    
    plot_multiacf(autocorr1 = autocorr_RNAs, label1 = "RNAs", autocorr2 = autocorr_proteins, label2 = "Proteins", title = "Gillespie simulation")



# Multiple simulations



def MultipleSimulationsAcf(simulation):
    """This function calculates the RNA and protein autocorrelation of N
    simulations and returns their values as lists."""

    dataframes_list = []
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('{}results_seed{}.csv'.format(simulation, n), sep=" ")
        removedwarmup_results = simulation_results[simulation_results['Time'] > warmup_time]
        dataframes_list.append(removedwarmup_results)
    
    autocorrs_RNAs = []
    
    autocorrs_Proteins = []  
      
    for dataframe in dataframes_list:

        RNAs = dataframe['Number of RNA molecules']
        proteins = dataframe['Number of proteins']
        time = dataframe['Time']
        
        if simulation == 'gillespie':
            
            xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], dt)
            
            f_RNAs = interp.interp1d(time, RNAs, kind='previous')
            yinterp_RNAs = f_RNAs(xvals)
            #plt.plot(time, RNAs, 'o', xvals, yinterp_RNAs, '-')
            #plt.show()
            
            f_proteins = interp.interp1d(time, proteins, kind='previous')
            yinterp_proteins = f_proteins(xvals)
            #plt.plot(time, proteins, 'o', xvals, yinterp_proteins, '-')
            #plt.show()
            
            autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
            autocorr_proteins = stattools.acf(yinterp_proteins, nlags = nlags, fft=False)
        
            autocorrs_RNAs.append(autocorr_RNAs)
            autocorrs_Proteins.append(autocorr_proteins)
        
        elif simulation == 'tauleap':
            
            autocorr_RNAs = stattools.acf(RNAs, nlags = nlags, fft=False) #1500, len(RNAs) - 1
            autocorr_proteins = stattools.acf(proteins, nlags = nlags, fft=False)
            
            autocorrs_RNAs.append(autocorr_RNAs)
            autocorrs_Proteins.append(autocorr_proteins)
   
    return autocorrs_RNAs, autocorrs_Proteins



def MultipleSimulationsAcf_Plot(title):
    """This function plots RNA autocorrelation and protein autocorrelation
    of N simulations respectively up and down a multiplot."""
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,7))
    for n, autocorr in enumerate(autocorrs_RNAs,1):
        ax[0].plot(np.arange(0,len(autocorr)), autocorr, linestyle = 'dotted', color='black')
        ax[0].set_ylabel('Autocorrelation')
        ax[0].set_xlabel('# of lags')
        ax[0].legend(['RNAs'])
        ax[0].set_title("{}".format(title))
        x = np.arange(0,len(autocorr))
        y = autocorr
        ax[0].text(x[-1],y[-1],"n={}".format(n))
        #ax[0].set_ylim(-0.2,1)
    for n, autocorr in enumerate(autocorrs_Proteins,1):
        ax[1].plot(np.arange(0,len(autocorr)), autocorr, linestyle = 'dotted', color='grey')
        ax[1].set_ylabel('Autocorrelation')
        ax[1].set_xlabel('# of lags')
        ax[1].legend(['Proteins'])
        x = np.arange(0,len(autocorr))
        y = autocorr
        ax[1].text(x[-1],y[-1],"n={}".format(n))
        #ax[1].set_ylim(-0.2,1)
        sns.despine(fig, bottom=False, left=False)
    plt.show()



if args.multiplesimulationsacfs_tauleap:

    autocorrs_RNAs, autocorrs_Proteins = MultipleSimulationsAcf(simulation = 'tauleap')
    
    MultipleSimulationsAcf_Plot(title = "Tau-leap Simulation")
    
if args.multiplesimulationsacfs_gillespie:
    
    autocorrs_RNAs, autocorrs_Proteins = MultipleSimulationsAcf(simulation = 'gillespie')
    
    MultipleSimulationsAcf_Plot(title = "Gillespie Simulation")
    



