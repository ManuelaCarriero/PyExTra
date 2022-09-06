# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 13:04:08 2022

@author: asus
"""

import argparse
import configparser
import sys
import ast 

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns
from collections import Counter 
import scipy.stats as st

from collections import namedtuple 



config = configparser.ConfigParser()



parser = argparse.ArgumentParser()



parser.add_argument("filename", help="read configuration file.")

parser.add_argument("-distribution", help="plot simulation states distribution and theoretical distribution.", action = "store_true")

parser.add_argument("-tauleap_distribution", help="plot simulation states distribution and theoretical distribution given by tauleap algorithm.", action = "store_true")

parser.add_argument("-all_distributions", help="plot simulation states distribution and theoretical distribution given by all algorithms.", action = "store_true")

parser.add_argument("-time_plot", help="plot gene activity and number of molecules as function of time.", action = "store_true")

parser.add_argument("-tauleaptime_plot", help="plot gene activity and number of molecules as function of time given by tauleap algorithm.", action = "store_true")

parser.add_argument("-remove_warmup", help="plot gene activity and number of molecules as function of time without warmup period.", action = "store_true")

parser.add_argument("-multiple_simulations", help="plot number of molecules as function of time for multiple simulations.", action = "store_true")

parser.add_argument("-remove_warmup_multiple_simulations", help="plot number of molecules as function of time for multiple simulations without warmup period.", action = "store_true")

parser.add_argument("-all_plots", help="makes all possible plots provided by plots.py", action = "store_true")



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
    
    rates = namedtuple("Rates",['ka', 'ki', 'k1', 'k2', 'k3', 'k4', 'k5'])
    rate = rates(ka = k_value['ka'], 
                 ki = k_value['ki'], 
                 k1 = k_value['k1'], 
                 k2 = k_value['k2'], 
                 k3 = k_value['k3'], 
                 k4 = k_value['k4'], 
                 k5 = k_value['k5'])
    
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



#%%

results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")

tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")



def MoleculesVsTimePlot(df):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time
    """
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    ax[0].plot(df['Time'], df['Gene activity'])
    ax[0].set_ylabel('Gene Activity')
    ax[0].set_xlabel('Time')
    
    if args.tauleaptime_plot:
        
        ax[0].text(0.9,0.3,r"$\tau$={}".format(dt), 
                   ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[0].transAxes)
        
    ax[0].text(0.9,0.8,"$K_a$=$n_a${}\n $K_i$=$n_i${}".format(rate.ka,rate.ki), 
               ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[0].transAxes)
    
    ax[1].plot(df['Time'], df['Number of RNA molecules'])
    ax[1].set_ylabel('# of RNA molecules')
    ax[1].set_xlabel('Time')
    ax[1].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1, rate.k2), 
               ha ='center', va = 'center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    
    ax[2].plot(df['Time'], df['Number of proteins'])
    ax[2].set_ylabel('# of proteins')
    ax[2].set_xlabel('Time')
    ax[2].text(0.9,0.8,"$K_3$=m{}\n $K_4$=p{}".format(rate.k3, rate.k4), 
               ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[2].transAxes)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()



def generate_RNA_distribution(df):
    """ This function creates a Counter with RNA state values 
    as keys and normalized residency time as values
    """
    RNA_distribution = Counter()
    for state, residency_time in zip(df['Number of RNA molecules'], df['Residency Time']):
        RNA_distribution[state] += residency_time
    
    total_time_observed = sum(RNA_distribution.values())
    for state in RNA_distribution:
        RNA_distribution[state] /= total_time_observed

    return RNA_distribution 



def generate_protein_distribution(df):
    """ This function creates a Counter with protein state values 
    as keys and normalized residency time as values
    """
    protein_distribution = Counter()
    for state, residency_time in zip(df['Number of proteins'], df['Residency Time']):
        protein_distribution[state] += residency_time
    
    total_time_observed = sum(protein_distribution.values())
    for state in protein_distribution:
        protein_distribution[state] /= total_time_observed
    
    return protein_distribution 



def StatesDistributionPlot(df):
    """ This function plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(df)
    
    protein_distribution = generate_protein_distribution(df)
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    values = np.arange(20)
    pmf = st.poisson(10).pmf(values) 
    ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
    ax[0].set_ylabel('Normalized residency time', fontsize=10)
    ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
    ax[0].bar(values, pmf, alpha=0.5)
    
    if args.distribution:
        simulation = "SSA"
    elif args.tauleap_distribution:
        simulation = "Tau-leap"
    
    ax[0].legend(["{} Simulation".format(simulation),"Poisson distribution"], fontsize=10)
    
    ax[1].bar(protein_distribution.keys(), protein_distribution.values())
    ax[1].set_ylabel('Normalized residency time', fontsize=10)
    ax[1].set_xlabel('Number of proteins', fontsize=10)
    ax[1].bar(values, pmf, alpha=0.5)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()



def AllStatesDistributionPlot():
    """ This function plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(df = results)
    
    protein_distribution = generate_protein_distribution(df = results)
    
    RNA_distribution_tauleap = generate_RNA_distribution(df = tauleap_results)
    
    protein_distribution_tauleap = generate_protein_distribution(df = tauleap_results)
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    values = np.arange(20)
    pmf = st.poisson(10).pmf(values) 
    ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
    ax[0].bar(RNA_distribution_tauleap.keys(), RNA_distribution_tauleap.values(), alpha=0.9)
    ax[0].set_ylabel('Normalized residency time', fontsize=10)
    ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
    ax[0].bar(values, pmf, alpha=0.5)
    
    ax[0].legend(["SSA","Tau-leap","Poisson distribution"], fontsize=10, loc = "upper right")
    
    ax[1].bar(protein_distribution.keys(), protein_distribution.values())
    ax[1].bar(protein_distribution_tauleap.keys(), protein_distribution_tauleap.values(), alpha=0.5)
    ax[1].set_ylabel('Normalized residency time', fontsize=10)
    ax[1].set_xlabel('Number of proteins', fontsize=10)
    ax[1].bar(values, pmf, alpha=0.5)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()

    
#%%

if args.distribution:
    StatesDistributionPlot(df = results)



if args.tauleap_distribution:
    StatesDistributionPlot(df = tauleap_results)
    


if args.all_distributions:
    AllStatesDistributionPlot()



if args.time_plot:
    MoleculesVsTimePlot(df = results)
    
    

if args.tauleaptime_plot:
    MoleculesVsTimePlot(df = tauleap_results)



removedwarmup_results = results[results['Time'] > warmup_time]



if args.remove_warmup:
    MoleculesVsTimePlot(df = removedwarmup_results)



#%% Multiple simulations



dataframes_list = []

for n in range(1,N+1):
    simulation_results = pd.read_csv('gillespieresults_seed{}.csv'.format(n), sep=" ")
    dataframes_list.append(simulation_results)



def MultipleSimulationsPlot(dataframes):
    """
    This function makes a multiplot with one column and two rows:
    top plot for the number of RNA molecules produced vs time 
    and down plot for the number of proteins produced vs time.

    Parameters
    ----------
    N : int > 0
        number of simulations.
    """
        
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,10))
    for dataframe in dataframes:
        ax[0].plot(dataframe['Time'], dataframe['Number of RNA molecules'])
        ax[0].set_ylabel('# of RNA molecules')
        ax[0].set_xlabel('Time')
        ax[1].plot(dataframe['Time'], dataframe['Number of proteins'])
        ax[1].set_ylabel('# of proteins')
        ax[1].set_xlabel('Time')
        sns.despine(fig, bottom=False, left=False)
    plt.show()


if args.multiple_simulations:
    MultipleSimulationsPlot(dataframes = dataframes_list)



#%%



removedwarmup_dataframes_list = []



for simulation_results in dataframes_list:
    removedwarmup_results = simulation_results[simulation_results['Time'] > warmup_time]
    removedwarmup_dataframes_list.append(removedwarmup_results)


if args.remove_warmup_multiple_simulations:
    MultipleSimulationsPlot(dataframes = removedwarmup_dataframes_list)



#%%



if args.all_plots:
    
    StatesDistributionPlot(df = results)
    
    MoleculesVsTimePlot(df = results)
    
    removedwarmup_results = results[results['Time'] > warmup_time]
    
    MoleculesVsTimePlot(df = removedwarmup_results)
    
    MultipleSimulationsPlot(dataframes = dataframes_list)
    
    MultipleSimulationsPlot(dataframes = removedwarmup_dataframes_list)
    