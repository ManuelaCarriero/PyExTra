# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:57:20 2022

@author: asus
"""
import configparser
import sys
import ast 

import numpy as np
import random as rn
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import typing 
from enum import Enum

from collections import Counter, namedtuple 

import scipy.stats as st




config = configparser.ConfigParser()
config.read('gillespie_configuration.ini')
#config.read(sys.argv[1])



def read_population():
    """This function reads population parameters from configuration file
    """
    
    state = config.get('POPULATION', 'state')
    
    def _convert_to_array(state):
        List = ast.literal_eval(state)
        return np.array(List)
    
    state = _convert_to_array(state)
    
    index = dict(config["INDEX"])
    
    for key,value in index.items():
        index[key] = int(value)
    
    active_genes = index['active_genes']
    inactive_genes = index['inactive_genes']
    RNAs = index['rnas']
    proteins = index['proteins']
    
    return state, active_genes, inactive_genes, RNAs, proteins




state, active_genes, inactive_genes, RNAs, proteins = read_population()




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
        simulation[key] = int(value)
      
    time_limit = simulation['time_limit'] 
    N = simulation['n']
    warmup_time = simulation['warmup_time']
    
    return time_limit, N, warmup_time



time_limit, N, warmup_time = read_simulation_parameters()

#%% 



def gene_activate(state):
    return state[inactive_genes]*rate.ka 
def gene_inactivate(state):
    return state[active_genes]*rate.ki

def RNA_increase(state):
    return state[active_genes]*rate.k1

def RNA_degrade(state):
    return state[RNAs]*rate.k2

def Protein_increase(state):
    return state[RNAs]*rate.k3

def Protein_degrade(state):
    return state[proteins]*rate.k4

def gene_degrade(state):
    return (state[active_genes]+state[inactive_genes])*rate.k5




transitions = [gene_activate, gene_inactivate, 
               RNA_increase, RNA_degrade, 
               Protein_increase, Protein_degrade]

class Transition(Enum):
    """Define all possible transitions"""
    GENE_ACTIVATE = 'gene activate'
    GENE_INACTIVATE = 'gene inactivate'
    RNA_INCREASE = 'RNA increase'
    RNA_DEGRADE = 'RNA degrade'
    PROTEIN_INCREASE = 'Protein increase'
    PROTEIN_DEGRADE = 'Protein degrade'
    GENE_DEGRADE = 'gene degrade'
    ABSORPTION = 'Absorption'

transition_names = [Transition.GENE_ACTIVATE, Transition.GENE_INACTIVATE, 
                    Transition.RNA_INCREASE, Transition.RNA_DEGRADE, 
                    Transition.PROTEIN_INCREASE, Transition.PROTEIN_DEGRADE]

class Observation(typing.NamedTuple):
    """ typing.NamedTuple class storing information
    for each event in the simulation"""
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition: Transition
    transition_rates: typing.Any

#%%

def update_state(event, state):
    """This method updates the initial state according to the event occured
    
    Parameters
    ----------
    event : member of Transition class
    state : ndarray
            ndarray with 4 dimensions: number of active genes, number of
            inactive genes, number of RNAs and number of proteins

    Returns
    -------
    updated_state : ndarray
                    ndarray with 4 dimensions: number of active genes, number of
                    inactive genes, number of RNAs and number of proteins after
                    the event has occured.
    """

    if event == Transition.GENE_ACTIVATE:
         state[active_genes] +=1
         state[inactive_genes] -=1
         state = state.copy()

         
    elif event == Transition.GENE_INACTIVATE:
        state[active_genes] -=1
        state[inactive_genes] +=1
        state = state.copy()

    elif event == Transition.RNA_INCREASE:
         state[RNAs] +=1
         state = state.copy()

    elif event == Transition.RNA_DEGRADE:
        state[RNAs] -=1
        state = state.copy()

    elif event == Transition.PROTEIN_INCREASE:
         state[proteins] +=1
         state = state.copy()

    elif event == Transition.PROTEIN_DEGRADE:
        state[proteins] -=1
        state = state.copy()
    
    elif event == Transition.GENE_DEGRADE:
        state[active_genes] = 0
        state[inactive_genes] = 0
        state = state.copy() 
    
    elif event == Transition.ABSORPTION:
        pass
    
    elif isinstance(event,str) or isinstance(event, str):
        raise TypeError("Do not use string ! Choose transitions from Transition enum members.")
    else:
        raise ValueError("Transition not recognized")

    updated_state = state

    return updated_state 







def gillespie_ssa(starting_state, transitions):
    
    state = starting_state 
    
    rates = [f(state) for f in transitions]
    
    total_rate = np.sum(rates)
    
    if total_rate > 0:
        
        time = np.random.exponential(1/total_rate)
        
        rates_array = np.array(rates)

        rates_array /= rates_array.sum()
    
        event = np.random.choice(transition_names, p=rates_array)
    
    else:
        
        time = np.inf
        
        event = Transition.ABSORPTION
    
    state = state.copy()
    
    updated_state = update_state(event, state)
    
    gillespie_result = [starting_state, updated_state, time, event, rates]
    
    return gillespie_result



def evolution(starting_state, time_limit, seed_number):
    observed_states = []
    state = starting_state
    total_time = 0.0
    
    
    np.random.seed(seed_number)

    
    
    while total_time < time_limit:
        
        gillespie_result = gillespie_ssa(starting_state = state, transitions = transitions)
        
        rates = gillespie_result[4]
        
        event = gillespie_result[3]
        
        time = gillespie_result[2]
        
        observation_state = gillespie_result[0]
        
        
        
        observation = Observation(observation_state, total_time, time, event, rates)
        
        
        
        observed_states.append(observation)
        
        # Update time
        total_time += time
        
        # Update starting state in gillespie algorithm
        state = state.copy()
        state = gillespie_result[1]

    return observed_states


time_limit = 100
simulation_results = evolution(starting_state = state, time_limit = time_limit, seed_number = 1)
#simulation_results
#simulation_results[:5]
#simulation_results[0]
#simulation_results[-5:]
#simulation_results_length = len(simulation_results)
#simulation_results_length

def remove_warmup(results):
    results_copy = results.copy()
    for observation in results_copy:
        if observation.time_of_observation < 20:
            results.remove(observation)
    return results

results = remove_warmup(results=simulation_results)
#results
#results_length = len(results)
#results_length




##############################################################################


def generate_RNA_distribution(results):
    """ This function creates a Counter with RNA state values 
    as keys and normalized residency time as values
    """
    RNA_distribution = Counter()
    for observation in results[0:-1]:
        state = observation.state[RNAs]
        residency_time = observation.time_of_residency
        RNA_distribution[state] += residency_time
    
    total_time_observed = sum(RNA_distribution.values())
    for state in RNA_distribution:
        RNA_distribution[state] /= total_time_observed

    return RNA_distribution 



def generate_protein_distribution(results):
    """ This function creates a Counter with protein state values 
    as keys and normalized residency time as values
    """
    protein_distribution = Counter()
    for observation in results[0:-1]:
        state = observation.state[proteins]
        residency_time = observation.time_of_residency
        protein_distribution[state] += residency_time
    
    total_time_observed = sum(protein_distribution.values())
    for state in protein_distribution:
        protein_distribution[state] /= total_time_observed
    
    return protein_distribution 






def StatesDistributionPlot():
    """ This method plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(results)
    protein_distribution = generate_protein_distribution(results)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    values = np.arange(20)
    pmf = st.poisson(5).pmf(values) 
    ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
    ax[0].set_ylabel('Normalized residency time', fontsize=16)
    ax[0].set_xlabel('Number of RNA molecules', fontsize=16)
    ax[0].bar(values, pmf, alpha=0.5)
    ax[0].legend(["Simulation","Poisson distribution"], fontsize=16)
    ax[1].bar(protein_distribution.keys(), protein_distribution.values())
    ax[1].set_ylabel('Normalized residency time', fontsize=16)
    ax[1].set_xlabel('Number of proteins', fontsize=16)
    ax[1].bar(values, pmf, alpha=0.5)
 


StatesDistributionPlot()



def create_dataframe(results):
    """ This method creates a dataframe with 4 columns:
        time of observation, gene activity, number of RNA molecules
        and number of proteins.          
    """
    time_of_observation = []
    number_of_RNA_molecules = []
    number_of_proteins = []
    gene_activity = []

    for observation in results:
        time_of_observation.append(observation.time_of_observation)
        number_of_RNA_molecules.append(observation.state[RNAs])
        number_of_proteins.append(observation.state[proteins])
        if observation.state[active_genes] > 0:
            gene_activity.append(1)
        else:
            gene_activity.append(0)
    
    d = {'Time': time_of_observation, 
         'Gene activity': gene_activity,
         'Number of RNA molecules': number_of_RNA_molecules, 
         'Number of proteins': number_of_proteins}
    
    results_dataframe = pd.DataFrame(d)
    
    return results_dataframe




def save_results(file_path):
    """This method saves dataframe in a tab separated CSV file
    
    Parameters
    
    file_path : str
                path to folder where the CSV file is saved
    """
    df = create_dataframe(results)
    df.to_csv(file_path, sep ='\t', index = None, header=True) 
    


save_results(file_path = r'C:\Users\asus\Desktop\results.csv')



def MoleculesVsTimePlot():
    """This method plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time
    """
    df = create_dataframe(results)
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    ax[0].plot(df['Time'], df['Gene activity'])
    ax[0].set_ylabel('Gene Activity')
    ax[0].set_xlabel('Time')
    ax[0].text(0.9,0.8,"$k_a$=$n_a$*{}\n $k_i$=$n_i$*{}".format(rate.ka,rate.ki), 
               ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[0].transAxes)
    ax[1].plot(df['Time'], df['Number of RNA molecules'])
    ax[1].set_ylabel('# of RNA molecules')
    ax[1].set_xlabel('Time')
    ax[1].text(0.9,0.8,"$k_1$=$n_a$*{}\n $k_2$=m*{}".format(rate.k1, rate.k2), 
               ha ='center', va = 'center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    ax[2].plot(df['Time'], df['Number of proteins'])
    ax[2].set_ylabel('# of proteins')
    ax[2].set_xlabel('Time')
    ax[2].text(0.9,0.8,"$k_3$=m*{}\n $k_4$=p*{}".format(rate.k3, rate.k4), 
               ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[2].transAxes)
    
    sns.despine(fig, trim=True, bottom=False, left=False)



MoleculesVsTimePlot()

def save_multiple_simulations_results(N, file_path=r'C:\\Users\asus\Desktop\{}.csv'):
    """This method saves dataframes of multiple simulations in tab separated CSV files
    each one named as "results_seedn" with n that is the number of the random seed.
    
    Parameters
    
    N : int
        number of simulations.
    
    file_path : str, default is r"C:\\Users\asus\Desktop\{}.csv"
                path to folder where files are saved. By default, it saves the files following the path \\Users\asus\Desktop.
    """
    results_list = []
    for n in range(1,N):
        result = evolution(starting_state = state, time_limit = time_limit, seed_number = n)
        results_list.append(result)
        
    dataframes_list = []
    for result in results_list:
        dataframe = create_dataframe(result)
        dataframes_list.append(dataframe)
    
    results_names = []
    for n in range(1,N):
        results_names.append("results_seed"+str(n))
    
    for dataframe, results in zip(dataframes_list, results_names):
        dataframe.to_csv(file_path.format(results), sep='\t', index = None, header=True)



save_multiple_simulations_results(N)

"""
#Alternative: use numpy function to speed up the code instead of range Python function.
#Compare the time
np.linspace(1,4,4, dtype = int)

%timeit save_multiple_simulations_results(N)
3.75 s ± 44 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit save_multiple_simulations_results_numpy(N)
4.92 s ± 23.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""

def MultipleSimulationsPlot(N):
    """
    This method makes a multiplot with one column and two rows:
    top plot for the number of RNA molecules produced vs time 
    and down plot for the number of proteins produced vs time.

    Parameters
    ----------
    N : int > 0
        number of simulations.
    """
    
    results_list = []
    for n in range(1,N):
        result = evolution(starting_state = state, time_limit = time_limit, seed_number = n)
        results_list.append(result)
        
    dataframes_list = []
    for result in results_list:
        dataframe = create_dataframe(result)
        dataframes_list.append(dataframe)
        
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,10))
    for dataframe in dataframes_list:
        ax[0].plot(dataframe['Time'], dataframe['Number of RNA molecules'])
        ax[0].set_ylabel('# of RNA molecules')
        ax[0].set_xlabel('Time')
        ax[1].plot(dataframe['Time'], dataframe['Number of proteins'])
        ax[1].set_ylabel('# of proteins')
        ax[1].set_xlabel('Time')



MultipleSimulationsPlot(N)


