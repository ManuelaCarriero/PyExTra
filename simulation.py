# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:41:16 2022

@author: asus
"""

#import argparse
import configparser
import ast 
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import typing 
from enum import Enum

from collections import Counter, namedtuple 
import scipy.stats as st



config = configparser.ConfigParser()

if len(sys.argv) == 1:
    config.read('gillespie_configuration.txt')
else:
    config.read(sys.argv[1])



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
    seed_number = simulation['seed_number']
    
    return time_limit, N, warmup_time, seed_number



time_limit, N, warmup_time, seed_number = read_simulation_parameters()



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
               Protein_increase, Protein_degrade,
               gene_degrade]



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
                    Transition.PROTEIN_INCREASE, Transition.PROTEIN_DEGRADE,
                    Transition.GENE_DEGRADE]



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
    """This function updates the initial state according to the event occured
    
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



simulation_results = evolution(starting_state = state, time_limit = time_limit, seed_number = seed_number)



#%%



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



def StatesDistributionPlot(results):
    """ This function plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(results)
    
    protein_distribution = generate_protein_distribution(results)
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    values = np.arange(20)
    pmf = st.poisson(5).pmf(values) 
    ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
    ax[0].set_ylabel('Normalized residency time', fontsize=10)
    ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
    ax[0].bar(values, pmf, alpha=0.5)
    
    ax[0].legend(["Simulation","Poisson distribution"], fontsize=10)
    
    ax[1].bar(protein_distribution.keys(), protein_distribution.values())
    ax[1].set_ylabel('Normalized residency time', fontsize=10)
    ax[1].set_xlabel('Number of proteins', fontsize=10)
    ax[1].bar(values, pmf, alpha=0.5)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()
 


StatesDistributionPlot(simulation_results)



def decide_timelimit(simulation_results):   
    
    user_answer = input("The actual time limit is {}. Do you want to increase simulation time limit ? [yes/no] : ".format(time_limit))
    
    user_answer = user_answer.replace(" ","")
    
    right_answers = ["yes","YES","no","NO"]
    
    if user_answer not in right_answers:
        
        print("NameError: valid answers are 'yes' or 'no'")
        
        user_answer=input("Do you want to increase simulation time limit ?[yes/no] : ")
    
    if user_answer == "yes" or user_answer == "YES":
        
        while user_answer != "no" and user_answer != "NO": 
            
            timelimit_answer = input("Choose simulation time limit [number] : ")

            def add_evolution(simulation_results, time_limit, seed_number):
                
                observed_states = simulation_results
                
                last_state = simulation_results[-1]
                
                state = last_state.state
                
                total_time = last_state.time_of_observation
                
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
            
            timelimit_answer = float(timelimit_answer)
        
            simulation_results = add_evolution(simulation_results = simulation_results, time_limit = timelimit_answer, seed_number = seed_number)  
         
            StatesDistributionPlot(simulation_results)
            
            user_answer = input("The actual time limit is {}. Do you want to increase again simulation time limit ? [yes/no] : ".format(timelimit_answer))
    
            user_answer = user_answer.replace(" ","")
    
            right_answers = ["yes","YES","no","NO"]
    
            if user_answer not in right_answers:
                
                print("NameError: valid answers are 'yes' or 'no'")
                
                user_answer=input("Do you want to increase again simulation time limit ?[yes/no] : ")
        
    else:
        
        simulation_results = simulation_results
        
        timelimit_answer = time_limit
    
    return simulation_results, timelimit_answer



simulation_results, timelimit_answer = decide_timelimit(simulation_results=simulation_results)
    


#%%



def create_dataframe(results):
    """ This function creates a dataframe with 4 columns:
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



def save_results(results, file_path):
    """This function saves dataframe in a tab separated CSV file
    
    Parameters
    
    file_path : str
                r'C: path to folder where the CSV file is saved
    """
    df = create_dataframe(results)
    df.to_csv(file_path, sep ='\t', index = None, header=True) 
    


save_results(results = simulation_results, file_path = r'C:\Users\asus\Desktop\results.csv')



def create_multiplesimulations_dataframes(N):
    """This function makes multiple simulations and creates a list
    of results dataframes (one results dataframe for each simulation)
    
    Parameters
    ----------
    N : int
        number of simulations.

    Returns
    -------
    list of results dataframe
    """
    
    results_list = []
    for n in range(1,N):
        result = evolution(starting_state = state, time_limit = timelimit_answer, seed_number = n)
        results_list.append(result)

    dataframes_list = []
    for result in results_list:
        dataframe = create_dataframe(result)
        dataframes_list.append(dataframe)
    
    return dataframes_list



dataframes_list = create_multiplesimulations_dataframes(N)



def save_multiplesimulations_results(N, file_path=r'C:\\Users\asus\Desktop\{}.csv'):
    """This function saves dataframes of multiple simulations in tab separated CSV files
    each one named as "results_seedn" with n that is the number of the random seed.
    
    Parameters
    
    N : int
        number of simulations.
    
    file_path : str, default is r"C:\\Users\asus\Desktop\{}.csv"
                path to folder where files are saved. By default, it saves the files following the path \\Users\asus\Desktop.
    """
    
    results_names = []
    for n in range(1,N):
        results_names.append("results_seed"+str(n))
    
    for dataframe, results in zip(dataframes_list, results_names):
        dataframe.to_csv(file_path.format(results), sep='\t', index = None, header=True)



save_multiplesimulations_results(N)



#%%



def steadystate_distribution(observation):
        return observation.time_of_observation > warmup_time

filtered_results = filter(steadystate_distribution, simulation_results)

removedwarmup_results = list(filtered_results)



def multiple_simulations_remove_warmup(N):
    """This function makes multiple simulations, removes warmup time points
    and creates a list of results dataframes (one results dataframe 
    for each simulation).
    

    Parameters
    ----------
    N : int
        number of simulations.

    Returns
    -------
    list of results dataframe
    """
    
    results_list = []
    for n in range(1,N):
        result = evolution(starting_state = state, time_limit = timelimit_answer, seed_number = n)
        results_list.append(result)
    
    removed_warmup_results_list =[]
    for result in results_list:
        filtered_results = filter(steadystate_distribution, result)
        removed_warmup = list(filtered_results)
        removed_warmup_results_list.append(removed_warmup)
        
    dataframes_list = []
    for result in removed_warmup_results_list:
        dataframe = create_dataframe(result)
        dataframes_list.append(dataframe)
    
    return dataframes_list



removed_warmup_dataframes = multiple_simulations_remove_warmup(N)    



def MoleculesVsTimePlot(results):
    """This function plots gene activity, the number of RNA molecules
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
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()



results = [simulation_results, removedwarmup_results]


    
MoleculesVsTimePlot(results = removedwarmup_results)



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



results = [dataframes_list, removed_warmup_dataframes]



MultipleSimulationsPlot(dataframes = removed_warmup_dataframes)

    
    