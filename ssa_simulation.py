# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:41:16 2022

@author: asus
"""

import argparse
import configparser
import ast 
#import sys

import numpy as np
import pandas as pd


import typing 
from enum import Enum, IntEnum

from collections import namedtuple 


import json
import jsonlines

import os
#from itertools import cycle
#import time


config = configparser.ConfigParser()

parser = argparse.ArgumentParser()

parser.add_argument("filename", help="read configuration file.")

parser.add_argument('-run', help='run Gillespie simulation given a configuration filename', action = "store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--time_limit", help="increase time limit", metavar='value', type = float)

args = parser.parse_args()



config.read(args.filename)

if args.verbose:
    print("I am reading the configuration file {}".format(args.filename))



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




actual_dir = os.getcwd()

file_path = r'{}\{}.csv'



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
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition: Transition
    transition_rates: typing.Any

class Index(IntEnum):
    state = 0
    time_of_observation = 1
    time_of_residency = 2
    transition = 3
    transition_rates = 4


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
                  
    elif event == Transition.GENE_INACTIVATE:
        state[active_genes] -=1
        state[inactive_genes] +=1
        
    elif event == Transition.RNA_INCREASE:
         state[RNAs] +=1
         
    elif event == Transition.RNA_DEGRADE:
        state[RNAs] -=1
        
    elif event == Transition.PROTEIN_INCREASE:
         state[proteins] +=1
         
    elif event == Transition.PROTEIN_DEGRADE:
        state[proteins] -=1
                
    elif event == Transition.GENE_DEGRADE:
        state[active_genes] = 0
        state[inactive_genes] = 0
            
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



def evolution(starting_state, starting_total_time, time_limit, seed_number):
    
    observed_states = []
    
    state = starting_state
    
    total_time = starting_total_time
    
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



class CustomizedEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Enum):
           return obj.name
        return json.JSONEncoder.default(self, obj)



if args.run: 
    
    simulation_results = evolution(starting_state = starting_state, starting_total_time = 0.0, time_limit = time_limit, seed_number = seed_number)
    
    for result in simulation_results:
        print(result)
        
    simulation_results_ = json.dumps(simulation_results, cls = CustomizedEncoder)

    with jsonlines.open('simulation_results.jsonl', mode='w') as writer:
        writer.write(simulation_results_)


    
if args.time_limit:
    
    with jsonlines.open('simulation_results.jsonl') as reader:
        simulation_results = reader.read()

    simulation_results = ast.literal_eval(simulation_results)
    last_event = simulation_results[-1][Index.transition]
    last_state = np.array(simulation_results[-1][Index.state])

    state = update_state(event = last_event, state = last_state) 

    added_simulation_results = evolution(starting_state = state, starting_total_time = simulation_results[-1][Index.time_of_observation] + simulation_results[-1][Index.time_of_residency], time_limit = args.time_limit, seed_number = seed_number)




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
    residency_time = []

    for observation in results:
        time_of_observation.append(observation.time_of_observation)
        number_of_RNA_molecules.append(observation.state[RNAs])
        number_of_proteins.append(observation.state[proteins])
        residency_time.append(observation.time_of_residency)
        if observation.state[active_genes] > 0:
            gene_activity.append(1)
        else:
            gene_activity.append(0)
    
    d = {'Time': time_of_observation, 
         'Gene activity': gene_activity,
         'Number of RNA molecules': number_of_RNA_molecules, 
         'Number of proteins': number_of_proteins,
         'Residency Time': residency_time}
    
    results_dataframe = pd.DataFrame(d)
    
    return results_dataframe



if args.run:

    df = create_dataframe(results = simulation_results)

if args.time_limit:
    
    df = create_dataframe(results = added_simulation_results)
    
    

"""
def progress(iterator):
    cycling = cycle("\|/")
    for element in iterator:
        print(next(cycling), end="\r")
        yield element
    print(" \r", end='')



for idx in progress(range(10)):
    time.sleep(0.5)
"""

"""
actual_dir = os.getcwdb()
if len(sys.argv) != 1 and args.verbose:
    print(" ")
    print("I am saving results into your current directory ({}) (simulation random seed = {}). ".format(actual_dir, seed_number))
"""


    
if args.run:

    df.to_csv(file_path.format(actual_dir,"gillespiesimulation_results"), sep =" ", index = None, header=True, mode = "w") 

if args.time_limit:
    
    df.to_csv(file_path.format(actual_dir,"added_gillespiesimulation_results"), sep =" ", index = None, header=True, mode = "w") 



"""
def progress(iterator):
    cycling = cycle("\|/")
    for element in iterator:
        print(next(cycling), end="\r")
        yield element
    print(" \r", end='')



for idx in progress(range(10)):
    time.sleep(0.5)



if len(sys.argv) != 1 and args.verbose:
    print(" ")
    print("Now I am doing {} different simulations (with random seed respectively from 1 to {}). ".format(N, N))



def progress(iterator):
    cycling = cycle("\|/")
    for element in iterator:
        print(next(cycling), end="\r")
        yield element
    print(" \r", end='')



for idx in progress(range(10)):
    time.sleep(0.5)
"""



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
    for n in range(1,N+1):
        result = evolution(starting_state = starting_state, starting_total_time = 0.0, time_limit = time_limit, seed_number = n)
        results_list.append(result)

    dataframes_list = []
    for result in results_list:
        dataframe = create_dataframe(result)
        dataframes_list.append(dataframe)
    
    return dataframes_list



dataframes_list = create_multiplesimulations_dataframes(N)

"""
if len(sys.argv) != 1 and args.verbose:
    print(" ")
    print("I am saving results into your current directory ({})".format(actual_dir))



def progress(iterator):
    cycling = cycle("\|/")
    for element in iterator:
        print(next(cycling), end="\r")
        yield element
    print(" \r", end='')



for idx in progress(range(10)):
    time.sleep(0.5)
"""



def save_multiplesimulations_results(N, file_path = file_path):
    """This function saves dataframes of multiple simulations in tab separated CSV files
    each one named as "results_seedn" with n that is the number of the random seed.
    
    Parameters
    
    N : int
        number of simulations.
    
    file_path : str, default is r"C:\\Users\asus\Desktop\{}.csv"
                path to folder where files are saved. By default, it saves the files following the path \\Users\asus\Desktop.
                You can change it in the configuration file.
    """
    
    results_names = []
    for n in range(1,N+1):
        results_names.append("gillespieresults_seed"+str(n))
    
    for dataframe, results in zip(dataframes_list, results_names):
        dataframe.to_csv(file_path.format(actual_dir,results), sep=" ", index = None, header=True)



save_multiplesimulations_results(N)



if args.verbose:
    print("My job is done. Enjoy data analysis !")