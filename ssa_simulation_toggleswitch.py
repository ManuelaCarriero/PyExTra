# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 15:45:11 2022

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


config.read('configuration_2genes.txt')

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
    
    active_gene1 = index['active_gene1']
    active_gene2 = index['active_gene2']
    inactive_gene1 = index['inactive_gene1']
    inactive_gene2 = index['inactive_gene2']
    
    RNAs_gene1 = index['rnas_gene1']
    proteins_gene1 = index['proteins_gene1']
    RNAs_gene2 = index['rnas_gene2']
    proteins_gene2 = index['proteins_gene2']
    
    return starting_state, active_gene1, active_gene2, inactive_gene1, inactive_gene2, RNAs_gene1, proteins_gene1, RNAs_gene2, proteins_gene2




starting_state, active_gene1, active_gene2, inactive_gene1, inactive_gene2, RNAs_gene1, proteins_gene1, RNAs_gene2, proteins_gene2 = read_population()




def read_k_values():
    """This function reads k parameters from configuration file
    """
    
    k_value = dict(config["RATES"])
    
    for key,value in k_value.items():
        k_value[key] = float(value)
    
    rates = namedtuple("Rates",['ka_1', 'ki_1', 'k1_1', 'k2_1', 'k3_1', 'k4_1', 'k5_1',
                                'ka_2', 'ki_2', 'k1_2', 'k2_2', 'k3_2', 'k4_2','k5_2'])
    rate = rates(ka_1 = k_value['ka_1'], 
                 ki_1 = k_value['ki_1'], 
                 k1_1 = k_value['k1_1'], 
                 k2_1 = k_value['k2_1'], 
                 k3_1 = k_value['k3_1'], 
                 k4_1 = k_value['k4_1'], 
                 ka_2 = k_value['ka_2'], 
                 ki_2 = k_value['ki_2'], 
                 k1_2 = k_value['k1_2'], 
                 k2_2 = k_value['k2_2'], 
                 k3_2 = k_value['k3_2'], 
                 k4_2 = k_value['k4_2'], 
                 k5_1 = k_value['k5_1'],
                 k5_2 = k_value['k5_1'])
    
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

def gene1_activate(state):
    
    state = state.copy()
    
    #Transition rate with this starting state
    trans_rate = (state[inactive_gene1])*(1/(1+state[proteins_gene2]))*rate.ka_1
    #Update state in case this transition is chosen
    state[active_gene1] +=1
    state[inactive_gene1] -=1
    
    new_state = state
    return [trans_rate, new_state]

def gene2_activate(state):
    
    state = state.copy()
    
    trans_rate = (state[inactive_gene2])*(1/(1+state[proteins_gene1]))*rate.ka_2
    state[active_gene2] +=1
    state[inactive_gene2] -=1

    new_state = state
    return [trans_rate, new_state]

def gene1_inactivate(state):
    
    state = state.copy()
    
    trans_rate = (state[active_gene1])*rate.ki_1
    state[active_gene1] -=1
    state[inactive_gene1] +=1

    new_state = state
    return [trans_rate, new_state]

def gene2_inactivate(state):
    
    state = state.copy()
    
    trans_rate = (state[active_gene2])*rate.ki_2
    state[active_gene2] -=1
    state[inactive_gene2] +=1
    new_state = state
    return [trans_rate, new_state]




def RNA_gene1_increase(state):
    
    state = state.copy()
    
    trans_rate = (state[active_gene1])*rate.k1_1
    state[RNAs_gene1] +=1
    new_state = state
    return [trans_rate, new_state]

def RNA_gene1_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[RNAs_gene1])*rate.k2_1
    state[RNAs_gene1] -=1
    new_state = state    
    return [trans_rate, new_state]

def RNA_gene2_increase(state):
    
    state = state.copy()
    
    trans_rate = (state[active_gene2])*rate.k1_2
    state[RNAs_gene2] +=1
    new_state = state
    return [trans_rate, new_state]

def RNA_gene2_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[RNAs_gene2])*rate.k2_2
    state[RNAs_gene2] -=1
    new_state = state
    return [trans_rate, new_state]




def Protein_gene1_increase(state):
    
    state = state.copy()
    
    trans_rate = (state[RNAs_gene1])*rate.k3_1
    state[proteins_gene1] +=1
    new_state = state
    return [trans_rate, new_state]

def Protein_gene1_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[proteins_gene1])*rate.k4_1
    state[proteins_gene1] -=1
    new_state = state
    return [trans_rate, new_state]

def Protein_gene2_increase(state):
    
    state = state.copy()
    
    trans_rate = (state[RNAs_gene2])*rate.k3_2
    state[proteins_gene2] +=1
    new_state = state
    return [trans_rate, new_state]

def Protein_gene2_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[proteins_gene2])*rate.k4_2
    state[proteins_gene2] -=1
    new_state = state
    return [trans_rate, new_state]




def gene1_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[active_gene1]+state[inactive_gene1])*rate.k5_1
    state[active_gene1] = 0
    state[inactive_gene1] = 0
    new_state = state
    return [trans_rate, new_state]

def gene2_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[active_gene2]+state[inactive_gene2])*rate.k5_2
    state[active_gene2] = 0
    state[inactive_gene2] = 0
    new_state = state
    return [trans_rate, new_state]




transitions = [gene1_activate, gene2_activate, gene1_inactivate, gene2_inactivate,
               RNA_gene1_increase, RNA_gene2_increase, RNA_gene1_degrade, RNA_gene2_degrade, 
               Protein_gene1_increase, Protein_gene2_increase, Protein_gene1_degrade, Protein_gene2_degrade,
               gene1_degrade, gene2_degrade]



class Transition(Enum):
    """Define all possible transitions"""
    GENE1_ACTIVATE = 'gene1 activate'
    GENE2_ACTIVATE = 'gene2 activate'
    GENE1_INACTIVATE = 'gene1 inactivate'
    GENE2_INACTIVATE = 'gene2 inactivate'
    RNA_gene1_INCREASE = 'RNA gene1 increase'
    RNA_gene1_DEGRADE = 'RNA gene1 degrade'
    PROTEIN_gene1_INCREASE = 'Protein gene1 increase'
    PROTEIN_gene1_DEGRADE = 'Protein gene1 degrade'
    RNA_gene2_INCREASE = 'RNA gene2 increase'
    RNA_gene2_DEGRADE = 'RNA gene2 degrade'
    PROTEIN_gene2_INCREASE = 'Protein gene2 increase'
    PROTEIN_gene2_DEGRADE = 'Protein gene2 degrade'
    GENE1_DEGRADE = 'gene1 degrade'
    GENE2_DEGRADE = 'gene2 degrade'
    ABSORPTION = 'Absorption'



transition_names = [Transition.GENE1_ACTIVATE, Transition.GENE2_ACTIVATE,  Transition.GENE1_INACTIVATE, Transition.GENE2_INACTIVATE,
                    Transition.RNA_gene1_INCREASE, Transition.RNA_gene2_INCREASE, Transition.RNA_gene1_DEGRADE, Transition.RNA_gene2_DEGRADE,
                    Transition.PROTEIN_gene1_INCREASE, Transition.PROTEIN_gene2_INCREASE, Transition.PROTEIN_gene1_DEGRADE, Transition.PROTEIN_gene2_DEGRADE,
                    Transition.GENE1_DEGRADE, Transition.GENE2_DEGRADE]




class Observation(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition: Transition
    rates: typing.Any

class Index(IntEnum):
    state = 0
    time_of_observation = 1
    time_of_residency = 2
    transition = 3

    
class index(IntEnum):
    trans_rate = 0
    updated_state = 1

#%%

def gillespie_ssa(starting_state):
    
    state = starting_state 
        
    transition_results = [f(state) for f in transitions]
    
    new_states = []

    for i in np.arange(0, len(transitions)):    
        new_states.append(transition_results[i][index.updated_state])
        
    dict_newstates = {k:v for k, v in zip(transition_names, new_states)}
    
    dict_newstates[Transition.ABSORPTION] = np.array([0,0,0,0,0,0,0,0])
    
    rates = []

    for i in np.arange(0, len(transitions)):    
        rates.append(transition_results[i][index.trans_rate])
    
    total_rate = np.sum(rates)
    
    if total_rate > 0:
        
        time = np.random.exponential(1/total_rate)
        
        rates_array = np.array(rates)

        rates_array /= rates_array.sum()
    
        event = np.random.choice(transition_names, p=rates_array)
    
    else:
        
        time = np.inf
        
        event = Transition.ABSORPTION
    
    updated_state = dict_newstates[event]
    
    gillespie_result = [starting_state, updated_state, time, event, rates]
    
    return gillespie_result



def evolution(starting_state, starting_total_time, time_limit, seed_number):
    
    observed_states = []
    
    state = starting_state
    
    total_time = starting_total_time
    
    np.random.seed(seed_number)

    while total_time < time_limit:
        
        gillespie_result = gillespie_ssa(starting_state = state)
        
        rates = gillespie_result[4]
        
        event = gillespie_result[3]
        
        time = gillespie_result[2]
        
        observation_state = gillespie_result[0] #starting state
            
        
        
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
    
    if args.verbose:
        
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
    
    state = last_state
    
    transition_results = [f(state) for f in transitions]
    
    new_states = []

    for i in np.arange(0, len(transitions)):    
        new_states.append(transition_results[i][index.updated_state])
    
    dict_newstates = {k:v for k, v in zip(transition_names, new_states)}
    
    updated_state = dict_newstates[last_event]
    
    state = updated_state
    
    added_simulation_results = evolution(starting_state = state, starting_total_time = simulation_results[-1][Index.time_of_observation] + simulation_results[-1][Index.time_of_residency], time_limit = args.time_limit, seed_number = seed_number)




#%%

def create_dataframe(results):
    """ This function creates a dataframe with 4 columns:
        time of observation, gene activity, number of RNA molecules
        and number of proteins.          
    """
    time_of_observation = []
    number_of_RNAs_gene1 = []
    number_of_RNAs_gene2 = []
    number_of_proteins_gene1 = []
    number_of_proteins_gene2 = []
    number_of_active_genes1 = []
    number_of_active_genes2 = []



    for observation in results:
        time_of_observation.append(observation.time_of_observation)
        number_of_RNAs_gene1.append(observation.state[RNAs_gene1])
        number_of_RNAs_gene2.append(observation.state[RNAs_gene2])
        number_of_proteins_gene1.append(observation.state[proteins_gene1])
        number_of_proteins_gene2.append(observation.state[proteins_gene2])
        number_of_active_genes1.append(observation.state[active_gene1])
        number_of_active_genes2.append(observation.state[active_gene2])

    d = {'Time': time_of_observation, 
         'Number of RNAs gene1': number_of_RNAs_gene1, 
         'Number of RNAs gene2': number_of_RNAs_gene2, 
         'Number of proteins gene1': number_of_proteins_gene1,
         'Number of proteins gene2': number_of_proteins_gene2,
         'Gene1 activity': number_of_active_genes1,
         'Gene2 activity': number_of_active_genes2}
    
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

    df.to_csv(file_path.format(actual_dir,"toggleswitch_gillespiesimulation_results"), sep =" ", index = None, header=True, mode = "w") 

if args.time_limit:
    
    df.to_csv(file_path.format(actual_dir,"added_gillespiesimulation_toggleswitch_results"), sep =" ", index = None, header=True, mode = "w") 



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
        results_names.append("gillespieresults_toggleswitch_seed"+str(n))
    
    for dataframe, results in zip(dataframes_list, results_names):
        dataframe.to_csv(file_path.format(actual_dir,results), sep=" ", index = None, header=True)



save_multiplesimulations_results(N)




print(" ")
print("My job is done. Enjoy data analysis !")
