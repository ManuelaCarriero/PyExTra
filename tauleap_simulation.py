# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:05:49 2022

@author: asus
"""

import argparse
import configparser
import ast 
import sys

import numpy as np
import pandas as pd



import typing 
from enum import Enum

import os

from collections import namedtuple 
#from itertools import cycle
import time
import datetime

# get the start time
st_sec = time.time()

st_date = datetime.datetime.now()




config = configparser.ConfigParser()

parser = argparse.ArgumentParser()

parser.add_argument("filename", help="read configuration file.")

parser.add_argument('-run', help='run Tau-leap simulation given a configuration filename', action = "store_true")
parser.add_argument('-run_multiplesimulations', help='run a number of N Gillespie simulations given a configuration filename', action = "store_true")

parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

args = parser.parse_args()



config.read(args.filename)

if args.verbose:
    print("I am reading the configuration file {}".format(args.filename))

#config.read('configuration.txt')

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



transitions = [RNA_increase, RNA_degrade, 
               Protein_increase, Protein_degrade]




class Transition(Enum):
    """Define all possible transitions"""
    GENE_ACTIVATE = 'gene activate'
    GENE_INACTIVATE = 'gene inactivate'
    RNA_INCREASE = 'RNA increase'
    RNA_DEGRADE = 'RNA degrade'
    PROTEIN_INCREASE = 'Protein increase'
    PROTEIN_DEGRADE = 'Protein degrade'



transition_names = [Transition.RNA_INCREASE, Transition.RNA_DEGRADE, 
                    Transition.PROTEIN_INCREASE, Transition.PROTEIN_DEGRADE]



class Observation(typing.NamedTuple):
    """ typing.NamedTuple class storing information
    for each event in the simulation"""
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    #transition: Transition
    transition_rates: typing.Any
    n_reactions: typing.Any
    mean: typing.Any

    

#%%
def update_state(state, n_reactions):
    
    state[RNAs] += n_reactions[0] 
    
    state[RNAs] -= n_reactions[1]
    
    state[proteins] += n_reactions[2]
    
    state[proteins] -= n_reactions[3]
    
    updated_state = state
    
    return updated_state


def tauleap(starting_state, transitions):
    
    state = starting_state 
    
    rates = np.array([f(state) for f in transitions])
    
    mean = rates * dt
    
    n_reactions = np.random.poisson(mean) 

    state = state.copy()
    
    updated_state = update_state(state, n_reactions)
    
    tauleap_result = [starting_state, updated_state, dt, rates, n_reactions, mean]
    
    return tauleap_result



def evolution(time_limit, seed_number):
    
    observed_states = []
    
    state = starting_state
    
    total_time = 0.0
    
    np.random.seed(seed_number)

    while total_time < time_limit:
        
        tauleap_result = tauleap(starting_state = state, transitions = transitions)
        
        rates = tauleap_result[3]
        
        dt = tauleap_result[2]
        
        observation_state = tauleap_result[0]
        
        n_reactions = tauleap_result[4]
        
        mean = tauleap_result[5]
        
        
        observation = Observation(observation_state, total_time, dt, rates, n_reactions, mean)
        
        
        observed_states.append(observation)
        
        # Update time
        total_time += dt
        
        # Update starting state in tau leaping algorithm
        state = state.copy()
        
        state = tauleap_result[1]
        

        
    return observed_states


if args.run:
    
    simulation_results = evolution(time_limit = time_limit, seed_number = seed_number)




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
    
    df.to_csv(file_path.format(actual_dir,"tauleapsimulation_results"), sep =" ", index = None, header=True, mode = "w") 


if args.run_multiplesimulations:
    
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
            result = evolution(time_limit = time_limit, seed_number = n)
            results_list.append(result)
    
        dataframes_list = []
        for result in results_list:
            dataframe = create_dataframe(result)
            dataframes_list.append(dataframe)
        
        return dataframes_list
    
    
    
    dataframes_list = create_multiplesimulations_dataframes(N)
    
    
    
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
            results_names.append("tauleapresults_seed"+str(n))
        
        for dataframe, results in zip(dataframes_list, results_names):
            dataframe.to_csv(file_path.format(actual_dir, results), sep=" ", index = None, header=True)
    
    
    
    save_multiplesimulations_results(N)

print("My job is done. Enjoy data analysis !")
# get the end time
et_sec = time.time()

et_date = datetime.datetime.now()

# get the execution time
elapsed_time_sec = et_sec - st_sec

elapsed_time_date = et_date - st_date

print(" ")
print('Execution time: {}s ({})'.format(elapsed_time_sec, elapsed_time_date))