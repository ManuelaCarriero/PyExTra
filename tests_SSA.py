# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:38:56 2023

@author: asus
"""

#from ssa_simulation import evolution

# Non posso farlo anche perché dipende da altre funzioni

# E poi se lo faccio dà questo errore


#usage:  [-h] [-run] [-run_multiplesimulations] [-v] [--time_limit value]
#        filename
#: error: the following arguments are required: filename
#An exception has occurred, use %tb to see the full traceback.

#SystemExit: 2

#C:\Users\asus\anaconda3\lib\site-packages\IPython\core\interactiveshell.py:3452: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
#  warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)

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
import datetime
import time





import pytest
import hypothesis
from hypothesis import given
import hypothesis.strategies as st







config = configparser.ConfigParser()
"""
parser = argparse.ArgumentParser()

parser.add_argument("filename", help="read configuration file.")

parser.add_argument('-run', help='run Gillespie simulation given a configuration filename', action = "store_true")
parser.add_argument('-run_multiplesimulations', help='run a number of N Gillespie simulations given a configuration filename', action = "store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--time_limit", help="increase time limit", metavar='value', type = float)

args = parser.parse_args()



config.read(args.filename)
"""
config.read('configuration.txt')
"""
if args.verbose:
    print("I am reading the configuration file {}".format(args.filename))
"""


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

def gene_activate(state):
    state = state.copy()
    
    trans_rate = state[inactive_genes]*rate.ka
    
    state[active_genes] +=1
    state[inactive_genes] -=1
    
    new_state = state
    
    return [trans_rate, new_state]

def gene_inactivate(state):
    state = state.copy()
    
    trans_rate = state[active_genes]*rate.ki
    
    state[active_genes] -=1
    state[inactive_genes] +=1
    
    new_state = state
    
    return [trans_rate, new_state]

def RNA_increase(state):
    state = state.copy()
    
    trans_rate = state[active_genes]*rate.k1
    
    state[RNAs] +=1
    
    new_state = state
    
    return [trans_rate, new_state]
    

def RNA_degrade(state):
    state = state.copy()
    
    trans_rate = state[RNAs]*rate.k2
    
    state[RNAs] -=1
    
    new_state = state
    
    return [trans_rate, new_state]


def Protein_increase(state):
    state = state.copy()
    
    trans_rate = state[RNAs]*rate.k3
    
    state[proteins] +=1
    
    new_state = state
    
    return [trans_rate, new_state]


def Protein_degrade(state):
    state = state.copy()
    
    trans_rate = state[proteins]*rate.k4
    
    state[proteins] -=1
    
    new_state = state
    
    return [trans_rate, new_state]


def gene_degrade(state):
    state = state.copy()
    
    trans_rate = (state[active_genes]+state[inactive_genes])*rate.k5
    
    state[active_genes] = 0
    state[inactive_genes] = 0
    
    new_state = state
    
    return [trans_rate, new_state]



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
    
class index(IntEnum):
    trans_rate = 0
    updated_state = 1

#%%

def gillespie_ssa(starting_state, transitions):
    
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





simulation_results = evolution(starting_state = starting_state, starting_total_time = 0.0, time_limit = time_limit, seed_number = seed_number)
simulation_results[-1]


#%%Tests

def test_no_increase_RNA_if_gene_is_inactive():
    """
    Test that there is no RNA molecules increase if gene is inactive.
    """
    for simulation in simulation_results:
        if simulation.state[inactive_genes] == 1:
            assert simulation.transition != Transition.RNA_INCREASE

def test_no_increase_Protein_if_nRNAs_are_zero():
    """
    Test that the number of proteins do not increase if the
    number of molecules is zero.
    """
    for simulation in simulation_results:
        if simulation.state[RNAs] == 0:
            assert simulation.transition != Transition.PROTEIN_INCREASE

def test_there_are_no_negative_number_of_molecules():
    """
    Test that there are not negative number of molecules
    """
    for simulation in simulation_results:
        assert simulation.state[active_genes] >= 0
        assert simulation.state[inactive_genes] >= 0
        assert simulation.state[RNAs] >= 0
        assert simulation.state[proteins] >= 0

#For configuration ka=0.01 ki=0.01 (the distribution of states is plotted with 
#time of residency on that state on the y axis)

def test_there_are_states_with_inactive_gene():
    inactivegenes_lst=[]
    for simulation in simulation_results:
        if simulation.state[inactive_genes] == 1:
            inactivegenes_lst.append(simulation)
    assert len(inactivegenes_lst) != 0



def test_there_are_states_with_zero_number_of_RNA_molecules():
    zeroRNAs_lst=[]
    for simulation in simulation_results:
        if simulation.state[RNAs] == 0:
            zeroRNAs_lst.append(simulation)
    assert len(zeroRNAs_lst) != 0            




 
    
    
