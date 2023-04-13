# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:01:24 2022

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

import random

from statsmodels.tsa import stattools
import scipy.interpolate as interp
import scipy.stats

import os
#from itertools import cycle
import time
import datetime

# get the start time
#start = time.time()
start_date = datetime.datetime.now()

config = configparser.ConfigParser()
"""
parser = argparse.ArgumentParser()

parser.add_argument("filename", help="read configuration file.")

parser.add_argument('-run', help='run Gillespie simulation given a configuration filename', action = "store_true")
parser.add_argument('-run_multiplesimulations', help='run a number of N Gillespie simulations given a configuration filename', action = "store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--time_limit", help="increase time limit", metavar='value', type = float)

args = parser.parse_args()
"""


config.read('configuration1.txt')



"""
if args.verbose:
    print("I am reading the configuration file {}".format(args.filename))
"""

"""
n=0
lst = [0.1]
while n < 24:
    for j in np.arange(0,24):
        i = lst[j]*0.1
        lst.append(i)
        n +=1
len(lst)
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




def gene_activate(state):
    
    state = state.copy()
    
    #Transition rate with this starting state
    trans_rate = (state[inactive_genes])*(1/(1+state[proteins]))*rate.ka
    #Update state in case this transition is chosen
    state[active_genes] +=1
    state[inactive_genes] -=1
    
    new_state = state
    return [trans_rate, new_state]

def gene_inactivate(state):
    
    state = state.copy()
    
    trans_rate = (state[active_genes])*rate.ki
    state[active_genes] -=1
    state[inactive_genes] +=1

    new_state = state
    return [trans_rate, new_state]





def RNA_increase(state):
    
    state = state.copy()
    
    trans_rate = (state[active_genes])*rate.k1
    state[RNAs] +=1
    new_state = state
    return [trans_rate, new_state]

def RNA_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[RNAs])*rate.k2
    state[RNAs] -=1
    new_state = state    
    return [trans_rate, new_state]



def Protein_increase(state):
    
    state = state.copy()
    
    trans_rate = (state[RNAs])*rate.k3
    state[proteins] +=1
    new_state = state
    return [trans_rate, new_state]

def Protein_degrade(state):
    
    state = state.copy()
    
    trans_rate = (state[proteins])*rate.k4
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








class Observation(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition: Transition
    rates: typing.Any


    
class index(IntEnum):
    trans_rate = 0
    updated_state = 1

#%%

def gillespie_ssa(starting_state):    
    
    state = starting_state 
    
    transitions = [gene_activate, gene_inactivate, 
                   RNA_increase, RNA_degrade, 
                   Protein_increase, Protein_degrade,
                   gene_degrade]
    
    
        
    transition_results = [f(state) for f in transitions]
    
    new_states = []

    for i in np.arange(0, len(transitions)):    
        new_states.append(transition_results[i][index.updated_state])
        
    transition_names = [Transition.GENE_ACTIVATE, Transition.GENE_INACTIVATE,
                        Transition.RNA_INCREASE, Transition.RNA_DEGRADE,
                        Transition.PROTEIN_INCREASE, Transition.PROTEIN_DEGRADE,
                        Transition.GENE_DEGRADE]
        
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





#Stiamo parlando dell'autorepressore quindi tutti i k influenzano
#l'andamento nel tempo del numero di RNA prodotte.
columns = ['acfs','k']

df_tot = pd.DataFrame(columns = columns)




actual_dir = os.getcwd()

file_path = r'{}\{}.csv'


#prova a mettere i parametri che cambi nel file di configurazione pari a 1.
a=3


np.random.seed(42)
rands=[]
for i in np.arange(1,1001):
    rand = scipy.stats.gamma.rvs(a, loc=0, scale=0.1, size=4, random_state=None)
    rand = rand.tolist()
    #[round(items,1) for items in rand]
    rands.append(rand)
    
    

for i in np.arange(0,999):   
    
    n_k1 = rands[i][0]
    n_k2 = rands[i][1]
    n_k3 = rands[i][2]
    n_k4 = rands[i][3]
    
       
    def read_k_values():
        """This function reads k parameters from configuration file
        """
        
        k_value = dict(config["RATES"])
        
        for key, value in k_value.items():
            
            if key == 'k1':
                k_value[key] = float(value)*n_k1
            elif key == 'k2':
                k_value[key] = float(value)*n_k2
            elif key == 'k3':
                k_value[key] = float(value)*n_k3
            elif key == 'k4':
                k_value[key] = float(value)*n_k4
            else:
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
    
    #%%

    simulation_results = evolution(starting_state = starting_state, starting_total_time = 0.0, time_limit = time_limit, seed_number = seed_number)
     

    
    
    
    #%%

    df = create_dataframe(results = simulation_results)
     
    dt=0.01
    nlags = 5000#20000
    
    time = df['Time']
    
    xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
    
    n_RNAs = df['Number of RNA molecules']
    f_RNAs = interp.interp1d(time, n_RNAs, kind='previous')
    yinterp_RNAs = f_RNAs(xvals)
    autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) 
    autocorr_RNAs = autocorr_RNAs.tolist()
    

    
     
    
    
    """
    n_proteins = df['Number of proteins']
    f_proteins = interp.interp1d(time, n_proteins, kind='previous')
    yinterp_proteins = f_proteins(xvals)
    autocorr_proteins = stattools.acf(yinterp_proteins, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
    autocorr_proteins = autocorr_proteins.tolist()
    
    autocorr = autocorr_RNAs + autocorr_proteins
    
    #autocorrs = [autocorr_RNAs,autocorr_proteins] 
    """
    rates = list(rate)
    
    df_tot.loc[i] = [autocorr_RNAs,rates]
    print(i)
                
                
    
    
    
df_tot.to_csv(file_path.format(actual_dir,"autorepressor_1000RNASacfs_seed42_scale0.1_nlags5000_k1k2k3k4"), sep =" ", index = None, header=True, mode = "w") 

    



print(" ")
print("My job is done. Autocorrelation values are now ready for MLP!")

# get the end time
#end = time.time()

end_date = datetime.datetime.now()

# get the execution time
#elapsed_time = end - start

elapsed_time_date = end_date - start_date

#print(" ")
#print('Execution time:', elapsed_time, 'seconds')

print(" ")
print('Execution time:', elapsed_time_date, 'seconds')

#Execution time: 1:57:42.144237 seconds 1000 simulazioni nlags 5000.
#Execution time: 3:21:08.792660 seconds 1000 simulazioni nlags 5000 autocorrelation RNAS + PROTEINS. 