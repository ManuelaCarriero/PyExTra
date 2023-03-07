# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:31:28 2023

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



# get the start time

st_sec = time.time()

st_date = datetime.datetime.now()



config = configparser.ConfigParser()

parser = argparse.ArgumentParser()

parser.add_argument("filename", help="read configuration file.")

parser.add_argument('-run', help='run Gillespie simulation given a configuration filename', action = "store_true")
parser.add_argument('-run_multiplesimulations', help='run a number of N Gillespie simulations given a configuration filename', action = "store_true")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--time_limit", help="increase time limit", metavar='value', type = float)

args = parser.parse_args()



config.read(args.filename)

#config.read('configuration_nfkb.txt')

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
    
    IKKa = index['ikka']
    IKKn = index['ikkn']
    nfkb_active = index['nfkb_active']
    nfkb_inactive = index['nfkb_inactive']
    IKalpha_active = index['ikalpha_active']
    IKalpha_inactive = index['ikalpha_inactive']
    RNAs = index['rnas']
    A20_active = index['a20_active']    
    A20_inactive = index['a20_inactive']
    
    return starting_state, IKKa, IKKn, nfkb_active, nfkb_inactive, IKalpha_active, IKalpha_inactive, RNAs, A20_active, A20_inactive




starting_state, IKKa, IKKn, nfkb_active, nfkb_inactive, IKalpha_active, IKalpha_inactive, RNAs, A20_active, A20_inactive = read_population()




def read_k_values():
    """This function reads k parameters from configuration file
    """
    
    k_value = dict(config["RATES"])
    
    for key,value in k_value.items():
        k_value[key] = float(value)
    
    rates = namedtuple("Rates",['ka', 'ki', 'k1', 'k1i', 'k4', 'k5', 'k2','k2i','k3','k3i'])
    rate = rates(ka = k_value['ka'], 
                 ki = k_value['ki'], 
                 k1 = k_value['k1'], 
                 k1i = k_value['k1i'], 
                 k4 = k_value['k4'], 
                 k5 = k_value['k5'], 
                 k2 = k_value['k2'],
                 k2i = k_value['k2i'], 
                 k3 = k_value['k3'], 
                 k3i = k_value['k3i'])
                 
                 
    
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

def IKK_activate(state):
    state = state.copy()
    
    trans_rate = state[IKKn]*rate.ka*(1/(1+state[A20_active]))
    
    state[IKKa] +=1
    state[IKKn] -=1
    
    new_state = state
    
    return [trans_rate, new_state]

def IKK_inactivate(state):
    state = state.copy()
    
    trans_rate = state[IKKa]*rate.ki*state[A20_active]
    
    state[IKKa] -=1
    state[IKKn] +=1
    
    new_state = state
    
    return [trans_rate, new_state]

def nfkb_activate(state):
    state = state.copy()
    
    trans_rate = state[IKKa]*rate.k1*state[nfkb_inactive]*(1/(1+state[IKalpha_active]))
    
    state[nfkb_active] +=1
    state[nfkb_inactive] -=1
    
    new_state = state
    
    return [trans_rate, new_state]

def nfkb_inactivate(state):
    state = state.copy()
    
    trans_rate = state[nfkb_active]*rate.k1i*state[IKalpha_active]
    
    state[nfkb_active] -=1
    state[nfkb_inactive] +=1
    
    new_state = state
    
    return [trans_rate, new_state]    

def RNA_increase(state):
    state = state.copy()
    
    trans_rate = state[nfkb_active]*rate.k4
    
    state[RNAs] +=1
    
    new_state = state
    
    return [trans_rate, new_state]

def RNA_degrade(state):
    state = state.copy()
    
    trans_rate = state[RNAs]*rate.k5
    
    state[RNAs] -=1
    
    new_state = state
    
    return [trans_rate, new_state]

def IKalpha_activate(state):
    state = state.copy()
    
    trans_rate = state[nfkb_active]*rate.k2*state[IKalpha_inactive]
    
    state[IKalpha_active] +=1
    state[IKalpha_inactive] -=1
    
    new_state = state
    
    return [trans_rate, new_state]


def IKalpha_inactivate(state):
    state = state.copy()
    
    trans_rate = state[IKKa]*rate.k2i*state[A20_active]*state[IKalpha_active]
    
    state[IKalpha_active] -=1
    state[IKalpha_inactive] +=1
    
    new_state = state
    
    return [trans_rate, new_state]


def A20_activate(state):
    state = state.copy()
    
    trans_rate = state[A20_inactive]*state[nfkb_active]*rate.k3
    
    state[A20_active] += 1
    state[A20_inactive] -= 1
    
    new_state = state
    
    return [trans_rate, new_state]

def A20_inactivate(state):
    state = state.copy()
    
    trans_rate = state[A20_active]*state[nfkb_inactive]*rate.k3i
    
    state[A20_active] -= 1
    state[A20_inactive] += 1
    
    new_state = state
    
    return [trans_rate, new_state]


transitions = [IKK_activate, IKK_inactivate, 
               nfkb_activate, nfkb_inactivate, 
               RNA_increase, RNA_degrade,
               IKalpha_activate, IKalpha_inactivate,
               A20_activate, A20_inactivate]



class Transition(Enum):
    """Define all possible transitions"""
    IKK_ACTIVATE = 'IKK activate'
    IKK_INACTIVATE = 'IKK inactivate'
    NFKB_ACTIVATE = 'NFKB activate'
    NFKB_INACTIVATE = 'NFKB inactivate'
    RNA_INCREASE = 'RNA increase'
    RNA_DEGRADE = 'RNA degrade'
    IKALPHA_ACTIVATE = 'Ikalpha activate'
    IKALPHA_INACTIVATE = 'Ikalpha inactivate'
    A20_ACTIVATE = 'A20 activate'
    A20_INACTIVATE = 'A20 inactivate'
    #ABSORPTION = 'Absorption'



transition_names = [Transition.IKK_ACTIVATE, Transition.IKK_INACTIVATE, 
                    Transition.NFKB_ACTIVATE, Transition.NFKB_INACTIVATE, 
                    Transition.RNA_INCREASE, Transition.RNA_DEGRADE,
                    Transition.IKALPHA_ACTIVATE, Transition.IKALPHA_INACTIVATE,
                    Transition.A20_ACTIVATE, Transition.A20_INACTIVATE]



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
    
    #new_states = []

    #for i in np.arange(0, len(transitions)):    
     #   new_states.append(transition_results[i][index.updated_state])
    
    new_states = [transition_results[i][index.updated_state] for i in np.arange(0, len(transitions))]
    
    dict_newstates = {k:v for k, v in zip(transition_names, new_states)}
    
    #dict_newstates[Transition.ABSORPTION] = np.array([0,0,0,0,0,0,0,0])
    
    #rates = []

    #for i in np.arange(0, len(transitions)):    
    #    rates.append(transition_results[i][index.trans_rate])
    
    rates = [transition_results[i][index.trans_rate] for i in np.arange(0,len(transitions))]
    
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
        
        if any(s<0 for s in state):
            return observed_states
        
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


#simulation_results = evolution(starting_state = starting_state, starting_total_time = 0.0, time_limit = time_limit, seed_number = seed_number)
#simulation_results[-1]


#simulation_results = evolution(starting_state = starting_state, starting_total_time = 0.0, time_limit = time_limit, seed_number = seed_number)
#simulation_results[1]
#nf_kb=[]
#for result in simulation_results:
#    if result.transition == Transition.NFKB_INACTIVATE:
#        nf_kb.append(result.transition)
#len(nf_kb) #ACTIVATE 31
#len(nf_kb) #INACTIVATE 30
#LE ALTRE SONO ALTRI TIPI DI REAZIONE.

#OLTRE ALL'RNA PRODOTTO PUOI VEDERE ANCHE QUANTE VOLTE SI è ATTIVATO L'NFKB.

#PROVA AD USARE DELLE EQUAZIONI DI CINETICA PIù SIMILI ALL'AUTOREPRESSORE.
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
    number_of_RNA_molecules = []
    #number_of_proteins = []
    nfkb_activity = []
    #residency_time = []

    for observation in results:
        time_of_observation.append(observation.time_of_observation)
        number_of_RNA_molecules.append(observation.state[RNAs])
        #number_of_proteins.append(observation.state[proteins])
        #residency_time.append(observation.time_of_residency)
        if observation.state[nfkb_active] > 0:
            nfkb_activity.append(1)

        else:
            nfkb_activity.append(0)

    
    d = {'Time': time_of_observation, 
         'Number of RNA molecules': number_of_RNA_molecules,
         'NFkB activity': nfkb_activity}
    
    results_dataframe = pd.DataFrame(d)
    
    return results_dataframe

#df = create_dataframe(results = simulation_results)




#df = df.iloc[14:]

#import matplotlib.pylab as plt

#import seaborn as sns

#plt.plot(df['Time'],df['Number of RNA molecules'])
#plt.plot(df['Time'],df['NFkB activity'])
#plt.xlabel('Time')
#plt.ylabel('# of molecules')
#sns.despine(bottom=False, left=False)

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

#df.to_csv(file_path.format(actual_dir,"gillespiesimulation_results"), sep =" ", index = None, header=True, mode = "w") 

    
if args.run:

    df.to_csv(file_path.format(actual_dir,"nfkb_gillespiesimulation_results"), sep =" ", index = None, header=True, mode = "w") 

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
            results_names.append("nfkb_gillespieresults_seed"+str(n))
        
        for dataframe, results in zip(dataframes_list, results_names):
            dataframe.to_csv(file_path.format(actual_dir,results), sep=" ", index = None, header=True)
    
    
    
    save_multiplesimulations_results(N)



print(" ")    
print("My job is done. Enjoy data analysis !")

# get the end time
et_sec = time.time()

et_date = datetime.datetime.now()

# get the execution time
elapsed_time_sec = et_sec - st_sec

elapsed_time_date = et_date - st_date

print(" ")
print('Execution time: {}s ({})'.format(elapsed_time_sec, elapsed_time_date))
