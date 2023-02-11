# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:39:34 2023

@author: asus
"""
import argparse
import configparser
import ast 
#import sys

import numpy as np
import pandas as pd

import math 

import typing 
from enum import Enum, IntEnum

from collections import namedtuple, Counter

import json
import jsonlines

import os
#from itertools import cycle
import time
import datetime
import math

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
               Protein_increase, Protein_degrade, gene_degrade]

gene_transitions = [gene_activate, gene_inactivate, gene_degrade]

tauleap_transitions = [RNA_increase, RNA_degrade, Protein_increase, Protein_degrade]



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
                    Transition.PROTEIN_INCREASE, Transition.PROTEIN_DEGRADE, Transition.GENE_DEGRADE]

gene_transition_names = [Transition.GENE_ACTIVATE, Transition.GENE_INACTIVATE, Transition.GENE_DEGRADE]

tauleap_transition_names = [Transition.RNA_INCREASE, Transition.RNA_DEGRADE, Transition.PROTEIN_INCREASE, Transition.PROTEIN_DEGRADE]



# Give to each one a different name to identify them better if debugging.
 
class ObservationI(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition_rates: typing.Any
    transition: Transition
    
class ObservationII(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition_rates: typing.Any
    transition: Transition
    
class ObservationIII(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition_rates: typing.Any
    transition: Transition
    rates_diff: typing.Any
    
class ObservationIV(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition_rates: typing.Any
    transition: Transition

class Observation_Hybrid(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    n_reactions: typing.Any
    mean: float
    transition_rates: typing.Any
    transition: Transition
    rates_diff: typing.Any
    
class Observation_SubHybridI(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    n_reactions: typing.Any
    mean: float
    transition_rates: typing.Any
    updated_state: typing.Any
    rates_diff: typing.Any

class Observation_SubHybridII(typing.NamedTuple):
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    n_reactions: typing.Any
    mean: float
    transition_rates: typing.Any
    updated_state: typing.Any
    rates_diff: typing.Any
    
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



def gillespie_ssa(starting_state, new_rates, rnchoice_transition_names):
    
    state = starting_state 
        
    transition_results = [f(state) for f in transitions]
    
    new_states = [transition_results[i][index.updated_state] for i in np.arange(0,len(transitions))]    
    
    dict_newstates = {k:v for k, v in zip(transition_names, new_states)}
    
    dict_newstates[Transition.ABSORPTION] = np.array([0,0,0,0,0,0,0,0])
    
    
    
    
    total_rate = np.sum(new_rates)
    

        
    time = np.random.exponential(1/total_rate)
    
    #transition_results = [f(state) for f in transitions]

    #new_rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]
    
    rates_array = np.array(new_rates)

    rates_array /= rates_array.sum()
    
    #If for some reason you run into these conditions,
    #you can use the magic command %debug and access to
    #the values of the variables to understand what is happening.
    
    if len(rnchoice_transition_names) != len(rates_array):
        raise ValueError("'a' and 'p' must have same size")
        
    if any(prob < 0 for prob in rates_array):
        raise ValueError("probabilities are not non-negative")
    

    event = np.random.choice(rnchoice_transition_names, p=rates_array)

        
    updated_state = dict_newstates[event]
    
    gillespie_result = [starting_state, updated_state, time, event, new_rates]
    
    return gillespie_result



def update_state(state, n_reactions):
    
    state[RNAs] += n_reactions[0] 
    
    state[RNAs] -= n_reactions[1]
    
    state[proteins] += n_reactions[2]
    
    state[proteins] -= n_reactions[3]
    
    updated_state = state
    
    return updated_state



def tauleap(starting_state, time):
    
    state = starting_state 
    
    transition_results = np.array([f(state) for f in tauleap_transitions], dtype=object)
    
    rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]    
    
    rates = np.array(rates) #se non lo converti in array dà errore TypeError: can't multiply sequence by non-int of type 'float'?
    
    mean = rates * time
    
    n_reactions = np.random.poisson(mean)

    state = state.copy()
    
    updated_state = update_state(state, n_reactions)
    
    tauleap_result = [starting_state, updated_state, rates, n_reactions, mean]
    
    return tauleap_result




def tauleap_mean(starting_state, time):
    
    state = starting_state 
    
    transition_results = np.array([f(state) for f in tauleap_transitions], dtype=object)
    
    rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]    
    
    rates = np.array(rates) #se non lo converti in array dà errore TypeError: can't multiply sequence by non-int of type 'float'?
    
    mean = rates * time
    
    return mean

def compute_rate(rates, new_state):
    
    transition_results = [f(new_state) for f in transitions]
    
    new_rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]
    
    sum_newrates = np.sum(new_rates)
    
    sum_oldrates = np.sum(rates)
    
    rates_diff = [np.abs((new_rate/sum_newrates)-(rate/sum_oldrates)) for new_rate,rate in zip(new_rates,rates)]

    return rates_diff
    
def halve_time(time):
    new_time = time/2
    return new_time





threshold=10



def evolution(starting_state, starting_total_time, time_limit, seed_number):
    
    observed_states = []
    
    state = starting_state
    
    total_time = starting_total_time
    
    np.random.seed(seed_number)
        
    gene_transition_results = [f(state) for f in gene_transitions]

    gene_rates = [gene_transition_results[i][index.trans_rate] for i in np.arange(0, len(gene_transition_results))]
    
    transition_results = [f(state) for f in transitions]
    
    rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]
    
    while total_time < time_limit:
        
        state = state.copy()
        
        state = state
        
        gene_rates = gene_rates.copy()
        
        gene_rates = gene_rates
        
        rates = rates.copy()
        
        rates = rates
            
        gillespie_result = gillespie_ssa(starting_state = state, new_rates = gene_rates, rnchoice_transition_names=gene_transition_names)
        
        gil_time = gillespie_result[2]
        
        updated_genestate = gillespie_result[1]
        
        
        
       
        tauleap_result = tauleap(starting_state = state, time = gil_time)
        
        updated_state = tauleap_result[1]
        
        updated_state = np.concatenate((updated_genestate[0:2],updated_state[2:4]),axis=None)
        
        rates_diff = compute_rate(rates=rates, new_state=updated_state)
    

            
        
        if  all(s >= 0 for s in updated_state) and all(rate_diff < threshold for rate_diff in rates_diff):
            
            transition = gillespie_result[3]

            n_reactions = tauleap_result[3]
            mean = tauleap_mean(starting_state=state, time = gil_time)

            
            
            observation_state = state
            updated_state = np.concatenate((updated_genestate[0:2],updated_state[2:4]), axis= None)

            
            
            
            observation = Observation_Hybrid(observation_state, total_time, gil_time, n_reactions, mean, rates, transition, rates_diff)
            

            
            observed_states.append(observation)
            
            #Update time
            total_time += gil_time
            
            #Update all rates, in particular gene rates for attempting an hybrid approach
                           
            RNAProteins_transition_results = [f(updated_state) for f in tauleap_transitions]
            
            RNAProtein_rates = [RNAProteins_transition_results[i][index.trans_rate] for i in np.arange(0, len(RNAProteins_transition_results))]
            
            gene_transition_results = [f(updated_state) for f in gene_transitions]
                       
            gene_rates = [gene_transition_results[i][index.trans_rate] for i in np.arange(0, len(gene_transition_results))]
            
            rates = np.concatenate((gene_rates[0:2], RNAProtein_rates, gene_rates[-1]),axis=None)

            
            #Update starting gene state and RNA Protein state 
            #in gillespie and tauleap algorithm

                
            gene_state = gillespie_result[1][0:2]
                
            RNAProtein_state = tauleap_result[1][2:4] 
            
            state = np.concatenate((gene_state, RNAProtein_state),axis=None)
            
            
        elif any(s < 0 for s in updated_state) or any(rate_diff >= threshold for rate_diff in rates_diff):
            


            

        
            
            

            
            time_count = 0
            
            new_time = gil_time
            
            
        
            transition_results = [f(state) for f in transitions]
            
            rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]

            gillespie_result = gillespie_ssa(starting_state = state, new_rates = rates, rnchoice_transition_names=transition_names)

            time_c = gillespie_result[2]
            
            
            
                
            while time_count < gil_time and new_time > 3*time_c:
                
                time = gil_time - time_count
        
                while any(s < 0 for s in updated_state) or any(rate_diff >= threshold for rate_diff in rates_diff): 

                    
                    new_time = halve_time(time=time)                                                                                       
        
                    tauleap_result= tauleap(starting_state = state, time = new_time)
                    
                    
                    
                    updated_state = tauleap_result[1]
                                                            
                    rates_diff = compute_rate(rates=rates, new_state=updated_state)
                        
    
                        
                            
  
                            
                
                n_reactions = tauleap_result[3]
                
                mean = tauleap_result[4]
            
                updated_state = np.concatenate((updated_genestate[0:2],updated_state[2:4]), axis= None)
                                        
                
                
                observation_state = state
                
                time = new_time
                
                
                observation = Observation_SubHybridI(observation_state, total_time, time, n_reactions, mean, rates, updated_state, rates_diff)
                

                
                observed_states.append(observation)
                
                #Update time
                time_count += new_time
                
                #Update starting rates in gillespie and tauleap algorithm
                
                RNAProteins_transition_results = [f(updated_state) for f in tauleap_transitions]
                
                RNAProtein_rates = [RNAProteins_transition_results[i][index.trans_rate] for i in np.arange(0, len(RNAProteins_transition_results))]
                
                rates = np.concatenate((gene_rates[0:2], RNAProtein_rates, gene_rates[-1]),axis=None)
                
                #Update starting gene state and RNA Protein state 
                #in gillespie and tauleap algorithm
    
                gene_state = gillespie_result[1][0:2]
                    
                RNAProtein_state = tauleap_result[1][2:4] 
                
                state = np.concatenate((gene_state, RNAProtein_state),axis=None)
                                   
                        

                    
                new_time = gil_time - time_count    
                    
                tauleap_result = tauleap(starting_state = state, time = gil_time - time_count)
                
                updated_state = tauleap_result[1]
                
                rates_diff = compute_rate(rates=rates, new_state=updated_state)

                if all(s >= 0 for s in updated_state) and all(rate_diff < threshold for rate_diff in rates_diff):
                
                    

                    n_reactions = tauleap_result[3]
                    
                    mean = tauleap_result[4]
                
                    updated_state = np.concatenate((updated_genestate[0:2],updated_state[2:4]), axis= None)
                                            
                    
                    
                    observation_state = state
                    
                    time = new_time
                    
                    
                    observation = Observation_SubHybridII(observation_state, total_time, time, n_reactions, mean, rates, updated_state, rates_diff)
                    

                    
                    observed_states.append(observation)
                    
                    #Update time
                    time_count += new_time
                    
                    #Update starting rates in gillespie and tauleap algorithm
                    
                    RNAProteins_transition_results = [f(updated_state) for f in tauleap_transitions]
                    
                    RNAProtein_rates = [RNAProteins_transition_results[i][index.trans_rate] for i in np.arange(0, len(RNAProteins_transition_results))]
                    
                    rates = np.concatenate((gene_rates[0:2], RNAProtein_rates, gene_rates[-1]),axis=None)
                    
                    
                    #Update starting gene state and RNA Protein state 
                    #in gillespie and tauleap algorithm
        
                    gene_state = gillespie_result[1][0:2]
                        
                    RNAProtein_state = tauleap_result[1][2:4] 
                    
                    state = np.concatenate((gene_state, RNAProtein_state),axis=None)
            
                if math.isclose(time_count, gil_time, rel_tol=0.5) or new_time < 3*time_c:
                    break
                        
               
            if math.isclose(time_count, gil_time, rel_tol=0.5):
                
                total_time += time_count
                
            else:
                
                transition_results = [f(state) for f in transitions]
                
                rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]

                gillespie_result = gillespie_ssa(starting_state = state, new_rates = rates, rnchoice_transition_names=transition_names)
                
                
                
                
                rates = gillespie_result[4]
                
                event = gillespie_result[3]
                
                time = gillespie_result[2]
                
                observation_state = gillespie_result[0]
                
                
                

                
                
                observation = ObservationIII(observation_state, total_time, time, rates, event, rates_diff)
                

                
                observed_states.append(observation)
                
                # Update time
                total_time += time
                
                # Update starting state in gillespie algorithm
                                
                state = state.copy()
                
                state = gillespie_result[1]
                
                # Update gene rates for attempting an hybrid step
                # and all rates in case of SSA
                
                transition_results = [f(state) for f in transitions]
                
                new_rates = [transition_results[i][index.trans_rate] for i in np.arange(0, len(transition_results))]

                
                #gene_rates = gene_rates.copy()                
                
                gene_rates = new_rates[0:2]
                
                gene_rate_degrade = new_rates[-1]
                
                RNAProtein_rates = new_rates[2:6]
                
                rates = np.concatenate((gene_rates, RNAProtein_rates, gene_rate_degrade),axis=None)
                
                gene_rates = np.concatenate((gene_rates, gene_rate_degrade),axis=None)
                
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
    
    with jsonlines.open('hybridsimulation_results.jsonl', mode='w') as writer:
        writer.write(simulation_results_)



 
if args.time_limit:
    
    with jsonlines.open('hybridsimulation_results.jsonl') as reader:
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
    number_of_RNAs = []
    number_of_proteins = []
    gene_activity = []
    residency_time = []




    for observation in results:
        time_of_observation.append(observation.time_of_observation)
        number_of_RNAs.append(observation.state[RNAs])
        number_of_proteins.append(observation.state[proteins])
        residency_time.append(observation.time_of_residency)
        if observation.state[active_genes] > 0:
            gene_activity.append(1)
        else:
            gene_activity.append(0)
        
    d = {'Time': time_of_observation, 
         'Number of RNA molecules': number_of_RNAs, 
         'Number of proteins': number_of_proteins,
         'Gene activity': gene_activity,
         'Residency Time': residency_time}
    
    results_dataframe = pd.DataFrame(d)
    
    return results_dataframe





if args.run:

    df = create_dataframe(results = simulation_results)
    if math.isinf(df.iloc[-1]['Residency Time']):
        df = df[:-1]

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

    df.to_csv(file_path.format(actual_dir,"hybridsimulation_results"), sep =" ", index = None, header=True, mode = "w") 

if args.time_limit:
    
    df.to_csv(file_path.format(actual_dir,"added_hybridsimulation_results"), sep =" ", index = None, header=True, mode = "w") 



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
            if math.isinf(dataframe.iloc[-1]['Residency Time']):
                dataframe = dataframe[:-1]
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
            results_names.append("hybridsimulation_results_seed"+str(n))
        
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

