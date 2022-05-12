# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:57:20 2022

@author: asus
"""

import numpy as np
import random as rn
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import typing 
from enum import Enum

from collections import Counter, namedtuple

import scipy.stats as st

# Define initial states

gene_state = 'inactive'
RNA_Protein_state = np.array([0,0]) # RNA, protein


# Define transitions and transition rates

class Transition(Enum):
    GENE_ACTIVATE = 'gene activate'
    GENE_INACTIVATE = 'gene inactivate'
    RNA_INCREASE = 'RNA increase'
    RNA_DEGRADE = 'RNA degrade'
    PROTEIN_INCREASE = 'Protein increase'
    PROTEIN_DEGRADE = 'Protein degrade'

def gene_activate(state):
    return 1 
def gene_inactivate(state):
    return 0.5 

def RNA_increase(state):
    return 1
def RNA_degrade(state):
    return state[0]*0.1
k2_value = 'm*0.1'
def Protein_increase(state):
    return state[0]*1
k3_value = 'm*1'
def Protein_degrade(state):
    return state[1]*1
k4_value = 'p*1'

active_gene_transitions = [gene_inactivate, RNA_increase, 
                           RNA_degrade, Protein_increase, 
                           Protein_degrade]

active_gene_transition_names = [Transition.GENE_INACTIVATE, 
                                Transition.RNA_INCREASE, 
                                Transition.RNA_DEGRADE, 
                                Transition.PROTEIN_INCREASE, 
                                Transition.PROTEIN_DEGRADE]

inactive_gene_transitions = [gene_activate, 
                             RNA_degrade, 
                             Protein_increase, 
                             Protein_degrade]

inactive_gene_transition_names = [Transition.GENE_ACTIVATE,
                                  Transition.RNA_DEGRADE, 
                                  Transition.PROTEIN_INCREASE, 
                                  Transition.PROTEIN_DEGRADE]

 

class Observation(typing.NamedTuple):
    """ typing.NamedTuple class storing information
    for each event in the simulation"""
    gene_state: typing.Any
    RNA_Protein_state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition: Transition


def update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state):
    """This method updates the initial state according to the event occured
    
    Parameters
    ----------
    active_gene_event : Transition class member
                        Transition that occurs when gene state is active.
    inactive_gene_event: Transition class member
                         Transition that occurs when gene state is inactive.
    gene_state: str
                gene state that can be active or inactive
    RNA_Protein_state: ndarray
                       ndarray that has number of RNAs as first dimension 
                       and number of proteins as
                       second dimension.

    Returns
    -------
    updated_state : ndarray, shape (gene_state, RNA_Protein_state)
                    ndarray that has the gene state as first dimension and 
                    RNA and Protein number of molecules as second dimension
                    after a given event is occured.
    """
    if inactive_gene_event == Transition.GENE_ACTIVATE and gene_state == 'inactive':
        gene_state = 'active'
        gene_state = gene_state[:]
        RNA_Protein_state = RNA_Protein_state
        RNA_Protein_state = RNA_Protein_state.copy() 
    elif active_gene_event == Transition.RNA_INCREASE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[0] +=1
        RNA_Protein_state = RNA_Protein_state.copy()
    elif active_gene_event == Transition.RNA_DEGRADE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[0] -=1
        RNA_Protein_state = RNA_Protein_state.copy()
    elif active_gene_event == Transition.PROTEIN_INCREASE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] +=1
        RNA_Protein_state = RNA_Protein_state.copy()
    elif active_gene_event == Transition.PROTEIN_DEGRADE and gene_state == 'active':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] -=1
        RNA_Protein_state = RNA_Protein_state.copy()
    elif active_gene_event == Transition.GENE_INACTIVATE and gene_state == 'active':
        gene_state = 'inactive'
        gene_state = gene_state[:]
        RNA_Protein_state = RNA_Protein_state
        RNA_Protein_state = RNA_Protein_state.copy()
    elif inactive_gene_event == Transition.RNA_DEGRADE and gene_state == 'inactive':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[0] -=1
        RNA_Protein_state = RNA_Protein_state.copy()
    elif inactive_gene_event == Transition.PROTEIN_INCREASE and gene_state == 'inactive' :
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] +=1
        RNA_Protein_state = RNA_Protein_state.copy()
    elif inactive_gene_event == Transition.PROTEIN_DEGRADE and gene_state == 'inactive':
        gene_state = gene_state
        gene_state = gene_state[:]
        RNA_Protein_state[1] -=1
        RNA_Protein_state = RNA_Protein_state.copy()
        
    elif isinstance(inactive_gene_event,str) or isinstance(active_gene_event, str):
        raise TypeError("Do not use string ! Choose transitions from Transition enum members.")
    else:
        raise ValueError("Transition not recognized")
    
    updated_state = np.array([gene_state,RNA_Protein_state], dtype=object)
    return updated_state 






def simulation(starting_gene_state, starting_RNA_Protein_state, time_limit):
    """ This method simulates RNA and Protein populations from a regulated gene.
    
    Parameters
    
    starting_gene_state : starting state of gene.
    starting_RNA_Protein_state : initial number of RNAs and proteins.
    time_limit : simulation time limit.

    Returns:
        observed_states : list
                        list of named tuples each one storing 
                        gene state, number of RNAs and proteins,
                        total time spent, time of residency in that state,
                        type of event for each iteration of simulation.
    """
    observed_states = []
    gene_state = starting_gene_state
    RNA_Protein_state = starting_RNA_Protein_state
    total_time = 0.0
    
    while total_time < time_limit:
        
        active_gene_rates = [f(RNA_Protein_state) for f in active_gene_transitions]
        inactive_gene_rates = [h(RNA_Protein_state) for h in inactive_gene_transitions]
        
        active_gene_total_rate = np.sum(active_gene_rates)
        inactive_gene_total_rate = np.sum(inactive_gene_rates)
        
        active_gene_time = np.random.exponential(1/active_gene_total_rate)
        inactive_gene_time = np.random.exponential(1/inactive_gene_total_rate)
        
        active_gene_event = rn.choices(active_gene_transition_names, weights=active_gene_rates)[0]
        inactive_gene_event = rn.choices(inactive_gene_transition_names, weights=inactive_gene_rates)[0]
        
        
        if gene_state == 'active':
            event = active_gene_event
            time = active_gene_time
        elif gene_state == 'inactive':
            event = inactive_gene_event
            time = inactive_gene_time
        
        
        observation = Observation(gene_state, RNA_Protein_state, total_time, time, event)
        
        observed_states.append(observation)
        
        
        total_time += time
        
        updated_state = update_state(active_gene_event, inactive_gene_event, gene_state, RNA_Protein_state)
        gene_state = updated_state[0]
        RNA_Protein_state = updated_state[1]
    
    return observed_states


time_limit = 100
results = simulation(starting_gene_state = gene_state, starting_RNA_Protein_state = RNA_Protein_state, time_limit = time_limit)

results 






def generate_RNA_distribution(results):
    """ This method creates a Counter with state values 
    as keys and normalized residency time as values
    """
    RNA_distribution = Counter()
    for observation in results:
        state = observation.RNA_Protein_state[0]
        residency_time = observation.time_of_residency
        RNA_distribution[state] += residency_time/time_limit
        
    total_time_observed = sum(RNA_distribution.values())
    for state in RNA_distribution:
        RNA_distribution[state] /= total_time_observed

    return RNA_distribution 

def generate_protein_distribution(results):
    protein_distribution = Counter()
    for observation in results:
        state = observation.RNA_Protein_state[1]
        residency_time = observation.time_of_residency
        protein_distribution[state] += residency_time/time_limit
    
    total_time_observed = sum(protein_distribution.values())
    for state in protein_distribution:
        protein_distribution[state] /= total_time_observed

    return protein_distribution 


def DistributionPlot():
    """ This method plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(results)
    protein_distribution = generate_protein_distribution(results)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    values = np.arange(20)
    pmf = st.poisson(10).pmf(values) 
    ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
    ax[0].set_ylabel('Normalized residency time', fontsize=16)
    ax[0].set_xlabel('Number of RNA molecules', fontsize=16)
    ax[0].bar(values, pmf, alpha=0.5)
    ax[0].legend(["Simulation","Poisson distribution"], fontsize=16)
    ax[1].bar(protein_distribution.keys(), protein_distribution.values())
    ax[1].set_ylabel('Normalized residency time', fontsize=16)
    ax[1].set_xlabel('Number of proteins', fontsize=16)
    ax[1].bar(values, pmf, alpha=0.5)
 

DistributionPlot()


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
        number_of_RNA_molecules.append(observation.RNA_Protein_state[0])
        number_of_proteins.append(observation.RNA_Protein_state[1])
        if observation.gene_state == 'active':
            gene_activity.append(1)
        else:
            gene_activity.append(0)
    
    d = {'Time': time_of_observation, 
         'Gene activity': gene_activity,
         'Number of RNA molecules': number_of_RNA_molecules, 
         'Number of proteins': number_of_proteins}
    
    results_dataframe = pd.DataFrame(d)
    
    return results_dataframe





def MoleculesPlot():
    """This method plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time
    """

    gene_transitions = [gene_activate, gene_inactivate]
    gene_rates = [f(gene_state) for f in gene_transitions] # [ka, ki]  
    gene_rates_namedtuple = namedtuple("Rates",['ka','ki'])
    ka_value = gene_rates[0]
    ki_value = gene_rates[1]
    gene_rate = gene_rates_namedtuple(ka=ka_value,ki=ki_value)
    
    
    RNA_transitions = [RNA_increase,RNA_degrade]
    RNA_rates = [f(RNA_Protein_state) for f in RNA_transitions] # [k1, k2]
    RNA_rates_namedtuple = namedtuple("Rates",['k1','k2'])
    k1_value = RNA_rates[0]
    RNA_rate = RNA_rates_namedtuple(k1 = k1_value, k2 = k2_value)
    
    Protein_rates_namedtuple = namedtuple("Rates",['k3','k4']) 
    Protein_rate = Protein_rates_namedtuple(k3 = k3_value, k4 = k4_value)
    
    df = create_dataframe(results)
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    ax[0].plot(df['Time'], df['Gene activity'])
    ax[0].set_ylabel('Gene Activity')
    ax[0].set_xlabel('Time')
    ax[0].text(0.9,0.8,"$k_a$={}\n $k_i$={}".format(gene_rate.ka,gene_rate.ki), 
               ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[0].transAxes)
    ax[1].plot(df['Time'], df['Number of RNA molecules'])
    ax[1].set_ylabel('# of RNA molecules')
    ax[1].set_xlabel('Time')
    ax[1].text(0.9,0.8,"$k_1$={}\n $k_2$={}".format(RNA_rate.k1, RNA_rate.k2), 
               ha ='center', va = 'center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    ax[2].plot(df['Time'], df['Number of proteins'])
    ax[2].set_ylabel('# of proteins')
    ax[2].set_xlabel('Time')
    ax[2].text(0.9,0.8,"$k_3$={}\n $k_4$={}".format(Protein_rate.k3, Protein_rate.k4), 
               ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[2].transAxes)
    
    sns.despine(fig, trim=True, bottom=False, left=False)



MoleculesPlot()
