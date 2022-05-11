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

from collections import Counter

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
    return 1 

def RNA_increase(state):
    return 1
def RNA_degrade(state):
    return state[0]*0.1
def Protein_increase(state):
    return 1
def Protein_degrade(state):
    return state[1]*0.1

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

# Define Observation class  

class Observation(typing.NamedTuple):
    gene_state: typing.Any
    RNA_Protein_state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition: Transition

def simulation(starting_gene_state, starting_RNA_Protein_state, time_limit):
    """ This method simulates RNA and Protein populations from a regulated gene.
    
    Parameters
    
    starting_state : initial number of DNA molecules, RNA molecules and proteins.
    time_limit : simulation time limit.

    Returns:
        updated number of RNA molecules, 
        total time spent, 
        time of residency in each state,
        the type of event regarding the gene
        the type of event regarding the RNA and protein molecules.
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
        
        
        
        # Update state
        
        
        
        if inactive_gene_event == Transition.GENE_ACTIVATE and gene_state == 'inactive':
            gene_state = 'active'
            gene_state = gene_state[:]
            RNA_Protein_state = RNA_Protein_state
            event = active_gene_event
            time = active_gene_time
            RNA_Protein_state = RNA_Protein_state.copy() 
        elif active_gene_event == Transition.RNA_INCREASE and gene_state == 'active':
            gene_state = gene_state
            gene_state = gene_state[:]
            RNA_Protein_state[0] +=1
            event = active_gene_event
            time = active_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        elif active_gene_event == Transition.RNA_DEGRADE and gene_state == 'active':
            gene_state = gene_state
            gene_state = gene_state[:]
            RNA_Protein_state[0] -=1
            event = active_gene_event
            time = active_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        elif active_gene_event == Transition.PROTEIN_INCREASE and gene_state == 'active':
            gene_state = gene_state
            gene_state = gene_state[:]
            RNA_Protein_state[1] +=1
            event = active_gene_event
            time = active_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        elif active_gene_event == Transition.PROTEIN_DEGRADE and gene_state == 'active':
            gene_state = gene_state
            gene_state = gene_state[:]
            RNA_Protein_state[1] -=1
            event = active_gene_event
            time = active_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        elif active_gene_event == Transition.GENE_INACTIVATE and gene_state == 'active':
            gene_state = 'inactive'
            gene_state = gene_state[:]
            RNA_Protein_state = RNA_Protein_state
            event = active_gene_event
            time = active_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        elif inactive_gene_event == Transition.RNA_DEGRADE and gene_state == 'inactive':
            gene_state = gene_state
            gene_state = gene_state[:]
            RNA_Protein_state[0] -=1
            event = inactive_gene_event
            time = inactive_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        elif inactive_gene_event == Transition.PROTEIN_INCREASE and gene_state == 'inactive' :
            gene_state = gene_state
            gene_state = gene_state[:]
            RNA_Protein_state[1] +=1
            event = inactive_gene_event
            time = inactive_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        elif inactive_gene_event == Transition.PROTEIN_DEGRADE and gene_state == 'inactive':
            gene_state = gene_state
            gene_state = gene_state[:]
            RNA_Protein_state[1] -=1
            event = inactive_gene_event
            time = inactive_gene_time
            RNA_Protein_state = RNA_Protein_state.copy()
        else:
            raise ValueError("transition not recognized")
        
        
        
        
    return observed_states


time_limit = 1000
results = simulation(starting_gene_state = gene_state, starting_RNA_Protein_state = RNA_Protein_state, time_limit = time_limit)


# Plot





def generate_RNA_distribution(results):
    """ This method creates a Counter with state values 
    as keys and normalized residency time as values
    
    Parameters
    
    result : list of Observation objects.

    Returns:
        Counter
          
    """
    RNA_distribution = Counter()
    for observation in results:
        state = observation.RNA_Protein_state[0]
        residency_time = observation.time_of_residency
        RNA_distribution[state] += residency_time/time_limit

    return RNA_distribution 

def generate_protein_distribution(results):
    protein_distribution = Counter()
    for observation in results:
        state = observation.RNA_Protein_state[1]
        residency_time = observation.time_of_residency
        protein_distribution[state] += residency_time/time_limit

    return protein_distribution 

RNA_distribution = generate_RNA_distribution(results)
protein_distribution = generate_protein_distribution(results)




fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
values = np.arange(20)
pmf = st.poisson(10).pmf(values) 
ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
ax[0].set_ylabel('Normalized residency time', fontsize=16)
ax[0].set_xlabel('Number of RNA molecules', fontsize=16)
ax[0].bar(values, pmf, alpha=0.5)
ax[1].bar(protein_distribution.keys(), protein_distribution.values())
ax[1].set_ylabel('Normalized residency time', fontsize=16)
ax[1].set_xlabel('Number of proteins', fontsize=16)
ax[1].bar(values, pmf, alpha=0.5)



def create_dataframe(results):
    """ This method creates a dataframe with 4 columns:
        time of observation, gene activity, number of RNA molecules
        and number of proteins.
    
    Parameters
    ----------
    
    result : list of Observation objects.

    Returns:
    ----------
    df: pandas dataframe
          
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
    
    df = pd.DataFrame(d)
    return df

df = create_dataframe(results)


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
ax[0].plot(df['Time'], df['Gene activity'])
ax[0].set_ylabel('Gene Activity')
ax[0].set_xlabel('Time')
ax[1].plot(df['Time'], df['Number of RNA molecules'])
ax[1].set_ylabel('# of RNA molecules')
ax[1].set_xlabel('Time')
ax[2].plot(df['Time'], df['Number of proteins'])
ax[2].set_ylabel('# of proteins')
ax[2].set_xlabel('Time')
sns.despine(fig, trim=True, bottom=False, left=False)
