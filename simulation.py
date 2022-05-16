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

state = np.array([0, 1, 0, 0])

#news = 0
index_n_active_genes = 0
index_n_inactive_genes = 1
index_n_RNAs = 2
index_n_proteins = 3

class Transition(Enum):
    """Define all possible transitions"""
    GENE_ACTIVATE = 'gene activate'
    GENE_INACTIVATE = 'gene inactivate'
    RNA_INCREASE = 'RNA increase'
    RNA_DEGRADE = 'RNA degrade'
    PROTEIN_INCREASE = 'Protein increase'
    PROTEIN_DEGRADE = 'Protein degrade'

rates = namedtuple("Rates",['ka', 'ki', 'k1', 'k2', 'k3', 'k4'])
rate = rates(ka = 1, ki = 0.5, k1 = 1, k2 = 0.1, k3 = 1, k4 = 1)

def gene_activate(state):
    return state[index_n_inactive_genes]*rate.ka
def gene_inactivate(state):
    return state[index_n_active_genes]*rate.ki

def RNA_increase(state):
    return state[index_n_active_genes]*rate.k1

def RNA_degrade(state):
    return state[index_n_RNAs]*rate.k2

def Protein_increase(state):
    return state[index_n_RNAs]*rate.k3

def Protein_degrade(state):
    return state[index_n_proteins]*rate.k4




transitions = [gene_activate, gene_inactivate, 
               RNA_increase, RNA_degrade, 
               Protein_increase, Protein_degrade]

transition_names = [Transition.GENE_ACTIVATE, Transition.GENE_INACTIVATE, 
                    Transition.RNA_INCREASE, Transition.RNA_DEGRADE, 
                    Transition.PROTEIN_INCREASE, Transition.PROTEIN_DEGRADE]




def update_state(event, state):
    """This method updates the initial state according to the event occured
    
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
         state[index_n_active_genes] +=1
         state[index_n_inactive_genes] -=1
         state = state.copy()

         
    elif event == Transition.GENE_INACTIVATE:
        state[index_n_active_genes] -=1
        state[index_n_inactive_genes] +=1
        state = state.copy()

    elif event == Transition.RNA_INCREASE:
         state[index_n_RNAs] +=1
         state = state.copy()

    elif event == Transition.RNA_DEGRADE:
        state[index_n_RNAs] -=1
        state = state.copy()

    elif event == Transition.PROTEIN_INCREASE:
         state[index_n_proteins] +=1
         state = state.copy()

    elif event == Transition.PROTEIN_DEGRADE:
        state[index_n_proteins] -=1
        state = state.copy()

    elif isinstance(event,str) or isinstance(event, str):
        raise TypeError("Do not use string ! Choose transitions from Transition enum members.")
    else:
        raise ValueError("Transition not recognized")

    updated_state = state

    return updated_state 




class Observation(typing.NamedTuple):
    """ typing.NamedTuple class storing information
    for each event in the simulation"""
    state: typing.Any
    time_of_observation: float
    time_of_residency: float
    transition: Transition



def gillespie_ssa(state, transitions, total_time):
    
    rates = [f(state) for f in transitions]
    
    total_rate = np.sum(rates)

    time = np.random.exponential(1/total_rate)
    
    
    event = rn.choices(transition_names, weights = rates)[0]

    updated_state = update_state(event, state)
        
    state = updated_state
 
    observation = Observation(state, total_time, time, event)

    return observation



def evolution(starting_state, time_limit):
    rn.seed(1)
    state = starting_state
    total_time = 0.0
    observed_states = []
    while total_time < time_limit:
        
        
        observation = gillespie_ssa(state, transitions, total_time)
        observed_states.append(observation)
        time = observation.time_of_residency
        
        total_time += time
        

    return observed_states



time_limit = 100
results = evolution(starting_state = state, time_limit = time_limit)
results[:5]





def generate_RNA_distribution(results):
    """ This method creates a Counter with state values 
    as keys and normalized residency time as values
    """
    RNA_distribution = Counter()
    for observation in results:
        state = observation.state[index_n_RNAs]
        residency_time = observation.time_of_residency
        RNA_distribution[state] += residency_time/time_limit
        
    total_time_observed = sum(RNA_distribution.values())
    for state in RNA_distribution:
        RNA_distribution[state] /= total_time_observed

    return RNA_distribution 

def generate_protein_distribution(results):
    protein_distribution = Counter()
    for observation in results:
        state = observation.state[index_n_proteins]
        residency_time = observation.time_of_residency
        protein_distribution[state] += residency_time/time_limit
    
    total_time_observed = sum(protein_distribution.values())
    for state in protein_distribution:
        protein_distribution[state] /= total_time_observed

    return protein_distribution 




def StatesDistributionPlot():
    """ This method plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(results)
    protein_distribution = generate_protein_distribution(results)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    values = np.arange(20)
    pmf = st.poisson(5).pmf(values) 
    ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
    ax[0].set_ylabel('Normalized residency time', fontsize=16)
    ax[0].set_xlabel('Number of RNA molecules', fontsize=16)
    ax[0].bar(values, pmf, alpha=0.5)
    ax[0].legend(["Simulation","Poisson distribution"], fontsize=16)
    ax[1].bar(protein_distribution.keys(), protein_distribution.values())
    ax[1].set_ylabel('Normalized residency time', fontsize=16)
    ax[1].set_xlabel('Number of proteins', fontsize=16)
    ax[1].bar(values, pmf, alpha=0.5)
 


StatesDistributionPlot()



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
        number_of_RNA_molecules.append(observation.state[index_n_RNAs])
        number_of_proteins.append(observation.state[index_n_proteins])
        if observation.state[index_n_active_genes] > 0:
            gene_activity.append(1)
        else:
            gene_activity.append(0)
    
    d = {'Time': time_of_observation, 
         'Gene activity': gene_activity,
         'Number of RNA molecules': number_of_RNA_molecules, 
         'Number of proteins': number_of_proteins}
    
    results_dataframe = pd.DataFrame(d)
    
    return results_dataframe




def save_results(file_path):
    """This method saves dataframe in a tab separated CSV file
    
    Parameters
    
    file_path : str
                path to folder where the CSV file is saved
    """
    df = create_dataframe(results)
    df.to_csv(file_path, sep ='\t', index = None, header=True) 
    


save_results(file_path = r'C:\Users\asus\Desktop\results.csv')



def MoleculesVsTimePlot():
    """This method plots gene activity, the number of RNA molecules
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
    
    sns.despine(fig, trim=True, bottom=False, left=False)



MoleculesVsTimePlot()
