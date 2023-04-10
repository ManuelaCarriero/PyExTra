# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:27:40 2023

@author: asus
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pylab as plt
import seaborn as sns

import configparser

from collections import namedtuple 

config = configparser.ConfigParser()

config.read('configuration.txt')

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



SSA_results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")

tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")

#%%

gene_activity = 1

# This is the case of a gene that is always active.

def CME(state, time, k1, k2, k3, k4):
    RNAs, proteins = state
    δRNAs = k1*gene_activity-k2*RNAs
    δproteins = k3*RNAs-k4*proteins
    return δRNAs, δproteins

time = np.linspace(0, 100, 100)
SSA_time = np.ascontiguousarray(SSA_results['Time'])
tauleap_time = np.ascontiguousarray(tauleap_results['Time'])

#gene_activity = np.ascontiguousarray(SSA_results['Gene activity'])
#gene_activity
#k1, k2, k3, k4 = 1.0, 0.1, 0.1, 1
k1, k2, k3, k4 = rate.k1, rate.k2, rate.k3, rate.k4
state0=(0.0, 0.0)

res = odeint(CME, y0=state0, t=time, args=(k1, k2, k3, k4))

RNAs_hat, proteins_hat = res.T

#%%

#gene_activity = np.ones(len(time))

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
ax[0].plot(time, RNAs_hat, linewidth=4)
ax[1].plot(time,proteins_hat)
sns.despine(fig, bottom=False, left=False)
plt.show()

#%%

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 20)) #figsize=(5, 10)
#ax[0].plot(time, gene_activity)
#ax[0].set_ylabel('Gene Activity')
#ax[0].set_xlabel('Time')

ax[0].plot(time, RNAs_hat, linewidth=5, color='black')
ax[0].plot(SSA_time, SSA_results['Number of RNA molecules'])
ax[0].plot(tauleap_time, tauleap_results['Number of RNA molecules'], marker = 'o', linestyle=':',markersize='1')#, linestyle='--', marker='o', label='line with marker'
ax[0].set_ylabel('# of RNA molecules',fontsize=20)
ax[0].set_xlabel('Time (a.u.)',fontsize=20)
ax[0].xaxis.set_tick_params(labelsize=20)
ax[0].yaxis.set_tick_params(labelsize=20)
ax[0].legend(["ODE","SSA","Tau-leap \u03C4 = {}".format(dt)], fontsize=20,loc='upper right')

ax[1].plot(time, proteins_hat, linewidth=5, color='black')
ax[1].plot(SSA_time, SSA_results['Number of proteins'])
ax[1].plot(tauleap_time, tauleap_results['Number of proteins'], marker = 'o', linestyle=':',markersize='1')
ax[1].set_ylabel('# of proteins',fontsize=20)
ax[1].set_xlabel('Time (a.u.)',fontsize=20)
ax[1].xaxis.set_tick_params(labelsize=20)
ax[1].yaxis.set_tick_params(labelsize=20)

sns.despine(fig, bottom=False, left=False)
plt.show()

#%%

hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")

hybrid_time = np.ascontiguousarray(hybrid_results['Time'])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5)) #figsize=(5, 10)
#ax[0].plot(time, gene_activity)
#ax[0].set_ylabel('Gene Activity')
#ax[0].set_xlabel('Time')

ax.plot(time, RNAs_hat, linewidth=4)
ax.plot(SSA_time, SSA_results['Number of RNA molecules'])
ax.plot(tauleap_time, tauleap_results['Number of RNA molecules'], marker = 'o', linestyle=':',markersize='1')
ax.plot(hybrid_time, hybrid_results['Number of RNA molecules'], color = 'm')
ax.set_ylabel('# of RNA molecules')
ax.set_xlabel('Time')
ax.legend(["ODE","SSA","Tau-leap \u03C4 = {}".format(dt),"Hybrid"])

#ax[1].plot(time, proteins_hat)
#ax[1].plot(SSA_time, SSA_results['Number of proteins'])
#ax[1].plot(tauleap_time, tauleap_results['Number of proteins'])
#ax[1].set_ylabel('# of proteins')
#ax[1].set_xlabel('Time')

sns.despine(fig, bottom=False, left=False)
plt.show()



