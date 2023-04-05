# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:45:47 2022

@author: asus
"""

import argparse
import configparser
import ast 
import sys
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from statsmodels.tsa import stattools
import scipy.interpolate as interp

import json
import jsonlines



config = configparser.ConfigParser()

parser = argparse.ArgumentParser()

parser.add_argument("filename", help="read configuration file.")

parser.add_argument('-plot_acf', help='plot acf of RNAs and Proteins in separated graphs (first protein synthesis model)', action = "store_true")
parser.add_argument('-plot_acfs', help='plot acf of RNAs and Proteins in the same graph', action = "store_true")

parser.add_argument('-plot_acf_RNA', help='plot acf of RNAs (SSA simulation results)', action = "store_true")


parser.add_argument('-plot_acf_nfkb', help='plot acf of RNAs as function of time produced through the NF-kB activity', action = "store_true")

parser.add_argument('-plot_acf_toggleswitch_RNA', help='plot acf of RNAs in Toggle Switch configuration (SSA)', action = "store_true")
parser.add_argument('-plot_acf_toggleswitch_protein', help='plot acf of Proteins in Toggle Switch configuration (SSA)', action = "store_true")
parser.add_argument('-plot_acf_toggleswitch_RNA_protein', help='plot acf of RNAs and proteins in separated graphs for Toggle Switch configuration (SSA)', action = "store_true")
parser.add_argument('--new_instance', help='plot acf of RNAs and proteins in separated graphs for Toggle Switch configuration (SSA) using results obtained from the flag -plot_acf_toggleswitch_RNA_protein', action = "store_true")

parser.add_argument('-plot_acf_toggleswitch_RNA_samplingtime', help='plot acf vs sampling time of RNAs in Toggle Switch configuration (SSA)', action = "store_true")

parser.add_argument('-plot_acf_toggleswitch_protein_samplingtime', help='plot acf vs sampling time of proteins in Toggle Switch configuration (SSA)', action = "store_true")



parser.add_argument('-plot_acf_autorepressor', help='plot acf of RNAs and Proteins in separated graphs for Autorepressor configuration (SSA)', action = "store_true")

parser.add_argument('-multiplesimulationsacfs_tauleap', help='plot acf of RNAs and Proteins in the same graph of multiple Tau-leap simulations', action = "store_true")
parser.add_argument('-multiplesimulationsacfs_gillespie', help='plot acf of RNAs and Proteins in the same graph of multiple SSA simulations', action = "store_true")
parser.add_argument('-multiplesimulationsacfs_hybrid', help='plot acf of RNAs and Proteins in the same graph of multiple SSA/Tau-leap simulations', action = "store_true")

parser.add_argument('-multiplesimulationsacfs_autorepressor', help='plot acf of RNAs and Proteins in the same graph of multiple SSA simulations in case of autorepressor model', action = "store_true")
parser.add_argument('-multiplesimulationsacfs_toggleswitch', help='plot acf of RNAs and Proteins in the same graph of multiple SSA simulations in case of toggle switch model', action = "store_true")
parser.add_argument('-multiplesimulationsacfs_firstmodel', help='plot acf of RNAs and Proteins in the same graph of multiple SSA simulations in case of first model', action = "store_true")

parser.add_argument('-multiplesimulations_nfkb', help='plot acf of nfkb RNA products as function of sampling time', action="store_true")

parser.add_argument('-multiplot_acf_toggleswitch_samplingtime', help='plot acf of RNAs and Proteins in the same graph of multiple SSA simulations in case of Toggle Switch', action = "store_true")



parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

args = parser.parse_args()



config.read(args.filename)



if args.filename == 'configuration.txt':

    
    
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
    
    
    
    nlags = config["ACF"].getint('n_lags')
    
    
    
elif args.filename == 'configuration_2genes.txt':
    
    
    
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
    
    
    
    nlags = config["ACF"].getint('n_lags')



#%%
 
def plot_acf(autocorr, title = ""):
    """This function plots autocorrelation
    as function of the number of lags."""
    
    plt.plot(np.arange(0,len(autocorr)), autocorr, linestyle = 'dotted', color='darkgrey')
    #plt.ylim(0,1.05)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title("{}".format(title))
    sns.despine(bottom=False, left=False)
    plt.show()

def plot_acf_samplingtime(autocorr, xvals, title = ""):
    """This function plots autocorrelation
    as function of sampling time."""
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    
    ax.plot(xvals[:len(autocorr)], autocorr, linestyle = 'dotted', color='black')
    #plt.xlim(0,1000)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'G($\tau$)')
    ax.set_title("{}".format(title))
    ax.axhline(y=0, color = 'lightgray')
    sns.despine(bottom=False, left=False)
    plt.show()

# Note
# lst = [1,2,3,4]
# lst[:3]
# Prende i primi 3: [1, 2, 3]

def autocorr(simulation, molecule = 'RNA', nlags = nlags):
    """This function returns autocorrelation values for RNAs and proteins 
     given the type of simulation."""
    
    results = pd.read_csv('{}simulation_results.csv'.format(simulation), sep=" ")
    
    if simulation == 'tauleap':
        
        results = results[results['Time'] > warmup_time]
        
        if molecule == 'RNA':

            RNAs = results['Number of RNA molecules']
            autocorr_RNAs = stattools.acf(RNAs, nlags = nlags, fft=False) #1500, len(RNAs) - 1
            
            return autocorr_RNAs

        else:
            
            proteins = results['Number of proteins']
            autocorr_proteins = stattools.acf(proteins, nlags = nlags, fft=False)
        
            return autocorr_proteins
        
    elif simulation == 'gillespie' or simulation == 'autorepressor_gillespie' or simulation == 'removed_warmup_nfkb_k20k2i0gillespie':
        
        #results = results[results['Time'] > warmup_time] 
        
        time = results['Time']
        xvals = np.arange(results['Time'].iloc[0], results['Time'].iloc[-1], dt)
        
        if molecule == 'RNA':

            RNAs = results['Number of RNA molecules']
            f_RNAs = interp.interp1d(time, RNAs, kind='previous')
            yinterp_RNAs = f_RNAs(xvals)
            autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
        
            return autocorr_RNAs, xvals
        
        else:
            
            proteins = results['Number of proteins']
            f_proteins = interp.interp1d(time, proteins, kind='previous')
            yinterp_proteins = f_proteins(xvals)        
            autocorr_proteins = stattools.acf(yinterp_proteins, nlags = nlags, fft=False)
        
            return autocorr_proteins, xvals 
    
    elif simulation == 'toggleswitch_gillespie':
        
        time = results['Time']
        xvals = np.arange(results['Time'].iloc[0], results['Time'].iloc[-1], dt)
        
        if molecule == 'RNA':
            
            RNAs_gene1 = results['Number of RNAs gene1']
    
            RNAs_gene2 = results['Number of RNAs gene2']
            
            f_RNAs_gene1 = interp.interp1d(time, RNAs_gene1, kind='previous')
            yinterp_RNAs_gene1 = f_RNAs_gene1(xvals)
            autocorr_RNAs_gene1 = stattools.acf(yinterp_RNAs_gene1, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
            
            f_RNAs_gene2 = interp.interp1d(time, RNAs_gene2, kind='previous')
            yinterp_RNAs_gene2 = f_RNAs_gene2(xvals)
            autocorr_RNAs_gene2 = stattools.acf(yinterp_RNAs_gene2, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
        
            return autocorr_RNAs_gene1, autocorr_RNAs_gene2, xvals
        
        else:
            
            proteins_gene1 = results['Number of proteins gene1']
            
            proteins_gene2 = results['Number of proteins gene2']
            
            f_proteins_gene1 = interp.interp1d(time, proteins_gene1, kind='previous')
            yinterp_proteins_gene1 = f_proteins_gene1(xvals)
            autocorr_proteins_gene1 = stattools.acf(yinterp_proteins_gene1, nlags = nlags, fft=False)
            
            f_proteins_gene2 = interp.interp1d(time, proteins_gene2, kind='previous')
            yinterp_proteins_gene2 = f_proteins_gene2(xvals)
            autocorr_proteins_gene2 = stattools.acf(yinterp_proteins_gene2, nlags = nlags, fft=False)
        
            return autocorr_proteins_gene1, autocorr_proteins_gene2, xvals 

if args.plot_acf_nfkb:#removed_warmup_nfkb_k20k2i0k30k3i0gillespiesimulation_results
    
    autocorr_RNAs, xvals = autocorr(simulation='removed_warmup_nfkb_k20k2i0gillespie', molecule = 'RNA', nlags = nlags)#nfkb_gillespie
    
    plot_acf_samplingtime(autocorr=autocorr_RNAs, xvals=xvals, title = "NF-kB products")
    

#autocorr = stattools.acf([1,2,3,4], nlags = 4, fft=False) #800, len(yinterp_RNAs) - 1
#autocorr 

if args.plot_acf:   
    
    autocorr_RNAs = autocorr(simulation = 'tauleap', molecule = 'RNA')
    
    autocorr_proteins = autocorr(simulation = 'tauleap', molecule = 'protein')
    
    
    
    plot_acf(autocorr=autocorr_RNAs, title = 'RNA autocorrelation Tau-leap simulation')
    
    plot_acf(autocorr=autocorr_proteins, title = 'Protein autocorrelation Tau-leap simulation')
    
    
    
    autocorr_RNAs = autocorr(simulation = 'gillespie', molecule = 'RNA')
    
    autocorr_proteins = autocorr(simulation = 'gillespie', molecule = 'protein')
    
    
    
    plot_acf(autocorr = autocorr_RNAs, title = 'RNA autocorrelation Gillespie simulation')
    
    plot_acf(autocorr = autocorr_proteins, title = 'Protein autocorrelation Gillespie simulation')


    
    

# RNAs and Proteins

def plot_multiacf(autocorr1, label1, autocorr2, label2, title = ""):
    """This function plots autocorrelation of two chemical species
    in the same graph. By default the title is empty."""
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.arange(0,len(autocorr1)), autocorr1, label = '{}'.format(label1), linestyle = 'dotted', color = 'blue')#black
    ax.plot(np.arange(0,len(autocorr2)), autocorr2, label = '{}'.format(label2), linestyle = 'dotted', color = 'cyan')#grey
    #plt.ylim(0,1.05)
    ax.set_xlabel('# of lags')
    ax.set_ylabel('Autocorrelation')
    ax.set_title("{}".format(title))
    ax.legend()
    sns.despine(fig, bottom=False, left=False)
    plt.show()
    
def plot_multiacf_samplingtime(autocorr1, label1, autocorr2, label2, title = ""):
    """This function plots autocorrelation of two chemical species
    in the same graph. By default the title is empty."""
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(xvals[:len(autocorr1)], autocorr1, label = '{}'.format(label1), linestyle = 'dotted', color = 'blue')#black
    ax.plot(xvals[:len(autocorr2)], autocorr2, label = '{}'.format(label2), linestyle = 'dotted', color = 'cyan')#grey
    #plt.ylim(0,1.05)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'G($\tau$)')
    ax.axhline(y=0, color = 'gray')
    ax.set_title("{}".format(title))
    ax.legend()
    sns.despine(fig, bottom=False, left=False)
    plt.show()
    
    

if args.plot_acfs:
    
    autocorr_RNAs = autocorr(simulation = 'tauleap', molecule = 'RNA')
    
    autocorr_proteins = autocorr(simulation = 'tauleap', molecule = 'protein')
        
    plot_multiacf(autocorr1 = autocorr_RNAs, label1 = "RNAs", autocorr2 = autocorr_proteins, label2 = "Proteins", title = "Tau-leap simulation")
    
    
    
    autocorr_RNAs = autocorr(simulation = 'gillespie', molecule = 'RNA')
    
    autocorr_proteins = autocorr(simulation = 'gillespie', molecule = 'protein')
    
    plot_multiacf(autocorr1 = autocorr_RNAs, label1 = "RNAs", autocorr2 = autocorr_proteins, label2 = "Proteins", title = "Gillespie simulation")



if args.plot_acf_RNA:
    
    autocorr_RNAs, xvals = autocorr(simulation = 'gillespie', molecule = 'RNA')
    
    plot_acf_samplingtime(autocorr = autocorr_RNAs, xvals=xvals, title = "")
    


if args.plot_acf_toggleswitch_RNA: 
    
    autocorr_RNAs_gene1, autocorr_RNAs_gene2 = autocorr(simulation = 'toggleswitch_gillespie', molecule = 'RNA')
    
    plot_multiacf(autocorr1=autocorr_RNAs_gene1, label1 = "RNAs gene1", autocorr2=autocorr_RNAs_gene2, label2 = "RNAs gene2", title = 'RNAs autocorrelation Toggle-Switch (Gillespie simulation)')

if args.plot_acf_toggleswitch_protein: 

    autocorr_proteins_gene1, autocorr_proteins_gene2 = autocorr(simulation = 'toggleswitch_gillespie', molecule = 'protein')
    
    plot_multiacf(autocorr1=autocorr_proteins_gene1, label1 = "proteins gene1", autocorr2=autocorr_proteins_gene2, label2 = "proteins gene2", title = 'Proteins autocorrelation Toggle-Switch (Gillespie simulation)')




if args.plot_acf_toggleswitch_RNA_samplingtime: 
    
    autocorr_RNAs_gene1, autocorr_RNAs_gene2, xvals = autocorr(simulation = 'toggleswitch_gillespie', molecule = 'RNA')
    
    plot_multiacf_samplingtime(autocorr1=autocorr_RNAs_gene1, label1 = "RNAs gene1", autocorr2=autocorr_RNAs_gene2, label2 = "RNAs gene2", title = 'RNAs autocorrelation Toggle-Switch (Gillespie simulation)')

if args.plot_acf_toggleswitch_protein_samplingtime: 

    autocorr_proteins_gene1, autocorr_proteins_gene2, xvals = autocorr(simulation = 'toggleswitch_gillespie', molecule = 'protein')
    
    plot_multiacf_samplingtime(autocorr1=autocorr_proteins_gene1, label1 = "proteins gene1", autocorr2=autocorr_proteins_gene2, label2 = "proteins gene2", title = 'Proteins autocorrelation Toggle-Switch (Gillespie simulation)')



def MultiPlot_Acf(title):
    """This function plots RNA autocorrelation and protein autocorrelation
    respectively up and down a multiplot."""
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,7))
    
    ax[0].plot(np.arange(0,len(autocorr_RNAs)), autocorr_RNAs, linestyle = 'dotted', color='black')
    ax[0].set_ylabel('Autocorrelation')
    ax[0].set_xlabel('# of lags')
    ax[0].legend(['RNAs'])
    ax[0].set_title("{}".format(title))
    ax[0].axhline(y=0, linewidth=1, color='lightgray')

    ax[1].plot(np.arange(0,len(autocorr_proteins)), autocorr_proteins, linestyle = 'dotted', color='grey')
    ax[1].set_ylabel('Autocorrelation')
    ax[1].set_xlabel('# of lags')
    ax[1].legend(['Proteins'])
    ax[1].axhline(y=0, linewidth=1, color='lightgray')
    sns.despine(fig, bottom=False, left=False)
    plt.show()
    
def MultiPlot_Acf_samplingtime(title):
    """This function plots RNA autocorrelation and protein autocorrelation
    as function of sampling time
    respectively up and down a multiplot ."""
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,7))
    
    ax[0].plot(xvals[:len(autocorr_RNAs)], autocorr_RNAs, linestyle = 'dotted', color='black')
    #plt.xlim(0,1000)
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylabel(r'G($\tau$)')
    ax[0].legend(['RNAs'])
    ax[0].set_title("{}".format(title))
    ax[0].axhline(y=0, color = 'lightgray')
    
    ax[1].plot(xvals[:len(autocorr_proteins)], autocorr_proteins, linestyle = 'dotted', color='black')
    #plt.xlim(0,1000)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'G($\tau$)')
    ax[1].legend(['Proteins'])
    ax[1].axhline(y=0, color = 'lightgray')
    
    sns.despine(bottom=False, left=False)
    plt.show()  

#if args.plot_acf_autorepressor:
    
#    autocorr_RNAs = autocorr(simulation = 'autorepressor_gillespie', molecule = 'RNA')

#    autocorr_proteins = autocorr(simulation = 'autorepressor_gillespie', molecule = 'protein')

#    MultiPlot_Acf(title = 'Autorepressor')    

if args.plot_acf_autorepressor:
    
    autocorr_RNAs, xvals = autocorr(simulation = 'autorepressor_gillespie', molecule = 'RNA')

    autocorr_proteins, xvals = autocorr(simulation = 'autorepressor_gillespie', molecule = 'proteins')
    
    MultiPlot_Acf_samplingtime(title = "Autorepressor model")
    



    
    



def MultiPlot_multiacf(title = ""):
    """This function plots autocorrelation of two chemical species
    in the same graph. RNA molecules and proteins produced by different genes
    are plotted respectively up and down a multiplot.
    By default the title is empty.
    """
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    
    ax[0].plot(np.arange(0,len(autocorr_RNAs_gene1)), autocorr_RNAs_gene1, label = 'RNAs gene1', linestyle = 'dotted', color='blue')
    ax[0].plot(np.arange(0,len(autocorr_RNAs_gene2)), autocorr_RNAs_gene2, label = 'RNAs gene2', linestyle = 'dotted', color='cyan')
    ax[0].set_ylabel('Autocorrelation')
    ax[0].set_xlabel('# of lags')
    ax[0].legend()
    ax[0].set_title("{}".format(title))
    ax[0].axhline(y=0, linewidth=1, color='lightgray')

    ax[1].plot(np.arange(0,len(autocorr_proteins_gene1)), autocorr_proteins_gene1, label = 'proteins gene1', linestyle = 'dotted', color='blue')
    ax[1].plot(np.arange(0,len(autocorr_proteins_gene2)), autocorr_proteins_gene2, label = 'proteins gene2', linestyle = 'dotted', color='cyan')
    ax[1].set_ylabel('Autocorrelation')
    ax[1].set_xlabel('# of lags')
    ax[1].legend()
    ax[1].axhline(y=0, linewidth=1, color='lightgray')
    sns.despine(fig, bottom=False, left=False)
    plt.show()
    
def MultiPlot_multiacf_samplingtime(title = ""):
    """This function plots autocorrelation of two chemical species
    in the same graph. RNA molecules and proteins produced by different genes
    are plotted respectively up and down a multiplot.
    By default the title is empty.
    """
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    
    ax[0].plot(xvals[:len(autocorr_RNAs_gene1)], autocorr_RNAs_gene1, label = 'RNAs gene1', linestyle = 'dotted', color='blue')
    ax[0].plot(xvals[:len(autocorr_RNAs_gene2)], autocorr_RNAs_gene2, label = 'RNAs gene2', linestyle = 'dotted', color='cyan')
    ax[0].set_ylabel(r'G($\tau$)')
    ax[0].set_xlabel(r'$\tau$')
    ax[0].legend()
    ax[0].set_title("{}".format(title))
    ax[0].axhline(y=0, linewidth=1, color='gray')

    ax[1].plot(xvals[:len(autocorr_proteins_gene1)], autocorr_proteins_gene1, label = 'proteins gene1', linestyle = 'dotted', color='blue')
    ax[1].plot(xvals[:len(autocorr_proteins_gene2)], autocorr_proteins_gene2, label = 'proteins gene2', linestyle = 'dotted', color='cyan')
    ax[1].set_ylabel(r'G($\tau$)')
    ax[1].set_xlabel(r'$\tau$')
    ax[1].legend()
    ax[1].axhline(y=0, linewidth=1, color='gray')
    sns.despine(fig, bottom=False, left=False)
    plt.show()


if args.multiplot_acf_toggleswitch_samplingtime:
    
    autocorr_RNAs_gene1, autocorr_RNAs_gene2, xvals = autocorr(simulation='toggleswitch_gillespie', molecule = 'RNA', nlags = nlags)
    
    autocorr_proteins_gene1, autocorr_proteins_gene2, xvals = autocorr(simulation='toggleswitch_gillespie', molecule = 'Proteins', nlags = nlags)

    MultiPlot_multiacf_samplingtime(title="Toggle Switch")
    
    
    
    
    
    

class CustomizedEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


    
if args.plot_acf_toggleswitch_RNA_protein and args.new_instance:
    
    with jsonlines.open('autocorr_RNAs_gene1.jsonl') as reader:
        autocorr_RNAs_gene1 = reader.read()
        
        autocorr_RNAs_gene1 = ast.literal_eval(autocorr_RNAs_gene1)
        
    with jsonlines.open('autocorr_RNAs_gene2.jsonl') as reader:
        autocorr_RNAs_gene2 = reader.read()
        
        autocorr_RNAs_gene2 = ast.literal_eval(autocorr_RNAs_gene2)
        
    with jsonlines.open('autocorr_proteins_gene1.jsonl') as reader:
        autocorr_proteins_gene1 = reader.read()
        
        autocorr_proteins_gene1 = ast.literal_eval(autocorr_proteins_gene1)
        
    with jsonlines.open('autocorr_proteins_gene2.jsonl') as reader:
        autocorr_proteins_gene2 = reader.read()
        
        autocorr_proteins_gene2 = ast.literal_eval(autocorr_proteins_gene2)
        
    MultiPlot_multiacf(title = 'Toggle-Switch (Gillespie simulation)')
    


if args.plot_acf_toggleswitch_RNA_protein and len(sys.argv) == 3: 
    
    autocorr_RNAs_gene1, autocorr_RNAs_gene2, yinterp_proteins_gene1, yinterp_proteins_gene2 = autocorr(simulation = 'toggleswitch_gillespie', molecule = 'RNA')
    
    print("Number of measurements: {}".format(len(yinterp_proteins_gene1)))
    
    autocorr_proteins_gene1, autocorr_proteins_gene2 = autocorr(simulation = 'toggleswitch_gillespie', molecule = 'protein')
    
    MultiPlot_multiacf(title = 'Toggle-Switch (Gillespie simulation)')
    
    autocorr_RNAs_gene1 = json.dumps(autocorr_RNAs_gene1, cls = CustomizedEncoder)

    with jsonlines.open('autocorr_RNAs_gene1.jsonl', mode='w') as writer:
        writer.write(autocorr_RNAs_gene1)
        
    autocorr_RNAs_gene2 = json.dumps(autocorr_RNAs_gene2, cls = CustomizedEncoder)

    with jsonlines.open('autocorr_RNAs_gene2.jsonl', mode='w') as writer:
        writer.write(autocorr_RNAs_gene2)   
        
    autocorr_proteins_gene1 = json.dumps(autocorr_proteins_gene1, cls = CustomizedEncoder)

    with jsonlines.open('autocorr_proteins_gene1.jsonl', mode='w') as writer:
        writer.write(autocorr_proteins_gene1)   

    autocorr_proteins_gene2 = json.dumps(autocorr_proteins_gene2, cls = CustomizedEncoder)

    with jsonlines.open('autocorr_proteins_gene2.jsonl', mode='w') as writer:
        writer.write(autocorr_proteins_gene2)               



#%% Multiple simulations



def MultipleSimulationsAcf(simulation):
    """This function calculates the RNA and protein autocorrelation of N
    simulations and returns their values as lists."""

    dataframes_list = []
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('{}results_seed{}.csv'.format(simulation, n), sep=" ")
        #removedwarmup_results = simulation_results[simulation_results['Time'] > warmup_time]
        dataframes_list.append(simulation_results)
    
    autocorrs_RNAs = []
    
    autocorrs_Proteins = []  
      
    for dataframe in dataframes_list:
        
        if simulation == 'nfkb_k20k2i0gillespiesimulation_':
            
            RNAs = dataframe['Number of RNA molecules']
            
            time = dataframe['Time']
            
            xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], dt)
   
            f_RNAs = interp.interp1d(time, RNAs, kind='previous')
            yinterp_RNAs = f_RNAs(xvals)
            
            autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
 
            autocorrs_RNAs.append(autocorr_RNAs)
            
            #return autocorrs_RNAs, xvals

        
        if simulation == 'ka0.1ki1gillespie' or simulation == 'hybrid' or simulation == 'gillespie_autorepressor_':
            
            RNAs = dataframe['Number of RNA molecules']
            proteins = dataframe['Number of proteins']
            time = dataframe['Time']
            xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], dt)
   
            f_RNAs = interp.interp1d(time, RNAs, kind='previous')
            yinterp_RNAs = f_RNAs(xvals)
            #plt.plot(time, RNAs, 'o', xvals, yinterp_RNAs, '-')
            #plt.show()
            
            f_proteins = interp.interp1d(time, proteins, kind='previous')
            yinterp_proteins = f_proteins(xvals)
            #plt.plot(time, proteins, 'o', xvals, yinterp_proteins, '-')
            #plt.show()
            
            autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
            autocorr_proteins = stattools.acf(yinterp_proteins, nlags = nlags, fft=False)
        
            autocorrs_RNAs.append(autocorr_RNAs)
            autocorrs_Proteins.append(autocorr_proteins)
            
            #return autocorrs_RNAs, autocorrs_Proteins, xvals
        
        elif simulation == 'tauleap':
            
            RNAs = dataframe['Number of RNA molecules']
            proteins = dataframe['Number of proteins']
            time = dataframe['Time']
            xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], dt)

            
            autocorr_RNAs = stattools.acf(RNAs, nlags = nlags, fft=False) #1500, len(RNAs) - 1
            autocorr_proteins = stattools.acf(proteins, nlags = nlags, fft=False)
            
            autocorrs_RNAs.append(autocorr_RNAs)
            autocorrs_Proteins.append(autocorr_proteins)
            
            #return autocorrs_RNAs, autocorrs_Proteins, xvals
            
        elif simulation == 'gillespie_toggleswitch_':
            
            RNAs_gene1 = dataframe['Number of RNAs gene1']
            
            proteins_gene1 = dataframe['Number of proteins gene1']
            
            time = dataframe['Time']
            
            xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], dt)
            
            f_RNAs_gene1 = interp.interp1d(time, RNAs_gene1, kind='previous')
            yinterp_RNAs_gene1 = f_RNAs_gene1(xvals)
            autocorr_RNAs_gene1 = stattools.acf(yinterp_RNAs_gene1, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
                   
            f_proteins_gene1 = interp.interp1d(time, proteins_gene1, kind='previous')
            yinterp_proteins_gene1 = f_proteins_gene1(xvals)
            autocorr_proteins_gene1 = stattools.acf(yinterp_proteins_gene1, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1



            autocorrs_RNAs.append(autocorr_RNAs_gene1)
            autocorrs_Proteins.append(autocorr_proteins_gene1)
            
    return autocorrs_RNAs, autocorrs_Proteins, xvals

            
            
   
    
"""
dataframe = pd.read_csv('removed_warmup_ka1ki0gillespieresults_seed2.csv', sep=" ")
RNAs = dataframe['Number of RNA molecules']
proteins = dataframe['Number of proteins']
time = dataframe['Time']
xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], dt)
dataframe['Time'].iloc[0]   
dataframe

f_RNAs = interp.interp1d(time, RNAs, kind='previous')
yinterp_RNAs = f_RNAs(xvals)
len(yinterp_RNAs)
len(xvals)
#plt.plot(time, RNAs, 'o', xvals, yinterp_RNAs, '-')
#plt.show()

f_proteins = interp.interp1d(time, proteins, kind='previous')
yinterp_proteins = f_proteins(xvals)
#plt.plot(time, proteins, 'o', xvals, yinterp_proteins, '-')
#plt.show()

autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) #800, len(yinterp_RNAs) - 1
autocorr_proteins = stattools.acf(yinterp_proteins, nlags = nlags, fft=False)

len(autocorr_RNAs)


dt=0.1
nlags = 10000
autocorrs_RNAs, autocorrs_Proteins, xvals = MultipleSimulationsAcf(simulation='removed_warmup_ka1ki0.5gillespie')
len(autocorrs_RNAs[0])
len(autocorrs_Proteins)
len(xvals)
"""
def MultipleSimulationsAcf_Plot(title):
    """This function plots RNA autocorrelation and protein autocorrelation
    of N simulations respectively up and down a multiplot."""
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,7))
    for n, autocorr in enumerate(autocorrs_RNAs,1):
        ax[0].plot(np.arange(0,len(autocorr)), autocorr, linestyle = 'dotted', color='black')
        ax[0].set_ylabel('Autocorrelation')
        ax[0].set_xlabel('# of lags')
        ax[0].legend(['RNAs'])
        ax[0].set_title("{}".format(title))
        #x = np.arange(0,len(autocorr))
        #y = autocorr
        #ax[0].text(x[-1],y[-1],"n={}".format(n))
        #ax[0].set_ylim(-0.2,1)
    for n, autocorr in enumerate(autocorrs_Proteins,1):
        ax[1].plot(np.arange(0,len(autocorr)), autocorr, linestyle = 'dotted', color='grey')
        ax[1].set_ylabel('Autocorrelation')
        ax[1].set_xlabel('# of lags')
        ax[1].legend(['Proteins'])
        #ax[1].set_ylim(0,1)
        #x = np.arange(0,len(autocorr))
        #y = autocorr
        #ax[1].text(x[-1],y[-1],"n={}".format(n))
        #ax[1].set_ylim(-0.2,1)
        sns.despine(fig, bottom=False, left=False)
    plt.show()

def MultipleSimulationsAcf_Plot_samplingtime(title = ""):
    """This function plots RNA autocorrelation and protein autocorrelation
    as function of sampling time,
    of N simulations respectively up and down a multiplot."""
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,7))
    for n, autocorr in enumerate(autocorrs_RNAs,1):
        ax[0].plot(xvals[:len(autocorr)], autocorr, linestyle = '-')#, color='black'
        ax[0].set_ylabel(r'G($\tau$)')
        #ax[0].xaxis.set_label_position('bottom')
        ax[0].set_xlabel(r'$\tau$')
        #◘ax[0].legend(['RNAs'])
        #ax[0].xaxis.set_label_position('top')
        #ax[0].set_xlabel('RNAs')
        ax[0].set_title("{}".format(title))
        #ax[0].set_title("{}".format(title))
        ax[0].axhline(y=0, color = 'gray')
        ax[0].text(0.9,0.9,"RNAs", 
                   ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[0].transAxes)
        #x = np.arange(0,len(autocorr))
        #y = autocorr
        #ax[0].text(x[-1],y[-1],"n={}".format(n))
        #ax[0].set_ylim(-0.2,1)
    for n, autocorr in enumerate(autocorrs_Proteins,1):
        ax[1].plot(xvals[:len(autocorr)], autocorr, linestyle = '-')#, color='black'
        ax[1].set_ylabel(r'G($\tau$)')
        #ax[1].xaxis.set_label_position('bottom')
        ax[1].set_xlabel(r'$\tau$')
        ax[1].text(0.9,0.9,"Proteins", 
                   ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[1].transAxes)
        #ax[1].xaxis.set_label_position('top')
        #ax[1].set_xlabel('Proteins')
        #ax[1].legend(['Proteins'])
        #ax[1].set_title("{}".format(title2), x=400,y=0.8)
        ax[1].axhline(y=0, color = 'gray')
        #ax[1].set_ylim(0,1)
        #x = np.arange(0,len(autocorr))
        #y = autocorr
        #ax[1].text(x[-1],y[-1],"n={}".format(n))
        #ax[1].set_ylim(-0.2,1)
        sns.despine(fig, bottom=False, left=False)
    plt.show()
    
def MultipleSimulationsOneAcf_Plot_samplingtime(title = ""):
    """This function plots autocorrelation of one chemical specie
    as function of sampling time"""
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
    for n, autocorr in enumerate(autocorrs_RNAs,1):
        #ax.plot(xvals[:len(autocorr)], autocorr, linestyle = '-')#, color='black'
        ax.plot(xvals[:len(autocorr)], autocorr[:len(xvals)], linestyle = '-')#, color='black'
        ax.set_ylabel(r'G($\tau$)')
        #ax[0].xaxis.set_label_position('bottom')
        ax.set_xlabel(r'$\tau$')
        #◘ax[0].legend(['RNAs'])
        #ax[0].xaxis.set_label_position('top')
        #ax[0].set_xlabel('RNAs')
        ax.set_title("{}".format(title))
        #ax[0].set_title("{}".format(title))
        ax.axhline(y=0, color = 'gray')

        sns.despine(fig, bottom=False, left=False)
    plt.show()
#MultipleSimulationsAcf_Plot_samplingtime("ACF")

if args.multiplesimulationsacfs_tauleap:

    autocorrs_RNAs, autocorrs_Proteins = MultipleSimulationsAcf(simulation = 'tauleap')
    
    MultipleSimulationsAcf_Plot(title = "Tau-leap Simulation")
    
if args.multiplesimulationsacfs_gillespie:
    
    autocorrs_RNAs, autocorrs_Proteins = MultipleSimulationsAcf(simulation = 'gillespie')
    
    MultipleSimulationsAcf_Plot(title = "Gillespie Simulation")
    
if args.multiplesimulationsacfs_hybrid:
    
    autocorrs_RNAs, autocorrs_Proteins = MultipleSimulationsAcf(simulation = 'hybrid')
    
    MultipleSimulationsAcf_Plot(title = "Hybrid Simulation")

#if args.multiplesimulationsacfs_autorepressor:
    
#    autocorrs_RNAs, autocorrs_Proteins = MultipleSimulationsAcf(simulation = 'gillespie_autorepressor_')
    
#    MultipleSimulationsAcf_Plot(title = "Autorepressor")


if args.multiplesimulationsacfs_firstmodel:
    
    autocorrs_RNAs, autocorrs_Proteins, xvals = MultipleSimulationsAcf(simulation = 'ka0.1ki1gillespie')
    
    MultipleSimulationsAcf_Plot_samplingtime(title = "First model ($ka=0.1$,$ki=1$)")



if args.multiplesimulations_nfkb:
    
    autocorrs_RNAs, autocorrs_Proteins, xvals = MultipleSimulationsAcf(simulation='nfkb_k20k2i0gillespiesimulation_')#nfkb_gillespie

    MultipleSimulationsOneAcf_Plot_samplingtime(title = "NF-kB products")


"""
if args.multiplesimulationsacfs_firstmodel:
    
    autocorrs_RNAs, autocorrs_Proteins, xvals = MultipleSimulationsAcf(simulation = 'removed_warmup_ka1ki0gillespie')
    
    autocorrs_RNAs_lengths = [len(autocorrs_RNAs[i]) for i in np.arange(0,len(autocorrs_RNAs))]
    
    min_length = min(autocorrs_RNAs_lengths)
    
    autocorrs_RNAs = [autocorr_RNAs[:min_length] for autocorr_RNAs in autocorrs_RNAs]
    
    autocorrs_Proteins = [autocorr_Proteins[:min_length] for autocorr_Proteins in autocorrs_Proteins]
    
    xvals = xvals[:min_length]
    
    MultipleSimulationsAcf_Plot_samplingtime(title = "First model ($ka=1$,$ki=0.5$)")
"""
#dt=0.1
#nlags=10000

#autocorrs_RNAs, autocorrs_Proteins, xvals = MultipleSimulationsAcf(simulation = 'removed_warmup_ka1ki0gillespie')

#autocorrs_RNAs_lengths = [len(autocorrs_RNAs[i]) for i in np.arange(0,len(autocorrs_RNAs))]

#min_length = min(autocorrs_RNAs_lengths)

#autocorrs_RNAs = [autocorr_RNAs[:min_length] for autocorr_RNAs in autocorrs_RNAs]

#autocorrs_Proteins = [autocorr_Proteins[:min_length] for autocorr_Proteins in autocorrs_Proteins]

#xvals = xvals[:min_length]

#MultipleSimulationsAcf_Plot_samplingtime(title = "First model ($ka=1$,$ki=0$)")

    
#lst=[1,2,3,4]    
#lst[:3]

if args.multiplesimulationsacfs_autorepressor:
    
    autocorrs_RNAs, autocorrs_Proteins, xvals = MultipleSimulationsAcf(simulation = 'gillespie_autorepressor_')
    
    MultipleSimulationsAcf_Plot_samplingtime(title = "Autorepressor model")
    
if args.multiplesimulationsacfs_toggleswitch:
    
    autocorrs_RNAs, autocorrs_Proteins, xvals = MultipleSimulationsAcf(simulation = 'gillespie_toggleswitch_')
    
    MultipleSimulationsAcf_Plot_samplingtime(title = "Toggle Switch model")
    
#PROVA CON LE LINEE.    



