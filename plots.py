# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 13:04:08 2022

@author: asus
"""

import argparse
import configparser
from collections import namedtuple 

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns
from collections import Counter 
import scipy.stats as st



config = configparser.ConfigParser()



parser = argparse.ArgumentParser()



parser.add_argument("filename", help="read configuration file.")

parser.add_argument("-theoretical_poisson", help="plot simulation states distribution and theoretical distribution of SSA results.", action = "store_true")

parser.add_argument("-distribution", help="plot simulation states distribution of SSA results.", action = "store_true")

parser.add_argument("-tauleap_distribution", help="plot simulation states distribution given by tauleap algorithm.", action = "store_true")

parser.add_argument("-hybrid_distribution", help="plot simulation states distribution given by SSA/Tau-leap algorithm.", action = "store_true")

parser.add_argument("-ssa_hybrid_distribution", help="plot simulation states distribution of first model using SSA.", action = "store_true")

parser.add_argument("-all_distributions", help="plot simulation states distribution and theoretical distribution given by all algorithms.", action = "store_true")

parser.add_argument("-ssa_autorepressor_distribution", help="plot simulation states distribution of autorepressor model using SSA.", action = "store_true")

parser.add_argument("-ssa_hybrid_autorepressor_distribution", help="plot SSA simulation states distribution vs hybrid simulation states distribution of autorepressor model.", action = "store_true")



parser.add_argument("-time_plot", help="plot gene activity and number of molecules as function of time of SSA results.", action = "store_true")

parser.add_argument("-two_ind_genes_time_plot", help="plot number of molecules as function of time for two indipendent genes.", action = "store_true")

parser.add_argument("-autorepressor_time_plot", help="plot number of molecules as function of time for autorepressor system.", action = "store_true")

parser.add_argument("-hybrid_autorepressor_time_plot", help="plot number of molecules as function of time for autorepressor system using hybrid simulation results.", action = "store_true")

parser.add_argument("-hybrid_toggleswitch_time_plot", help="plot number of molecules as function of time for autorepressor system using hybrid simulation results.", action = "store_true")

parser.add_argument("-toggleswitch_time_plot", help="plot number of molecules as function of time for toggle switch system.", action = "store_true")

parser.add_argument("-tauleaptime_plot", help="plot gene activity and number of molecules as function of time of tauleap algorithm results.", action = "store_true")

parser.add_argument("-hybridtime_plot", help="plot gene activity and number of molecules as function of time of SSA/Tau-leap algorithm results.", action = "store_true")



parser.add_argument("-remove_warmup", help="plot gene activity and number of molecules as function of time without warmup period.", action = "store_true")

parser.add_argument("-multiple_simulations", help="plot number of molecules as function of time for multiple simulations. It has to be called after one of the flags that refers to a time plot.", action = "store_true")

parser.add_argument("-remove_warmup_multiple_simulations", help="plot number of molecules as function of time for multiple simulations without warmup period.", action = "store_true")



parser.add_argument("-all_plots", help="makes all possible plots provided by plots.py", action = "store_true")



args = parser.parse_args()

config.read(args.filename)



if args.two_ind_genes_time_plot or args.toggleswitch_time_plot or args.hybrid_toggleswitch_time_plot:
    
    def read_k_values_2_ind_genes():
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
    
    rate = read_k_values_2_ind_genes()

else:
           
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

#results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")

#tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")



def MoleculesVsTimePlot(df):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time
    """
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    ax[0].plot(df['Time'], df['Gene activity'])
    ax[0].set_ylabel('Gene Activity')
    ax[0].set_xlabel('Time')
    
    if args.tauleaptime_plot:
        
        ax[0].text(0.9,0.3,r"$\tau$={}".format(dt), 
                   ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[0].transAxes)
        
    ax[0].text(0.9,0.8,"$K_a$=$n_i${}\n $K_i$=$n_a${}".format(rate.ka,rate.ki), 
               ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[0].transAxes)
    
    ax[1].plot(df['Time'], df['Number of RNA molecules'])
    ax[1].set_ylabel('# of RNA molecules')
    ax[1].set_xlabel('Time')
    ax[1].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1, rate.k2), 
               ha ='center', va = 'center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    
    ax[2].plot(df['Time'], df['Number of proteins'])
    ax[2].set_ylabel('# of proteins')
    ax[2].set_xlabel('Time')
    ax[2].text(0.9,0.8,"$K_3$=m{}\n $K_4$=p{}".format(rate.k3, rate.k4), 
               ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[2].transAxes)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()


    
def MoleculesVsTimePlot_2_ind_genes(df):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time in 
    the case of two indipendent genes.
    """
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
    
    ax[0].plot(df['Time'], df['Number of RNAs gene1'], label = 'gene 1', color = 'blue')
    ax[0].plot(df['Time'], df['Number of RNAs gene2'], label = 'gene 2', color = 'cyan')
    ax[0].set_ylabel('# of RNA molecules')
    ax[0].set_xlabel('Time')
    ax[0].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}\n $K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1_1, rate.k2_1, rate.k1_2, rate.k2_2), 
               ha ='center', va = 'center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[0].transAxes)
    
    ax[0].legend(loc = 2, prop={'size': 7})
    
    ax[1].plot(df['Time'], df['Number of proteins gene1'], color = 'blue')
    ax[1].plot(df['Time'], df['Number of proteins gene2'], color = 'cyan')
    ax[1].set_ylabel('# of proteins')
    ax[1].set_xlabel('Time')
    ax[1].text(0.9,0.8,"$K_3$=m{}\n $K_4$=p{}\n $K_3$=m{}\n $K_4$=p{}".format(rate.k3_1, rate.k4_1, rate.k3_2, rate.k4_2), 
               ha='center', va='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()
    


def MoleculesVsTimePlot_toggleswitch(df):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time in 
    the case of two indipendent genes.
    """
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    
    ax[0].plot(df['Time'], df['Gene1 activity'], label = 'gene 1', color = 'blue')
    ax[0].plot(df['Time'], df['Gene2 activity'], label = 'gene 2', color = 'cyan', alpha = 0.5)
    ax[0].set_ylabel('Genes activity')
    ax[0].set_xlabel('Time')
    ax[0].text(0.9,0.8,"$K_a$=$n_i${}\n $K_i$=$n_a${}".format(rate.ka_1, rate.ki_1), 
               ha ='center', va = 'center', fontsize=7, bbox=dict(facecolor='white', alpha=0.9, edgecolor = 'blue'),
               transform = ax[0].transAxes)
    ax[0].text(0.9,0.3,"$K_a$=$n_i${}\n $K_i$=$n_a${}".format(rate.ka_2, rate.ki_2), 
               ha ='center', va = 'bottom', fontsize=7, bbox=dict(facecolor='white', alpha=0.9, edgecolor = 'cyan'),
               transform = ax[0].transAxes)
    
    ax[0].legend(loc = 2, prop={'size': 7})
    
    ax[1].plot(df['Time'], df['Number of RNAs gene1'], label = 'gene 1', color = 'blue')
    ax[1].plot(df['Time'], df['Number of RNAs gene2'], label = 'gene 2', color = 'cyan')
    ax[1].set_ylabel('# of RNA molecules')
    ax[1].set_xlabel('Time')
    ax[1].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}\n $K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1_1, rate.k2_1, rate.k1_2, rate.k2_2), 
               ha ='center', va = 'center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    
    ax[2].plot(df['Time'], df['Number of proteins gene1'], color = 'blue')
    ax[2].plot(df['Time'], df['Number of proteins gene2'], color = 'cyan')
    ax[2].set_ylabel('# of proteins')
    ax[2].set_xlabel('Time')
    ax[2].text(0.9,0.8,"$K_3$=m{}\n $K_4$=p{}\n $K_3$=m{}\n $K_4$=p{}".format(rate.k3_1, rate.k4_1, rate.k3_2, rate.k4_2), 
               ha='center', va='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[2].transAxes)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()



def generate_RNA_distribution(df):
    """ This function creates a Counter with RNA state values 
    as keys and normalized residency time as values
    """
    RNA_distribution = Counter()
    for state, residency_time in zip(df['Number of RNA molecules'], df['Residency Time']):
        RNA_distribution[state] += residency_time
    
    total_time_observed = sum(RNA_distribution.values())
    for state in RNA_distribution:
        RNA_distribution[state] /= total_time_observed

    return RNA_distribution 



def generate_protein_distribution(df):
    """ This function creates a Counter with protein state values 
    as keys and normalized residency time as values
    """
    protein_distribution = Counter()
    for state, residency_time in zip(df['Number of proteins'], df['Residency Time']):
        protein_distribution[state] += residency_time
    
    total_time_observed = sum(protein_distribution.values())
    for state in protein_distribution:
        protein_distribution[state] /= total_time_observed
    
    return protein_distribution 



def StatesDistributionPlot(df,df_1=None):
    """ This function plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(df)
    
    protein_distribution = generate_protein_distribution(df)
    

    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    
    if args.hybrid_distribution or args.tauleap_distribution or args.distribution:


        ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
        ax[0].set_ylabel('Normalized residency time', fontsize=10)
        ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
        if args.theoretical_poisson:
            values = np.arange(20)
            pmf = st.poisson(10).pmf(values) 
            ax[0].bar(values, pmf, alpha=0.5)
        ax[0].set_title('First basic model (ka={},ki={})'.format(rate.ka,rate.ki), fontsize=10)
     
        if args.hybrid_distribution:
            simulation = "Hybrid Simulation"
        elif args.tauleap_distribution:
            simulation = "Tau-leap"
        elif args.distribution:
            simulation = "SSA"
        
        ax[0].legend(["{} Simulation".format(simulation)], fontsize=10)
        
        ax[1].bar(protein_distribution.keys(), protein_distribution.values())
        ax[1].set_ylabel('Normalized residency time', fontsize=10)
        ax[1].set_xlabel('Number of proteins', fontsize=10)
        ax[1].legend(["{} Simulation".format(simulation)], fontsize=10)
        if args.theoretical_poisson:
            values = np.arange(20)
            pmf = st.poisson(1).pmf(values) 
            ax[1].bar(values, pmf, alpha=0.5)
    
    elif args.ssa_autorepressor_distribution:
        
        simulation = "SSA"

        ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
        ax[0].set_ylabel('Normalized residency time', fontsize=10)
        ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
        ax[0].set_title('Autorepressor (ka={}, ki={})'.format(rate.ka,rate.ki), fontsize=14)
        
        ax[0].legend(["{} Simulation".format(simulation)], fontsize=10)
        
        ax[1].bar(protein_distribution.keys(), protein_distribution.values())
        ax[1].set_ylabel('Normalized residency time', fontsize=10)
        ax[1].set_xlabel('Number of proteins', fontsize=10)
        
    elif args.ssa_hybrid_autorepressor_distribution or args.ssa_hybrid_distribution:
        
         RNA_distribution_hybrid = generate_RNA_distribution(df_1)
        
         protein_distribution_hybrid = generate_protein_distribution(df_1)
         
         #simulation = "SSA vs Hybrid"

         ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
         ax[0].bar(RNA_distribution_hybrid.keys(), RNA_distribution_hybrid.values(), alpha=0.7)
         ax[0].set_ylabel('Normalized residency time', fontsize=10)
         ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
         ax[0].set_title('Autorepressor (ka={}, ki={})'.format(rate.ka,rate.ki), fontsize=14)
         
         ax[0].legend(["SSA Simulation","Hybrid simulation"], fontsize=10)
         
         ax[1].bar(protein_distribution.keys(), protein_distribution.values())
         ax[1].bar(protein_distribution_hybrid.keys(), protein_distribution_hybrid.values(), alpha=0.7)
         ax[1].set_ylabel('Normalized residency time', fontsize=10)
         ax[1].set_xlabel('Number of proteins', fontsize=10)
    
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()



def AllStatesDistributionPlot():
    """ This function plots the probability distribution of 
    observing each state
    """
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")

    RNA_distribution = generate_RNA_distribution(df = results)
    
    protein_distribution = generate_protein_distribution(df = results)
    
    tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")
    
    RNA_distribution_tauleap = generate_RNA_distribution(df = tauleap_results)
    
    protein_distribution_tauleap = generate_protein_distribution(df = tauleap_results)
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    values = np.arange(20)
    pmf = st.poisson(10).pmf(values) 
    ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
    ax[0].bar(RNA_distribution_tauleap.keys(), RNA_distribution_tauleap.values(), alpha=0.9)
    ax[0].set_ylabel('Normalized residency time', fontsize=10)
    ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
    ax[0].bar(values, pmf, alpha=0.5)
    
    ax[0].legend(["SSA","Tau-leap","Poisson distribution"], fontsize=10, loc = "upper right")
    
    ax[1].bar(protein_distribution.keys(), protein_distribution.values())
    ax[1].bar(protein_distribution_tauleap.keys(), protein_distribution_tauleap.values(), alpha=0.5)
    ax[1].set_ylabel('Normalized residency time', fontsize=10)
    ax[1].set_xlabel('Number of proteins', fontsize=10)
    ax[1].bar(values, pmf, alpha=0.5)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()
    
    

   
def ssa_gene_activity_distribution():
    """
    This function plots the states of gene.
    """
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    gene_activity = results['Gene activity']
    
    gene_activity = np.ascontiguousarray(gene_activity)
    
    plt.hist(gene_activity)
    
    gene_activity = gene_activity.tolist()
    print("Number of times gene is active {}".format(gene_activity.count(1)))
    print("Number of times gene is inactive {}".format(gene_activity.count(0)))
    
    plt.show()

ssa_gene_activity_distribution() 

def hybrid_gene_activity_distribution():
    """
    This function plots the states of gene.
    """
    results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    gene_activity = results['Gene activity']
    
    gene_activity = np.ascontiguousarray(gene_activity)
    
    plt.hist(gene_activity)

    gene_activity = gene_activity.tolist()
    print("Number of times gene is active {}".format(gene_activity.count(1)))
    print("Number of times gene is inactive {}".format(gene_activity.count(0)))
 
    plt.show()   
    
hybrid_gene_activity_distribution()
    
#%%

if args.distribution:
    
    ssa_results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = ssa_results)
   
    


if args.hybrid_distribution:
    
    hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = hybrid_results)
    
    
    
if args.ssa_hybrid_distribution:
    
    ssa_results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")    

    hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = ssa_results, df_1 = hybrid_results)
    



if args.tauleap_distribution:
    
    tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = tauleap_results)



if args.ssa_autorepressor_distribution:
    
    autorepressor_results = pd.read_csv('autorepressor_gillespiesimulation_results.csv', sep=" ")

    StatesDistributionPlot(df = autorepressor_results)



if args.ssa_hybrid_autorepressor_distribution:
    
    ssa_autorepressor_results = pd.read_csv('hybridsimulation_autorepressor_results.csv', sep=" ")
    hybrid_autorepressor_results = pd.read_csv('autorepressor_gillespiesimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = ssa_autorepressor_results, df_1 = hybrid_autorepressor_results)


if args.all_distributions:
    
    AllStatesDistributionPlot()

#%% Time plot and multiple simulations

dataframes_list = []

if args.time_plot:
    
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")

    MoleculesVsTimePlot(df = results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespieresults_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
    


if args.two_ind_genes_time_plot:
    
    results = pd.read_csv('gillespiesimulation_2_ind_genes_results.csv', sep=" ")

    MoleculesVsTimePlot_2_ind_genes(df = results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespieresults_2_ind_genes_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)



if args.autorepressor_time_plot:
    
    results = pd.read_csv('autorepressor_gillespiesimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespie_autorepressor_results_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
        
if args.hybrid_autorepressor_time_plot:
    
    results = pd.read_csv('hybridsimulation_autorepressor_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('hybrid_autorepressor_results_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)

    
if args.toggleswitch_time_plot:
    
    results = pd.read_csv('toggleswitch_gillespiesimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot_toggleswitch(df = results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespieresults_toggleswitch_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)


if args.hybrid_toggleswitch_time_plot:
    
    results = pd.read_csv('hybridsimulation_toggleswitch_results.csv', sep=" ")
    
    MoleculesVsTimePlot_toggleswitch(df = results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('hybrid_toggleswitch_results_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)




if args.tauleaptime_plot:
    
    tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = tauleap_results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('tauleapresults_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)



if args.hybridtime_plot:
    
    hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = hybrid_results)
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('hybridresults_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)



def MultipleSimulationsPlot(dataframes):
    """
    This function makes a multiplot with one column and two rows:
    top plot for the number of RNA molecules produced vs time 
    and down plot for the number of proteins produced vs time.

    Parameters
    ----------
    N : int > 0
        number of simulations.
    """
        
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,10))
    for dataframe in dataframes:
        ax[0].plot(dataframe['Time'], dataframe['Number of RNA molecules'])
        ax[0].set_ylabel('# of RNA molecules')
        ax[0].set_xlabel('Time')
        ax[1].plot(dataframe['Time'], dataframe['Number of proteins'])
        ax[1].set_ylabel('# of proteins')
        ax[1].set_xlabel('Time')
        sns.despine(fig, bottom=False, left=False)
    plt.show()
    
    

def MultipleSimulationsPlot_2_ind_genes(dataframes):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time in 
    the case of two indipendent genes.
    """
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
    for df in dataframes:
    
        ax[0].plot(df['Time'], df['Number of RNAs gene1'], label = 'gene 1')
        ax[0].plot(df['Time'], df['Number of RNAs gene2'], label = 'gene 2')
        ax[0].set_ylabel('# of RNA molecules')
        ax[0].set_xlabel('Time')
        ax[0].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}\n $K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1_1, rate.k2_1, rate.k1_2, rate.k2_2), 
                   ha ='center', va = 'center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[0].transAxes)
        
        ax[0].legend()
        
        ax[1].plot(df['Time'], df['Number of proteins gene1'])
        ax[1].plot(df['Time'], df['Number of proteins gene2'])
        ax[1].set_ylabel('# of proteins')
        ax[1].set_xlabel('Time')
        ax[1].text(0.9,0.8,"$K_3$=m{}\n $K_4$=p{}\n $K_3$=m{}\n $K_4$=p{}".format(rate.k3_1, rate.k4_1, rate.k3_2, rate.k4_2), 
                   ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[1].transAxes)
        
        sns.despine(fig, bottom=False, left=False)
    plt.show()
    


def MultipleSimulationsPlot_toggleswitch(dataframes):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time in 
    the case of two indipendent genes.
    """
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    
    for df in dataframes:
    
        ax[0].plot(df['Time'], df['Gene1 activity'], label = 'gene 1', color = 'blue')
        ax[0].plot(df['Time'], df['Gene2 activity'], label = 'gene 2', color = 'cyan')
        ax[0].set_ylabel('Genes activity')
        ax[0].set_xlabel('Time')
        ax[0].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}\n $K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1_1, rate.k2_1, rate.k1_2, rate.k2_2), 
                   ha ='center', va = 'center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[0].transAxes)
        
        ax[0].legend(loc = 2, prop={'size': 7})
        
        ax[1].plot(df['Time'], df['Number of RNAs gene1'], label = 'gene 1', color = 'blue')
        ax[1].plot(df['Time'], df['Number of RNAs gene2'], label = 'gene 2', color = 'cyan')
        ax[1].set_ylabel('# of RNA molecules')
        ax[1].set_xlabel('Time')
        ax[1].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}\n $K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1_1, rate.k2_1, rate.k1_2, rate.k2_2), 
                   ha ='center', va = 'center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[1].transAxes)
        
        ax[2].plot(df['Time'], df['Number of proteins gene1'], color = 'blue')
        ax[2].plot(df['Time'], df['Number of proteins gene2'], color = 'cyan')
        ax[2].set_ylabel('# of proteins')
        ax[2].set_xlabel('Time')
        ax[2].text(0.9,0.8,"$K_3$=m{}\n $K_4$=p{}\n $K_3$=m{}\n $K_4$=p{}".format(rate.k3_1, rate.k4_1, rate.k3_2, rate.k4_2), 
                   ha='center', va='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[2].transAxes)
        
        sns.despine(fig, bottom=False, left=False)
    plt.show()
    

if args.two_ind_genes_time_plot and args.multiple_simulations:
    MultipleSimulationsPlot_2_ind_genes(dataframes = dataframes_list)
    
if args.toggleswitch_time_plot and args.multiple_simulations:
    MultipleSimulationsPlot(dataframes = dataframes_list)
    
if any([args.time_plot, args.tauleaptime_plot, args.hybridtime_plot]) and args.multiple_simulations:
    MultipleSimulationsPlot(dataframes = dataframes_list)



#%% Remove warmup part

if args.remove_warmup:
    
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    
    removedwarmup_results = results[results['Time'] > warmup_time]
    
    MoleculesVsTimePlot(df = removedwarmup_results)



removedwarmup_dataframes_list = []

for simulation_results in dataframes_list:
    removedwarmup_results = simulation_results[simulation_results['Time'] > warmup_time]
    removedwarmup_dataframes_list.append(removedwarmup_results)


if args.remove_warmup_multiple_simulations:
    MultipleSimulationsPlot(dataframes = removedwarmup_dataframes_list)



#%%



if args.all_plots:
    
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = results)
    
    MoleculesVsTimePlot(df = results)
    
    removedwarmup_results = results[results['Time'] > warmup_time]
    
    MoleculesVsTimePlot(df = removedwarmup_results)
    
    MultipleSimulationsPlot(dataframes = dataframes_list)
    
    MultipleSimulationsPlot(dataframes = removedwarmup_dataframes_list)
    
    tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = tauleap_results)
    
    MoleculesVsTimePlot(df = tauleap_results)
    
    removedwarmup_results = tauleap_results[tauleap_results['Time'] > warmup_time]
    
    MoleculesVsTimePlot(df = removedwarmup_results)
    
#    MultipleSimulationsPlot(dataframes = dataframes_list)
    
#    MultipleSimulationsPlot(dataframes = removedwarmup_dataframes_list)

    