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
import scipy.interpolate as interp
import math

import matplotlib.pylab as plt
import seaborn as sns
from collections import Counter 
import scipy.stats as st

import os


config = configparser.ConfigParser()



parser = argparse.ArgumentParser()



parser.add_argument("filename", help="read configuration file.")

parser.add_argument("-theoretical_poisson", help="plot simulation states distribution and theoretical distribution of SSA results.", action = "store_true")

parser.add_argument("-distribution", help="plot simulation states distribution of SSA results.", action = "store_true")

parser.add_argument("-tauleap_distribution", help="plot simulation states distribution given by tauleap algorithm.", action = "store_true")

parser.add_argument("-ssa_tauleap_distribution", help="plot simulation states distribution given by SSA vs Tau-leap algorithms", action = "store_true")

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

parser.add_argument("-nfkb_timeplot", help="plot nfkb gene expression.", action = "store_true")


parser.add_argument("-plot_mean",help="plot mean of N simulations", action = "store_true")


parser.add_argument("-remove_warmup_time_plot", help="plot number of molecoles vs time given by SSA removing initial warmup period.", action = "store_true")

parser.add_argument("-remove_warmup_tauleaptime_plot", help="plot number of molecoles vs time given by tauleap removing initial warmup period", action = "store_true")

parser.add_argument("-remove_warmup", help="plot gene activity and number of molecules as function of time without warmup period.", action = "store_true")


parser.add_argument("-multiple_simulations", help="plot number of molecules as function of time for multiple simulations. It has to be called after one of the flags that refers to a time plot.", action = "store_true")

parser.add_argument("-remove_warmup_multiple_simulations", help="plot number of molecules as function of time for multiple simulations without warmup period.", action = "store_true")



#parser.add_argument("-all_plots", help="makes all possible plots provided by plots.py", action = "store_true")



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



def MoleculesVsTimePlot(df, df_1=None):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time
    """
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    ax[0].plot(df['Time'], df['Gene activity'])
    ax[0].set_ylabel('Gene Activity')
    ax[0].set_xlabel('Time (a.u.)')
    ax[0].set_yticks([0,1])
    ax[0].set_yticklabels(['inactive','active'])

    if args.tauleaptime_plot:
        
        ax[0].text(0.9,0.3,r"$\tau$={}".format(dt), 
                   ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
                   transform = ax[0].transAxes)
        
    ax[0].text(0.9,0.8,"$K_a$=$n_i${}\n $K_i$=$n_a${}".format(rate.ka,rate.ki), 
               ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[0].transAxes)
    
    ax[1].plot(df['Time'], df['Number of RNA molecules'])
    ax[1].set_ylabel('# of RNA molecules')
    ax[1].set_xlabel('Time (a.u.)')
    ax[1].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1, rate.k2), 
               ha ='center', va = 'center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    
    ax[2].plot(df['Time'], df['Number of proteins'])
    ax[2].set_ylabel('# of proteins')
    ax[2].set_xlabel('Time (a.u.)')
    ax[2].text(0.9,0.8,"$K_3$=m{}\n $K_4$=p{}".format(rate.k3, rate.k4), 
               ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[2].transAxes)
    
    sns.despine(fig, bottom=False, left=False)
    plt.show()

def MoleculesVsTimePlot_nfkb(df):
    """This function plots gene activity, the number of RNA molecules
    produced vs time and the number of proteins produced vs time
    """
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
    #ax[0].plot(df['Time'], df['NFkB activity'])
    #ax[0].set_ylabel('NFkB Activity')
    #ax[0].set_xlabel('Time (a.u.)')
    #ax[0].set_yticks([0,1])
    #ax[0].set_yticklabels(['inactive','active'])
    
    ax.plot(df['Time'], df['Number of RNA molecules'])
    ax.set_ylabel('# of RNA molecules')
    ax.set_xlabel('Time (a.u.)')
    
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
    ax[0].set_yticks([0,1])
    ax[0].set_yticklabels(['inactive','active'])
    ax[0].set_ylabel('Genes activity')
    ax[0].set_xlabel('Time (a.u.)')
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
    ax[1].set_xlabel('Time (a.u.)')
    ax[1].text(0.9,0.8,"$K_1$=$n_a${}\n $K_2$=m{}\n $K_1$=$n_a${}\n $K_2$=m{}".format(rate.k1_1, rate.k2_1, rate.k1_2, rate.k2_2), 
               ha ='center', va = 'center', fontsize=7, bbox=dict(facecolor='white', alpha=0.5),
               transform = ax[1].transAxes)
    
    ax[2].plot(df['Time'], df['Number of proteins gene1'], color = 'blue')
    ax[2].plot(df['Time'], df['Number of proteins gene2'], color = 'cyan')
    ax[2].set_ylabel('# of proteins')
    ax[2].set_xlabel('Time (a.u.)')
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



def find_warmup_methodI(data = 'gillespieresults_seed{}.csv'):
    """This function returns the first number of molecules of RNAs
    and proteins to remove because in the warmup period"""
    
    dataframes_list = []
    


    for n in range(1,N+1):
        simulation_results = pd.read_csv(data.format(n), sep=" ")
        dataframes_list.append(simulation_results)
    
    RNAs_list = []
    
    #Occorre interpolare perché i dati non sono stati presi tutti ad intervalli regolari
    
    for dataframe in dataframes_list:
        
        time = dataframe['Time']
        
        xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
        
        RNAs = dataframe['Number of RNA molecules']
        
        f_RNAs = interp.interp1d(time, RNAs, kind='previous')
        yinterp_RNAs = f_RNAs(xvals)
        
        RNAs_list.append(yinterp_RNAs)
    
    
    
    RNAs_arrays = np.array(RNAs_list, dtype=object)
    
    lengths = []
    for a in RNAs_arrays:
        lengths.append(len(a))
        
    
    
    min_RNAs_arrays = [RNAs_array[:min(lengths)] for RNAs_array in RNAs_arrays]
    
    
       
    mean_RNAs = np.mean(min_RNAs_arrays, axis=0)
    

    
    xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
    
    min_time_array = xvals[:min(lengths)]
    

    
    
    
    for i in np.arange(0,len(mean_RNAs)):
        if math.isclose(mean_RNAs[i], mean_RNAs[i+800],abs_tol=0.4):
            n_r = i
            break
    

    
    Proteins_list = []
    
    for dataframe in dataframes_list:
        
        Proteins_arrays = np.ascontiguousarray(dataframe['Number of proteins'])
        
        Proteins_list.append(Proteins_arrays)
    
    Proteins_arrays = np.array(Proteins_list, dtype=object)
    
    lengths = []
    for a in Proteins_arrays:
        lengths.append(len(a))
        
    
    
    min_Proteins_arrays = [Proteins_array[:min(lengths)] for Proteins_array in Proteins_arrays]
    
    
        
    mean_Proteins = np.mean(min_Proteins_arrays, axis=0)
    
    xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
    
    min_time_array = xvals[:min(lengths)]
    
    for i in np.arange(0,len(mean_Proteins)):
        if math.isclose(mean_Proteins[i], mean_Proteins[i+800],abs_tol=0.4):
            n_p = i
            break
        
    return n_r, n_p




def find_warmupRNAS_methodI(data = 'gillespieresults_seed{}.csv'):
    """This function returns the first number of molecules of RNAs
    and proteins to remove because in the warmup period"""
    
    dataframes_list = []
    


    for n in range(1,N+1):
        simulation_results = pd.read_csv(data.format(n), sep=" ")
        dataframes_list.append(simulation_results)
    
    RNAs_list = []
    
    #Occorre interpolare perché i dati non sono stati presi tutti ad intervalli regolari
    
    for dataframe in dataframes_list:
        
        time = dataframe['Time']
        
        xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
        
        RNAs = dataframe['Number of RNA molecules']
        
        f_RNAs = interp.interp1d(time, RNAs, kind='previous')
        yinterp_RNAs = f_RNAs(xvals)
        
        RNAs_list.append(yinterp_RNAs)
    
    
    
    RNAs_arrays = np.array(RNAs_list, dtype=object)
    
    lengths = []
    for a in RNAs_arrays:
        lengths.append(len(a))
        
    
    
    min_RNAs_arrays = [RNAs_array[:min(lengths)] for RNAs_array in RNAs_arrays]
    
    
       
    mean_RNAs = np.mean(min_RNAs_arrays, axis=0)
    

    
    xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
    
    min_time_array = xvals[:min(lengths)]
    

    
    
    
    for i in np.arange(0,len(mean_RNAs)):
        if math.isclose(mean_RNAs[i], mean_RNAs[i+800],abs_tol=0.4):
            n_r = i
            break
        
    return n_r




def find_warmup_methodII(data = 'gillespieresults_seed{}.csv'):
    """This function returns the first number of molecules of RNAs
    and proteins to remove because in the warmup period. The method
    used calculates the number of molecules at t+dt, the number
    of molecules at t+2dt and the number of molecules at t+3dt.
    When the difference is lower than 0.4, the number of molecules are the
    ones in the warmup period."""
    
    dataframes_list = []
    


    for n in range(1,N+1):
        simulation_results = pd.read_csv(data.format(n), sep=" ")
        dataframes_list.append(simulation_results)
    
    RNAs_list = []
    
    #Occorre interpolare perché i dati non sono stati presi tutti ad intervalli regolari
    
    for dataframe in dataframes_list:
        
        time = dataframe['Time']
        
        xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
        
        RNAs = dataframe['Number of RNA molecules']
        
        f_RNAs = interp.interp1d(time, RNAs, kind='previous')
        yinterp_RNAs = f_RNAs(xvals)
        
        RNAs_list.append(yinterp_RNAs)
    
    
    
    RNAs_arrays = np.array(RNAs_list, dtype=object)
    
    lengths = []
    for a in RNAs_arrays:
        lengths.append(len(a))
        
    
    
    min_RNAs_arrays = [RNAs_array[:min(lengths)] for RNAs_array in RNAs_arrays]
    
    
       
    mean_RNAs = np.mean(min_RNAs_arrays, axis=0)
    

    
    xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
    
    #min_time_array = xvals[:min(lengths)]
    

    
    di = 5
    
    for i in np.arange(0,len(mean_RNAs)):
        if math.isclose(mean_RNAs[i], mean_RNAs[i+di],abs_tol=0.4) and math.isclose(mean_RNAs[i+di], mean_RNAs[i+2*di],abs_tol=0.4) and math.isclose(mean_RNAs[i+2*di], mean_RNAs[i+3*di],abs_tol=0.4):
            n_r = i
            break
    

    
    Proteins_list = []
    
    for dataframe in dataframes_list:
        
        Proteins_arrays = np.ascontiguousarray(dataframe['Number of proteins'])
        
        Proteins_list.append(Proteins_arrays)
    
    Proteins_arrays = np.array(Proteins_list, dtype=object)
    
    lengths = []
    for a in Proteins_arrays:
        lengths.append(len(a))
        
    
    
    min_Proteins_arrays = [Proteins_array[:min(lengths)] for Proteins_array in Proteins_arrays]
    
    
        
    mean_Proteins = np.mean(min_Proteins_arrays, axis=0)
    
    xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
    
    #min_time_array = xvals[:min(lengths)]
    
    for i in np.arange(0,len(mean_Proteins)):
        if math.isclose(mean_Proteins[i], mean_Proteins[i+di],abs_tol=0.4) and math.isclose(mean_Proteins[i+di],mean_Proteins[i+2*di],abs_tol=0.4) and math.isclose(mean_Proteins[i+2*di],mean_Proteins[i+3*di],abs_tol=0.4):
            n_p = i
            break
        
    return n_r, n_p

#========================================================
"""
#Remove warmup in case of first protein synthesis modeling ka 1 ki 0
N=64
data = 'ka1ki0gillespieresults_seed{}.csv'

#find warmup-period
n_r, n_p = find_warmup(data = data)

    
dataframes_list = []


#remove warmup
for n in range(1,N+1):
    simulation_results = pd.read_csv(data.format(n), sep=" ")
    simulation_results = simulation_results.iloc[n_r:]
    dataframes_list.append(simulation_results)

#prepare names of new data files    
results_names = []
for n in range(1,N+1):
    results_names.append("removed_warmup_ka1ki0gillespieresults_seed"+str(n))
    
#save removed warmup data
    
actual_dir = os.getcwd()

file_path = r'{}\{}.csv'

for dataframe, results in zip(dataframes_list, results_names):
    dataframe.to_csv(file_path.format(actual_dir,results), sep=" ", index = None, header=True)


#df = pd.read_csv('ka1ki0gillespieresults_seed1.csv', sep=" ")
#df
#df.iloc[4:]

#Remove warmup in case of first protein synthesis modeling ka 1 ki 0.5
N=64
data = 'ka1ki0.5gillespieresults_seed{}.csv'

#find warmup-period
n_r, n_p = find_warmup(data = data)

    
dataframes_list = []


#remove warmup
for n in range(1,N+1):
    simulation_results = pd.read_csv(data.format(n), sep=" ")
    simulation_results = simulation_results.iloc[n_r:]
    dataframes_list.append(simulation_results)

#prepare names of new data files    
results_names = []
for n in range(1,N+1):
    results_names.append("removed_warmup_ka1ki0.5gillespieresults_seed"+str(n))
    
#save removed warmup data
    
actual_dir = os.getcwd()

file_path = r'{}\{}.csv'

for dataframe, results in zip(dataframes_list, results_names):
    dataframe.to_csv(file_path.format(actual_dir,results), sep=" ", index = None, header=True)




#Remove warmup in case of NF-kB products in case of one simulation
df = pd.read_csv("nfkb_k20k2i0gillespiesimulation_results.csv", sep=" ")
df = df.iloc[100:]
df.to_csv(file_path.format(actual_dir,"removed_warmup_nfkb_k20k2i0gillespiesimulation_results"), sep =" ", index = None, header=True, mode = "w") 

N=64
data = 'nfkb_k20k2i0gillespiesimulation_results_seed{}.csv'

#find warmup-period
#n_r = find_warmupRNAS(data = data)

    
dataframes_list = []


#remove warmup
for n in range(1,N+1):
    simulation_results = pd.read_csv(data.format(n), sep=" ")
    simulation_results = simulation_results.iloc[100:]
    dataframes_list.append(simulation_results)

#prepare names of new data files    
results_names = []
for n in range(1,N+1):
    results_names.append("removed_warmup_nfkb_k20k2i0gillespiesimulation_results_seed"+str(n))
    
#save removed warmup data
    
actual_dir = os.getcwd()

file_path = r'{}\{}.csv'

for dataframe, results in zip(dataframes_list, results_names):
    dataframe.to_csv(file_path.format(actual_dir,results), sep=" ", index = None, header=True)

"""


#results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
#results[0:10]
#results = results[:10]
#results = results.iloc[10:]
#results



"""
#===============================================================

n_r, n_p = find_warmup_method1(data='ka1ki0.5gillespieresults_seed{}.csv')

results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
results = results[n_r:]
plt.plot(results['Time'], results['Number of proteins'])


plt.plot(results['Time'],results['Number of RNA molecules'])

#Delete the first rows RNAs
results_RNAs = results.drop('Number of proteins',axis=1)
results_RNAs_ssa = results_RNAs[n_r:]

plt.plot(results_RNAs_ssa['Time'],results_RNAs_ssa['Number of RNA molecules'])

#Delete the first rows proteins
results_proteins = results.drop('Number of RNA molecules',axis=1)
results_proteins_ssa = results_proteins[n_p:]

#========================================================

n_r, n_p = remove_warmup(data = 'tauleapresults_seed{}.csv')

results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")

#Delete the first rows RNAs
results_RNAs = results.drop('Number of proteins',axis=1)
results_RNAs_tauleap = results_RNAs[n_r:]

#Delete the first rows proteins
results_proteins = results.drop('Number of RNA molecules',axis=1)
results_proteins_tauleap = results_proteins[n_p:]

#========================================================

n_r, n_p = remove_warmup(data = 'hybridsimulation_results_seed{}.csv')

results = pd.read_csv('hybridsimulation_results.csv', sep=" ")

#Delete the first rows RNAs
results_RNAs = results.drop('Number of proteins',axis=1)
results_RNAs_hybrid = results_RNAs[n_r:]

#Delete the first rows proteins
results_proteins = results.drop('Number of RNA molecules',axis=1)
results_proteins_hybrid = results_proteins[n_p:]

#========================================================
"""
def StatesDistributionPlot(df_1=None, df_2=None, df_3=None,df_4=None):
    """ This function plots the probability distribution of 
    observing each state
    """
    RNA_distribution = generate_RNA_distribution(df_1)
    
    protein_distribution = generate_protein_distribution(df_2)
    

    
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
        
    elif args.ssa_hybrid_autorepressor_distribution:
        
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
         
    elif args.ssa_hybrid_distribution or args.ssa_tauleap_distribution:
        
         RNA_distribution_hybrid = generate_RNA_distribution(df_3)
        
         protein_distribution_hybrid = generate_protein_distribution(df_4)
         
         #simulation = "SSA vs Hybrid"

         ax[0].bar(RNA_distribution.keys(), RNA_distribution.values())
         ax[0].bar(RNA_distribution_hybrid.keys(), RNA_distribution_hybrid.values(), alpha=0.7)
         ax[0].set_ylabel('Normalized residency time', fontsize=10)
         ax[0].set_xlabel('Number of RNA molecules', fontsize=10)
         ax[0].set_title('First protein synthesis model (ka={}, ki={})'.format(rate.ka,rate.ki), fontsize=14)
         
         if args.ssa_hybrid_distribution:
             
             ax[0].legend(["SSA simulation","Hybrid simulation"], fontsize=10)
             
         elif args.ssa_tauleap_distribution:
             
             ax[0].legend(["SSA simulation","Tauleap simulation \u03C4 = {}".format(dt)], fontsize=10)
         
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
    
    

"""   
def ssa_gene_activity_distribution():
    
    #This function plots the states of gene.
    
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
    
    #This function plots the states of gene.
    
    results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    gene_activity = results['Gene activity']
    
    gene_activity = np.ascontiguousarray(gene_activity)
    
    plt.hist(gene_activity)

    gene_activity = gene_activity.tolist()
    print("Number of times gene is active {}".format(gene_activity.count(1)))
    print("Number of times gene is inactive {}".format(gene_activity.count(0)))
 
    plt.show()   
    
hybrid_gene_activity_distribution()
#Nel caso dell'ibrido rimane più attivo quindi può essere la causa del fatto
#che quando dovremmo osservare degli stati a 0 questi non ci sono.
#35818-29977 = 5841 volte più attivo nell'ibrido nel caso ka=0.01 e ki=0.01. 
#5914-4183 = 1731.



def generate_geneactivity_distribution(df):
    #This function creates a Counter with active gene state values 
    #as keys and normalized residency time as values
    
    activitygene_distribution = Counter()
    for state, residency_time in zip(df['Gene activity'], df['Residency Time']):
        activitygene_distribution[state] += residency_time
    
    total_time_observed = sum(activitygene_distribution.values())
    for state in activitygene_distribution:
        activitygene_distribution[state] /= total_time_observed

    return activitygene_distribution 


def ssa_gene_activity_distribution_PLOT():
    
    ssa_results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    
    ssa_geneactivity_distribution = generate_geneactivity_distribution(df = ssa_results)
        
    hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    
    hybrid_geneactivity_distribution = generate_geneactivity_distribution(df = hybrid_results)
        
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
    ax.bar(ssa_geneactivity_distribution.keys(), ssa_geneactivity_distribution.values(), width = 0.1)
    ax.bar(hybrid_geneactivity_distribution.keys(), hybrid_geneactivity_distribution.values(), width = 0.1, alpha=0.7)
    ax.set_ylabel('Normalized residency time', fontsize=10)
    ax.set_xlabel('Gene state', fontsize=10)
    ax.legend(["SSA Simulation","Hybrid simulation"], fontsize=10)
    #ax.set_xticklabels([0,1])
    ax.set_xticks([0,1])
    plt.show()

ssa_gene_activity_distribution_PLOT()
"""    
    
    
    
#%%

if args.distribution:
    
    ssa_results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df_1 = ssa_results, df_2 = ssa_results)
   
    


if args.hybrid_distribution:
    
    hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    
    StatesDistributionPlot(df = hybrid_results)
    


if args.ssa_tauleap_distribution:
    
    ssa_results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")    

    tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")
    
    results_RNAs_ssa = ssa_results['Number of RNA molecules'].iloc[100:]

    results_proteins_ssa = ssa_results['Number of proteins'].iloc[100:] 
    
    results_RNAs_tauleap = tauleap_results['Number of RNA molecules'].iloc[100:]
    
    results_proteins_tauleap = tauleap_results['Number of proteins'].iloc[100:]
    
    #StatesDistributionPlot(df = ssa_results, df_1 = tauleap_results)
    StatesDistributionPlot(df_1=results_RNAs_ssa,
                           df_2=results_proteins_ssa, 
                           df_3=results_RNAs_tauleap, 
                           df_4=results_proteins_tauleap)


    
if args.ssa_hybrid_distribution:
    
    ssa_results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")   
    
    results_RNAs_ssa = ssa_results['Number of RNA molecules'].iloc[100:]

    results_proteins_ssa = ssa_results['Number of proteins'].iloc[100:] 

    hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    
    results_RNAs_hybrid = ssa_results['Number of RNA molecules'].iloc[100:]

    results_proteins_hybrid = ssa_results['Number of proteins'].iloc[100:] 
    
    #StatesDistributionPlot(df = ssa_results, df_1 = hybrid_results)
    
    StatesDistributionPlot(df_1=results_RNAs_ssa,
                           df_2=results_proteins_ssa, 
                           df_3=results_RNAs_hybrid, 
                           df_4=results_proteins_hybrid)


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

#%% Time plot 



if args.time_plot:
    
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")

    MoleculesVsTimePlot(df = results)
    

 
if args.remove_warmup_time_plot:
    
    MoleculesVsTimePlot(df = results_RNAs_ssa, df_1=results_proteins_ssa)


if args.two_ind_genes_time_plot:
    
    results = pd.read_csv('gillespiesimulation_2_ind_genes_results.csv', sep=" ")

    MoleculesVsTimePlot_2_ind_genes(df = results)
    




if args.autorepressor_time_plot:
    
    results = pd.read_csv('autorepressor_gillespiesimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = results)
    
    #for n in range(1,N+1):
    #    simulation_results = pd.read_csv('gillespie_autorepressor_results_seed{}.csv'.format(n), sep=" ")
    #    dataframes_list.append(simulation_results)
        
if args.hybrid_autorepressor_time_plot:
    
    results = pd.read_csv('hybridsimulation_autorepressor_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = results)
    
    #for n in range(1,N+1):
    #    simulation_results = pd.read_csv('hybrid_autorepressor_results_seed{}.csv'.format(n), sep=" ")
    #    dataframes_list.append(simulation_results)

    
if args.toggleswitch_time_plot:
    
    results = pd.read_csv('toggleswitch_gillespiesimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot_toggleswitch(df = results)
    



if args.hybrid_toggleswitch_time_plot:
    
    results = pd.read_csv('hybridsimulation_toggleswitch_results.csv', sep=" ")
    
    MoleculesVsTimePlot_toggleswitch(df = results)
    
    #for n in range(1,N+1):
    #    simulation_results = pd.read_csv('hybrid_toggleswitch_results_seed{}.csv'.format(n), sep=" ")
    #    dataframes_list.append(simulation_results)




if args.tauleaptime_plot:
    
    tauleap_results = pd.read_csv('tauleapsimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = tauleap_results, df_1 = tauleap_results)
    
if args.remove_warmup_tauleaptime_plot:
    
    MoleculesVsTimePlot(df = results_RNAs_tauleap, 
                        df_1=results_proteins_tauleap)
    




if args.hybridtime_plot:
    
    hybrid_results = pd.read_csv('hybridsimulation_results.csv', sep=" ")
    
    MoleculesVsTimePlot(df = hybrid_results)
    
if args.nfkb_timeplot:
    
    nfkb_results = pd.read_csv('nfkb_k20k2i0k30k3i0gillespiesimulation_results.csv', sep=" ")
    
    #nfkb_results = nfkb_results.iloc[100:]
    
    MoleculesVsTimePlot_nfkb(df = nfkb_results)



#%%Multiple simulations

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




dataframes_list = []    



if args.multiple_simulations and args.two_ind_genes_time_plot:
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespieresults_2_ind_genes_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
        
    MultipleSimulationsPlot_2_ind_genes(dataframes = dataframes_list)
    
if args.multiple_simulations and args.toggleswitch_time_plot:
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespieresults_toggleswitch_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
        
    MultipleSimulationsPlot(dataframes = dataframes_list)
    

   
 
if args.multiple_simulations and args.time_plot: 
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespieresults_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
        
    MultipleSimulationsPlot(dataframes = dataframes_list)
    
if args.multiple_simulations and args.tauleaptime_plot:
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('tauleapresults_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
    
    MultipleSimulationsPlot(dataframes = dataframes_list)
    
if args.multiple_simulations and args.hybridtime_plot:
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('hybridresults_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
    
    MultipleSimulationsPlot(dataframes = dataframes_list)
    
    




#%%
"""
data = [[240, 240, 239],
        [250, 249, 237], 
        [242, 239, 237],
        [240, 234, 233]]

np.mean(data, axis=0)
"""
if args.plot_mean:
    
    dataframes_list = []
    
    N=64
    for n in range(1,N+1):
        simulation_results = pd.read_csv('gillespieresults_seed{}.csv'.format(n), sep=" ")
        dataframes_list.append(simulation_results)
    
    
    
    """    
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    results
    results['Number of RNA molecules']
    """
    
    #time = autorepressor_results['Time']
    
    #xvals = np.arange(autorepressor_results['Time'].iloc[0], autorepressor_results['Time'].iloc[-1], 0.01)
    
    
    
    #f_RNAs = interp.interp1d(time, RNAs, kind='previous')
    #yinterp_RNAs = f_RNAs(xvals)
    
    RNAs_list = []
    
    #Occorre interpolare perché i dati non sono stati presi tutti ad intervalli regolari
    
    for dataframe in dataframes_list:
        
        time = dataframe['Time']
        
        xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
        
        RNAs = dataframe['Number of RNA molecules']
        
        f_RNAs = interp.interp1d(time, RNAs, kind='previous')
        yinterp_RNAs = f_RNAs(xvals)
        
        RNAs_list.append(yinterp_RNAs)
        
        #RNAs_arrays = np.ascontiguousarray(dataframe['Number of RNA molecules'])
        
        #RNAs_list.append(RNAs_arrays)
    
    RNAs_arrays = np.array(RNAs_list, dtype=object)
    
    lengths = []
    for a in RNAs_arrays:
        lengths.append(len(a))
        
    
    
    min_RNAs_arrays = [RNAs_array[:min(lengths)] for RNAs_array in RNAs_arrays]
    
    
       
    mean_RNAs = np.mean(min_RNAs_arrays, axis=0)
    
    #time_array = np.ascontiguousarray(results['Time'])
    #min_time_array=time_array[:min(lengths)]
    
    xvals = np.arange(dataframe['Time'].iloc[0], dataframe['Time'].iloc[-1], 0.01)
    
    min_time_array = xvals[:min(lengths)]
    
    plt.plot(min_time_array,mean_RNAs)
    
    mean_RNAs[:500]
    
    np.where(mean_RNAs > 6.)
    
    mean_RNAs[101]
    
    mean_RNAs[2418]
    
    mean_RNAs[2418:]
    
    
    #Select index where the stationary distribution starts
    for i in np.arange(0,len(mean_RNAs)):
        if math.isclose(mean_RNAs[i], mean_RNAs[i+800],abs_tol=0.4):
            n = i
            break
    
    
    
    plt.plot(min_time_array[n:],mean_RNAs[n:])
        #n = np.where(np.abs(mean_RNAs[i] - mean_RNAs[i+500]) < 0.8)
        #mean_RNAs = mean_RNAs[n:]
    
    results = pd.read_csv('gillespiesimulation_results.csv', sep=" ")
    results = results.iloc[:n]
    
    plt.plot(results['Time'],results['Number of RNA molecules'])
    
    
    
    
    Proteins_list = []
    
    for dataframe in dataframes_list:
        
        Proteins_arrays = np.ascontiguousarray(dataframe['Number of proteins'])
        
        Proteins_list.append(Proteins_arrays)
    
    Proteins_arrays = np.array(Proteins_list, dtype=object)
    
    lengths = []
    for a in Proteins_arrays:
        lengths.append(len(a))
        
    
    
    min_Proteins_arrays = [Proteins_array[:min(lengths)] for Proteins_array in Proteins_arrays]
    
    
        
    mean_Proteins = np.mean(min_Proteins_arrays, axis=0)
    
    time_array = np.ascontiguousarray(results['Time'])
    min_time_array=time_array[:min(lengths)]
    
    plt.plot(min_time_array,mean_Proteins)
    mean_Proteins[900:1000]
    
    np.where(mean_Proteins == 80.)
    
    mean_Proteins[350]

#%% Remove warmup part after having observed the time plot

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


"""
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
"""
    