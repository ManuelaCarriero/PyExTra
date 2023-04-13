# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:59:04 2022

@author: asus
"""

import pandas as pd

import numpy as np

from statsmodels.tsa import stattools
import scipy.interpolate as interp
from scipy.stats.mstats import mquantiles

import matplotlib.pylab as plt

import seaborn as sns

import argparse
import configparser

import ast



parser = argparse.ArgumentParser()

parser.add_argument("-stats_acf", help="Plot RNAs and proteins autocorrelations.", action = "store_true")
parser.add_argument("-stats_acf_toggleswitch", help="Plot RNAs and proteins autocorrelations of toggle switch model.", action = "store_true")
parser.add_argument('-f')
args = parser.parse_args()

#%% Multiple Simulations Analysis

N=64
#warmup_time = 20
dt = 10#0.01
nlags=10000

def CreateDataframesList(simulation):
    """This function calculates the RNA and protein autocorrelation of N
    simulations and returns their values as lists."""

    dataframes_list = []
    
    for n in range(1,N+1):
        simulation_results = pd.read_csv('{}results_seed{}.csv'.format(simulation, n), sep=" ")
        #removedwarmup_results = simulation_results[simulation_results['Time'] > warmup_time]
        dataframes_list.append(simulation_results)
    
    return dataframes_list

dataframes_list = CreateDataframesList(simulation='removed_warmup_nfkb_k20k2i0gillespiesimulation_')

#dataframes_list = CreateDataframesList(simulation='removed_warmup_ka1ki0.5gillespie')

#dataframes_list = CreateDataframesList(simulation='gillespie_autorepressor_')

#dataframes_list = CreateDataframesList(simulation='gillespie_toggleswitch_')

#dataframes_list = CreateDataframesList(simulation='ka0.1ki1gillespie')
"""
dataframes_list_hybrid_Imodel = CreateDataframesList(simulation='hybrid')

dataframes_list_hybrid_autorepressor_VIIIapproach = CreateDataframesList(simulation='hybrid_autorepressor_')

dataframes_list_hybrid_autorepressor_VIIapproach = CreateDataframesList(simulation='hybrid_autorepressor_')
"""


def acf_list(dataframes_list, molecule):
    """
    This function divides time signal of multiple simulations into parts
    given by dt and calculates autocorrelation for each of these

    Parameters
    ----------
    dataframes_list : list of dataframes
        list of dataframes where each dataframes contains simulations
        results for each random seed of the multiple simulations.
    molecule : str
        molecules of which you want to calculate the acf. The string to
        put in is not arbitrary since you have to choose the name 
        of the dataframe column title refered to molecule measurements.  

    Returns
    -------
    autocorrs_RNAs : list
        list of acfs calculated for each time interval of all simulations.

    """
    autocorrs_RNAs = []
    
    xvals_lst = []
    
    for df in dataframes_list:
        
        xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
        time = df['Time'] 
        RNAs = df[molecule]
        f_RNAs = interp.interp1d(time, RNAs, kind='previous')
        yinterp_RNAs = f_RNAs(xvals)
        #nlags = len(yinterp_RNAs)
        
        #Tipo
        #yinterp_RNAs = f_RNAs(xvals[0])
        #yinterp_RNAs = f_RNAs(xvals[0])
        
        autocorr_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) 

        autocorrs_RNAs.append(autocorr_RNAs)
        
        xvals_lst.append(xvals)
        
        
        
    
    return autocorrs_RNAs, xvals_lst
    
#Se fai un'interpolazione con dt = 100, poi avrai che ytinerp è time_limit= 4000/100 = 40 
#e quindi anche usi un lag di 2000, l'autocorrelazione avrà al massimo 40 punti.

#First model or autorepressor
#autocorrs_RNAs, xvals_lst = acf_list(dataframes_list = dataframes_list, molecule = 'Number of RNA molecules' )
#autocorrs_proteins, xvals_lst = acf_list(dataframes_list = dataframes_list, molecule = 'Number of proteins' )

#Toggle Switch
#autocorrs_RNAs, xvals_lst = acf_list(dataframes_list = dataframes_list, molecule = 'Number of RNAs gene1' )
#autocorrs_proteins, xvals_lst = acf_list(dataframes_list = dataframes_list, molecule = 'Number of proteins gene1')

#NFkB
autocorrs_RNAs, xvals_lst = acf_list(dataframes_list = dataframes_list, molecule = 'Number of RNA molecules' )



lst_lengths_RNAs = [len(autocorr) for autocorr in autocorrs_RNAs]
#lst_lengths_proteins = [len(autocorr) for autocorr in autocorrs_proteins]

#lst_lenghts_xvals = [len(xval) for xval in xvals_lst]

#len(autocorrs_RNAs)
#len(autocorrs_RNAs[0])
#min(lst_lengths)
"""
autocorrs_RNAs_ts = acf_list(dataframes_list = dataframes_list_ts, molecule = 'Number of RNAs gene1' )
autocorrs_RNAs_ar_hybridVIII = acf_list(dataframes_list = dataframes_list_hybrid_autorepressor_VIIIapproach, molecule = 'Number of RNA molecules' )
autocorrs_RNAs_ar_hybridVII = acf_list(dataframes_list = dataframes_list_hybrid_autorepressor_VIIapproach, molecule = 'Number of RNA molecules' )

autocorrs_RNAs_ar_hybridVIII
autocorrs_RNAs_ar
"""
#for i in autocorrs_RNAs_ar_hybridVIII:
#    print(len(i))

def compute_quantiles1(autocorrs, lst_lengths):
    """
    This function computes quantiles for the set of autocorrelations
    distributions at each time points. 

    Parameters
    ----------
    autocorrs : list
        list of numpy arrays containing autocorrelation values 
        at precise time points for each simulation in dataframe list.

    Returns
    -------
    quantiles_list : list
        List of quantiles computed for each time points of all simulations.

    """
    
    

    quantiles_list = []
    
    autocorrs_RNAs_list_of_lists = []
    
    autocorrs_RNAs_list = []
    
  
    
    for j in np.arange(0,min(lst_lengths),1):
        
        
        
        for i in np.arange(0,len(autocorrs),1):
            
            autocorrs_RNAs_list.append(autocorrs[i][j])
            
        a = autocorrs_RNAs_list.copy()
        
        autocorrs_RNAs_list_of_lists.append(a)
        
        del autocorrs_RNAs_list[:]
            
        
        
        
            
            
    autocorrs_RNAs_list_of_lists    
    
    for autocorrs in autocorrs_RNAs_list_of_lists:
        
        quantile = mquantiles(np.array(autocorrs), prob=[0.25,0.50,0.75])
        
        quantiles_list.append(quantile)
    
    return quantiles_list     



quantiles_list_RNAs = compute_quantiles1(autocorrs = autocorrs_RNAs, lst_lengths=lst_lengths_RNAs)
#quantiles_list_proteins = compute_quantiles1(autocorrs = autocorrs_proteins, lst_lengths=lst_lengths_proteins)



#len(quantiles_list_RNAs)
#quantiles_list_ar
#len(quantiles_list_ar)

#df =  pd.read_csv('gillespie_autorepressor_results_seed4.csv', sep=" ")
#xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
#len(xvals)
"""
quantiles_list_ts = compute_quantiles(autocorrs = autocorrs_RNAs_ts)
quantiles_list_ar_hybridVIIIapproach = compute_quantiles(autocorrs = autocorrs_RNAs_ar_hybridVIII)
quantiles_list_ar_hybridVIIapproach = compute_quantiles(autocorrs = autocorrs_RNAs_ar_hybridVII)
"""
#xvals
#len(quantiles_list_ar)

#df = pd.read_csv('gillespie_toggleswitch_results_seed1.csv',sep=" ")
#xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)

#df = pd.read_csv('ka1ki0gillespieresults_seed1.csv',sep=" ")
#xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)

df = pd.read_csv('removed_warmup_nfkb_k20k2i0gillespiesimulation_results_seed1.csv',sep=" ")
xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)

#min_length = min(lst_lenghts_xvals)
#min_length
#xvals = np.arange(0, min_length, dt)
#xvals

#len(xvals)
def Plot_Quantiles(title, quantiles_list):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    
    #n_series = np.arange(0,len(quantiles_list)).tolist()
    
    #xvals = np.arange(0, min_length, dt)
    
    xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
    
    #Assumo che xvals di una qualsiasi simulazione sarà sicuramente 
    #maggiore della lunghezza dei punti di autocorrelazione.
    xvals = xvals[:len(quantiles_list)]
    
    for quantile, n in zip(quantiles_list, xvals):
    
        lower_quartile = quantile[0]
        median = quantile[1]
        upper_quartile = quantile[2]
        
        x = np.array([n])
        y = np.array([median])
        
        plt.axhline(y=0, linewidth=1, color='lightgray')
        
        plt.vlines(x, lower_quartile, upper_quartile, color = 'black')
        
        plt.plot(x, y, '.', color = 'black')
        
        #plt.ylabel('Autocorrelation', size = 14)
        plt.ylabel(r'G($\tau$)')
        plt.xlabel(r'$\tau$')
        
        #ax.set_xlim(0,200)
        #plt.xlabel('$t_{tot}/dt$', size = 14)
        
        #array = np.arange(0,(1000/dt)+10,10)
        #lst = array.tolist()
        #lst = [int(x) for x in lst]
        #ax.xaxis.set_ticks(lst)
        #ax.set_xticklabels(lst)
        
        plt.title(title,size=14)
        
        sns.despine(fig, bottom=False, left=False)
        
    plt.show()
    
Plot_Quantiles(title="NF-kB products", quantiles_list = quantiles_list_RNAs)

if args.stats_acf_toggleswitch:    
    
    Plot_Quantiles(title = 'Toggle Switch (RNAs) 64 SSA simulations', quantiles_list = quantiles_list_RNAs)

    Plot_Quantiles(title = 'First model (RNAs) 64 SSA simulations', quantiles_list = quantiles_list_RNAs)
    
    Plot_Quantiles(title = 'Toggle Switch (Proteins) 64 SSA simulations', quantiles_list = quantiles_list_proteins)

#quantiles_list_ar = quantiles_list_ar[1:]    
#Plot_Quantiles(title = 'Autorepressor (RNAs) 64 SSA simulations', quantiles_list = quantiles_list_RNAs_ar)
#Plot_Quantiles(title = 'Autorepressor (proteins) 64 SSA simulations', quantiles_list = quantiles_list_proteins_ar)
    
"""
Plot_Quantiles(title = 'Toggle Switch (RNAs gene1)', quantiles_list = quantiles_list_ts)

Plot_Quantiles(title = 'Autorepressor (RNAs)', quantiles_list = quantiles_list_ar_hybridVIIIapproach)

Plot_Quantiles(title = 'Autorepressor (RNAs)', quantiles_list = quantiles_list_ar_hybridVIIapproach)
"""

#df = pd.read_csv('removed_warmup_ka1ki0.5gillespieresults_seed1.csv', sep=" ")
#df['Time']

#len(df['Time'])

def MultiPlot_Quantiles(title, quantiles_list1, quantiles_list2):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    
    #n_series = np.arange(0,len(quantiles_list)).tolist()
    
    #xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
    
    xvals = np.arange(0, len(df['Time']), dt)
    
    #xvals = np.arange(0, min_length, dt)
    
    #Assumo che xvals di una qualsiasi simulazione sarà sicuramente 
    #maggiore della lunghezza dei punti di autocorrelazione.
    xvals1 = xvals[:len(quantiles_list1)]
    
    for quantile, n in zip(quantiles_list1, xvals1):
    
        lower_quartile = quantile[0]
        median = quantile[1]
        upper_quartile = quantile[2]
        
        x = np.array([n])
        y = np.array([median])
        
        ax[0].plot(x, y, '.', color = 'black')
        
        ax[0].legend(['RNAs'])
        
        ax[0].axhline(y=0,linewidth=0.1, color='lightgray')
        
        ax[0].vlines(x, lower_quartile, upper_quartile, color = 'black')
        
        #ax[0].plot(x, y, '.', color = 'black')
        
        #plt.ylabel('Autocorrelation', size = 14)
        ax[0].set_ylabel(r'G($\tau$)')
        ax[0].set_xlabel(r'$\tau$')
        #ax[0].set_xlim(None,1000)
        
        #ax[0].set_xlim(0,200)
        #plt.xlabel('$t_{tot}/dt$', size = 14)
        
        #array = np.arange(0,(1000/dt)+10,10)
        #lst = array.tolist()
        #lst = [int(x) for x in lst]
        #ax.xaxis.set_ticks(lst)
        #ax.set_xticklabels(lst)
        
        ax[0].set_title(title,size=14)

    
    xvals2 = xvals[:len(quantiles_list2)]
        
    for quantile, n in zip(quantiles_list2, xvals2):
    
        lower_quartile = quantile[0]
        median = quantile[1]
        upper_quartile = quantile[2]
        
        x = np.array([n])
        y = np.array([median])
        
        ax[1].plot(x, y, '.', color = 'black')

        ax[1].legend(['Proteins'])
        
        ax[1].axhline(y=0, linewidth=0.1, color='lightgray')#linewidth=1,
        
        ax[1].vlines(x, lower_quartile, upper_quartile, color = 'black')
        
        #ax[1].plot(x, y, '.', color = 'black')
        
        #plt.ylabel('Autocorrelation', size = 14)
        ax[1].set_ylabel(r'G($\tau$)')
        ax[1].set_xlabel(r'$\tau$')
        #ax[1].set_xlim(None,1000)
        
        #ax[0].set_xlim(0,200)
        #plt.xlabel('$t_{tot}/dt$', size = 14)
        
        #array = np.arange(0,(1000/dt)+10,10)
        #lst = array.tolist()
        #lst = [int(x) for x in lst]
        #ax.xaxis.set_ticks(lst)
        #ax.set_xticklabels(lst)
        
        #ax[1].set_title(title,size=14)
        
        sns.despine(fig, bottom=False, left=False)
        
    plt.show()

if args.stats_acf:
    
    MultiPlot_Quantiles(title="Toggle Switch 64 SSA simulations", quantiles_list1=quantiles_list_RNAs, quantiles_list2=quantiles_list_proteins)    
    


#%% One simulation (first calculate acf for this simulation 
#and then divide in sub-blocks and calculate quantiles)

config = configparser.ConfigParser()

config.read('configuration.txt')

config.read('configuration_2genes.txt')
#Dividing simulations in dts.

N = config["SIMULATION"].getint('n_simulations') # N=16
N
#warmup_time = 20
dt = config["SIMULATION"].get('dt') #100
dt = ast.literal_eval(dt)
dt
time_limit = config["SIMULATION"].getint('time_limit')

nlags = config["ACF"].getint('n_lags')




#df = pd.read_csv('autorepressor_gillespiesimulation_results.csv', sep=" ")
df = pd.read_csv('toggleswitch_gillespiesimulation_results.csv', sep=" ")
"""
df = pd.read_csv('hybridsimulation_autorepressor_results.csv',sep=" ")




config.read('configuration_2genes.txt')

#Dividing simulations in dts.

N = config["SIMULATION"].getint('n_simulations') # N=16

#warmup_time = 20
dt = config["SIMULATION"].get('dt') #100
dt = ast.literal_eval(dt)
dt
time_limit = config["SIMULATION"].getint('time_limit')

nlags = config["ACF"].getint('n_lags')




df = pd.read_csv('toggleswitch_gillespiesimulation_results.csv', sep=" ")
"""

        
xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1],dt)
time = df['Time'] 
RNAs = df['Number of RNA molecules']
RNAs = df['Number of RNAs gene1']
f_RNAs = interp.interp1d(time, RNAs, kind='previous')

yinterp_RNAs = f_RNAs(xvals)
    
autocorrs_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) 

len(autocorrs_RNAs)



Dt= 5000#400
quantiles = []

steps_size = [Dt]*(len(autocorrs_RNAs)//Dt)

for m,n,k in zip(np.arange(0,len(autocorrs_RNAs),Dt), np.arange(0,len(autocorrs_RNAs),Dt), steps_size):
    
    quantile = mquantiles(autocorrs_RNAs[m:n+k], prob=[0.25,0.50,0.75])
    
    quantiles.append(quantile)
    
quantiles[-1] #array([0.0247893 , 0.02594543, 0.02841868])
quantiles[0] #array([0.8982566 , 0.93280979, 0.9675491 ])
quantiles[1]
quantiles[2]
quantiles[3] #array([-0.18, -0.1 , -0.02])   
len(quantiles)
mquantiles([-0.1,-0.2,0], prob=[0.25,0.50,0.75])

mquantiles(autocorrs_RNAs[16000:20000], prob=[0.25,0.50,0.75])

def compute_quantiles1sim():
    
    steps_size = [Dt]*(len(autocorrs_RNAs)//Dt)

    for m,n,k in zip(np.arange(0,len(autocorrs_RNAs),Dt), np.arange(0,len(autocorrs_RNAs),Dt), steps_size):
        
        quantile = mquantiles(autocorrs_RNAs[m:n+k], prob=[0.25,0.50,0.75])
        
        quantiles.append(quantile)
    
    return quantiles

quantiles = compute_quantiles1sim()
quantiles[-1]
mquantiles(autocorrs_RNAs[19600:20000], prob=[0.25,0.50,0.75])
mquantiles(autocorrs_RNAs[1200:1600], prob=[0.25,0.50,0.75])
autocorrs_RNAs[19600:20000]
autocorrs_RNAs[1200:1600]

#Testing

quantiles[-1] == mquantiles(autocorrs_RNAs[19600:20000], prob=[0.25,0.50,0.75])#array([ True,  True,  True])

quantiles[3] == mquantiles(autocorrs_RNAs[1200:1600], prob=[0.25,0.50,0.75])#array([False, False, False])

quantiles[0]
mquantiles(autocorrs_RNAs[0:400], prob=[0.25,0.50,0.75])

quantiles[1]
mquantiles(autocorrs_RNAs[400:800], prob=[0.25,0.50,0.75])


quantiles[2]
mquantiles(autocorrs_RNAs[800:1200], prob=[0.25,0.50,0.75])

quantiles[3]
mquantiles(autocorrs_RNAs[1200:1600], prob=[0.25,0.50,0.75])

lst = [0,1,2,3,4,5,6,7,8,9,10]
lst[0:3]
lst[3:5]

ind_values = np.arange(0,20000+400,400)
ind_values

ind_mvalues = []

for m,n,k in zip(np.arange(0,len(autocorrs_RNAs),Dt), np.arange(0,len(autocorrs_RNAs),Dt), steps_size):
    
    quantile = mquantiles(autocorrs_RNAs[m:n+k], prob=[0.25,0.50,0.75])
    
    ind_mvalues.append(m)
    
ind_mvalues
np.array(ind_mvalues)

mquantiles(autocorrs_RNAs[98000:98700], prob=[0.25,0.50,0.75])
plt.hist(autocorrs_RNAs[10000:10500])

mquantiles(autocorrs_RNAs[0:5000],prob=[0.25,0.50,0.75])#array([0.38810798, 0.50040722, 0.65598728])
mquantiles(autocorrs_RNAs[5000:10000],prob=[0.25,0.50,0.75])#array([0.30048941, 0.30954446, 0.32381294])
mquantiles(autocorrs_RNAs[10000:15000],prob=[0.25,0.50,0.75])#array([0.157741  , 0.19020666, 0.25220907])




ind_nkvalues = []

for m,n,k in zip(np.arange(0,len(autocorrs_RNAs),Dt), np.arange(0,len(autocorrs_RNAs),Dt), steps_size):
    
    quantile = mquantiles(autocorrs_RNAs[m:n+k], prob=[0.25,0.50,0.75])
    
    ind_nkvalues.append(n+k)

ind_nkvalues[0]    
ind_nkvalues[1]
np.array(ind_nkvalues)








def Plot_Quantiles1sim(title):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    
    #xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
    
    #xvals = xvals[:len(quantiles)]
    
    for quantile, n in zip(quantiles, np.arange(0,len(quantiles))):# np.arange(0,len(quantiles))
    
        lower_quartile = quantile[0]
        median = quantile[1]
        upper_quartile = quantile[2]
        
        x = np.array([n])
        y = np.array([median])
        
        plt.axhline(y=0, linewidth=1, color='gray')
        
        plt.vlines(x, lower_quartile, upper_quartile, color = 'black')
        
        plt.plot(x, y, '.', color = 'black')
        
        plt.ylabel(r'G($\tau$)', size = 14)
        
        plt.xlabel('$n_{lags}/Dt$', size = 14)
        
        #plt.xlabel(r'$\tau$')
        
        plt.title(title, size = 15)
        
        sns.despine(fig, bottom=False, left=False)
        
    plt.show()


Plot_Quantiles1sim(title = 'Autorepressor (RNAs)')
Plot_Quantiles1sim(title='Toggle Switch (RNAs)')

#%%Prova a calcolare l'autocorrelazione per ogni sottoblocco 
config = configparser.ConfigParser()

config.read('configuration_2genes.txt')

dt = config["SIMULATION"].get('dt') #100
dt = ast.literal_eval(dt)
dt

nlags = config["ACF"].get('n_lags')
nlags = ast.literal_eval(nlags)
nlags

df = pd.read_csv('toggleswitch_gillespiesimulation_results.csv', sep=" ")
time = df['Time']
xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1],dt)
RNAs = df['Number of RNAs gene1']
time_limit = config["SIMULATION"].getint('time_limit')




RNAs = np.ascontiguousarray(RNAs)
len(RNAs)
time = np.ascontiguousarray(time)
time
len(time)
time[500:1000]

plt.plot(time,RNAs)


Dt= 500#400
acfs = []

steps_size = [Dt]*(len(time)//Dt)

for m,n,k in zip(np.arange(0,len(time),Dt), np.arange(0,len(time),Dt), steps_size):
    
    #quantile = mquantiles(autocorrs_RNAs[m:n+k], prob=[0.25,0.50,0.75])
    
    #quantiles.append(quantile)
    
    RNAs = df['Number of RNAs gene1']
    
    RNAs = np.ascontiguousarray(RNAs)
    
    time = df['Time']
    
    time = np.ascontiguousarray(time)
    
    time = time[m:n+k]
    
    RNAs = RNAs[m:n+k]
    
    f_RNAs = interp.interp1d(time, RNAs, kind='previous')
    
    xvals = np.arange(time[0], time[-1],dt)

    yinterp_RNAs = f_RNAs(xvals)
        
    autocorrs_RNAs = stattools.acf(yinterp_RNAs, nlags = nlags, fft=False) 
    acfs.append(autocorrs_RNAs)
    
len(acfs)

lst = []
a = np.array([1,2,3,4])    
b = np.array([4,5,6,7])    

lst.append(a)
lst.append(b)
lst


quantiles_lst = []
for i in acfs:
    quantiles = mquantiles(i,prob=[0.25,0.50,0.75])
    quantiles_lst.append(quantiles)
    
len(quantiles_lst)



def Plot_Quantiles1sim(title):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    
    #xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], dt)
    
    #xvals = xvals[:len(quantiles)]
    
    for quantile, n in zip(quantiles_lst, np.arange(0,len(quantiles_lst))):# np.arange(0,len(quantiles))
    
        lower_quartile = quantile[0]
        median = quantile[1]
        upper_quartile = quantile[2]
        
        x = np.array([n])
        y = np.array([median])
        
        plt.axhline(y=0, linewidth=1, color='gray')
        
        plt.vlines(x, lower_quartile, upper_quartile, color = 'black')
        
        plt.plot(x, y, '.', color = 'black')
        
        plt.ylabel(r'G($\tau$)', size = 14)
        
        plt.xlabel('n', size = 14)
        
        #plt.xlabel(r'$\tau$')
        
        plt.title(title, size = 15)
        
        sns.despine(fig, bottom=False, left=False)
        
    plt.show()


Plot_Quantiles1sim(title = 'Autorepressor (RNAs)')
Plot_Quantiles1sim(title='Toggle Switch (RNAs)')


