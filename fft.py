# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:12:02 2023

@author: asus
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import scipy.interpolate as interp
from scipy.fft import fft, fftfreq, fftshift
import scipy
from scipy import fftpack


df = pd.read_csv('toggleswitch_gillespiesimulation_results.csv',sep= " ")


#RNAs

RNAs = df['Number of RNAs gene1']

time = df['Time']

xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], 0.01)
f_RNAs = interp.interp1d(time, RNAs, kind='previous')
yinterp_RNAs = f_RNAs(xvals)

# Fourier Transform results
N_RNAs = yinterp_RNAs.size

yf_RNAs = fft(yinterp_RNAs)
xf_RNAs = fftfreq(N_RNAs, d=0.01) 

len(xf_RNAs)



# Fourier Transform log scale
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
ax.plot(scipy.fftpack.fftshift(xf_RNAs), scipy.fftpack.fftshift(np.abs(yf_RNAs)), color='red')
#ax.plot(scipy.fftpack.fftshift(xf_RNAs), scipy.fftpack.fftshift(yf_RNAs_mean[:len(xf_RNAs)]))#scipy.fftpack.fftshift(xf_RNAs[:len(yf_RNAs_mean)]), scipy.fftpack.fftshift(yf_RNAs_mean)
ax.set_xlabel('Frequency (a.u.)')
ax.set_ylabel('FFT Amplitude')
ax.set_title('Toggle Switch model Fourier Transform (RNAs gene 1)')
ax.set_xlim(0.001,1)
#If you put the logarithm you can not consider 0 as starting point.
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_ylim(0,400000)
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
sns.despine(bottom=False, left=False)
plt.show()  




#Proteins

proteins = df['Number of proteins gene1']

time = df['Time']

xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], 0.01)
f_proteins = interp.interp1d(time, proteins, kind='previous')
yinterp_proteins = f_proteins(xvals)

# Fourier Transform results
N_proteins = yinterp_proteins.size

yf_proteins = fft(yinterp_proteins)
xf_proteins = fftfreq(N_proteins, d=0.01) 

len(xf_proteins)



# Fourier Transform log scale
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
ax.plot(scipy.fftpack.fftshift(xf_proteins), scipy.fftpack.fftshift(np.abs(yf_proteins)), color='red')
#ax.plot(scipy.fftpack.fftshift(xf_RNAs), scipy.fftpack.fftshift(yf_RNAs_mean[:len(xf_RNAs)]))#scipy.fftpack.fftshift(xf_RNAs[:len(yf_RNAs_mean)]), scipy.fftpack.fftshift(yf_RNAs_mean)
ax.set_xlabel('Frequency (a.u.)')
ax.set_ylabel('FFT Amplitude')
ax.set_title('Toggle Switch model Fourier Transform (Proteins gene 1)')
ax.set_xlim(0.0001,1)
#If you put the logarithm you can not consider 0 as starting point.
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_ylim(0, 8000000)
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
sns.despine(bottom=False, left=False)
plt.show() 



#%% If you want to make the mean Fourier Transform



N=64
data = 'gillespieresults_seed{}.csv'

dataframe_list = []
for n in range(1,N+1):
    simulation_results = pd.read_csv(data.format(n), sep=" ") 
    dataframe_list.append(simulation_results)
    
yf_RNAs_lst = []
xf_RNAs_lst = []
    
for df in dataframe_list:
    
    RNAs = df['Number of RNA molecules']

    time = df['Time']

    xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], 0.01)
    f_RNAs = interp.interp1d(time, RNAs, kind='previous')
    yinterp_RNAs = f_RNAs(xvals)
    
    # Fourier Transform results
    N_RNAs = yinterp_RNAs.size

    yf_RNAs = fft(yinterp_RNAs)
    xf_RNAs = fftfreq(N_RNAs, d=0.01) 
    
    yf_RNAs_lst.append(yf_RNAs)
    xf_RNAs_lst.append(xf_RNAs)
    
len(yf_RNAs_lst[0])      

lengths=[]
for lst in yf_RNAs_lst:
    lengths.append(len(lst))
len(lengths)    
lengths

min_length = min(lengths)

new_yf_RNAs_lst = []
abs_lst = []
for lst in yf_RNAs_lst:
    for i in np.arange(0,min_length):
        abs_i = np.abs(lst[i])
        abs_lst.append(abs_i)
    a = abs_lst.copy()
    new_yf_RNAs_lst.append(a)
    del abs_lst[:]

#Check    
len(new_yf_RNAs_lst)
len(new_yf_RNAs_lst[0])
#Array
new_yf_RNAs_array = np.array(new_yf_RNAs_lst, dtype=object)


len(new_yf_RNAs_array)
type(new_yf_RNAs_array)
len(new_yf_RNAs_array[0])
a = np.array([[4,8],[2,4]])

a.mean(axis=0)

#Mean
#yf_RNAs_mean = new_yf_RNAs_array.mean(axis=0)

yf_RNAs_mean = np.mean(new_yf_RNAs_array,axis=0)
len(yf_RNAs_mean) 



#RNAs

RNAs = df['Number of RNA molecules']

time = df['Time']

xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], 0.01)
f_RNAs = interp.interp1d(time, RNAs, kind='previous')
yinterp_RNAs = f_RNAs(xvals)

# Fourier Transform results
N_RNAs = yinterp_RNAs.size

#yf_RNAs = fft(yinterp_RNAs)
xf_RNAs = fftfreq(N_RNAs, d=0.01) 


#Proteins

proteins = df['Number of proteins']

time = df['Time']

xvals = np.arange(df['Time'].iloc[0], df['Time'].iloc[-1], 0.01)
f_proteins = interp.interp1d(time, proteins, kind='previous')
yinterp_proteins = f_proteins(xvals)

# Fourier Transform results
N_proteins = yinterp_proteins.size

#yf_proteins = fft(yinterp_proteins)
xf_proteins = fftfreq(N_proteins, d=0.01) 

len(xf_proteins)





#%% Example of FFT signal that changes frequency
t_n=1
N=100000
xa = np.linspace(0, t_n, num=N)

xb = np.linspace(0, t_n/4, num=N//4)

frequencies = [4, 30, 60, 90]
y1a, y1b = np.sin(2*np.pi*frequencies[0]*xa), np.sin(2*np.pi*frequencies[0]*xb)
y2a, y2b = np.sin(2*np.pi*frequencies[1]*xa), np.sin(2*np.pi*frequencies[1]*xb)
y3a, y3b = np.sin(2*np.pi*frequencies[2]*xa), np.sin(2*np.pi*frequencies[2]*xb)
y4a, y4b = np.sin(2*np.pi*frequencies[3]*xa), np.sin(2*np.pi*frequencies[3]*xb)

composite_signal2 = np.concatenate([3*y1b, y2b, y3b, y4b])

plt.plot(xa, composite_signal2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
sns.despine(bottom=False, left=False)
plt.show()

time = xa
signal = composite_signal2

yf = fft(signal)
xf = fftfreq(signal.size,d=time[0]-time[1])

plt.plot(xf, np.abs(yf))

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


N = 100000
T = t_n / N
f_s = 1/T


f_values2, fft_values2 = get_fft_values(composite_signal2, T, N, f_s)

plt.plot(f_values2,fft_values2)
plt.xlim(0,140)

plt.plot(scipy.fftpack.fftshift(xf),scipy.fftpack.fftshift(np.abs(yf)))
# se moltiplichi per 2.0/N ottieni il grafico del tutorial.
# penso che lo fa per avere una scala più ridotta ?
plt.xlim(0,140)
plt.ylabel('FFT Amplitude')
plt.xlabel('Frequency (Hz)')
sns.despine(bottom=False, left=False)
plt.show()
