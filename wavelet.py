# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 10:46:22 2023

@author: asus
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as plticker# has classes for tick-locating and -formatting
import seaborn as sns
from statsmodels.tsa import stattools
import scipy.interpolate as interp
from scipy.fft import fft, fftfreq, fftshift
import scipy
from scipy import fftpack
import scaleogram as scg 
import pywt

#%% Signal that changes frequency as the last time (matshow)

x = np.linspace(0,1/4,2500)
y1 = np.sin(2*np.pi*x*4)
#plt.plot(x,y1)
y2 = np.sin(2*np.pi*x*30)
#plt.plot(x,y2)
y3 = np.sin(2*np.pi*x*60)
#plt.plot(x,y3)
y4 = np.sin(2*np.pi*x*90)
#plt.plot(x,y4)

# Using only the first two waves 
signal = np.concatenate([y1,y2])
time = np.linspace(0,1,5000)
plt.plot(time,signal)

# Using only the first three waves 
signal = np.concatenate([y1,y2,y3])
time = np.linspace(0,1,7500)
plt.plot(time,signal)

# Using all waves
signal = np.concatenate([y1,y2,y3,y4])
N = 10000
time = np.linspace(0,1,N)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(time,signal)
sns.despine(fig, bottom=False, left=False)
plt.show()



scales = np.arange(1,500)
waveletname = 'cgau1'
coef, freqs=pywt.cwt(signal, scales, waveletname)





len(abs(coef)) # 799 = length of scale
len(abs(coef[0])) #10000 = length of data








fig, ax = plt.subplots(figsize=(12, 5))
matshow_ = ax.matshow(np.abs(coef), aspect = 'auto', cmap='Reds',
                      vmax=abs(coef).max(), vmin=0) 
fig.colorbar(matshow_)
plt.gca().xaxis.tick_bottom() # it puts x axis from top to bottom of figure.



ax.set_xlim(xmin=0, xmax=N)

loc = plticker.MultipleLocator(base=2500)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

def numfmt(x, pos): # your custom formatter function: divide by 1000
    s = '{}'.format(x / N)
    return s
yfmt = plticker.FuncFormatter(numfmt) # create your custom formatter function
plt.gca().xaxis.set_major_formatter(yfmt) 

ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)

plt.show() 




#%% Sognal that changes frequency as the last time (imshow)
x = np.linspace(0,1/4,2500)
y1 = np.sin(2*np.pi*x*4)
#plt.plot(x,y1)
y2 = np.sin(2*np.pi*x*30)
#plt.plot(x,y2)
y3 = np.sin(2*np.pi*x*60)
#plt.plot(x,y3)
y4 = np.sin(2*np.pi*x*90)
#plt.plot(x,y4)

# Using only the first two waves 
signal = np.concatenate([y1,y2])
time = np.linspace(0,1,5000)
plt.plot(time,signal)

# Using only the first three waves 
signal = np.concatenate([y1,y2,y3])
time = np.linspace(0,1,7500)
plt.plot(time,signal)

# Using all waves
signal = np.concatenate([y1,y2,y3,y4])
N = 10000
time = np.linspace(0,1,N)
plt.plot(time,signal)

scales = np.arange(1,1500)
waveletname = 'cmorl1.5-1.0'#cmorl1.5-1.0
coef, freqs=pywt.cwt(signal,scales, waveletname)







fig, ax = plt.subplots(figsize=(12, 5))
imshow_ = plt.imshow(np.abs(coef), interpolation='bilinear', cmap='Reds', aspect='auto') #extent=[0, len(x), 1, len(scales)],
ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales',fontsize=15)
ax.set_xlabel('Time', fontsize=15)
fig.colorbar(imshow_)

plt.gca().xaxis.tick_bottom() # it puts x axis from top to bottom of figure.

ax.set_xlim(xmin=0, xmax=N)
loc = plticker.MultipleLocator(base=2500)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
def numfmt(x, pos): # your custom formatter function: divide by 1000
    s = '{}'.format(x / N)
    return s
yfmt = plticker.FuncFormatter(numfmt) # create your custom formatter function
plt.gca().xaxis.set_major_formatter(yfmt) 

ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)

plt.show() 




#%% Signal that changes frequency as the last time (contourf)
x = np.linspace(0,1/4,2500)
y1 = np.sin(2*np.pi*x*4)
#plt.plot(x,y1)
y2 = np.sin(2*np.pi*x*30)
#plt.plot(x,y2)
y3 = np.sin(2*np.pi*x*60)
#plt.plot(x,y3)
y4 = np.sin(2*np.pi*x*90)
#plt.plot(x,y4)

# Using only the first two waves 
signal = np.concatenate([y1,y2])
time = np.linspace(0,1,5000)
plt.plot(time,signal)

# Using only the first three waves 
signal = np.concatenate([y1,y2,y3])
time = np.linspace(0,1,7500)
plt.plot(time,signal)

# Using all waves
signal = np.concatenate([y1,y2,y3,y4])
N = 10000
time = np.linspace(0,1,N)
plt.plot(time,signal)

scales = np.arange(1,500)
waveletname = 'cgau1'
coef, freqs=pywt.cwt(signal, scales, waveletname)







fig, ax = plt.subplots(figsize=(12, 5))



contourf_ = ax.contourf(time, scales, np.abs(coef), cmap=plt.cm.Reds)#extend='both',


ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
fig.colorbar(contourf_)
plt.show()



fig, ax = plt.subplots(figsize=(12, 5))
contourf_ = ax.contourf(time, scales, np.abs(coef), cmap=plt.cm.Reds)#extend='both',
ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
ax.invert_yaxis()
fig.colorbar(contourf_)
plt.show()



#%% Example of signal that changes frequency in just one time interval

#Time domain signal
time = np.linspace(-1, 1, 200, endpoint=False)
signal  = np.cos(2 * np.pi * 7 * time) + np.real(np.exp(-7*(time-0.4)**2)*np.exp(1j*2*np.pi*2*(time-0.4)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(time,signal)
sns.despine(fig, bottom=False, left=False)
plt.show()

#Setting parameters for Continous Wavelet Transform
scales = np.arange(1,128)
waveletname='cmorl1.5-1.0'
coef, freqs=pywt.cwt(signal, scales, waveletname)



#contourf

fig, ax = plt.subplots(figsize=(12, 2))
contourf_ = ax.contourf(time, scales, np.abs(coef), cmap=plt.cm.Reds)#extend='both',
ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_xlim(-1,1)
ax.invert_yaxis()
fig.colorbar(contourf_)
plt.show()

fig, ax = plt.subplots(figsize=(12, 2))
contourf_ = ax.contourf(time, freqs, np.abs(coef), cmap=plt.cm.Reds)#extend='both',
ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_xlim(-1,1)
ax.invert_yaxis()
fig.colorbar(contourf_)
plt.show()




#matshow
fig, ax = plt.subplots(figsize=(12, 5))
matshow_ = ax.matshow(np.abs(coef), extent=[-1, 1, 31, 1], aspect = 'auto', cmap='Reds',
                      vmax=abs(coef).max(), vmin=0) # removing extent=[-1, 1, 1, 31] it turns right.
fig.colorbar(matshow_)

plt.gca().xaxis.tick_bottom() # it puts x axis from top to bottom of figure.
loc = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)
plt.show() 





#imshow
fig, ax = plt.subplots(figsize=(12, 5))
imshow_ = plt.imshow(np.abs(coef), cmap='Reds', aspect='auto', extent=[-1, 1, 31, 1],
           vmax=abs(coef).max(), vmin=0)  # extent=[-1, 1, 1, 31],
fig.colorbar(imshow_)

loc = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)
plt.show() 




#%%Sinusoidal wave with constant frequency.

x = np.linspace(0,1,1000)
y = np.sin(2*np.pi*x*4)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x,y)
sns.despine(fig, bottom=False, left=False)
plt.show()



#Setting parameters for Continous Wavelet Transform
scales = np.arange(1,256)
waveletname='cgau1'
coef, freqs=pywt.cwt(y, scales, waveletname)



#contourf in log viene più quello che ci aspettiamo
fig, ax = plt.subplots(figsize=(12, 2))
contourf_ = ax.contourf(x, scales, np.log(np.abs(coef)), cmap=plt.cm.Reds)#extend='both',
ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
fig.colorbar(contourf_)
plt.show()



#matshow
fig, ax = plt.subplots(figsize=(12, 5))
matshow_ = ax.matshow(np.abs(coef),  aspect = 'auto', cmap='Reds',
                      vmax=abs(coef).max(), vmin=0) 
fig.colorbar(matshow_)
plt.gca().xaxis.tick_bottom() # it puts x axis from top to bottom of figure.

#loc = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
#ax.xaxis.set_major_locator(loc)
ax.set_xlim(0,1000)
def numfmt(x, pos): # your custom formatter function: divide by 1000
    s = '{}'.format(x / 1000)
    return s
yfmt = plticker.FuncFormatter(numfmt) # create your custom formatter function
plt.gca().xaxis.set_major_formatter(yfmt) 

ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)
ax.invert_yaxis()
plt.show() 


#imshow
fig, ax = plt.subplots(figsize=(12, 5))
imshow_ = plt.imshow(np.abs(coef), extent=[min(x), max(x), min(scales), max(scales)], cmap='Reds', aspect='auto',
           vmax=abs(coef).max(), vmin=0)  
fig.colorbar(imshow_)
ax.set_title('Wavelet Transform of Signal (${}$)'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)
ax.invert_yaxis()
plt.show() 




#%% Just to know.

import numpy as np
import ssqueezepy
from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq
from ssqueezepy.visuals import imshow

t = np.linspace(-1, 1, 200, endpoint=False)
sig = (np.cos(2 * np.pi * 7 * t) +
       np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4))))

wavelet = Wavelet(('gmw', {'beta': 4}))
Wx, scales = cwt(sig, wavelet, padtype='zero')

freqs = scale_to_freq(scales, wavelet, N=len(sig), fs=1/(t[1] - t[0]))
imshow(Wx, abs=1, yticks=freqs,  xticks=t,
       xlabel="time [sec]", ylabel="frequency [Hz]")

ssqueezepy.wavs()
ssqueezepy.Wavelet('cmhat').info()



#%% Our molecules data

autorepressor_results = pd.read_csv('autorepressor_gillespiesimulation_results.csv',sep= " ")
autorepressor_results.columns
RNAs = autorepressor_results['Number of RNA molecules']
proteins = autorepressor_results['Number of proteins']
time = autorepressor_results['Time']

xvals = np.arange(autorepressor_results['Time'].iloc[0], autorepressor_results['Time'].iloc[-1], 0.01)

f_RNAs = interp.interp1d(time, RNAs, kind='previous')
yinterp_RNAs = f_RNAs(xvals)

f_proteins = interp.interp1d(time, proteins, kind='previous')
yinterp_proteins = f_proteins(xvals)

df_RNAs = pd.DataFrame(zip(yinterp_RNAs,xvals), columns = ['RNAs','Time'])
df_proteins = pd.DataFrame(zip(yinterp_proteins,xvals), columns = ['proteins','Time'])


signal_RNAs = df_RNAs['RNAs']
signal_proteins = df_proteins['proteins']

#####################Time Domain Signal
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax.plot(df_RNAs['Time'], df_RNAs['RNAs'])
ax.set_ylabel('# of RNAs')
ax.set_xlabel('Time')
ax.set_title('Interpolated data (dt=0.01)')
sns.despine(fig, bottom=False, left=False)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax.plot(df_proteins['Time'], df_proteins['proteins'])
ax.set_ylabel('# of proteins')
ax.set_xlabel('Time')
ax.set_title('Interpolated data (dt=0.01)')
sns.despine(fig, bottom=False, left=False)
plt.show()
###################################################

signal_RNAs = np.ascontiguousarray(signal_RNAs)

signal_proteins = np.ascontiguousarray(signal_proteins)

scales = np.arange(1,128)
waveletname = 'cmorl1.5-1.0'
coef_RNAs, freqs_RNAs = pywt.cwt(signal_RNAs, scales, waveletname)
coef_proteins, freqs_proteins = pywt.cwt(signal_proteins, scales, waveletname)




#contourf
fig, ax = plt.subplots(figsize=(12, 5))

contourf_ = ax.contourf(xvals, scales, np.abs(coef_RNAs), cmap=plt.cm.Reds)#extend='both',

ax.set_title('Wavelet Transform of Signal (${}$) RNAs'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
fig.colorbar(contourf_)
ax.invert_yaxis()
plt.show()



fig, ax = plt.subplots(figsize=(12, 5))

contourf_ = ax.contourf(xvals, scales, np.abs(coef_proteins), cmap=plt.cm.Reds)#extend='both',

ax.set_title('Wavelet Transform of Signal (${}$) Proteins'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
fig.colorbar(contourf_)
ax.invert_yaxis()
plt.show()




#imshow
fig, ax = plt.subplots(figsize=(12, 5))
imshow_ = ax.imshow(np.abs(coef_RNAs), cmap='Reds', aspect='auto')  #extent=[min(xvals), max(xvals), min(scales), max(scales)], , vmax=abs(coef).max(), vmin=0
ax.set_title('Wavelet Transform of Signal (${}$) RNAs'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)

ax.set_xlim(0,100000)
def numfmt(x, pos): # your custom formatter function: divide by 1000
    s = '{}'.format(int(x / 100))
    return s
yfmt = plticker.FuncFormatter(numfmt) # create your custom formatter function
plt.gca().xaxis.set_major_formatter(yfmt) 

fig.colorbar(imshow_)
plt.show() 





fig, ax = plt.subplots(figsize=(12, 5))
imshow_ = ax.imshow(np.abs(coef_proteins),cmap='Reds', aspect='auto')  #extent=[min(xvals), max(xvals), min(scales), max(scales)], vmax=np.abs(coef_proteins).max(), vmin=0
ax.set_title('Wavelet Transform of Signal (${}$) Proteins'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)

ax.set_xlim(0,100000)
def numfmt(x, pos): # your custom formatter function: divide by 1000
    s = '{}'.format(int(x / 100))
    return s
yfmt = plticker.FuncFormatter(numfmt) # create your custom formatter function
plt.gca().xaxis.set_major_formatter(yfmt) 

fig.colorbar(imshow_)
plt.show() 





#matshow
fig, ax = plt.subplots(figsize=(12, 5))
matshow_ = ax.matshow(np.abs(coef_RNAs),  aspect = 'auto', cmap='Reds',
                      vmax=abs(coef).max(), vmin=0) 
fig.colorbar(matshow_)
plt.gca().xaxis.tick_bottom() # it puts x axis from top to bottom of figure.

#loc = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
#ax.xaxis.set_major_locator(loc)
ax.set_xlim(0,100000)
def numfmt(x, pos): # your custom formatter function: divide by 1000
    s = '{}'.format(int(x / 100))
    return s
yfmt = plticker.FuncFormatter(numfmt) # create your custom formatter function
plt.gca().xaxis.set_major_formatter(yfmt) 

ax.set_title('Wavelet Transform of Signal (${}$) RNAs'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)
#ax.invert_yaxis()
plt.show() 



fig, ax = plt.subplots(figsize=(12, 5))
matshow_ = ax.matshow(np.abs(coef_proteins),  aspect = 'auto', cmap='Reds',
                      vmax=abs(coef).max(), vmin=0) 
fig.colorbar(matshow_)
plt.gca().xaxis.tick_bottom() # it puts x axis from top to bottom of figure.

ax.set_xlim(0,100000)
def numfmt(x, pos): # your custom formatter function: divide by 1000
    s = '{}'.format(int(x / 100))
    return s
yfmt = plticker.FuncFormatter(numfmt) # create your custom formatter function
plt.gca().xaxis.set_major_formatter(yfmt) 

ax.set_title('Wavelet Transform of Signal (${}$) Proteins'.format(waveletname), fontsize=20)
ax.set_ylabel('Scales', fontsize=15)
ax.set_xlabel('Time', fontsize=15)
#ax.invert_yaxis()
plt.show() 

