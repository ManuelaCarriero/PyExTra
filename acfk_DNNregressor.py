# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:03:04 2022

@author: asus
"""

# Load data
import pandas as pd
import numpy as np
import ast
import datetime

#Build the model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from keras.callbacks import EarlyStopping

# To plot the training and test loss curves:
import matplotlib.pyplot as plt
import seaborn as sns


import os


start_date = datetime.datetime.now()

data = pd.read_csv('autorepressor_1000RNAsacfs_seed42_scale0.5_nlags5000_k1k2k3k4.csv',sep= " ")
data.columns



#Here we have a multi-output regression model

ks = np.ascontiguousarray(data['k'])

lst_k = []
for k in ks:
    lst_k.append(ast.literal_eval(k))

#Remove parameters that does not change.

for i in np.arange(0,len(lst_k)):
    lst_k[i].pop(0)
for i in np.arange(0,len(lst_k)):
    lst_k[i].pop(0)
for i in np.arange(0,len(lst_k)):
    lst_k[i].pop(4)



ks = np.array(lst_k) #output variables 



acfs = np.ascontiguousarray(data['acfs'])
lst_acfs = []
for acf in acfs:
    lst_acfs.append(ast.literal_eval(acf))

acfs = np.array(lst_acfs) #input variables


end_date = datetime.datetime.now()



elapsed_time_date = end_date - start_date

print(" ")
print('Execution time:', elapsed_time_date, 'seconds')



print(np.shape(acfs))#(999, 5001), (200, 20001)
print(np.shape(ks)) #(999, 4), (198, 7)




start_date = datetime.datetime.now()
seed = 7
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(acfs, ks, test_size=0.05, shuffle=True, random_state=seed)


dim1 = len(acfs[0])
           
model = Sequential()
model.add(Dense(4000, input_dim=dim1, activation='relu'))#, kernel_initializer='he_uniform',
model.add(Dropout(0.25))
model.add(Dense(units=1000, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(units=250, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.30))
model.add(Dense(units=4))


print(model.summary())


model.compile(loss='mae', optimizer='adam')

#Fit the model

history = model.fit(X_train, y_train,  batch_size=128, validation_data=(X_test, y_test),verbose=2, epochs=50)

end_date = datetime.datetime.now()

elapsed_time_date = end_date - start_date

print(" ")
print('Execution time:', elapsed_time_date, 'seconds')





test_loss = model.evaluate(X_test, y_test) 




prediction = model.predict(X_test)


print(r2_score(y_test,prediction)) 
r2score_test = r2_score(y_test,prediction)
#0.11228924302246548

r2score_test = r2_score(y_test,prediction,multioutput='variance_weighted')

r2score_test#0.17247737615920744

# One set of parameters

plt.plot(y_test[0],prediction[0], ".")
ident = [0.0, max(y_test[0])]
plt.plot(ident,ident)

# For all data

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_test)):
    ident = [0.0, 4]
    ax.plot(ident,ident)
    ax.plot(y_test[i],prediction[i], ".")
ax.set_ylabel("True parameters")
ax.set_xlabel("Predicted parameters")
ax.set_title("Test data")
    


y_test = [np.round(item,2) for item in y_test]
prediction = [np.round(item,2) for item in prediction]

y_test[0]
prediction[0]




prediction_train = model.predict(X_train)

print(r2_score(y_train, prediction_train))
#0.7937693970512397

r2score_train = r2_score(y_train, prediction_train,multioutput='variance_weighted')
r2score_train#0.7928748533596189



y_trainflat = y_train.flatten()
prediction_trainflat = prediction_train.flatten()
corr_train, _ = pearsonr(y_trainflat, prediction_trainflat)
corr_train#0.8987982424713679

y_testflat = y_test.flatten()
prediction_flat = prediction.flatten()
corr_test, _ = pearsonr(y_testflat, prediction_flat)
corr_test #0.503818344436295





# For 50 data points

prediction_train50 = model.predict(X_train[0:50])
print(r2_score(y_train[0:50], prediction_train50))
r2score_train50 = r2_score(y_train[0:50], prediction_train50)



# One set of parameters

plt.plot(y_train[0],prediction_train[0], ".")
ident = [0.0, max(y_train[0])]
plt.plot(ident,ident)

# For all data

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_train)):
    ident = [0.0, 8]
    ax.plot(ident,ident)
    ax.plot(y_train[i],prediction_train[i], ".")
ax.set_ylabel("True parameters")
ax.set_xlabel("Predicted parameters")
ax.set_title("Train data")
    





    
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = np.arange(1, len(val_loss)+1)
epochs = epochs.tolist()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

ax.plot(epochs, loss, "-", color="darkblue", label='Training loss')
ax.plot(epochs, val_loss, "-", color="cyan", label='Validation loss')
ax.legend()
sns.despine(fig, bottom=False, left=False)
plt.show()







# serialize model to JSON
model_json = model.to_json()
with open("model_autorepressor_1000RNASacfs_seed42_scale0.5_nlags5000_k1k2k3k4withoutkikak5.json","w") as json_file:
    json_file.write(model_json)
    
# serialize model to HDF5
model.save_weights("model_autorepressor_1000RNASacfs_seed42_scale0.5_nlags5000_k1k2k3k4withoutkikak5.h5")
print("Saved model to disk")








#scale=0.5

# load json and create model
json_file = open('model_autorepressor_1000RNASacfs_seed42_scale0.5_nlags5000_k1k2k3k4.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_autorepressor_1000RNASacfs_seed42_scale0.5_nlags5000_k1k2k3k4.h5")
print("Loaded model from disk")

data_r = pd.read_csv('autorepressor_1000RNASacfs_seed42_scale0.5_nlags5000_k1k2k3k4.csv',sep= " ")

ks = np.ascontiguousarray(data_r['k'])

lst_k = []
for k in ks:
    lst_k.append(ast.literal_eval(k))

ks = np.array(lst_k) #output variables 



acfs = np.ascontiguousarray(data_r['acfs'])
lst_acfs = []
for acf in acfs:
    lst_acfs.append(ast.literal_eval(acf))

acfs = np.array(lst_acfs) #input variables

seed = 7
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(acfs, ks, test_size=0.05, shuffle=True, random_state=seed)



# evaluate loaded model on test data
loaded_model.compile(loss='mae', optimizer='adam')
prediction_train = loaded_model.predict(X_train)
print(r2_score(y_train, prediction_train, multioutput='variance_weighted'))
#0.6918805022894133
prediction_test = loaded_model.predict(X_test)
print(r2_score(y_test, prediction_test, multioutput='variance_weighted'))
#0.17144665739202822



y_trainflat = y_train.flatten()
prediction_trainflat = prediction_train.flatten()
corr_train, _ = pearsonr(y_trainflat, prediction_trainflat)
corr_train#0.9182048067103267

y_testflat = y_test.flatten()
prediction_testflat = prediction_test.flatten()
corr_test, _ = pearsonr(y_testflat, prediction_testflat)
corr_test#0.7598667662847687


# For all data

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_train)):
    ident = [0.0, 5]
    ax.plot(ident,ident)
    ax.plot(y_train[i],prediction_train[i], ".")
ax.set_ylabel("True parameters")
ax.set_xlabel("Predicted parameters")
ax.set_title("Train data")
    
    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_test)):
    ident = [0.0, 4]
    ax.plot(ident,ident)
    ax.plot(y_test[i],prediction_test[i], ".")
ax.set_ylabel("True parameters")
ax.set_xlabel("Predicted parameters")
ax.set_title("Test data")
    
    
    
    
#scale=0.1    

# load json and create model
json_file = open('model_autorepressor_1000RNASacfs_seed42_scale0.1_nlags5000_k1k2k3k4.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_modelr01 = model_from_json(loaded_model_json)

# load weights into new model
loaded_modelr01.load_weights("model_autorepressor_1000RNASacfs_seed42_scale0.1_nlags5000_k1k2k3k4.h5")
print("Loaded model from disk")

data_r01 = pd.read_csv('autorepressor_1000RNASacfs_seed42_scale0.1_nlags5000_k1k2k3k4.csv',sep= " ")

ks = np.ascontiguousarray(data_r01['k'])

lst_k = []
for k in ks:
    lst_k.append(ast.literal_eval(k))

ks = np.array(lst_k) #output variables 



acfs = np.ascontiguousarray(data_r01['acfs'])
lst_acfs = []
for acf in acfs:
    lst_acfs.append(ast.literal_eval(acf))

acfs = np.array(lst_acfs) #input variables

seed = 7
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(acfs, ks, test_size=0.05, shuffle=True, random_state=seed)



# evaluate loaded model on test data
loaded_modelr01.compile(loss='mae', optimizer='adam')
prediction_train = loaded_modelr01.predict(X_train)
print(r2_score(y_train, prediction_train, multioutput='variance_weighted'))
#0.23654368868339354

prediction_test = loaded_modelr01.predict(X_test)
print(r2_score(y_test, prediction_test, multioutput='variance_weighted'))
#0.16954647235370415

#y_trainflat = y_train.flatten()
#prediction_trainflat = prediction_train.flatten()
#np.corrcoef(y_trainflat, prediction_trainflat)[0,1]

r2_score(y_train, prediction_train, multioutput='raw_values')
np.average(r2_score(y_train, prediction_train, multioutput='raw_values'))

#Fa più fatica a riconoscere i valori che non cambiano.
y_train[0][6]
prediction_train[0][6]

y_train[0]
prediction_train[0]


y_trainflat = y_train.flatten()
prediction_trainflat = prediction_train.flatten()
corr_train, _ = pearsonr(y_trainflat, prediction_trainflat)
corr_train#0.9333620872844718

y_testflat = y_test.flatten()
prediction_testflat = prediction_test.flatten()
corr_test, _ = pearsonr(y_testflat, prediction_testflat)
corr_test#0.9311876291423031




# For all data

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_train)):
    ident = [0.0, 2]
    ax.plot(ident,ident)
    ax.plot(y_train[i],prediction_train[i], ".")
ax.set_ylabel("True parameters")
ax.set_xlabel("Predicted parameters")
ax.set_title("Train data")
    
    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_test)):
    ident = [0.0, 1.5]
    ax.plot(ident,ident)
    ax.plot(y_test[i],prediction_test[i], ".")
ax.set_ylabel("True parameters")
ax.set_xlabel("Predicted parameters")
ax.set_title("Test data")


















#Save important information
columns = ['r2score_test','r2score_alltraindata','r2score_train','execution_time']
df_tot = pd.DataFrame(columns = columns)
actual_dir = os.getcwd()
file_path = r'{}\{}.csv'
df_tot.loc[0] = [r2score_test, r2score_train, r2score_train50, elapsed_time_date]

df_tot.to_csv(file_path.format(actual_dir,"INFOautorepressor_1000RNASacfs_seed42_scale0.1_nlags5000_k1k2k3k4"), sep =" ", index = None, header=True, mode = "w") 




#RNAS+PROTEINS

# load json and create model
json_file = open('model_autorepressor_1000RNASPROTEINSacfs_seed42_scale0.5_nlags5000_k1k2k3k4.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_autorepressor_1000RNASPROTEINSacfs_seed42_scale0.5_nlags5000_k1k2k3k4.h5")
print("Loaded model from disk")

data_rp = pd.read_csv('autorepressor_1000RNASPROTEINSacfs_seed42_scale0.5_nlags5000_k1k2k3k4.csv',sep= " ")

ks = np.ascontiguousarray(data_rp['k'])

lst_k = []
for k in ks:
    lst_k.append(ast.literal_eval(k))

ks = np.array(lst_k) #output variables 



acfs = np.ascontiguousarray(data_rp['acfs'])
lst_acfs = []
for acf in acfs:
    lst_acfs.append(ast.literal_eval(acf))

acfs = np.array(lst_acfs) #input variables

seed = 7
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(acfs, ks, test_size=0.05, shuffle=True, random_state=seed)

# evaluate loaded model on test data
loaded_model.compile(loss='mae', optimizer='adam')
prediction_train = loaded_model.predict(X_train)
print(r2_score(y_train, prediction_train, multioutput='variance_weighted'))
#0.6810341514255214

prediction_test = loaded_model.predict(X_test)
print(r2_score(y_test, prediction_test, multioutput='variance_weighted'))
#0.2839312209038598



# For all data

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_train)):
    ident = [0.0, 8]
    ax.plot(ident,ident)
    ax.plot(y_train[i],prediction_train[i], ".")
    
    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
for i in np.arange(0,len(y_test)):
    ident = [0.0, 4]
    ax.plot(ident,ident)
    ax.plot(y_test[i],prediction_test[i], ".")