# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:31:44 2018

@author: yeswanth.kuruba
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
import numpy as np
import pandas as pd
import ADConfig

def data_preparing(data, time_encoder, forecasting_length):
    window_size = ADConfig.seq_length
#    data = signal.detrend(data1)
#    trend = data1-data
#    model = LinearRegression() 
#    model.fit(np.reshape(np.arange(len(data1)),(1,-1)),np.reshape(trend,(1,-1)))
    df = np.array(data).reshape((len(data), 1))
    # train the standardization
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)
    # standardization the dataset and print the first 5 rows
    normalized = scaler.transform(df)
    kept_value = normalized.flatten().tolist()
    kept_values = []
    for i in range(len(kept_value)):
        kept_values.append([kept_value[i]]+time_encoder[i].tolist())
    
    X,Y = [],[]
    for i in range(len(kept_values) - window_size):
        X.append(kept_values[i:i + window_size])
        Y.append([kept_values[i + window_size:i + window_size + forecasting_length][0][0]])
            
    if(forecasting_length>1):
        # To be able to concat on inner dimension later on:
        X = np.expand_dims(X[:-forecasting_length+1], axis=2)
        Y = np.expand_dims(Y[:-forecasting_length+1], axis=2)
    else:
        # To be able to concat on inner dimension later on:
        X = np.expand_dims(X, axis=2)
        Y = np.expand_dims(Y, axis=2)
    
    X = X.reshape(X.shape[0],X.shape[1],X.shape[3])
    
    
    Xmax = float(scaler.data_max_)
    Xmin = float(scaler.data_min_)
#    coef = model.coef_
#    intercept = model.intercept_
    X_train = X[:int(len(X)*ADConfig.split_train)].transpose((1, 0, 2))
    Y_train = Y[:int(len(Y)*ADConfig.split_train)].transpose((1, 0, 2))
    X_test = X[int(len(X)*ADConfig.split_train):int(len(X)*(ADConfig.split_train+ADConfig.split_test))].transpose((1, 0, 2))
    Y_test = Y[int(len(Y)*ADConfig.split_train):int(len(Y)*(ADConfig.split_train+ADConfig.split_test))].transpose((1, 0, 2))
    X_val = X[int(len(X)*(ADConfig.split_train+ADConfig.split_test)):].transpose((1, 0, 2))
    Y_val = Y[int(len(X)*(ADConfig.split_train+ADConfig.split_test)):].transpose((1, 0, 2))
    Xfull = X.transpose((1, 0, 2))
    Yfull = Y.transpose((1, 0, 2))
    
    return Xfull, Yfull, X_train, Y_train, X_test, Y_test, X_val, Y_val, Xmax, Xmin

def data_preparing_(data, forecasting_length):
    window_size = ADConfig.seq_length
    # train the standardization
    data = data.fillna(0)
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    scaler_filename = "MinMaxscaler.save"
    joblib.dump(scaler, scaler_filename) 
    kept_values = data_scaled
    X,Y = [],[]
    for i in range(len(kept_values) - window_size):
        X.append(np.array(kept_values.iloc[i:i + window_size]))
        Y.append(np.array(kept_values[i + window_size:i + window_size + forecasting_length]))
            
    if(forecasting_length>1):
        # To be able to concat on inner dimension later on:
        X = np.expand_dims(X[:-forecasting_length+1], axis=2)
        Y = np.expand_dims(Y[:-forecasting_length+1], axis=2)
    else:
        # To be able to concat on inner dimension later on:
        X = np.expand_dims(X, axis=2)
        Y = np.expand_dims(Y, axis=2)
    
    X = X.reshape(X.shape[0],X.shape[1],X.shape[3])
    Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3])    
    
#    Xmax = scaler.data_max_
#    Xmin = scaler.data_min_
#    coef = model.coef_
#    intercept = model.intercept_
    X_train = X[:int(len(X)*ADConfig.split_train)].transpose((1, 0, 2))
    Y_train = Y[:int(len(Y)*ADConfig.split_train)].transpose((1, 0, 2))
    X_test = X[int(len(X)*ADConfig.split_train):int(len(X)*(ADConfig.split_train+ADConfig.split_test))].transpose((1, 0, 2))
    Y_test = Y[int(len(Y)*ADConfig.split_train):int(len(Y)*(ADConfig.split_train+ADConfig.split_test))].transpose((1, 0, 2))
    X_val = X[int(len(X)*(ADConfig.split_train+ADConfig.split_test)):].transpose((1, 0, 2))
    Y_val = Y[int(len(X)*(ADConfig.split_train+ADConfig.split_test)):].transpose((1, 0, 2))
    Xfull = X.transpose((1, 0, 2))
    Yfull = Y.transpose((1, 0, 2))
    
    return Xfull, Yfull, X_train, Y_train, X_test, Y_test, X_val, Y_val, scaler_filename


def new_data_preparing(data,Xmax, Xmin, window_size, time_encoders):
    df = np.array(data).reshape((len(data), 1))
    kept_values = []
    for i in range(len(df)):
        kept_values.append(((df[i] - Xmin) / (Xmax - Xmin)).tolist()+time_encoders[i].tolist())
    X,Y = [],[]
    for i in range(len(kept_values) - window_size):
        X.append(kept_values[i:i + window_size])
        Y.append([kept_values[i + window_size:i + window_size + 1][0][0]])
    X = np.array(X).transpose((1, 0, 2))
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0],1,1).transpose((1, 0, 2))
    return X,Y

def rescalig(Y,Xmax,Xmin):
        X = Y * (Xmax - Xmin) + Xmin
        return X

def AnomalyScore(Y, Ypred, eta, anomaly_counter, isanomaly):
    if(isanomaly == "Y"):
        if(np.abs(Y - Ypred) >= 2.58*eta):
            X_percentage_ab = 100
        elif(Y-Ypred >=0 and Y - Ypred < 2.58*eta):
            Xmin = Ypred + ADConfig.confidance_interval*eta
            Xmax = Ypred + 2.58*eta
            X_percentage_ab = ((Ypred - Xmin) / (Xmax - Xmin))
            if(X_percentage_ab<=1):
                X_percentage_ab = X_percentage_ab*100
            else:
                X_percentage_ab =  100
        else:
            Xmax = Ypred - ADConfig.confidance_interval*eta
            Xmin = Ypred - 2.58*eta
            X_percentage_ab = ((Ypred - Xmin) / (Xmax - Xmin))
            if(X_percentage_ab<=1):
                X_percentage_ab = X_percentage_ab*100
            else:
                X_percentage_ab =  100
    else:
        X_percentage_ab = 0
    
    eplison = 0.0001
    if(Y == 0):
        Y = Y+eplison
    X_percentage_re = np.absolute(((Ypred -Y)/Y))
    if(X_percentage_re<=1):
        X_percentage_re = X_percentage_re*100
    else:
        X_percentage_re = 100
    
    X_percentage_dis = 1 / float(1 + np.exp(- (anomaly_counter/10.0)))
    X_percentage_dis = ((X_percentage_dis - 0.5) / (1 - 0.5))
    if(X_percentage_dis<=1):
        X_percentage_dis = X_percentage_dis*100
    else:
        X_percentage_dis = 100

    X_percentage = X_percentage_ab*(0.65)+X_percentage_re*(0.20)+X_percentage_dis*(0.15)
    
    if(isanomaly == "Y"):
        return X_percentage
    else:
        return 0

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff, axis = 0)

