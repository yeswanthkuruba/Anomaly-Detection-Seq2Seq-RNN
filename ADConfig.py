# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:46:19 2018

@author: yeswanth.kuruba
"""
import uuid

seq_length = 168
forecasting_length = 1
Model_Forecasting = 1
split_train = 0.75
split_test = 0.15
confidance_interval = 1.64
if seq_length > 200:
    high_seqlen_flag = 1
else:
    high_seqlen_flag = 0   

# Number of stacked recurrent cells, on the neural depth axis.
hidden_dim_list_enc = [16, 32, 12]
hidden_dim_list_dec = [16, 32, 12]
dropout_list = [0.5,0.5,0.5]
layers_stacked_count_enc = len(hidden_dim_list_enc)
layers_stacked_count_dec = len(hidden_dim_list_dec)

hidden_dim = hidden_dim_list_enc[0]
out_hidden_dim = hidden_dim_list_dec[-1]
# Optmizer:
learning_rate = 0.007  # Small lr helps not to diverge during training.

nb_iters = 600
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting


early_stop = 150

checkpoint_file = "C:\\Users\\yeswanth.kuruba\\OneDrive - Subex Limited\\Projects\\Anomaly Detection\\model1\\model.ckpt"
checkpoint_file = str(uuid.uuid1())
