# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:57:40 2018

@author: yeswanth.kuruba
"""

import PreProcessing
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='-1'
import tensorflow as tf
import ADConfig
from pathlib import Path
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import numpy as np
import time, uuid
import pandas as pd
#folder_name = "BALANCE TRANSFER//"
#files = ["Int BT Refund_Amt", "Int BT Refund_Count", "Local_In_Count", "Local_Out_Amt", "Local_Out_Count"]
##files = ["Int BT Refund_Amt", "Int BT Refund_Count", "Int_BT_Amt", "Int_BT_Count", "Local_In_Amt", "Local_In_Count", "Local_Out_Amt", "Local_Out_Count"]
filename = "outgoing_duration"
test_days = 20
data_freq = "hourly"
if(data_freq == "daily"):
    freq = 7
else:
   freq = 24


def outlierReplace(data):
    mean = data_t.mean()
    std = data_t.std()  
    for i in range(len(data)):
        if(np.abs(data[i]- mean) >= 3*std):
            data[i] = mean
    return data
from sklearn.preprocessing import OneHotEncoder

data_all = pd.read_csv(filename+".csv")
data_all["timestamp"] = [ x+' '+str(y).zfill(2)+':00' for x,y in zip(data_all["timestamp"],data_all["hour"]) ]
data_all["timestamp"] = pd.to_datetime(data_all["timestamp"], format = '%d-%m-%Y %H:%M')
data_all.rename(columns = {data_all.columns[-1]:'value'}, inplace = True)
country_codes = [56,51,1,54,57,591,53,55,52,34,593,58,598,49,1809,1829,44,61,33,39,595]
country_codes = [53,216,252,387,290,881,960,232,235,1284,269,1876]
for code in country_codes:
    start = time.time()
    data = data_all[data_all["country_code"]==code]
    data.set_index("timestamp", inplace = True)
    data = data.resample("H").sum().reset_index().fillna(0)
    data['weekday'] = data['timestamp'].dt.dayofweek
    data['weekend'] = np.where(data['weekday']<5, 0, 1)
    data['hour_day'] = data['timestamp'].dt.hour
    enc = OneHotEncoder()
    enc.fit(data[['weekend','hour_day']])  
    time_encoder = enc.transform(data[['weekend','hour_day']]).toarray()
    data_t = data[["value"]]
    kpim = filename+str(code)
       
    forecast_length = 1
    Xfull, Yfull, Xtr, Ytr, Xts, Yts, Xv, Yv, Xmax, Xmin = PreProcessing.data_preparing(data_t, time_encoder, forecast_length)
    # Internal neural network parameters
    seq_length = Xfull.shape[0]
    out_length = Yfull.shape[0]
    
    # Output dimension (e.g.: multiple signals at once, tied in time)
    output_dim = Yfull.shape[-1]
    input_dim = Xfull.shape[-1]
    
    Xtr_f = np.ones((1, Xtr.shape[1], input_dim-1))
    Xts_f = np.ones((1, Xts.shape[1], input_dim-1))
    Xv_f = np.ones((1, Xv.shape[1], input_dim-1))
    Xfull_f = np.ones((1, Xfull.shape[1], input_dim-1))
        
    # Backward compatibility for TensorFlow's version 0.12:
    try:
        tf.nn.seq2seq = tf.contrib.legacy_seq2seq
        tf.nn.rnn_cell = tf.contrib.rnn
        tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
        print("TensorFlow's version : 1.0 (or more)")
    except:
        print("TensorFlow's version : 0.12")
       
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    with tf.variable_scope("Encoder") as scope:
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
            for t in range(seq_length)
        ]
        dec_inp = [
            tf.placeholder(tf.float32, shape=(None, 1), name="inp_{}".format(t))
            for t in range(seq_length)
        ]
        feats = [
            tf.placeholder(tf.float32, shape=(None, input_dim-1), name="dec_31_feats_{}".format(t))
            for t in range(out_length)
        ]
        
        # Th encoder cell, multi-layered with dropout
        if ADConfig.high_seqlen_flag == 1:
            cells_enc = []
            for i in range(ADConfig.layers_stacked_count_enc):
                cell_enc = LSTMCell(ADConfig.hidden_dim_list_enc[i])
                cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=1.0-ADConfig.dropout_list[i])    
                cells_enc.append(cell_enc)
            cell_enc = tf.contrib.rnn.MultiRNNCell(cells_enc)
            
        else:
            cells_enc = []
            for i in range(ADConfig.layers_stacked_count_enc):
                cell_enc = GRUCell(ADConfig.hidden_dim_list_enc[i])
                cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob=1.0-ADConfig.dropout_list[i])    
                cells_enc.append(cell_enc)
            cell_enc = tf.contrib.rnn.MultiRNNCell(cells_enc)
            
        encoder_outputs, encoder_final_state = tf.contrib.rnn.static_rnn(cell_enc,
                                                  inputs=enc_inp, dtype=tf.float32) 
        
                                         
    with tf.variable_scope("Decoder") as scope:
        # Decoder: inputs    
        dummy_zero_input = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO_dummy_zero_input")]
    
        # Decoder: expected outputs
        expected_sparse_output = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="expected_sparse_output_".format(t))
            for t in range(out_length)
        ]
    
        w_out = tf.Variable(tf.random_normal([ADConfig.out_hidden_dim, output_dim]))
        b_out = tf.Variable(tf.random_normal([output_dim]))
        
        # The decoder, also multi-layered
        if ADConfig.high_seqlen_flag == 1:
            cells_dec = []
            for i in range(ADConfig.layers_stacked_count_dec):
                cell_dec = LSTMCell(ADConfig.hidden_dim_list_dec[i])
                cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob=1.0-ADConfig.dropout_list[i])    
                cells_dec.append(cell_dec)
            cell_dec = tf.contrib.rnn.MultiRNNCell(cells_dec)
            
        else:
            cells_dec = []
            for i in range(ADConfig.layers_stacked_count_dec):
                cell_dec = GRUCell(ADConfig.hidden_dim_list_dec[i])
                cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob=1.0-ADConfig.dropout_list[i])    
                cells_dec.append(cell_dec)
            cell_dec = tf.contrib.rnn.MultiRNNCell(cells_dec)
    		
        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:
                next_cell_state = encoder_final_state            
                next_input = dummy_zero_input[0] 
            else:  
                next_cell_state = cell_state
                prev_output = tf.add(tf.matmul(cell_output, w_out), b_out)
                prev_output = tf.reshape(prev_output, (1, -1, 1))
                next_input = tf.concat([prev_output, feats], axis=2)
                next_input = tf.reshape(next_input, (-1, input_dim))
                
            elements_finished = (time >= out_length)
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)        
        
        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell_dec, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()
        
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        
        decoder_predictions = tf.add(tf.matmul(decoder_outputs_flat, w_out), b_out)
        reshaped_outputs = tf.reshape(decoder_predictions, (out_length, -1, output_dim))
    
    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        
        epsilon = 0.01  # Smoothing factor, helps SMAPE to be well-behaved near zero    
        true_o = expected_sparse_output
        pred_o = reshaped_outputs
        summ = tf.maximum(tf.abs(true_o) , epsilon) 
        smape_ = tf.abs(pred_o - true_o) / summ #* 2.0    
        sloss =tf.losses.compute_weighted_loss(smape_, loss_collection=None)
        loss = sloss   
    
    with tf.variable_scope('Optimizer'):  # AdamOptimizer
        optimizer = tf.train.RMSPropOptimizer(
            ADConfig.learning_rate, decay=ADConfig.lr_decay, momentum=ADConfig.momentum)
        train_op = optimizer.minimize(loss)
    
    
    def train_batch(Xtr, Ytr):
        """
        Training step that optimizes the weights
        provided some batch_size X and Y examples from the dataset.
        """
        Xtr_f = np.ones((1, Xtr.shape[1], input_dim-1))
        feed_dict = {enc_inp[t]: Xtr[t] for t in range(len(enc_inp))}
        feed_dict.update({feats[t]: Xtr_f[t] for t in range(len(feats))} )
        feed_dict.update({expected_sparse_output[t]: Ytr[
                         t] for t in range(len(expected_sparse_output))})
        _, loss_t = sess.run([train_op, loss], feed_dict)
        return loss_t
    
    
    def test_batch(Xts, Yts):
        """
        Test step, does NOT optimizes. Weights are frozen by not
        doing sess.run on the train_op.
        """    
        Xts_f = np.ones((1, Xts.shape[1], input_dim-1))
        feed_dict = {enc_inp[t]: Xts[t] for t in range(len(enc_inp))}
        feed_dict.update({feats[t]: Xts_f[t] for t in range(len(feats))} )
        feed_dict.update({expected_sparse_output[t]: Yts[
                         t] for t in range(len(expected_sparse_output))})
        loss_t = sess.run([loss], feed_dict)
        return loss_t[0]
    
    def model_build(Xtr, Ytr, Xts, Yts, checkpoint_file):
        train_losses = []
        test_losses = []
        for t in range(ADConfig.nb_iters + 1):
            train_loss = train_batch(Xtr, Ytr)
            train_losses.append(train_loss)
            test_loss = test_batch(Xts, Yts)
            test_losses.append(test_loss)
            
            print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t,
                                                                       ADConfig.nb_iters, train_loss, test_loss))
            #print(test_loss,np.min(test_losses[-30:]))
            if(test_loss<=np.min(test_losses) and len(test_losses)>100):
                saver.save(sess, checkpoint_file)
                print("model saved to path {}, train loss: {}, \tTEST loss: {}".format(checkpoint_file, train_loss, test_loss))
            if(np.min(test_losses)<np.min(test_losses[-ADConfig.early_stop:]) and len(test_losses)>ADConfig.early_stop):
                print("Stopped due to Early Stopping : Step {}/{}, train loss: {}, \tTEST loss: {}".format(t,ADConfig.nb_iters, train_loss, test_loss))
                break
        print("Fin. train loss: {}, \tTEST loss: {}".format(train_loss, test_loss))
    
    folder = str(uuid.uuid1())
    
    folder_anomaly = folder+'_anomaly'
    checkpoint_file = folder_anomaly+'/model.ckpt'
    path = Path(checkpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True) 
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_build(Xtr, Ytr, Xts, Yts, checkpoint_file)
    
    saver_rest = tf.train.Saver()
    with tf.Session() as sess:
        saver_rest.restore(sess, checkpoint_file)
        ### MONTE CARLO METHODS APPLICATION  FOR FULL/ENTIRE DATA
        Xv_f = np.ones((1, Xv.shape[1], input_dim-1))
        Xfull_f = np.ones((1, Xfull.shape[1], input_dim-1))

        feed_dict = {enc_inp[t]: Xfull[t] for t in range(seq_length)}
        feed_dict.update({feats[t]: Xfull_f[t] for t in range(len(feats))} )
        outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
        Yfull_original = PreProcessing.rescalig(Yfull,Xmax,Xmin)
        outputs_original = PreProcessing.rescalig(outputs,Xmax,Xmin)
        smape_ = PreProcessing.smape(Yfull, outputs)  
        confidence_score = (1 - smape_)*100
        
        uncertainty=[]
        for rept in range(0, 250):
            outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
            diff = ((Yfull - outputs) ** 2).mean()
            uncertainty.append(diff)
        eta1 = np.sqrt(np.mean(uncertainty))
        print("\n eta1 = ", eta1)
    
        #validation Errors
        feed_dict = {enc_inp[t]: Xv[t] for t in range(seq_length)}
        feed_dict.update({feats[t]: Xv_f[t] for t in range(len(feats))} )
        outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
        eta2 = np.sqrt(((Yv - outputs) ** 2).mean())
        print("\n eta2 = ", eta2)
        
        eta = np.sqrt( ((eta1)**2) + ((eta2)**2) )
        print("\n Eta = ", eta)
    
    data_t = data["value"]
    week = data["value"]
    Ymean = np.mean(data_t)
    Xfull, Yfull, Xtr, Ytr, Xts, Yts, Xv, Yv, Xmax, Xmin = PreProcessing.data_preparing(data_t, time_encoder, forecast_length)
    
    Xtesting = np.concatenate((Xts, Xv), axis=1)
    Ytesting = np.concatenate((Yts, Yv), axis=1)
    
    saver_rest = tf.train.Saver()
    with tf.Session() as sess:
        saver_rest.restore(sess, checkpoint_file)
        model_build(Xfull, Yfull, Xtesting, Ytesting, checkpoint_file)
    anomaly_counter = 0
    saver_rest = tf.train.Saver()
    with tf.Session() as sess:
        saver_rest.restore(sess, checkpoint_file)
        feed_dict = {enc_inp[t]: Xfull[t] for t in range(seq_length)}
        feed_dict.update({feats[t]: Xfull_f[t] for t in range(len(feats))} )
        outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
        anomaly_type_list = []
        isanomaly_list = []
        anomaly_score_list = []
        output_list = []
        upper_bound_list = []
        lower_bound_list = []
        actual_list = []
        SYSTEM_GENERATED_FL = []
        KPIA_DELETE_FL = []
        KPIA_VERSION_ID = []
        PTN_ID = []
        KPIM_ID = []
        outputs = outputs.reshape([-1])
        Yfull_ = Yfull.reshape([-1])
        for output,actual in zip(outputs,Yfull_):
            upper_bound = float(output+(ADConfig.confidance_interval*eta))
            lower_bound = float(output-(ADConfig.confidance_interval*eta))
            if(actual>output):
                anomaly_type = "upward"
            else:
                anomaly_type = "downward"
            if(actual<=upper_bound and actual>= lower_bound):
                isanomaly = "N"
                anomaly_counter = anomaly_counter+1
                anomaly_score = PreProcessing.AnomalyScore(actual,output,eta,anomaly_counter,isanomaly)
            else:
                isanomaly = "Y"
                anomaly_counter = 0
                anomaly_score = PreProcessing.AnomalyScore(actual,output,eta,anomaly_counter,isanomaly)
    
            output_ = PreProcessing.rescalig(output,Xmax,Xmin)
            upper_bound_ = PreProcessing.rescalig(upper_bound,Xmax,Xmin)
            lower_bound_ = PreProcessing.rescalig(lower_bound,Xmax,Xmin)
            actual_ = PreProcessing.rescalig(actual,Xmax,Xmin)
            if(lower_bound_<0):
                lower_bound_ = 0
            anomaly_type_list.append(anomaly_type)
            isanomaly_list.append(isanomaly)
            anomaly_score_list.append(anomaly_score)
            output_list.append(output_)
            upper_bound_list.append(upper_bound_)
            lower_bound_list.append(lower_bound_)
            actual_list.append(actual_)
            SYSTEM_GENERATED_FL.append("N")
            KPIA_DELETE_FL.append("N")
            KPIA_VERSION_ID.append(1)
            PTN_ID.append(1)
            KPIM_ID.append(kpim)
        out = pd.DataFrame([KPIM_ID,output_list,actual_list,upper_bound_list,lower_bound_list,isanomaly_list,anomaly_score_list,anomaly_type_list,SYSTEM_GENERATED_FL,KPIA_DELETE_FL,KPIA_VERSION_ID,PTN_ID])
        out_ = out.transpose()
        out_.columns = ["KPIM_ID", "predicted", "value","upper_bound", "lower_bound", "isAnomaly", "anomaly_score", "anomaly_type","SYSTEM_GENERATED_FL","KPIA_DELETE_FL","KPIA_VERSION_ID","PTN_ID"]      
        out_.to_csv("KPI_ANOMALY_DETECTION_Telefonica_08_06_all.csv", mode="a", header = False)
        print(time.time()-start)

