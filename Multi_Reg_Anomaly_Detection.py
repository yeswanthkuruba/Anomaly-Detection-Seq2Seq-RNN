# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 07:35:33 2018

@author: yeswanth.kuruba
"""

import PreProcessing
import tensorflow as tf
import ADConfig
from sklearn.externals import joblib
from pathlib import Path
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import numpy as np
import time, uuid
import pandas as pd

start = time.time()
data = pd.read_pickle("multiple_time series_data.pkl")
forecast_length = 1
Xfull, Yfull, Xtr, Ytr, Xts, Yts, Xv, Yv, scaler_filename = PreProcessing.data_preparing_(data,forecast_length)
# Internal neural network parameters
seq_length = Xfull.shape[0]
out_length = Yfull.shape[0]        
# Output dimension (e.g.: multiple signals at once, tied in time)
output_dim = Yfull.shape[-1]
input_dim = Xfull.shape[-1]        

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
        tf.placeholder(tf.float32, shape=(None, output_dim), name="inp_{}".format(t))
        for t in range(seq_length)
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
        for t in range(out_length)]        
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
            next_input = tf.add(tf.matmul(cell_output, w_out), b_out)          

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
#            Xtr_f = np.ones((1, Xtr.shape[1], input_dim-1))
    feed_dict = {enc_inp[t]: Xtr[t] for t in range(len(enc_inp))}
#            feed_dict.update({feats[t]: Xtr_f[t] for t in range(len(feats))} )
    feed_dict.update({expected_sparse_output[t]: Ytr[
                     t] for t in range(len(expected_sparse_output))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


def test_batch(Xts, Yts):
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op.
    """    
#            Xts_f = np.ones((1, Xts.shape[1], input_dim-1))
    feed_dict = {enc_inp[t]: Xts[t] for t in range(len(enc_inp))}
#            feed_dict.update({feats[t]: Xts_f[t] for t in range(len(feats))} )
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

print("model build")
saver_rest = tf.train.Saver()
with tf.Session() as sess:
    saver_rest.restore(sess, checkpoint_file)
#            ## MONTE CARLO METHODS APPLICATION  FOR FULL/ENTIRE DATA
    feed_dict = {enc_inp[t]: Xfull[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
    scaler = joblib.load(scaler_filename) 
    Yfull_original =  scaler.inverse_transform(Yfull[0])
    Yfull_original = Yfull_original.reshape(1,Yfull_original.shape[0],Yfull_original.shape[1])
    outputs_original =  scaler.inverse_transform(outputs[0])
    outputs_original = outputs_original.reshape(1,outputs_original.shape[0],outputs_original.shape[1])
    smape_scale = PreProcessing.smape(Yfull, outputs)  
    smape_original = PreProcessing.smape(Yfull_original, outputs_original)  
    confidence_score_scale = (1 - smape_scale)*100
    confidence_score_original = (1 - smape_original)*100
    
    uncertainty=[]
    for rept in range(0, 100):
        outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
        diff = np.mean((Yfull - outputs) ** 2,axis = 1)
        uncertainty.append(diff)
    eta1 = np.sqrt(np.mean(np.array(uncertainty),axis = 0))
    print(eta1.shape)

    #validation Errors
    feed_dict = {enc_inp[t]: Xv[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
    eta2 = np.sqrt(np.mean((Yv - outputs) ** 2,axis = 1))
    print(eta2.shape)
    eta = np.sqrt( ((eta1)**2) + ((eta2)**2) )
    print(eta.shape)

#Xfull, Yfull, Xtr, Ytr, Xts, Yts, Xv, Yv, scaler_filename = PreProcessing.data_preparing_(data, forecast_length)

Xtesting = np.concatenate((Xts, Xv), axis=1)
Ytesting = np.concatenate((Yts, Yv), axis=1)

saver_rest = tf.train.Saver()
with tf.Session() as sess:
    saver_rest.restore(sess, checkpoint_file)
    model_build(Xfull, Yfull, Xtesting, Ytesting, checkpoint_file)

print("model build")
saver_rest = tf.train.Saver()
with tf.Session() as sess:
    saver_rest.restore(sess, checkpoint_file)
    feed_dict = {enc_inp[t]: Xfull[t] for t in range(seq_length)}
    outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
    etas = np.array([[eta[0]]*outputs.shape[1]])
    upper_bound = outputs+(ADConfig.confidance_interval*eta)
    lower_bound = outputs-(ADConfig.confidance_interval*eta)
    
    outputs_ = outputs.reshape([-1])
    actual_ = Yfull.reshape([-1])
    eta_ = etas.reshape([-1])
    upper_bound_ = upper_bound.reshape([-1])
    lower_bound_ = lower_bound.reshape([-1])
    
    scaler = joblib.load(scaler_filename) 
    outputs_rs =  scaler.inverse_transform(outputs[0])
    upper_bound_rs =  scaler.inverse_transform(upper_bound[0])
    lower_bound_rs =  scaler.inverse_transform(lower_bound[0])
    actual_rs =  scaler.inverse_transform(Yfull[0])
    
    outputs_rs = outputs_rs.reshape([-1])
    actual_rs = actual_rs.reshape([-1])
    upper_bound_rs = upper_bound_rs.reshape([-1])
    lower_bound_rs = lower_bound_rs.reshape([-1])
    
    anomaly_type_list = []
    isanomaly_list = []
    anomaly_score_list = []
    output_list = []
    upper_bound_list = []
    lower_bound_list = []
    actual_list = []
    KPIM_ID = []
    
    loop = outputs.shape[-1]
    j = 0
    anomaly_counter_ = [0]*loop
    for i in range(len(outputs_)):
        if(i%loop == 0):
            j = 0
        upper_bound = upper_bound_[i]
        lower_bound = lower_bound_[i]
        output = outputs_[i]
        actual = actual_[i]
        eta = eta_[j]
        if(actual>output):
            anomaly_type = "upward"
        else:
            anomaly_type = "downward"
        if(actual<=upper_bound and actual>= lower_bound):
            isanomaly = "N"
            anomaly_counter_[j] = anomaly_counter_[j]+1
            anomaly_score = PreProcessing.AnomalyScore(actual,output,eta,anomaly_counter_[j],isanomaly)
        else:
            isanomaly = "Y"    
            anomaly_score = PreProcessing.AnomalyScore(actual,output,eta,anomaly_counter_[j],isanomaly)
            anomaly_counter_[j] = 0
            
        if(lower_bound<0):
            lower_bound = 0
        anomaly_type_list.append(anomaly_type)
        isanomaly_list.append(isanomaly)
        anomaly_score_list.append(anomaly_score)
        output_list.append(outputs_rs[i])
        upper_bound_list.append(upper_bound_rs[i])
        lower_bound_list.append(lower_bound_rs[i])
        actual_list.append(actual_rs[i])
        KPIM_ID.append(j)
        j += 1
    out = pd.DataFrame([KPIM_ID,actual_list,output_list,upper_bound_list,lower_bound_list,isanomaly_list,anomaly_score_list,anomaly_type_list])
    out_ = out.transpose()
    out_.columns = ["KPIM_ID", "value", "predicted", "upper_bound", "lower_bound", "isAnomaly", "anomaly_score", "anomaly_type"]      
    out_.to_csv("ANOMALY_DETECTION_MR.csv", mode="a", header = False)
    print(time.time()-start)        


