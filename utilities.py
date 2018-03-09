import os
import pandas as pd
import numpy as np
import subprocess
import tensorflow as tf

def normalize(data, mean=0.0, std=1.0):
    n_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return n_data * (std + 1e-8) + mean

#outputs parameters of Beta distribution
def policy_network(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=3, 
        size=32, 
        activation=tf.tanh,
        output_activation=None
        ):
    
     with tf.variable_scope(scope):        
        out = input_placeholder
        for _ in range(n_layers):
            out = tf.layers.dense(inputs=out, units=size, activation=activation)
        out_1 = tf.layers.dense(inputs=out, units=size, activation=activation)         
       
        out_cont = tf.layers.dense(inputs=out_1, units=output_size, activation=output_activation)
        alpha = tf.log(tf.exp(out_cont)+1.0)+1.00        
        
        out_cont1 = tf.layers.dense(inputs=out_1, units=output_size, activation=output_activation) 
        beta = tf.log(tf.exp(out_cont1)+1.0)+1.00           
        
        return alpha,beta
        
#template for a generic environment class for the multi-agent contextual bandit setting
#the environment object should sample states from the state distribution,and
#fetch rewards for the actions submitted by the agents
class Environment:
    def __init__(self,seed=32):
      

    #sample state based on initial distibution
    def samplestates(self, batch_size):
      
        return s
    
    #implement function that takes actions of all the agents and outputs corresponding rewards
    def get_rewards(self, actions):
        
            return rewards
