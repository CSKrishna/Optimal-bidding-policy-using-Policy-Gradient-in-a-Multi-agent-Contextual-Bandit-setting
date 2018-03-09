import numpy as np
import tensorflow as tf
from utilities import *
   
class Agent(object):    
    def __init__(self,env,n_layers,size,lr,name):            
            self.ob_dim=env.state_dim
            self.ac_dim=env.action_dim
            self.name=name
            self.n_layers=n_layers
            self.size=size
            self.lr=lr
            self.c1=0.0
            self.c2=100.0
            self.HR=HR
            self.build()
            
    def _create_placeholders(self):        
        with tf.variable_scope(self.name):
            self.actions=tf.placeholder(shape=[None, self.ac_dim],name="actions",dtype=tf.float32)           
            self.states=tf.placeholder(shape=[None, self.ob_dim],name="state",dtype=tf.float32) 
            self.adv=tf.placeholder(shape=[None],name="adv",dtype=tf.float32) 
            
    def _policy_parameters(self):        
        alpha,beta=policy_network(self.states,self.ac_dim,self.name,n_layers=self.n_layers,size=self.size)
        return alpha,beta
    
    def _compute_logprob(self,alpha,beta):
        with tf.variable_scope(self.name):
            sy_z1 = (self.actions-self.c1)/self.c2
            logprob=tf.reduce_sum((alpha-1.0)*tf.log(sy_z1)+(beta-1.0)*tf.log(1-sy_z1)+tf.lgamma(alpha+beta)-tf.lgamma(alpha)-tf.lgamma(beta),axis=1) 
        return logprob
    
    def _create_feed_dict(self,states,advantages,actions):
        return {self.states: states,                    
                self.adv: advantages,                      
                self.actions:actions}
    
    def _add_loss_op(self,logprob):        
            total_loss = -tf.reduce_mean(logprob*self.adv) + tf.reduce_mean(logprob)*0.1
            update_op = tf.train.AdamOptimizer(self.lr).minimize(total_loss)
            return total_loss,update_op
        
    def build(self):
        self._create_placeholders()
        self.alpha,self.beta=self._policy_parameters()
        self.logprob=self._compute_logprob(self.alpha,self.beta)
        self.loss,self.train_op=self._add_loss_op(self.logprob)
    
    def sample_actions(self,session,states):
        alpha,beta = session.run([self.alpha,self.beta],feed_dict={self.states:states})          
        prices = np.random.beta(alpha,beta)*self.c2+self.c1
        mean_prices = alpha/(alpha+beta)*self.c2+self.c1
        return prices,mean_prices
    
    def improve_policy(self,session,states,advantages,actions):
        feed_dict=self._create_feed_dict(states,advantages,actions)
        loss,_ = session.run([self.loss,self.train_op],feed_dict=feed_dict)
        return loss
