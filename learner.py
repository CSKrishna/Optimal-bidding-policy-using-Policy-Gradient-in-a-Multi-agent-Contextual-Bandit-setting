import numpy as np
import tensorflow as tf
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
from utilities import *
from agent import *   


def get_rewards(env,ag1_prices,ag2_prices):
    actions=np.concatenate((ag1_prices,ag2_prices),axis=1)             
    rewards = env.get_rewards(actions)  
    ag1_rewards = rewards[:,0]
    ag2_rewards = rewards[:,1]
    return ag1_rewards,ag2_rewards
    

 #learn Nash Eq policy & reward for an agent given another agent's deterministic policy
def get_Nash_reward(sess,agent1_Nash,agent2,env):
    batch_size=50
    
    for itr in range(1000):
        s = env.samplestates(batch_size)        
        
        ag1_prices,_=agent1_Nash.sample_actions(sess,s)
        
        # we now sample actions deterministically for agent 2
        _,ag2_prices=agent2.sample_actions(sess,s)
        
        ag1_rewards,ag2_rewards=get_rewards(env,ag1_prices,ag2_prices)
      
        ag1_adv = normalize(ag1_rewards)                 
            
        _=agent1_Nash.improve_policy(sess,s,ag1_adv,ag1_prices)  
    
    #get smart reward for policy learnt by agent
    _,_,m1_m,_,_,_=get_smart_rewards(sess,agent1_Nash,agent2,env)
    
    return m1_m
     

        
        
        
        

#given agents' policies, compute the degree of deviation from the true Nash equilibrium
def assess_policy_accuracy(sess,agent1,agent1_Nash,agent2,agent2_Nash,env):
    #keep agent 2's policies fixed and learn agent 1's policies from scratch
    print("Computing Nash Soln for Agent 1")
    ag1_nash=get_Nash_reward(sess,agent1_Nash,agent2,env)
    print("Agent1's Nash Profit: " + repr(ag1_nash))     
 
    _,_,m1_m,m2_m,_,_=get_smart_rewards(sess,agent1,agent2,env)
    print("Agent1's MARL profit: " + repr(m1_m))
    
    
    ag1_acc= m1_m/(ag1_nash)
    print("Agent1 Accuracy: "+ repr(ag1_acc))
    
    #keep agent 1's policies fixed and learn agent 2's policies from scratch
    print("Computing Nash Soln for Agent 2")    
    ag2_nash=get_Nash_reward(sess,agent2_Nash,agent1,env)
    print("Agent2's Nash Profit: " + repr(ag2_nash))    
    
    print("Agent2's MARL profit: " + repr(m2_m))
    ag2_acc= (m2_m)/(ag2_nash)
    print("Agent2 Accuracy: "+ repr(ag2_acc))
    
    return ag1_acc,ag2_acc
    
    
    
#get rewards when both agents bid at their smart policies
def get_smart_rewards(sess,agent1,agent2,env):       
   
    s = env.getNextObservations(5000)      
    
    ag1_prices,ag1_mean=agent1.sample_actions(sess,s)
    ag2_prices,ag2_mean=agent2.sample_actions(sess,s)

    m1,m2=get_rewards(env,ag1_prices,ag2_prices)
    m1_m,m2_m=get_rewards(env,ag1_mean,ag2_mean)   
   
   
    ag1_p = np.mean(ag1_mean,axis = 0)
    ag2_p = np.mean(ag2_mean,axis = 0)  
    
    return np.mean(m1),np.mean(m2),np.mean(m1_m),np.mean(m2_m),ag1_p,ag2_p
    
    
    
         

#============================================================================================#
# Use Policy Gradient Theorem to modify both agents' policies
#============================================================================================#
def train_PG(exp_name='',
             batch_size = 250,
             n_episodes=25000,           
             learning_rate=1e-3,              
             logdir=None,              
             seed=0,
             # network arguments
             n_layers=2,
             size=64             
             ):
    
    env = Environment() 
    agent1=Agent(env,n_layers,size,learning_rate,"agent1")
    agent2=Agent(env,n_layers,size,learning_rate,"agent2")
    agent1_Nash=Agent(env,3,32,1e-2,"agent1_Nash")
    agent2_Nash=Agent(env,3,32,1e-2,"agent2_Nash")
    

    start = time.time()   
    
    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)  
   

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    n_iter = n_episodes // batch_size    
   
    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    
    #========================================================================================#
    # Training Loop
    #========================================================================================#
      
    
    
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        #simulate a batch of temperature-gas price states
        s = env.samplestatess(batch_size)    
         
        ag1_prices,_ =agent1.sample_actions(sess,s)
        ag2_prices,_ =agent2.sample_actions(sess,s)
                

        #====================================================================================#                                       # Feed agents' actions into the market simulator and obtain corresponding rewards
        #====================================================================================#             
        #Convert agent RTM actions to corresponding prices
        ag1_rewards,ag2_rewards=get_rewards(env,ag1_prices,ag2_prices)
       
        #====================================================================================#
        #                          
        # Advantage Normalization
        #====================================================================================#
        ag1_adv = normalize(ag1_rewards)
        ag2_adv = normalize(ag2_rewards)           
        
        
        #====================================================================================#
        #                         
        # Performing the Policy Update
        #====================================================================================#
        #update policy parameters for agent1
        #if (itr % 20 < 10):
        loss1=agent1.improve_policy(sess,s,ag1_adv,ag1_prices)            
        #update policy parameters for agent2    
        #else:  
        loss2=agent2.improve_policy(sess,s,ag2_adv,ag2_prices)
            
        
        # Log diagnostics        
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageProfit_agt1", np.mean(ag1_rewards))
        logz.log_tabular("AverageProfit_agt2", np.mean(ag2_rewards))    
        
            
        logz.log_tabular("Agt1_StdReturn", np.std(ag1_rewards))
        logz.log_tabular("Agt2_StdReturn", np.std(ag2_rewards))
        
        logz.log_tabular("Agt1_MaxReturn", np.max(ag1_rewards))
        logz.log_tabular("Agt2_MaxReturn", np.max(ag2_rewards))
        
        logz.log_tabular("Agt1_MinReturn", np.min(ag1_rewards))
        logz.log_tabular("Agt2_MinReturn", np.min(ag2_rewards))
        
      
        logz.dump_tabular()
        logz.pickle_tf_vars()
       
    m1,m2,m1_m,m2_m,ag1_p,ag2_p=get_smart_rewards(sess,agent1,agent2,env)    
    print("Agent1 Stochastic Profit: "+ repr(m1))
    print("Agent2 Stochastic Profit: "+ repr(m2))
    
    print("Agent1 Deterministic Profit: "+ repr(m1_m))
    print("Agent2 Deterministic Profit: "+ repr(m2_m))

    print("Agent1 Mean Price")
    print (ag1_p)
    print("Agent2 Prices")
    print (ag2_p) 
    
    print("Assessing degree of deviation from Nash Eq")
    ag1_imp,ag2_imp=assess_policy_accuracy(sess,agent1,agent1_Nash,agent2,agent2_Nash,env)
    print("Agent1 Accuracy: "+ repr(ag1_imp))
    print("Agent2 Accuracy: "+ repr(ag2_imp))

   
        
        
def main():
    import argparse
    parser = argparse.ArgumentParser()   
    parser.add_argument('--exp_name', type=str, default='vpg')   
    parser.add_argument('--n_episodes', '-n', type=int, default=25000)
    parser.add_argument('--batch_size', '-b', type=int, default=5)      
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2)    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=3)
    parser.add_argument('--size', '-s', type=int, default=64) 

                     
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_'  + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    
    return args, logdir

def train_func(args, seed, logdir):    
    train_PG(
        exp_name=args.exp_name,     
        batch_size=args.batch_size,
        n_episodes=args.n_episodes,        
        learning_rate=args.learning_rate,       
        logdir=os.path.join(logdir,'%d'%seed),        
        seed=seed,
        n_layers=args.n_layers,
        size=args.size        
        )                               
    
if __name__ == "__main__":    
    args, logdir = main()    
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    for e in range(args.n_experiments):
        seed = args.seed +  e        
        print('Running experiment with seed %d'%seed)       
        p = Process(target=train_func,args= (args,seed,logdir,))
        p.start()
        p.join()
        
       
        

        
        
        
        
       
 
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    