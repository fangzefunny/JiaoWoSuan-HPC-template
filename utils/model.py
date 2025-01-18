import numpy as np 
import pandas as pd 
 
from scipy.special import softmax 
from scipy.stats import gamma, uniform, beta

from utils.fit import *
from utils.env_fn import rl_reversal
from utils.viz import *

eps_ = 1e-13
max_ = 1e+13

# ------------------------------#
#        Axulliary funcs        #
# ------------------------------#

flatten = lambda l: [item for sublist in l for item in sublist]

def get_param_name(params, block_types=['sta', 'vol'], feedback_types=['gain', 'loss']):
    return flatten([flatten([[f'{key}_{j}_{i}' for key in params]
                                               for i in feedback_types])
                                               for j in block_types])

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    return np.exp(x) 

sigmoid = lambda x: 1 / (1+clip_exp(-x))

# ------------------------------#
#         Agent wrapper         #
# ------------------------------#

class wrapper:
    '''Agent wrapper

    We use the wrapper to

        * Fit
        * Simulate
        * Evaluate the fit 
    '''

    def __init__(self, agent, env_fn):
        self.agent  = agent
        self.env_fn = env_fn
        self.use_hook = False
    
    # ------------ fit ------------ #

    def fit(self, data, method, alg, pool=None, p_priors=None,
            init=False, seed=2021, verbose=False, n_fits=40):
        '''Fit the parameter using optimization 
        '''

        # get functional inputs 
        fn_inputs = [self.loss_fn, 
                     data, 
                     self.agent.p_bnds,
                     self.agent.p_pbnds, 
                     self.agent.p_name,
                     self.agent.p_priors if p_priors is None else p_priors,
                     method,
                     alg, 
                     init,
                     seed,
                     verbose]
        
        if pool:
            sub_fit = fit_parallel(pool, *fn_inputs, n_fits=n_fits)
        else: 
            sub_fit = fit(*fn_inputs)  

        return sub_fit      

    def loss_fn(self, params, sub_data, p_priors=None):
        '''Total likelihood

        Fit individual:
            Maximum likelihood:
            log p(D|θ) = log \prod_i p(D_i|θ)
                       = \sum_i log p(D_i|θ )
            or Maximum a posterior 
            log p(θ|D) = \sum_i log p(D_i|θ ) + log p(θ)
        '''
        # negative log likelihood
        tot_loglike_loss  = -np.sum([self.loglike(params, sub_data[key])
                    for key in sub_data.keys()])
        # negative log prior 
        tot_logprior_loss = 0 if p_priors==None else \
            -self.logprior(params, p_priors)
        # sum
        return tot_loglike_loss + tot_logprior_loss

    def loglike(self, params, block_data):
        '''Likelihood for one sample
        -log p(D_i|θ )
        In RL, each sample is a block of experiment,
        Because it is independent across experiment.
        '''
        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)
        ll   = 0
       
        ## loop to simulate the responses in the block 
        for _, row in block_data.iterrows():

            # predict stage: obtain input
            ll += env.eval_fn(row, subj)

        return ll
          
    def logprior(self, params, p_priors):
        '''Add the prior of the parameters
        '''
        lpr = 0
        for pri, param in zip(p_priors, params):
            lpr += np.max([pri.logpdf(param), -max_])
        return lpr

    # ------------ evaluate ------------ #

    def eval(self, data, params):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            sim_data.append(self.eval_block(block_data, params))
        return pd.concat(sim_data, ignore_index=True)
    
    def eval_block(self, block_data, params):

        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = ['ll'] + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # record some insights of the model
            # for v in self.agent.voi:
            #     pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # simulate the data 
            ll = env.eval_fn(row, subj)
            
            # record the stimulated data
            pred_data.loc[t, 'll'] = ll

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)

    # ------------ simulate ------------ #

    def sim(self, data, params, rng):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            for v in self.env_fn.voi:
                if v in block_data.columns:
                    block_data = block_data.drop(columns=v)
            sim_data.append(self.sim_block(block_data, params, rng))
        
        return pd.concat(sim_data, ignore_index=True)

    def sim_block(self, block_data, params, rng):

        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = self.env_fn.voi + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # simulate the data 
            subj_voi = env.sim_fn(row, subj, rng)

            # record some insights of the model
            for i, v in enumerate(self.agent.voi):
                pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # if register hook to get the model insights
            if self.use_hook:
                for k in self.insights.keys():
                    self.insights[k].append(eval(f'subj.get_{k}()'))

            # record the stimulated data
            for i, v in enumerate(env.voi): 
                pred_data.loc[t, v] = subj_voi[i]

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)
    
    def register_hooks(self, *args):
        self.use_hook = True 
        self.insights = {k: [] for k in args}

# ------------------------------#
#         Memory buffer         #
# ------------------------------#

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = {k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

# ------------------------------#
#          Base model           #
# ------------------------------#

class baseAgent:
    '''Base Agent'''
    name     = 'base'
    n_params = 0
    p_bnds   = None
    p_pbnds  = []
    p_name   = []  
    n_params = 0 
    p_priors = None 
    # value of interest, used for output
    # the interesting variable in simulation
    voi      = []
    
    def __init__(self, nA, params):
        self.nA = nA 
        self.load_params(params)
        self._init_believes()
        self._init_buffer()

    def load_params(self, params): 
        return NotImplementedError

    def _init_buffer(self):
        self.mem = simpleBuffer()
    
    def _init_believes(self):
        self._init_critic()
        self._init_actor()
        self._init_dists()

    def _init_critic(self): pass 

    def _init_actor(self): pass 

    def _init_dists(self):  pass

    def learn(self): 
        return NotImplementedError

    def policy(self, m, **kwargs):
        '''Control problem
            create a policy that map to
            a distribution of action 
        '''
        return NotImplementedError

class RL(baseAgent):
    name     = 'RL'
    p_bnds   = None
    p_pbnds  = [(-10, -.15), (-2, 2)]
    p_name   = ['alpha', 'beta']
    p_priors = []
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), lambda x: clip_exp(x)]
    n_params = len(p_name)
    voi      = []
    color    = viz.r2 
   
    def load_params(self, params):
        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]
        # assign the parameter
        self.alpha = params[0]
        self.beta  = params[1]

        
    def _init_critic(self):
        self.p1     = 1/2
        self.p_S   = np.array([1-self.p1, self.p1]) 

    def learn(self):
        self._learn_critic()

    def _learn_critic(self):
        s, t, f = self.mem.sample('s', 't_type', 'f_type')
        delta = s-self.p1
        o = 'pos' if delta>0 else 'neg'
        self.o = o 
        self.p1 += self.alpha * delta 
        self.p_S = np.array([1-self.p1, self.p1])

    def policy(self, m, **kwargs):
        self.pi = softmax(self.beta*self.p_S*m)
        return self.pi

    def get_pS1(self):
        return self.p_S[1]
    
    def get_pi1(self):
        return self.pi[1]
    
