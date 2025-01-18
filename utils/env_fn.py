import numpy as np 
import pandas as pd 

eps_ = 1e-13
max_ = 1e+13
    
class rl_reversal:
    nS = 1
    nA = 2
    voi = ['a', 'acc', 'r']

    def __init__(self, block_type='sta_vol'):
        assert block_type in ['cont', 'sta_vol', 'vol_sta']
        self.block_type = block_type

    # ---------- Initialization ---------- #

    def instan(self, seed=1234):
        '''Instantiate the environment
        '''

        rng = np.random.default_rng(seed)
        n_trials = 180
        psi     = np.zeros(n_trials)
        state   = np.zeros(n_trials)
        if self.block_type == 'sta_vol':
            psi[:90]     = .75
            psi[90:110]  = .2
            psi[110:130] = .8
            psi[130:150] = .2
            psi[150:170] = .8
            psi[170:180] = .2
            trial_type   = ['sta']*90 + ['vol']*90
        else:
            psi[:20]     = .2
            psi[20:40]   = .8
            psi[40:60]   = .2
            psi[60:80]   = .8
            psi[80:90]   = .2
            psi[90:]     = .75
            trial_type   = ['vol']*90 + ['sta']*90
        
        # get the state of each trial 
        for i in range(n_trials):
            state[i] = int(rng.random() < psi[i])

        # get the input magnitude
        mag0 = rng.integers(1, 99, size=n_trials)
        mag1 = rng.integers(1, 99, size=n_trials)
        fb_type = ['gain']*180

        block = {
            'state':      state,
            'm0':         mag0,
            'm1':         mag1,
            'trial_type': trial_type, 
            'feedback_type': fb_type, 
            'block_type': [self.block_type]*n_trials,
            'stage'  :    ['train']*n_trials, 
            'trial'  :    list(range(n_trials)),
        }

        block_data = pd.DataFrame.from_dict(block)

        return block_data

    # ---------- Interaction functions ---------- #
    @staticmethod
    def eval_fn(row, subj):
    
        # see state 
        stage  = row['stage']
        s      = int(row['state'])
        m0     = row['m0']
        m1     = row['m1']
        m      = np.array([m0, m1])
        t_type = row['trial_type']
        f_type = row['feedback_type'] 
        pi     = subj.policy(m,
                    t_type=t_type,
                    f_type=f_type)
        a      = int(row['a'])
        ll     = np.log(pi[a]+eps_)

        # save the info and learn 
        if stage == 'train':
            subj.mem.push({
                's': s, 
                'a': a,
                't_type': t_type, 
                'f_type': f_type,
            })
            subj.learn()

        return ll
    
    @staticmethod
    def sim_fn(row, subj, rng):
        
        # see state 
        stage  = row['stage']
        s      = int(row['state'])
        m0     = row['m0']
        m1     = row['m1']
        m      = np.array([m0, m1])
        t_type = row['trial_type']
        f_type = row['feedback_type'] 
        pi     = subj.policy(m,
                    t_type=t_type,
                    f_type=f_type)
        a      = int(rng.choice(rl_reversal.nA, p=pi)) 
        r      = (a==s)*eval(f'm{int(a)}')

        # save the info and learn 
        if stage == 'train':
            subj.mem.push({
                's': s, 
                'a': a,
                't_type': t_type, 
                'f_type': f_type,
            })
            subj.learn()

        return a, pi[s].copy(), r