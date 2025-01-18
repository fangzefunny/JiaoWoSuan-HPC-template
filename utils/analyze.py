import os 
import numpy as np 
import pandas as pd 
from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm

from tabulate import tabulate
import pingouin as pg 

import seaborn as sns 
import matplotlib.pyplot as plt 

from utils.model import *
from utils.viz import viz 


# path to the current file 
path = os.path.dirname(os.path.abspath(__file__))

def model_fit(models, method='mle'):
    feedbacks = ['gain', 'loss']
    crs = {}
    for m in models:
        for feedback in feedbacks:
            fname = f'{path}/../simulations/exp1data/{m}/sim-{feedback}_exp1data-{method}-idx0.csv'
            data  = pd.read_csv(fname)
            n_param = eval(m).n_params
            subj_Lst = data['sub_id'].unique()

            nlls, aics = [], [] 
            for sub_id in subj_Lst:
                sel_data = data.query(f'sub_id=="{sub_id}" & feedback_type=="{feedback}"')
                inll = sel_data['logLike'].sum() 
                nlls.append(inll)
                if np.isnan(inll): print(inll) 
                aics.append(2*inll + 2*n_param)

            crs[m] = {'nll': nlls, 'aic': aics}
    return crs

def model_cmp(quant_crs):
    crs = ['nll', 'aic']
    pairs = [['gagModel', 'risk'],
             ['gagModel', 'mix_pol_3w'],
             ['risk',  'mix_pol_3w']]
    
    for cr in crs:
        print(f'''
            ------------- {cr} ------------- ''')
        for p in pairs:
            x = quant_crs[p[0]][cr]
            y = quant_crs[p[1]][cr]
            res = ttest_ind(x, y)
            print(f'''
            {p[0]}-{p[1]}: 
                {p[0]}:{np.mean(x):.3f}, {p[1]}:{np.mean(y):.3f}
                t={res[0]:.3f} p={res[1]:.3f}''')

def t_test(x_data, y_data, paired=False, title=''):
    df = pg.ttest(x_data, y_data, paired=paired)
    dof = df.loc[:, 'dof'].values[0]
    t   = df.loc[:, 'T'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    cohen_d = df.loc[:, 'cohen-d'].values[0]
    pair_str = '-paired' if paired==True else ''
    print(f'{title} \tt{pair_str}({dof:.3f})={t:.3f}, p={pval:.3f}, cohen-d={cohen_d:.3f}')

def corr(x_data, y_data, title=''):
    df = pg.corr(x_data, y_data)
    n = df.loc[:, 'n'].values[0]
    r   = df.loc[:, 'r'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    print(f'{title} \tr({n})={r:.3f}, p={pval:.3f}')

def anova(dv, between, data, all_table=False):
    df = pg.anova(dv=dv, between=between, data=data).rename(columns={'p-unc': 'punc'})
    if all_table:
       print(df.round(3).to_string())
    else:
        sig_df = df.query('punc<=.05')
        dof2 = int(df.query('Source=="Residual"')['DF'].values[0])
        other_min_p = df.query('punc>.05')['punc'].min()
        for _, row in sig_df.iterrows():
            title = row['Source']
            dof   = int(row['DF'])
            F     = row['F']
            p     = row['punc']
            np2   = row['np2']
            print(f'\t{title}:\tF({dof}, {dof2})={F:.3f}, p={p:.3f}, np2={np2:.3f}')
        print(f'\tOther: \tp>={other_min_p:.3f}')

def linear_regression(x, y, add_intercept=False, title='', x_var='x', y_var='y'):
    df = pg.linear_regression(X=x, y=y, add_intercept=add_intercept)
    beta0 = df['coef'][0]
    beta1 = df['coef'][1]
    pval  = df['pval'][1]
    print(f'{title}\t{y_var}={beta1:.3f}{x_var}+{beta0:.3f},\n\tp={pval:.3f}')

def get_advantage(agent):
    if agent=='human':
        fname = '../data/exp1_data.csv'
    else:
        fname = f'../simulations/exp1data/{agent}/sim-map.csv'
    data = pd.read_csv(fname)
    data['group'] = data['group'].map(
        {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'}
    )
    data['m0'] = data['m0']*100
    data['m1'] = data['m1']*100
    # get correct action 
    data['pS0'] = data['psi_truth'].apply(lambda x: 1-x)
    data['pS1'] = data['psi_truth'].apply(lambda x: x)
    # get correct action 
    data['cor_a'] = data.apply(
        lambda x: x['state'] if x['feedback_type']=='gain' else int(1-x['state']) 
    , axis=1)
    # get action based on EU strategy
    data['a_eu'] = data.apply(
        lambda x: np.argmax([x['pS0']*x['m0'], x['pS1']*x['m1']])    
    , axis=1)
    data['r0'] = data.apply(
        lambda x: (x['state']==0)*x['m0']
    , axis=1) 
    data['r1'] = data.apply(
        lambda x: (x['state']==1)*x['m1']
    , axis=1) 
    data['b'] = data.apply(
        lambda x: (x['r0']+x['r1']) / 2
    , axis=1) 
    # get action based on MO strategy
    data['a_mo'] = data.apply(
        lambda x: np.argmax([x['m0'], x['m1']])
    , axis=1)
    data['a_ha'] = data.shift(1)['a']
    data['adv0']  = data.apply(
        lambda x: x['r0'] - x['b'] 
    , axis=1)
    data['adv1']  = data.apply(
        lambda x: x['r1'] - x['b'] 
    , axis=1)
    data['adv']  = data.apply(
        lambda x: x[f"adv{int(x['a'])}"]
    , axis=1)
    data['hit']  = data.apply(
        lambda x: x['a']==x['cor_a']
    , axis=1)
    data['hit_eu'] = data.apply(
        lambda x:  x['a_eu']==x['cor_a']    
    , axis=1)
    data['r_eu'] = data.apply(
        lambda x: x[f'adv{x["a_eu"]}']
    , axis=1)
    data['hit_mo'] = data.apply(
        lambda x:  x['a_mo']==x['cor_a']    
    , axis=1)
    data['r_mo'] = data.apply(
        lambda x: x[f'adv{x["a_mo"]}']
    , axis=1)
    data['adv-mo'] = data.apply(
        lambda x: x['adv'] - x['r_mo']
    , axis=1)
    data['hit_ha'] = data.apply(
        lambda x:  x['a_ha']==x['cor_a'] if x['trial']>0 else False  
    , axis=1)
    data['r_ha'] = data.apply(
        lambda x: x[f'adv{int(x["a_ha"])}'] if x['trial']>0 else 0
    , axis=1)
    return data

def main_effect(pivot_table, pred, cond1, cond2,
            tar=['l1', 'l2', 'l3', 'l4'], 
            notes=['exp utility', 'reward probability', 'magnitude', 'habit']):
    nr, nc = 1, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*3.7, nr*4), sharey=True, sharex=True)
    for_title = t_test(pivot_table, cond1, cond2, tar=tar)
    for idx in range(nc):
        ax  = axs[idx]
        sns.boxplot(x=pred, y=f'{tar[idx]}', data=pivot_table,
                        palette=viz.Palette, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f'{notes[idx]} {for_title[idx]}')
        # if idx == 1: ax.legend(bbox_to_anchor=(1.4, 0), loc='lower right')
        # else: ax.get_legend().remove()
    plt.tight_layout()
    plt.show()

def f_twoway(data, fac1, fac2, tar=['l1', 'l2', 'l3']):
    ## the significant test 
    for_title = []
    for i in tar:
        model = ols(f'{i} ~ C({fac1}) + C({fac2}) + C({fac1}):C({fac2})', data).fit()
        res = anova_lm(model).loc[f'C({fac1}):C({fac2})', ['F', 'PR(>F)']]
        if res[1] < .01:
            for_title.append('**')
        elif res[1] < .05:
            for_title.append('*')
        else:
            for_title.append('')
        print(f'{i} f-two way: f={res[0]:.4f}, p-val:{res[1]:.4f}')
    return for_title

def intersect_effect(pivot_table, fac1, fac2,
            tar=['l1', 'l2', 'l3', 'l4'], 
            notes=['exp utility', 'reward probability', 'magnitude', 'habit']):
    nr, nc = 1, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*3.7, nr*4), sharey=True)
    for_title = f_twoway(pivot_table, fac1, fac2, tar)
    for idx in range(nc):
        ax  = axs[idx]
        sns.boxplot(x=fac1, y=f'{tar[idx]}', data=pivot_table,
                        hue=fac2, palette=viz.Palette, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f'{notes[idx]} {for_title[idx]}')
        if idx == nc-1: ax.legend(bbox_to_anchor=(1.6, .5), loc='right')
        else: ax.get_legend().remove()
    plt.tight_layout()
    plt.show()

def get_advantage(agent):
    if agent=='human':
        fname = '../data/exp1_data.csv'
    else:
        fname = f'../simulations/exp1data/{agent}/sim-map.csv'
    data = pd.read_csv(fname)
    data['group'] = data['group'].map(
        {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'}
    )
    data['m0'] = data['m0']*100
    data['m1'] = data['m1']*100
    # get correct action 
    data['pS0'] = data['psi_truth'].apply(lambda x: 1-x)
    data['pS1'] = data['psi_truth'].apply(lambda x: x)
    # get correct action 
    data['cor_a'] = data.apply(
        lambda x: x['state'] if x['feedback_type']=='gain' else int(1-x['state']) 
    , axis=1)
    # get action based on EU strategy
    data['a_eu'] = data.apply(
        lambda x: np.argmax([x['pS0']*x['m0'], x['pS1']*x['m1']])    
    , axis=1)
    data['r0'] = data.apply(
        lambda x: (x['state']==0)*x['m0']
    , axis=1) 
    data['r1'] = data.apply(
        lambda x: (x['state']==1)*x['m1']
    , axis=1) 
    data['r'] = data['rawRew']*100
    data['b'] = data.apply(
        lambda x: (x['r0']+x['r1']) / 2
    , axis=1) 
    # get action based on MO strategy
    data['a_mo'] = data.apply(
        lambda x: np.argmax([x['m0'], x['m1']])
    , axis=1)
    data['a_ha'] = data.shift(1)['a']
    data['adv0']  = data.apply(
        lambda x: x['r0'] - x['b'] 
    , axis=1)
    data['adv1']  = data.apply(
        lambda x: x['r1'] - x['b'] 
    , axis=1)
    data['adv']  = data.apply(
        lambda x: x[f"adv{int(x['a'])}"]
    , axis=1)
    data['hit']  = data.apply(
        lambda x: x['a']==x['cor_a']
    , axis=1)
    data['hit_eu'] = data.apply(
        lambda x:  x['a_eu']==x['cor_a']    
    , axis=1)
    data['r_eu'] = data.apply(
        lambda x: x[f'adv{x["a_eu"]}']
    , axis=1)
    data['hit_mo'] = data.apply(
        lambda x:  x['a_mo']==x['cor_a']    
    , axis=1)
    data['r_mo'] = data.apply(
        lambda x: x[f'adv{x["a_mo"]}']
    , axis=1)
    data['adv-mo'] = data.apply(
        lambda x: x['adv'] - x['r_mo']
    , axis=1)
    data['hit_ha'] = data.apply(
        lambda x:  x['a_ha']==x['cor_a'] if x['trial']>0 else False  
    , axis=1)
    data['r_ha'] = data.apply(
        lambda x: x[f'adv{int(x["a_ha"])}'] if x['trial']>0 else 0
    , axis=1)
    return data


