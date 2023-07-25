'''
VMP 21-07-2023:
plot the results of the grid search (bayesian models). 
'''

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import re
RANDOM_SEED=413
rng = np.random.default_rng(RANDOM_SEED)
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

# some helper functions
def gather_inference_data(tuple_list: list): 
    inference_dict={}
    for idata, question_id in tuple_list: 
        post = idata.posterior['mu'].values.flatten()
        # calculate mean and hdi
        post_mean = np.mean(post)
        hdi_lower, hdi_upper = az.hdi(post, hdi_prob=0.94)
        # append to list
        inference_dict[question_id]={'post_mean':post_mean, 
                                    'hdi_upper':hdi_upper,
                                    'hdi_lower':hdi_lower}
    return inference_dict

def gather_dataframe(inference_dict: dict,
                     reference_sub: pd.DataFrame): 
    df=pd.DataFrame.from_dict(inference_dict, orient='index')
    df=df.sort_values(by=['post_mean'])
    df['Question ID'] = df.index
    df=df.merge(reference_sub, on='Question ID', how='inner')
    df=df.reset_index()
    df['index']=df.index
    return df    

def plot_posterior(df,
                   outname=''):
    fig, ax = plt.subplots()
    for i, row in df.iterrows(): 
        color='tab:orange' if row['hdi_lower'] <= 0 <= row['hdi_upper'] else 'tab:blue'
        ax.scatter(row['post_mean'], i, color=color)
        ax.hlines(i, row['hdi_lower'], row['hdi_upper'], color=color)
    ax.axvline(0, color='black', linestyle='--')  
    ax.set_yticks(df['index'])
    ax.set_yticklabels(df['Question'])
    plt.xlabel('posterior mean')
    plt.ylabel('question')
    plt.title('Plot of post_mean with 94% HDI')
    if outname: 
        plt.savefig(f'fig/{outname}.png',
                    dpi=300,
                    bbox_inches='tight')
    else: 
        plt.show();
    plt.close()

# overall data
reference_data=pd.read_csv('processed/supreme_area_start.csv')
reference_sub=reference_data[['Question ID','Question']].drop_duplicates()
dir='data/mdl'
filenames=os.listdir(dir)

### 1. space models
# get the files
space_filenames=[f for f in filenames if 'space' in f]
space_files=[(az.from_netcdf(os.path.join(dir, f)), int(re.search(r'\d+', f).group())) for f in space_filenames]
# gather inferencedata in a dictionary
dict_space=gather_inference_data(space_files)
df_space=gather_dataframe(dict_space, reference_sub)
# plot inferencedata
plot_posterior(df_space,
               outname='space')

# plot 
### 2. time models
time_filenames=[f for f in filenames if 'time' in f]
time_files=[(az.from_netcdf(os.path.join(dir, f)), int(re.search(r'\d+', f).group())) for f in time_filenames]
dict_time=gather_inference_data(time_files)
df_time=gather_dataframe(dict_time, reference_sub)
plot_posterior(df_time,
               outname='time')
