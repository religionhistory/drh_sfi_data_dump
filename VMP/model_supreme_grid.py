'''
VMP 2023-07-21: 
fit 3 different models;
1. space
2. time
3. space + time
in all cases we predict yes / no answer
run over grid of all questions (see plot_grid.py for results)
'''

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pandas as pd 
import xarray as xr
import math

RANDOM_SEED=413
rng = np.random.default_rng(RANDOM_SEED)

# read data
df=pd.read_csv('processed/supreme_area_start.csv')

# select columns
df['log_sq_km']=np.log(df['area_sq_km'])
df['bin_log_sq_km']=pd.cut(df['log_sq_km'],bins=4,labels=False)
df=df.dropna(subset=['Answers'])
df['start_year_norm'] = (df['start_year'] - df['start_year'].min()) / (df['start_year'].max() - df['start_year'].min())
df['Answers']=df['Answers'].astype(int)
question_ids = df['Question ID'].unique()

# loop over questions for space
for idx in question_ids: 
    df_sub=df[df['Question ID'] == idx]
    df_sub=df_sub.sort_values(by=['log_sq_km'])
    # fit space model
    with pm.Model() as model_space:
        beta0 = pm.Normal('intercept', 0, 5)
        beta1 = pm.Normal('mu', 0, 5)
        mu = beta0 + beta1 * df_sub['log_sq_km'].values
        pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
        trace_space = pm.sample(2000,
                            tune=2000,
                            target_accept=0.9,
                            random_seed=rng)
        pm.sample_posterior_predictive(trace_space, 
                                       extend_inferencedata=True,
                                       random_seed=rng)
    # fit time model
    with pm.Model() as model_time: 
        beta0 = pm.Normal('intercept', 0, 5)
        beta1 = pm.Normal('mu', 0, 5)
        mu = beta0 + beta1 * df_sub['start_year_norm'].values
        pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
        trace_time = pm.sample(2000,
                               tune=2000,
                               target_accept=0.9,
                               random_seed=rng)
        pm.sample_posterior_predictive(trace_space, 
                                       extend_inferencedata=True,
                                       random_seed=rng)

    with pm.Model() as model_mixed:
        beta0 = pm.Normal('intercept', 0, 5)
        beta1 = pm.Normal('space', 0, 5)
        beta2 = pm.Normal('time', 0, 5) # does not like this at all
        mu = beta0 + beta1 * df_sub['log_sq_km'].values + beta2 * df_sub['start_year_norm'].values
        pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
        trace_mixed = pm.sample(2000,
                                tune=2000,
                                target_accept=0.9,
                                random_seed=rng,
                                idata_kwargs={'log_likelihood': True})
        pm.sample_posterior_predictive(trace_mixed, extend_inferencedata=True, random_seed=rng)

    # save idata
    trace_space.to_netcdf(f'data/mdl/idata_space_{idx}.nc')
    trace_time.to_netcdf(f'data/mdl/idata_time_{idx}.nc')
    trace_mixed.to_netcdf(f'data/mdl/idata_mixed_{idx}.nc')
    df_sub.to_csv(f'data/mdl/df_{idx}.csv', index=False)