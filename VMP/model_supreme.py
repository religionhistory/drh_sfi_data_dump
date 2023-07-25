'''
VMP 2023-07-21:
Really tricky; 
Clearly, the "space" model is better than the "time" model--
but it seems like there is a strong interaction.
Question is how to interpret and report this, and if this
is a better way of understanding the data than the 
single-predictor models...
(also... this is just 1 outcome). 
'''

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pandas as pd 
import xarray as xr
#https://www.pymc.io/projects/examples/en/latest/howto/api_quickstart.html
#https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/diagnostics_and_criticism/posterior_predictive.html
#https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html
RANDOM_SEED=413
rng = np.random.default_rng(RANDOM_SEED)
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

# read data
df=pd.read_csv('processed/supreme_area_start.csv')

# select columns
df['log_sq_km']=np.log(df['area_sq_km'])
df['bin_log_sq_km']=pd.cut(df['log_sq_km'],bins=4,labels=False)
df=df.dropna(subset=['Answers'])
df['start_year_norm'] = (df['start_year'] - df['start_year'].min()) / (df['start_year'].max() - df['start_year'].min())
df['Answers']=df['Answers'].astype(int)

# select a question
# unquestionably good
# knowledge of this world
# communicates with the living
df[['Question ID','Question']].drop_duplicates()
df_sub=df[df['Question ID'] == 4836] # unquestionably good

# sort values for ease
df_sub=df_sub.sort_values(by=['log_sq_km'])

### 1. pure "area" model ###
## priors ## 
with pm.Model() as model_space:
    beta0 = pm.Normal('intercept', 0, 5)
    beta1 = pm.Normal('mu', 0, 5)
    mu = beta0 + beta1 * df_sub['log_sq_km'].values
    pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
    trace_space = pm.sample_prior_predictive(samples=50, random_seed=rng)

_, ax = plt.subplots()

x = xr.DataArray(np.linspace(df_sub['log_sq_km'].min(), 
                             df_sub['log_sq_km'].max(), 50), 
                 dims=["plot_dim"])
prior = trace_space.prior
y = prior["intercept"] + prior["mu"] * x
ax.plot(x, y.stack(sample=("chain", "draw")), c="k", alpha=0.4)

ax.set_xlabel("log sq km")
ax.set_ylabel("mu")
ax.set_title("Prior predictive checks -- Flat priors");

## posterior / posterior predictive ##
with model_space: 
    trace_space.extend(pm.sample(2000, 
                                 tune=2000, 
                                 target_accept=0.9,
                                 random_seed=rng,
                                 idata_kwargs={'log_likelihood': True}))
    pm.sample_posterior_predictive(trace_space, extend_inferencedata=True, random_seed=rng)

# check the trace & summary
az.plot_trace(trace_space);
az.summary(trace_space)

# check the posterior
post = trace_space.posterior
mu_pp = post["intercept"] + post["mu"] * xr.DataArray(df_sub['log_sq_km'], dims=["obs_id"])

_, ax = plt.subplots()

ax.plot(
    df_sub['log_sq_km'], mu_pp.mean(("chain", "draw")), label="Mean outcome", color="C1", alpha=0.6
)

az.plot_hdi(df_sub['log_sq_km'], 
            mu_pp)

ax.set_xlabel(r"log $km^2$ ($1km^2 = 0$)")
ax.set_ylabel("logit / log-odds");

''' interpretation
at x=0 (1 km2): ~25% chance of being unquestionably good
at x=15 (300.000 km2): ~65% chance of being unquestionably good
the effect is significant (i.e., the 95% HDI does not include 0)
most certainty around 400-20.000 km2.
'''

# plot the transformed (i.e. in percent)
# why is this not as clean as I would like it to be?
_, ax = plt.subplots()

ax.plot(
    df_sub['log_sq_km'], 
    inv_logit(mu_pp.mean(('chain', 'draw'))), 
    label="Mean outcome", 
    color="C1", 
    alpha=0.6
)

ax.scatter(
    x=df_sub['log_sq_km'],
    y=df_sub['Answers'],
    marker="x",
    color="#A69C75",
    alpha=0.8,
    label="Observed outcomes",
)

az.plot_hdi(df_sub['log_sq_km'], 
            inv_logit(mu_pp))

ax.set_xlabel(r"log $km^2$ ($1km^2 = 0$)")
ax.set_ylabel("fraction unquestionably good");

### 2. pure TIME model ###
df_sub=df_sub.sort_values('start_year_norm')
with pm.Model() as model_time: 
    beta0 = pm.Normal('intercept', 0, 5)
    beta1 = pm.Normal('mu', 0, 5)
    mu = beta0 + beta1 * df_sub['start_year_norm'].values
    pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
    trace_time = pm.sample(2000,
                           tune=2000,
                           target_accept=0.9,
                           random_seed=rng,
                           idata_kwargs={'log_likelihood': True})
    pm.sample_posterior_predictive(trace_time, 
                                   extend_inferencedata=True, 
                                   random_seed=rng)

az.plot_trace(trace_time) # sampling looks fine 
az.summary(trace_time) # not significant

post=trace_time.posterior
mu_pp=post['intercept']+post['mu']*xr.DataArray(df_sub['start_year_norm'], dims=['obs_id'])
_, ax = plt.subplots()

ax.plot(
    df_sub['start_year_norm'], inv_logit(mu_pp.mean(('chain', 'draw'))), label="Mean outcome", color="C1", alpha=0.6
)

ax.scatter(
    x=df_sub['start_year_norm'],
    y=df_sub['Answers'],
    marker="x",
    color="#A69C75",
    alpha=0.8,
    label="Observed outcomes",
)

az.plot_hdi(df_sub['start_year_norm'], 
            inv_logit(mu_pp))

ax.set_xlabel(r"normalized start year ($0 \approx -4000 BCE, 1 \approx 2000 CE$)")
ax.set_ylabel("fraction unquestionably good");

''' interpretation
not significant effect (does cross 0)
although, clearly trending. 
crazy amount of uncertainty in low end (lack of observations)
'''

### 3. combined model ###
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

# seems like space remains significant
# after conditioning on time (which is also a bit positive)
az.plot_trace(trace_mixed);
az.summary(trace_mixed) 

''' interpretation
seems like space and time are not strongly correlated
at least our estimates remain almost entirely unchanged
from the "invididual" models. 

still: 
space is significant, and time is trending. 
'''

### 4. interaction model ###
# hmmm---very significant interaction
# so maybe not just space--but space + something that time is a proxy for...
with pm.Model() as model_interaction:
    beta0=pm.Normal('intercept', 0, 5)
    beta1=pm.Normal('space', 0, 5)
    beta2=pm.Normal('time', 0, 5)
    beta3=pm.Normal('interaction', 0, 5)
    mu=beta0+beta1*df_sub['log_sq_km'].values+beta2*df_sub['start_year_norm'].values+beta3*df_sub['log_sq_km'].values*df_sub['start_year_norm'].values
    pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
    trace_interaction = pm.sample(2000, 
                                  tune=2000, 
                                  target_accept=0.9,
                                  random_seed=rng,
                                  idata_kwargs={'log_likelihood': True})
    pm.sample_posterior_predictive(trace_interaction, extend_inferencedata=True, random_seed=rng)
    
az.plot_trace(trace_interaction);
az.summary(trace_interaction)

# interpretation really hard here ... 

### 5. model comparison ###
df_comp_loo = az.compare(
    {"time": trace_time, 
     "space": trace_space,
     "mixed": trace_mixed,
     "interaction": trace_interaction})
df_comp_loo # really likes the interaction model