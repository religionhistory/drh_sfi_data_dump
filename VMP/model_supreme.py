import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pandas as pd 
#https://www.pymc.io/projects/examples/en/latest/howto/api_quickstart.html
#https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/diagnostics_and_criticism/posterior_predictive.html
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
df_sub=df[df['Question ID'] == 4836]

# fit the model
with pm.Model() as model:
    beta0 = pm.Normal('intercept', 0, 5)
    beta1 = pm.Normal('mu', 0, 5)
    mu = beta0 + beta1 * df_sub['log_sq_km'].values
    pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
    idata = pm.sample(2000)

# check the trace
az.plot_trace(idata);

# check the summary data
az.summary(idata)
### interpretation here ###
# intercept: something at 1 log_sq_km
# mu: increase in probability of increse in 1 log_sq_km
with model:
    idata.extend(pm.sample_posterior_predictive(idata))

# need a better way to plot this 
fig, ax = plt.subplots()
az.plot_ppc(idata, ax=ax)
ax.axvline(df_sub['Answers'].mean(), ls="--", color="r", label="True mean")
ax.legend(fontsize=10);

# also plotting posterior draws against data would be good
# here we have a clear significant positive effect

####### condition on time ########
with pm.Model() as model_time:
    beta0 = pm.Normal('intercept', 0, 5)
    beta1 = pm.Normal('space', 0, 5)
    beta2 = pm.Normal('time', 0, 5) # does not like this at all
    mu = beta0 + beta1 * df_sub['log_sq_km'].values + beta2 * df_sub['start_year_norm'].values
    pm.Bernoulli('obs', p = pm.invlogit(mu), observed = df_sub['Answers'].values)
    idata = pm.sample(2000)

# seems like space remains significant
# after conditioning on time (which is also a bit positive)
az.plot_trace(idata);

## intercept here:
# * time=0 (-4000 BCE)
# * space=0 (1 square kilometer)
# invlogit(-1.46)=0.18 (18%)
## time here
# * time from 0 to 1 (4000 BCE to 2000+ CE)
# invlogit(intercept+0.7)=0.30 (30%)
# so going from the oldest to the most recent
# moves us from 18% to 30% (12%)
## space here (think more about this)
# space from 0 to 1 basically going 
# making the space 2.718 times larger
# so e.g. going from 1.000-1.000.000 sqkm (or 1-1.000)
# means moving 6.9 points on this axis
# that is basically 0.69 so also 
# from 18% to 30%. 
az.summary(idata) # space significant, time tending--but not significant



def invlogit(x):
    return 1 / (1 + np.exp(-x))
invlogit(-1.5+0.7) # 