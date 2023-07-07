import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

monitoring_long=pd.read_csv('data/complete_long.csv')

df_aggregated=monitoring_long.groupby(['Entry ID', 'Date', 'World Region'])['Answer values'].sum().reset_index(name='Answer values')
df_aggregated['Answer values']=df_aggregated['Answer values'].astype(int)
df_aggregated.sort_values('Answer values', ascending=False)

# lineplot in time
# but also takes space into account
'''
Southwest Asia
Europe
Africa
East Asia
South America
South Asia
Southeast Asia
Central Eurasia
Oceania-Australia
North America
'''

world_region='Southwest Asia' 
world_region='Europe' 
world_region='Africa'
df_sub=df_aggregated[df_aggregated['World Region'] == world_region]

fig, ax = plt.subplots()
sns.lineplot(x=df_sub['Date'],
             y=df_sub['Answer values'],
             data=df_sub)
plt.show();

# what if we only care about time 
df_aggregated=df_aggregated[['Entry ID', 'Date', 'Answer values']].drop_duplicates()

fig, ax = plt.subplots()
sns.lineplot(x=df_aggregated['Date'],
             y=df_aggregated['Answer values'],
             data=df_aggregated)
plt.show();

# points in time (first occurence)
df_mintime=monitoring_long.groupby('Entry ID')['Date'].min().reset_index(name='Date')
monitoring_no_region=monitoring_long.drop(columns=['World Region']).drop_duplicates()
df_mintime=df_mintime.merge(monitoring_no_region, on=['Entry ID', 'Date'], how='inner')
df_aggmin=df_mintime.groupby(['Entry ID', 'Date'])['Answer values'].sum().reset_index(name='Answer values')
df_aggmin['Answer values']=df_aggmin['Answer values'].astype(int)

sns.stripplot(x=df_aggmin['Date'],
                y=df_aggmin['Answer values'],
                data=df_aggmin,
                jitter=0.2)
plt.xticks(rotation=90)
plt.show();