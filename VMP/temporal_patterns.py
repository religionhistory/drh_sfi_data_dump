import numpy as np
import pandas as pd
import seaborn as sns

monitoring_long=pd.read_csv('data/complete_long.csv')
monitoring_long=monitoring_long[['Entry ID', 'Date', 'World Region', 'Standardized Question', 'Answer values']].

# aggregate pre
df_aggregated=monitoring_long.groupby(['Entry ID', 'Date', 'World Region'])['Answer values'].sum().reset_index(name='Answer values')
df_aggregated['Answer values']=df_aggregated['Answer values'].astype(int)
df_aggregated.sort_values('Answer values', ascending=False)


test=monitoring_long[monitoring_long['Entry ID'] == 688]
test

# lineplot in time
sns.lineplot(x=df_aggregated['Date'],
             y=df_aggregated['Answer values'],
             data=df_aggregated)

# lineplot in space + time
sns.lineplot(x=df_aggregated['Date'],
             y=df_aggregated['Answer values'],
             hue=df_aggregated['World Region'],
             data=df_aggregated)

agg_time=monitoring_long.groupby(['Date', 'World Region'])['Answer values'].mean().reset_index(name='Answer values')