import numpy as np 
import pandas as pd 

# load stuff
d = pd.read_csv('../SIMON_PROCESSED/processed_monitoring.csv')
d['weight']=1.0
d=d.sort_values('Entry ID')
metadata = pd.read_csv('../metadata.csv')

# metadata
entry_simon = d['Entry ID'].unique() 
entry_meta = metadata['Entry ID'].unique()

# check whether all entries are in metadata
d_meta=d.merge(metadata, on='Entry ID', how='inner')
d_meta['weight']=d_meta.groupby(['Date', 'World Region'])['weight'].transform(lambda x: x / x.sum())
d_meta.to_csv('processed/with_metadata.csv', index=False)

# aggregate
sum_entries=d_meta.groupby('Entry ID')['weight'].sum().reset_index(name='weight')
monit_wide=d_meta.drop(columns=['weight', 'Date', 'World Region', 'NGA'])
monit_wide=monit_wide.drop_duplicates()
monit_wide=monit_wide.merge(sum_entries, on='Entry ID', how='inner')
monit_wide.to_csv('processed/weight_collapsed.csv', index=False)

# all of the ones that we lost
#lost_entries = np.setdiff1d(entry_simon, entry_meta)
#dsub = d[d['Entry ID'].isin(lost_entries)]
#dsub['weight']=monit_wide['weight'].mean()
#dmain=pd.concat([monit_wide, dsub])
#dmain.to_csv('processed/weight_collapsed.csv', index=False)