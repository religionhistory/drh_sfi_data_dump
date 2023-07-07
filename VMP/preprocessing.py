'''
Really basic preprocessing
'''

# libraries 
import pandas as pd 
import numpy as np

# load data 
metadata=pd.read_csv('data/metadata.csv')
monitoring_questions=pd.read_csv('data/monitoring_questions.csv')
metadata=metadata.drop(columns=['NGA'])
metadata=metadata.drop_duplicates()

### preprocessing ###
# remove weird question
problem_question='Supernatural beings care about other:'
monitoring_questions=monitoring_questions[monitoring_questions['Standardized Question'] != problem_question]

# replace -1 with nan
monitoring_questions['Answer values']=monitoring_questions['Answer values'].replace(-1, np.nan)

# only group poll
monitoring_group=monitoring_questions[(monitoring_questions['Poll'].isin(['Religious Group (v5)', 'Religious Group (v6)']))]

# NB: some of these also have "No" as parent
monit_parent_true=monitoring_group[monitoring_group['Parent answer'] == 'Yes']

# select questions
monit_focusq=monit_parent_true[['Standardized Question ID', 
                                'Standardized Question', 
                                'Entry ID', 
                                'Entry name',
                                'Answer values']]
monit_focusq=monit_focusq.rename(columns={'Standardized Question ID': 'Question ID',
                                          'Standardized Question': 'Question',
                                          'Answer values': 'Answers'})

# drop inconsistency (NB: not optimal)
# check whether we are dropping something stupid. 
monit_consistent=monit_focusq.drop_duplicates(subset=["Entry ID",
                                                      "Question ID"], keep=False)

# save 
monit_consistent.to_csv('data/monitoring_basic_preprocessing.csv', index=False)

### preprocessing with metadata ###
### NB: removes n=12 entries ###
d=pd.merge(monit_consistent, metadata, on='Entry ID', how='inner')
d=d.sort_values(['Entry ID', 'Question ID'], ascending=True) 
question_ids = d[['Question ID']].drop_duplicates().sort_values('Question ID')['Question ID'].tolist()

# Convert Question ID into column names prefixed by 'Q'
monit_wide=monit_consistent.pivot(index='Entry ID', 
                                  columns='Question ID', 
                                  values='Answers').reset_index()

# add metadata 
monit_wide=pd.merge(monit_wide, metadata, on='Entry ID', how='inner')
monit_wide['weight']=1.0
monit_wide['weight']=monit_wide.groupby(['Date', 'World Region'])['weight'].transform(lambda x: x / x.sum())
monit_wide.to_csv('data/monitoring_weighted_preprocessing.csv', index=False)

### new dataset (n of features) ###
# exclude the ones with nan
complete=20
monitoring_complete=monitoring_group[monitoring_group['Answer values'].notna()]
complete_records=monitoring_complete.groupby('Entry ID').size().reset_index(name='count')
complete_records=complete_records[complete_records['count'] == complete]
monitoring_complete=monitoring_complete.merge(complete_records, on='Entry ID', how='inner')

# find entries that have more than one answer to a question
duplicates=monitoring_complete.groupby(['Entry ID', 'Standardized Question ID']).size().reset_index(name='count')
duplicates=duplicates[duplicates['count'] > 1]
duplicates=duplicates['Entry ID'].unique()
monitoring_complete=monitoring_complete[~monitoring_complete['Entry ID'].isin(duplicates)]

# make something non-wide
monitoring_long=monitoring_complete.merge(metadata, on='Entry ID', how='inner')
monitoring_long.to_csv('data/complete_long.csv', index=False)

# make something wide
complete_wide=monitoring_complete.pivot(index='Entry ID',
                                        columns='Standardized Question ID',
                                        values='Answer values').reset_index()

# now merge with metadata
complete_wide=pd.merge(complete_wide, metadata, on='Entry ID', how='inner')
complete_wide.to_csv('data/complete_meta.csv', index=False)