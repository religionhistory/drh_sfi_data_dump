'''
Really basic preprocessing
'''

# libraries 
import pandas as pd 
import numpy as np

# load data 
metadata=pd.read_csv('data/metadata.csv')
monitoring_questions=pd.read_csv('data/monitoring_questions.csv')

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
monit_wide=monit_wide.drop(columns='NGA')
monit_wide.to_csv('data/monitoring_weighted_preprocessing.csv', index=False)
