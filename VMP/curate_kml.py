from kml_size import extract_polygons, calculate_area
import pandas as pd 
import numpy as np

# first load the new data
df=pd.read_csv('../supreme_high_gods.csv')
df.columns

# take out the columns we need
df=df[['Standardized Question ID',
        'Standardized Question',
        'Standardized Parent question',
        'Poll',
        'Answer values',
        'Entry name',
        'Entry ID',
        'Region name',
        'Parent answer value',
        'start_year']]

# rename columns
df=df.rename(columns={
    'Standardized Question ID': 'Question ID',
    'Standardized Question': 'Question',
    'Standardized Parent question': 'Parent question',
    'Answer values': 'Answers',
    'Parent answer value': 'Parent answer'})

# replace -1 with nan
df['Answers']=df['Answers'].replace(-1, np.nan)

# only group poll
df=df[(df['Poll'].isin(['Religious Group (v5)', 'Religious Group (v6)']))]

########### PARENT NO ###########
# first let us create a dataframe with all of the entries 
# that have "NO" for the parent
df.groupby('Parent answer')['Entry ID'].nunique() # no=158, yes=373

# some of these have both "YES" and "NO"
# in these cases we will only take "YES"
parents=df[['Parent answer', 'Entry ID']].drop_duplicates()
duplicate_parents=parents[parents.duplicated(subset=['Entry ID'], keep=False)]
duplicate_parents=duplicate_parents.sort_values('Entry ID')
duplicate_parent_entries=duplicate_parents['Entry ID'].unique()

# create dataframe of entries with "NO" as parent
# these are all automatically coded as "NO"
df_parent_no=df[df['Parent answer'] == 0.0]
df_parent_no=df_parent_no[~df_parent_no['Entry ID'].isin(duplicate_parent_entries)]

########## PARENT YES ###########
df_parent_yes=df[df['Parent answer'] == 1.0]
len(df_parent_yes)

# investigate all with multiple answers
inconsistent_coding=df_parent_yes.groupby(['Entry ID', 'Question ID']).size().reset_index(name='counts').sort_values('counts', ascending=False)
inconsistent_coding=inconsistent_coding[inconsistent_coding['counts'] > 1]
df_parent_yes

# remove these combinations for now
for index, row in inconsistent_coding.iterrows():
    df_parent_yes=df_parent_yes[~((df_parent_yes['Entry ID'] == row['Entry ID']) & (df_parent_yes['Question ID'] == row['Question ID']))]

# I think we do not actually want to remove NAN now
# we are just going to be modeling one feature at a time 
# so we will want to use all the available data possible
df_total=pd.concat([df_parent_no, df_parent_yes])


###### KML ######

# To use the function, simply call it with the path to the KML file you want to parse:
import os
from bs4 import BeautifulSoup
kml_path='data/kml_files'
kml_files=os.listdir(kml_path)
# there is one weird DS_STORE file in there
kml_files=[i for i in kml_files if i.endswith('.kml')]
area_dict={}
for i in kml_files: 
    file=os.path.join(kml_path, i)
    # extract the region name of the file
    kml=open(file, 'r', errors='replace').read()
    soup=BeautifulSoup(kml, 'lxml-xml')
    name=soup.find('name').text

    # exxtract area of polygons
    polygons = extract_polygons(file)
    square_km_sum=0
    for polygon in polygons:
        area = calculate_area(polygon)
        square_km_sum+=area
    
    area_dict[name]=square_km_sum
    
area_df=pd.DataFrame.from_dict(area_dict, orient='index', columns=['area_sq_km'])
area_df['Region name']=area_df.index

####### MERGE #######
df_area=df_total.merge(area_df, on='Region name', how='inner')
len(df_area)
len(df_total) # we are only losing 17 observations--not bad

# save
df_area.to_csv('processed/supreme_area_start.csv', index=False)