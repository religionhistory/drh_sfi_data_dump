import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# recreate something like the figure we had
df=pd.read_csv('processed/supreme_area_start.csv')

# drop NAN for now
df=df.dropna(subset=['Answers'])

# recode sqkm to log
df['log_sq_km']=np.log(df['area_sq_km'])

# first look at AREA vs. TIME
# sort-of surprising that this does not correlate?
# have I done something crazy here?
df_sqkm_year=df[['Entry ID', 'log_sq_km', 'start_year']].drop_duplicates()
sns.regplot(x='start_year', y='log_sq_km', data=df_sqkm_year)

# look at this only between say -2000 and 1500
# then we do have some (weak) correlation
df_sqkm_year=df_sqkm_year[(df_sqkm_year['start_year'] < 1500) &
                          (df_sqkm_year['start_year'] > -2000)]
sns.regplot(x='start_year', y='log_sq_km', data=df_sqkm_year)

#### correlations against time #####
# bin this in time and do rough plot
# we can do this prettier definitely 
# make this nice in the morning
df['binned']=pd.cut(df['start_year'], bins=6)

#### correlations against space #####
## each of these should be a boxplot actually
df['bin_log_sq_km']=pd.cut(df['log_sq_km'], bins=6)
g = sns.FacetGrid(df, col="Question", col_wrap=3, sharey=False, height=4)
g.map(sns.pointplot, 'bin_log_sq_km', 'Answers', order=None, scale=2)
g.set_titles("{col_name}", size=10)
g.set_xticklabels(rotation=45)
plt.show()

## this is totally wrong 

#### we actually need sort-of a good model ####
