'''
so far this implements really basic lasso regression
does not take into account missing data (removes)
and is not bayesian 

questions: 
(1) should we standardize (e.g. z-score?). suggested: https://www.kirenz.com/post/2019-08-12-python-lasso-regression-auto/
'''

# crazy lasso setup. 
# https://machinelearningmastery.com/lasso-regression-with-python/
# https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/pymc3_howto/lasso_block_update.html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

RANDOM_SEED=413
rng = np.random.default_rng(RANDOM_SEED)
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

# read data
df=pd.read_csv('processed/supreme_area_start.csv')
df=df[df['Parent answer']==1.0]

# now we need wide format;
df_sub=df[['Question ID', 'Entry ID', 'Answers']]
df_wide=df_sub.pivot(index='Entry ID', 
                     columns='Question ID', 
                     values='Answers') # 524 total
df_wide=df_wide.dropna() # 308
df_wide.columns.name = None
df_wide=df_wide.reset_index()

# now add time and space
df['start_year_norm']=(df['start_year'] - df['start_year'].min()) / (df['start_year'].max() - df['start_year'].min())
df['log_sq_km']=np.log(df['area_sq_km'])
df_time_space=df[['Entry ID', 'start_year_norm', 'log_sq_km']].drop_duplicates()
df_time_space.groupby('Entry ID').size().reset_index(name='count').sort_values('count')
df_wide=df_wide.merge(df_time_space, on='Entry ID', how='inner')

# ignore problems with duplicates for now and just run the analysis
# such that we have the pipeline for when we get the actual data ...
model = Lasso(alpha=1.0) # check up on values
A=df_wide.drop(columns='Entry ID').to_numpy()
X, y = A[:, :-2], A[:, -1]

## automatic hyper-param ##
## can we get uncertainty? ##
cv = RepeatedKFold(n_splits=10,
                   n_repeats=3,
                   random_state=1231)
model = LassoCV(alphas=np.arange(0, 1, 0.01),
                cv=cv,
                n_jobs=-1)
model.fit(X, y)
model.alpha_ # 0.06
coef = model.coef_ 
coef_df = pd.DataFrame({'coef': coef})

## join with questions ##
coef_df['Question ID'] = df_wide.drop(columns=['Entry ID', 'start_year_norm', 'log_sq_km']).columns
df_questions=df[['Question ID', 'Question']].drop_duplicates()
coef_df = coef_df.merge(df_questions, on='Question ID', how='inner')

## plot ##
coef_df=coef_df.sort_values('coef', ascending=False)
coef_df=coef_df.reset_index()
coef_df['index']=coef_df.index
def plot_coef(df, 
              outname=''):
    fig, ax = plt.subplots()
    ax.scatter(df['coef'], df['Question'], color='tab:blue')
    ax.set_yticks(df['index'])
    ax.set_yticklabels(df['Question'])
    plt.xlabel('Coef on log_sq_km')
    plt.ylabel('Question')
    plt.title('Lasso')
    if outname: 
        plt.savefig(f'fig/{outname}.png', 
                    dpi=300,
                    bbox_inches='tight')
    else: 
        plt.show();
    plt.close()

plot_coef(coef_df,
          outname='lasso_coef_conditional')

'''interpreation: 
two most positive adjusts estimate of 
area size up by factor of more than e (2.7)

two most negative adjusts estimate of 
area size down by almost factor of e (2.7)
'''

## now do the same including time ##
X, y = A[:, :-1], A[:, -1]
cv = RepeatedKFold(n_splits=10,
                   n_repeats=3,
                   random_state=1231)
model = LassoCV(alphas=np.arange(0, 1, 0.01),
                cv=cv,
                n_jobs=-1)
model.fit(X, y)
best_alpha=model.alpha_ 
coef = model.coef_ 
coef_df = pd.DataFrame({'coef': coef})

## join with questions ##
coef_df['Question ID'] = df_wide.drop(columns=['Entry ID', 'log_sq_km']).columns
df_questions=df[['Question ID', 'Question']].drop_duplicates()
df_questions = df_questions._append({'Question ID': 'start_year_norm', 'Question': 'Start Year Normalized'}, ignore_index=True)
coef_df = coef_df.merge(df_questions, on='Question ID', how='inner')

## plot ##
coef_df=coef_df.sort_values('coef', ascending=False)
coef_df=coef_df.reset_index()
coef_df['index']=coef_df.index
plot_coef(coef_df,
          outname='lasso_coef_time_conditional')

#### test how good the model is with train/test split ####
# different--but consistent--results
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=413)
reg=Lasso(alpha=best_alpha)
reg.fit(X_train, y_train)
## R squared 
# captures nothing (0.14, -0.02)
print('R squared training', round(reg.score(X_train, y_train), 2)) # 11.07
print('R squared test set', round(reg.score(X_test, y_test), 2)) # 9.17
## MSE
# really high (16.18--so 4 log sq km off)
from sklearn.metrics import mean_squared_error
pred_train=reg.predict(X_train)
mse_train=mean_squared_error(y_train, pred_train)
print('MSE training set', round(mse_train, 2)) # 4: still huge
## basic plot
# one problem is all of the "no to all" answers
# we should consider whether it makes sense to 
# include these;
# there could be a lot of noise there 
# but excluding will remove a lot of data...
# NB: come back and save this plot
plt.plot(y_train, pred_train, 'o')
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.savefig('fig/y_true_y_pred_control_time.png',
            dpi=300)
## plot and save coefficients 
coef=reg.coef_
coef_df=pd.DataFrame({'coef': coef})
coef_df['Question ID']=df_wide.drop(columns=['Entry ID', 'log_sq_km']).columns
df_questions=df[['Question ID', 'Question']].drop_duplicates()
df_questions=df_questions._append({'Question ID': 'start_year_norm', 'Question': 'Start Year Normalized'}, ignore_index=True)
coef_df=coef_df.merge(df_questions, on='Question ID', how='inner')
coef_df=coef_df.sort_values('coef', ascending=False)
coef_df=coef_df.reset_index()
coef_df['index']=coef_df.index
plot_coef(coef_df,
          outname='lasso_coef_time_train_test_conditional') 