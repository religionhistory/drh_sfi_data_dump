'''
Main take-aways: 
1. Correlations
* Rituals again.
* Sex, taboos, elders
* murder
* etc.

More structure perhaps?

2. PCA


3. Feature clustering

4. Observation clustering
'''

# libraries 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.experimental import enable_iterative_imputer
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#### NB: we should include the data that we do not have weights for ####
#### we can just give it "mean" weight ####
# load preprocessed data
monit_wide=pd.read_csv('data/monitoring_weighted_preprocessing.csv')

# get the questions
questions=monit_wide.columns
questions=[int(x) for x in questions if x not in ['Entry ID', 'World Region', 'Date', 'weight']]

# get the "weight" column
sum_entries=monit_wide.groupby('Entry ID')['weight'].sum().reset_index(name='weight')
monit_wide=monit_wide.drop(columns=['weight', 'Date', 'World Region'])
monit_wide=monit_wide.drop_duplicates()
monit_wide=monit_wide.merge(sum_entries, on='Entry ID', how='inner')

# load the unweighted data
entries_with_weight= monit_wide['Entry ID'].unique()
monit_consistent=pd.read_csv('data/monitoring_basic_preprocessing.csv')
entries_without_weight=monit_consistent[~monit_consistent['Entry ID'].isin(entries_with_weight)]
entries_without_weight_wide=entries_without_weight.pivot(index='Entry ID',
                                                         columns='Question ID',
                                                         values='Answers').reset_index()
entries_without_weight_wide.columns = entries_without_weight_wide.columns.astype(str)

# set weight of these to the mean of the other weight
# take matrix and weights out 
entries_without_weight_wide['weight']=monit_wide['weight'].mean()
total_entries=pd.concat([monit_wide, entries_without_weight_wide])
total_entries=total_entries.sort_values('Entry ID')

# take out matrix and weights
monit_matrix=total_entries.drop(columns=['Entry ID', 'weight']).to_numpy()
weights=total_entries['weight'].to_numpy()

### imputation ###
lr = LogisticRegression()
imp = IterativeImputer(estimator=lr,
                       missing_values=np.nan, 
                       max_iter=100, 
                       verbose=2, 
                       imputation_order='roman',
                       random_state=0)
X=imp.fit_transform(monit_matrix)

#### correlations in the imputed data ####
# we just need the nice labels now ....
def weighted_corr(x, w):
    """Compute the weighted correlation matrix of a 2D array"""
    
    # Mean
    mean = np.sum(x * w[:, None], axis=0) / np.sum(w)
    
    # Covariance
    cov = np.zeros((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            cov[i, j] = np.sum(w * (x[:, i] - mean[i]) * (x[:, j] - mean[j])) / np.sum(w)
    
    # Correlation
    stddev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stddev, stddev)
    
    return corr

# Compute the weighted correlation matrix
questions=list(monit_consistent['Question'].unique())
#df = pd.DataFrame(X, columns=questions)
weighted_correlation_matrix = weighted_corr(X, weights)
df_corr = pd.DataFrame(weighted_correlation_matrix, columns=questions)
df_corr.index = questions

# Create a clustermap
sns.clustermap(df_corr, 
               annot=False, 
               cmap="RdBu_r", 
               center=0,
               cbar=False,
               cbar_pos=None)

# Show the plot
plt.show()



# Compute the correlation matrix
correlation_matrix = df.corr()

# Create a clustermap
sns.clustermap(correlation_matrix, 
               annot=False, 
               cmap="RdBu_r", 
               center=0,
               cbar=False,
               cbar_pos=None)

# Show the plot
plt.show()

### Fit PCA models with varying number of components ###
explained_variances = []
n_components_range=range(10)
for n_components in n_components_range:
    pca = PCA(n_components=n_components)
    pca.fit(X)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    explained_variances.append(explained_variance)

# Plot the explained variances
plt.figure(figsize=(8, 5))
plt.plot(n_components_range, explained_variances, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Plot')
plt.grid(True)
plt.xticks(n_components_range)
plt.show()

## result 1: strongest evidence for just 1 component
## typically looking for inflection point which here
## is clearly just 1 component
## 45% of variance explained by just 1 component
## if you have just one of these traits--
## you are just much more likely to have the others.
pca=PCA(n_components=1)
pca.fit(X)
pca.components_[0] # all negative (i.e., could have all been positive)

## of course there might be interesting caveats to this
## for instance; when we add the second component
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X)
X_pca=pca.transform(X)
# which features drive this?
pc1_dict={}
pc2_dict={}
for num, ele in enumerate(questions):
    pc1_dict[ele]=pca.components_[0][num]
    pc2_dict[ele]=pca.components_[1][num] 
# sort dictionaries by absoulte value
pc1_dict={k: v for k, v in sorted(pc1_dict.items(), key=lambda item: abs(item[1]), reverse=True)}
pc2_dict={k: v for k, v in sorted(pc2_dict.items(), key=lambda item: abs(item[1]), reverse=True)}
pc1_dict # first one still just everything negative
pc2_dict # second one is more interesting (and third one might be too).

#### feature clustering ####
# Transpose the data matrix so features are rows
X_T = X.T

# Calculate the pairwise distances between features using Jaccard distance
distance_matrix = pdist(X_T, metric='jaccard')

# Perform hierarchical clustering
linked = linkage(distance_matrix, 'single')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=list(questions),
           distance_sort='descending',
           show_leaf_counts=True)
plt.xticks(rotation=90)
plt.show();

#### observation clustering ####
clusterer = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='jaccard', linkage='average')
clusterer.fit(X)

# You can plot the dendrogram and decide on a threshold for the number of clusters
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(clusterer, truncate_mode='level', p=3)

n_clusters = 2 # this is just an example value, choose based on the dendrogram
clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='jaccard', linkage='average')
cluster_labels = clusterer.fit_predict(X)
silhouette_score(X, cluster_labels, metric='jaccard')
# n=2: 0.755
# n=3: 0.751
# n=4: 0.701
# n=5: 0.679

## prefers n=20 clusters (which is the smallest amount possible here) ##
## in this case only 2 observations are in the "weird" cluster ##
## Old Norse Fornsed + Jews in South Arabia ##
## Supernatural monitoring only prosocial (apparently weird) ##
weird_cluster=monit_wide.loc[cluster_labels == 1]
weird_cluster=weird_cluster.merge(monit_consistent, on='Entry ID', how='inner')
weird_cluster['Entry name'].unique()
weird_cluster.groupby('Question')['Answers'].mean()

## what if we go to three clusters? ##
## which is almost as good as n=2 ##
## now we just get two clusters of things that are different ##
n_clusters = 3
clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='jaccard', linkage='average')
cluster_labels = clusterer.fit_predict(X)
silhouette_score(X, cluster_labels, metric='jaccard')
weird_cluster_1=monit_wide.loc[cluster_labels == 0]
weird_cluster_1=weird_cluster_1.merge(monit_consistent, on='Entry ID', how='inner')
weird_cluster_1['Entry name'].unique() # new weirdoes
weird_cluster_2=monit_wide.loc[cluster_labels == 1]
weird_cluster_2=weird_cluster_2.merge(monit_consistent, on='Entry ID', how='inner')
weird_cluster_2['Entry name'].unique() # same as before

## so what is special about this new cluster? ##
## Gods that just care about rituals ## 
weird_cluster_1.groupby('Question')['Answers'].mean()


##### meta-data quest #####





# check whether we have metadata for this
unique_entry_monitor=parent_y['Entry ID'].unique()
unique_entry_meta=metadata['Entry ID'].unique()
len(set(unique_entry_monitor).intersection(unique_entry_meta))
not_in_meta = list(set(unique_entry_monitor) - set(unique_entry_meta))

# anyways, join it and lets see 
merged=parent_y_sub.merge(metadata, on='Entry ID', how='inner')
merged['Answer values']=merged['Answer values'].replace(-1, np.nan)

# aggregate in space and time
df_agg_space=merged.groupby(['Standardized Question', 'World Region'])['Answer values'].mean().reset_index()

### AGGREGATE IN SPACE ###
# should 
def quick_spatial_boxplot(df: pd.DataFrame, 
                          question_list: list=[],
                          hue: list='Question Short'):
    if question_list: 
        df=df[df[hue].isin(question_list)]
        
    sns.catplot(data=df, 
                 x='Answers values',
                 y='World Region', 
                 hue=hue,
                 kind='bar')
    
    plt.show();

questions=df_agg_space['Standardized Question'].unique()
selected_question=questions[0]
df_selected=df_agg_space[df_agg_space['Standardized Question'] == selected_question]
plt.bar(df_selected['World Region'], df_selected['Answer values'])
plt.xticks(rotation=90)
plt.show();

sns.catplot(df=df_agg_space,
            x='Answer values',
            y='World Region',
            hue='World Region',
            kind='bar')

quick_spatial_boxplot(df_agg_space,
                      question_list)
    
def quick_temporal_lineplot(df: pd.DataFrame, 
                            question_list: list=[],
                            hue: list='Question Short',
                            identifier: str='',
                            outpath: str='../fig/spatiotemporal'):
    if question_list: 
        df=df[df[hue].isin(question_list)]
        
    sns.lineplot(data=df, 
                 x='Date',
                 y='Answers', 
                 hue=hue)
    
    plt.savefig(os.path.join(outpath, f"temporal_{identifier}.png"),
                bbox_inches='tight')
    plt.close()

sns.pointplot(x='Date', y='recode', data=merged_binary, hue='Standardized Question ID')
