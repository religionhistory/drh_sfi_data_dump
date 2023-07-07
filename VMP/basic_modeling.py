'''
Main take-aways: 
1. correlations
Murder cluster (coreligionists, other religions, other polities)
Ritual cluster (which is VERY WEAKLY related to other practices)
Basically everything else is just medium correlated.
Not a lot of structure or clear communities.

2. PCA
best explanation for the data is just 1 component which is basically
just "having everything / not having anything". Reflects the fact that
all of these are positively correlated (not a single negative correlation).
This would definitely be the first thing to say. 

if we force a second dimension it wants: 
+ shirk risk
- murder other polities
- murder other religions
+ personal hygiene
- murder coreligionists
+ taboos
so the second most informative dimension is a contrast between
caring about risk, taboos, hygiene and caring about murder. 

3. Feature clustering
Clusters it slightly differently,
but generally also just prefers as few clusters as possible
which is in concert with PCA.

4. Observation clustering
Cannot really split the data "well"--
it produces these splits where n=388 religions are in one cluster
and then n=2 religions are in the other cluster. 
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

# load preprocessed data
monit_consistent=pd.read_csv('data/monitoring_basic_preprocessing.csv')
monit_consistent=pd.read_csv('../SIMON_PROCESSED/processed_monitoring.csv')
monit_consistent=monit_consistent.sort_values(['Entry ID'], ascending=True)

# pivot
monit_wide=monit_consistent.pivot(index='Entry ID', 
                                 columns='Question ID', 
                                 values='Answers').reset_index()

# take out the matrix
columns_to_drop=['Entry ID', 'Entry name', 'Date', 'World Region', 'NGA']
monit_matrix=monit_wide.drop(columns=['Entry ID']).to_numpy()

### imputation ###
# iterative imputer: estimate features from all others (iterated round-robin)
# https://scikit-learn.org/stable/modules/impute.html#iterative-imputer
# single imputation currently--but can easily be extended to multiple
# if we are interested in how much the uncertainty affects the results
# I have just tried a couple of random seeds, and seems pretty consistent
# different orders can be considered as well (e.g. "random")
# and one could try e.g. a random forest classifier. 
# for now this will be good enough to get an idea. 
lr = LogisticRegression()
imp = IterativeImputer(estimator=lr,
                       missing_values=np.nan, 
                       max_iter=100, 
                       verbose=2, 
                       imputation_order='roman',
                       random_state=0)
X=imp.fit_transform(monit_matrix)

#### correlations in the imputed data ####
questions=monit_consistent['Question'].unique()
df = pd.DataFrame(X, columns=questions)

# Compute the correlation matrix
correlation_matrix = df.corr()
correlation_matrix

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
