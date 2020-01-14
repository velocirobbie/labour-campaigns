#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(color_codes=True)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


from useful_functions import *

opposition = ["con", "ld", "ukip", "grn", "snp"]
parties = ["lab"] + opposition

year = 19
compare_year = 17

election_results = read_in_election_results()

census = read_in_census()

#marginals = calc_marginal_within(0.15, ge17)

# Combine latest election with census data
ge17_census = election_results[year].merge(
    census.drop(columns=["Region", "Constituency"]),
    left_index=True,
    right_index=True,
    validate="one_to_one",
)

features = [
    "c11PopulationDensity",
    "c11HouseOwned",
    "c11CarsNone",
    "c11EthnicityWhite",
    "c11Unemployed",
    "c11Retired",
    "c11FulltimeStudent",
    "c11Age65to74",
    "c11DeprivedNone",
]

constit_demog = ge17_census[["Constituency"] + features]
constit_demog = constit_demog.dropna()

constit_improvement = score_campaigns_uns(election_results[year],
                                          election_results[compare_year])

n_constits = len(constit_demog)

constit_scores = pd.DataFrame(
    {
        "Constituency": constit_demog["Constituency"],
        "score_std": np.zeros(n_constits),
        "score_dist": np.zeros(n_constits),
        "score_posincluster": np.zeros(n_constits),
        "score_overall": np.zeros(n_constits),
    },
    index=constit_demog.index,
)


def cluster_and_score_constits(n_clusters, metric, scores, constit_demog, features):
    clusters, cluster_labels = cluster_constituencies_kmeans(
        n_clusters, constit_demog, features
    )
    cluster_summary = gather_data(clusters, metric, election_results[year])

    n_constits = len(cluster_summary)
    cluster_summary = cluster_summary.sort_values(by="sigma_from_mean", ascending=False)

    for place, i in enumerate(cluster_summary.index):
        ons_id = cluster_summary.loc[i, "ons_id"]
        scores.loc[ons_id, "score_std"] += cluster_summary.loc[i, "sigma_from_mean"]
        scores.loc[ons_id, "score_dist"] += cluster_summary.loc[i, "dist_from_mean"]
        scores.loc[ons_id, "score_posincluster"] += cluster_summary.loc[
            i, "pos_in_cluster"
        ]
        scores.loc[ons_id, "score_overall"] += (n_constits - place) / n_constits

    return scores


def ensemble_cluster_and_score_constits(
    epochs, n_clusters, metric, scores, constit_demog, features
):
    for i in range(epochs):
        print(i)
        scores = cluster_and_score_constits(
            n_clusters, metric, scores, constit_demog, features
        )

    cols = ["score_std", "score_dist", "score_overall", "score_posincluster"]
    for col in cols:
        scores[col] /= epochs
    return scores.sort_values(
        by=["score_posincluster", "score_overall"], ascending=False
    )


def build_similarity_matrix(epochs, n_clusters, constit_demog, features):
    n_constits = len(constit_demog)
    # similarity_matrix = {constit:np.zeros(n_constits) for constit in constit_demog['Constituency']}
    # similarity_matrix = pd.DataFrame(similarity_matrix,index=constit_demog['Constituency'])
    similarity_matrix = np.zeros((n_constits, n_constits))
    lookup = {constit: i for i, constit in enumerate(constit_demog.index)}

    for epoch in range(epochs):
        clusters, cluster_labels = cluster_constituencies_kmeans(
            n_clusters, constit_demog, features
        )
        for cluster in clusters.values():
            n = len(cluster)
            for i in range(n - 1):
                # constit1 = constit_demog.loc[cluster[i],'Constituency']
                constit1 = lookup[cluster[i]]
                for j in range(i + 1, n):
                    # constit2 = constit_demog.loc[cluster[j],'Constituency']
                    constit2 = lookup[cluster[j]]
                    similarity_matrix[constit1, constit2] += 1
                    similarity_matrix[constit2, constit1] += 1
    return similarity_matrix


def print_where(where_matrix):
    where_matrix = np.array(where_matrix)
    for i, j in where_matrix.T:
        print(
            constit_demog.iloc[i]["Constituency"],
            ",",
            constit_demog.iloc[j]["Constituency"],
        )


### cluster and score many times, show table of average scores
epochs = 10
print(ensemble_cluster_and_score_constits(epochs, 40, constit_improvement, constit_scores, constit_demog, features))


### Cluster many times, pick pairs of constits that are most often similar
# epochs = 100
# n_clusters = 250
# similarity_matrix = build_similarity_matrix(epochs, n_clusters, constit_demog, features)

# print(similarity_matrix)
# best = np.max(similarity_matrix)
# where = np.array(np.where(similarity_matrix==best))

# for i,j in where.T:
#    if i < j:
#        print(constit_demog.iloc[i]['Constituency'],constit_demog.iloc[j]['Constituency'])
# print(np.where(similarity_matrix == max(similarity_matrix)))

from sklearn import preprocessing

C = constit_demog.index
X = constit_demog[features]
X_scaled = preprocessing.scale(X)

from sklearn.metrics import pairwise_distances

d_matrix = pairwise_distances(X_scaled)
print(d_matrix)
print_where(np.where(d_matrix == np.max(d_matrix)))

constit_improvement = constit_improvement.loc[C]
scale_change = preprocessing.scale(constit_improvement["difference"])

change_matrix = scale_change[:, np.newaxis] - scale_change
print(change_matrix)

significance = np.divide(change_matrix, d_matrix, where=d_matrix != 0)
scores = np.sum(significance, 1)

best = np.argsort(scores)
for i in range(len(C)):
    print(scores[best[-i - 1]], name_from_onsid(C[best[-i - 1]], election_results[year]))


"""
# make a useful database `data`, that calculates the how good the campaign was relevant to the others in the cluster. Basically trying to remove the naitonal campaign so we can figure out which local campaigns did well.

ge17_cluster = ge17.merge(cluster_summary.rename(columns={'constituency':'Constituency'}),on="Constituency")

cols = ['Constituency','marginal','sigma_from_mean','dist_from_mean','change',
        'cluster','cluster_size','winner','con','lab','ld','snp']
ge17_cluster[cols].sort_values(by='dist_from_mean',ascending=False).head(5)

best_labour_campaigns = ge17_cluster[cols].sort_values(by='dist_from_mean',ascending=False).head(10)['Constituency'].values
best_labour_campaigns



def find_worst_similar_constit(best,cluster_summary):
    cluster = int(cluster_summary['cluster'][cluster_summary['constituency']==best])
    clusterdata = cluster_summary[cluster_summary['cluster']==cluster]
    smallest_swing = clusterdata['change'].idxmin()
    return cluster_summary['constituency'].loc[smallest_swing]

for i in range(len(best_labour_campaigns)):
    print(best_labour_campaigns[i],',',find_worst_similar_constit(best_labour_campaigns[i],cluster_summary))


# In[ ]:


# From the above data, the best campaigns are in quite large clusters, so hard to pick out all the information. Milton Keynes South performed well, and is in a small cluster so I've chosen that as an example.
# 
# ## Is this useful???
# 
# Declare the cluster you are interested in and the following cells will make some useful stats on it...
# 
# All the constituencies in the group improved their votes for labour, but by varying amounts.

cluster = 10


# In[9]:


cluster_summary[cluster_summary['cluster']==cluster].sort_values(by='dist_from_mean',ascending=False)


# In[10]:


ge17_cluster[ge17_cluster['cluster']==cluster].sort_values(by='dist_from_mean',ascending=False)


# In[11]:


# In[27]:


sns.distplot(cluster_summary[cluster_summary['cluster']==cluster]['change'])


# In[ ]:


import geopandas as gpd

map_df = gpd.read_file("map.shp")
map_df.index = census.index

def highlight_constits(map_df,constits):
    ids = []
    for constit in constits:
        ids += [ onsid_from_name(constit,election_results[year]) ]
    highlight_map = map_df
    highlight_map['color'] = 0.9
    L = len(ids)
    for i,id_ in enumerate(ids):
        highlight_map['color'][id_] = i*0.5/L 
    return highlight_map


# In[16]:


constits = cluster_summary[cluster_summary['cluster']==cluster].sort_values(by='dist_from_mean',ascending=False)['constituency']
constits


# In[17]:


hmap = highlight_constits(map_df,constits)
f, ax = plt.subplots(1, figsize=(10, 10))
ax = hmap.plot(column=hmap['color'],ax=ax,linewidth=0.1)
ax.set_axis_off()


# In[18]:


fig, ax = plt.subplots(len(constits),2,figsize=(10,5*len(constits)))

for i,constit in enumerate(constits):
    onsid = onsid_from_name(constit,election_results[year])
    #plt.figure(i)
    
    election_results[compare_year][parties].loc[onsid].plot.bar(color=['r','b','y','m','g','y'],title=constit+'2015',ax=ax[i,0])
    
    election_results[year][parties].loc[onsid].plot.bar(color=['r','b','y','m','g','y'],title=constit+'2017',ax=ax[i,1])
    #sns.distplot(cluster_summary[cluster_summary['cluster']==i]['change'])


# In[19]:


constit_demog[constit_demog['Constituency'].isin(constits)]


# In[20]:


# Actual labour forcast/expected votes
# 
# How much money per constituency campaign 
# 
# Incumbency ???
# 
# House prices 
# 
# Local press 
# 
# How does marginal effect result?
# Dont ignore anomolies that aren't marginal 
# 
# Noise of model, ensemble results 


Constits are connected by demographic similarity and different election performace
Constit score = sum of improvememnt on votes / distance 

"""
