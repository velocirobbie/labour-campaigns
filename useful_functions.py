#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(color_codes=True)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

def read_in_election_results():
    ge15 = pd.read_csv("general_election-uk-2015-results.csv")
    ge15 = ge15.rename({"Constituency ID": "ons_id"}, axis=1)
    ge15= ge15[ge15['Region'] != 'Northern Ireland']
    ge15.index = ge15['ons_id']

    ge10 = pd.read_csv("general_election-uk-2010-results.csv")
    ge10 = ge10.rename({"Constituency ID": "ons_id"}, axis=1)
    ge10= ge10[ge10['Region'] != 'Northern Ireland']
    ge10.index = ge15['ons_id']

    ge15 = ge15.drop(columns='ons_id')

    ge17 = pd.read_csv('2017 UKPGE electoral data 4.csv',encoding = "ISO-8859-1")
    ge17 = ge17.rename(columns={'ONS Code':'ons_id'})
    total_votes = ge17[['ons_id','Valid votes']].groupby('ons_id').sum()

    p = {'ukip':'UKIP',
         'ld':'Liberal Democrats',
         'lab':'Labour',
         'con':'Conservative',
         'snp':'SNP',
         'grn':'Green Party'}

    ge17 = ge17.merge(total_votes.rename(columns={'Valid votes':'total_votes'}), on='ons_id')
    for party in p:
        mask = ge17['Party Identifer'] == p[party]
        ge17_party = ge17[mask].rename(columns={'Valid votes':party})
        ge17 = ge17.merge(ge17_party[['ons_id',party]],on='ons_id',how='left').fillna(0)
        ge17[party+'_pc'] = ge17[party] / ge17['total_votes']

    ge17 = ge17[ ge17['Party Identifer']=='Labour' ]
    ge17.index = ge17['ons_id']
    ge17 = ge17.drop(columns='ons_id')
    
    return ge10, ge15, ge17

# some useful functions
def onsid_from_name(name,ge17):
    return ge17.index[ge17['Constituency']==name][0]

def name_from_onsid(onsid,ge17):
    return ge17['Constituency'].loc[onsid]


# ### Estimate marginal seats
# If labour were won by a certain percentage, or were within a certain percentage of winning the seat (margin = 0.15). Rough estimate of seats of interest
# 
# This is so we can identify constituencies that actually had a campaign run in them.

# In[5]:

def calc_marginal_within(margin,ge17):
    opposition = ['con','ld','ukip','grn','snp']
    parties = ['lab'] + opposition

    ge17['winner'] = ge17[parties].T.apply(lambda x: x.nlargest(1).idxmin())
    ge17['second'] = ge17[parties].T.apply(lambda x: x.nlargest(2).idxmin())
    ge17['third']  = ge17[parties].T.apply(lambda x: x.nlargest(3).idxmin())
    ge17['marginal'] = False

    for row in ge17.T:
        win_party = ge17.loc[row,'winner']
        win_vote  = ge17.loc[row,win_party+'_pc']
        sec_party = ge17.loc[row,'second']
        sec_vote  = ge17.loc[row,sec_party+'_pc']
        thi_party = ge17.loc[row,'third']
        thi_vote  = ge17.loc[row,thi_party+'_pc']
        if win_vote - sec_vote < margin:
            if 'lab' in [win_party,sec_party]:
                ge17.loc[row,'marginal'] = True
        if win_vote - thi_vote < margin:
            if thi_party == 'lab':
                ge17.loc[row,'marginal'] = True

    return ge17['marginal']

def read_in_census():
    census = pd.read_csv("census_file.csv")
    census.index = census['ons_id']
    census = census.drop(columns='ons_id')
    return census

def get_clusters(names,labels):
    n_clusters = max(labels)
    clusters = {}
    for cluster in range(n_clusters):
        mask = labels == cluster
        similar = list(names[mask])
        clusters[cluster] = similar
    return clusters

def cluster_constituencies_kmeans(n_clusters,constit_demog, features):
    # i don't think this is neccesary
    from sklearn import preprocessing
    X = constit_demog[features]
    X_scaled = preprocessing.scale(X)

    from sklearn.cluster import KMeans

    # Ive declared the random_state just to make sure we can compare things
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    kmeans_clusters = get_clusters(constit_demog.index,
                                   kmeans.labels_)
    return kmeans_clusters, kmeans.labels_

def score_campaigns_difference(election, prev_election):
    lab_change = pd.DataFrame([prev_election['lab'],election['lab']],
                               index=['prev_election','election_result']).T
    lab_change['change'] = lab_change['election_result'] - lab_change['prev_election']
    return lab_change

def gather_data(clusters, constituency_scores,ge17):
    data = pd.DataFrame()

    for cluster in clusters:
        constituencies = clusters[cluster]
        swings = []
        for constit in constituencies:
            swings += [constituency_scores['change'].loc[constit]]
        sorted_swings = np.sort(swings)

        for constit,swing in zip(constituencies,swings):
            if len(swings) == 1:
                sigma_from_mean = 0
                pos_in_cluster = 0.5
            else:
                sigma_from_mean = (swing-np.mean(swings))/np.std(swings)
                pos_in_cluster = np.where(sorted_swings==swing)[0][0] / (len(swings) -1)
            line = pd.DataFrame({
                'ons_id':constit,
                'constituency':name_from_onsid(constit,ge17),
                'cluster':cluster,
                'cluster_size':len(swings),
                'mean':np.mean(swings),
                'std':np.std(swings),
                'change':swing,
                'sigma_from_mean':sigma_from_mean,
                'dist_from_mean':swing - np.mean(swings),
                'pos_in_cluster': pos_in_cluster
            },index=[0])
            data = data.append(line,ignore_index=True,sort=False)
    return data

