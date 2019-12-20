#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns

sns.set(color_codes=True)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def read_in_election_results():
    ge10 = readge10("data/general_election-uk-2010-results.csv")
    ge15 = readge15("data/general_election-uk-2015-results.csv")
    ge17 = readge17("data/2017 UKPGE electoral data 4.csv")
    ge19 = readge19("data/ge2019.csv")
    return ge10, ge15, ge17, ge19


def readge15(datafile):
    ge15 = pd.read_csv(datafile)
    ge15 = ge15.rename({"Constituency ID": "ons_id"}, axis=1)
    ge15 = ge15[ge15["Region"] != "Northern Ireland"]
    ge15.index = ge15["ons_id"]
    ge15 = ge15.drop(columns="ons_id")
    return ge15


def readge10(datafile):
    ge10 = pd.read_csv(datafile)
    ge10 = ge10.rename({"Constituency ID": "ons_id"}, axis=1)
    ge10 = ge10[ge10["Region"] != "Northern Ireland"]
    return ge10


def readge17(datafile):
    ge17 = pd.read_csv(datafile, encoding="ISO-8859-1")
    ge17 = ge17.rename(columns={"ONS Code": "ons_id"})
    ge17["Election Year"] = 2017
    total_votes = ge17[["ons_id", "Valid votes"]].groupby("ons_id").sum()

    p = {
        "ukip": "UKIP",
        "ld": "Liberal Democrats",
        "lab": "Labour",
        "con": "Conservative",
        "snp": "SNP",
        "grn": "Green Party",
    }

    ge17 = ge17.merge(
        total_votes.rename(columns={"Valid votes": "total_votes"}), on="ons_id"
    )
    for party in p:
        mask = ge17["Party Identifer"] == p[party]
        ge17_party = ge17[mask].rename(columns={"Valid votes": party})
        ge17 = ge17.merge(
            ge17_party[["ons_id", party]], on="ons_id", how="left"
        ).fillna(0)
        ge17[party + "_pc"] = ge17[party] / ge17["total_votes"]

    ge17 = ge17[ge17["Party Identifer"] == "Labour"]
    ge17.index = ge17["ons_id"]
    ge17 = ge17.drop(columns="ons_id")
    ge17["winner"] = ge17[p.keys()].T.apply(lambda x: x.nlargest(1).idxmin())
    return ge17


def readge19(datafile):
    ge19 = pd.read_csv(datafile, encoding="ISO-8859-1")

    ge19 = ge19.rename({"ONSConstID": "ons_id"}, axis=1)
    ge19 = ge19.dropna(subset=["ons_id"])
    ge19.index = ge19["ons_id"]

    parties = {
        "con": "CON",
        "lab": "LAB",
        "ld": "LIBDEM",
        "grn": "GRN",
        "snp": "SNP",
        "pc": "PC",
        "ukip": "UKIP",
        "bxp": "BXP",
        "other": "OTHER",
    }

    for party in parties:
        ge19 = ge19.rename({parties[party]: party}, axis=1)
        ge19[party] = ge19[party].astype(str)
        ge19[party] = ge19[party].str.replace(",", "")
        ge19[party] = ge19[party].astype(float)

    ge19["total"] = ge19.fillna(0)[parties].sum(1)

    for party in parties:
        ge19[party + "_pc"] = ge19[party] / ge19["total"]

    ge19["Election Year"] = 2019
    ge19["winner"] = ge19[parties.keys()].T.apply(lambda x: x.nlargest(1).idxmin())
    return ge19


def onsid_from_name(name, ge17):
    return ge17.index[ge17["Constituency"] == name][0]


def name_from_onsid(onsid, ge17):
    return ge17["Constituency"].loc[onsid]


def calc_marginal_within(margin, ge17):
    opposition = ["con", "ld", "ukip", "grn", "snp"]
    parties = ["lab"] + opposition

    ge17["winner"] = ge17[parties].T.apply(lambda x: x.nlargest(1).idxmin())
    ge17["second"] = ge17[parties].T.apply(lambda x: x.nlargest(2).idxmin())
    ge17["third"] = ge17[parties].T.apply(lambda x: x.nlargest(3).idxmin())
    ge17["marginal"] = False

    for row in ge17.T:
        win_party = ge17.loc[row, "winner"]
        win_vote = ge17.loc[row, win_party + "_pc"]
        sec_party = ge17.loc[row, "second"]
        sec_vote = ge17.loc[row, sec_party + "_pc"]
        thi_party = ge17.loc[row, "third"]
        thi_vote = ge17.loc[row, thi_party + "_pc"]
        if win_vote - sec_vote < margin:
            if "lab" in [win_party, sec_party]:
                ge17.loc[row, "marginal"] = True
        if win_vote - thi_vote < margin:
            if thi_party == "lab":
                ge17.loc[row, "marginal"] = True

    return ge17["marginal"]


def read_in_census():
    census = pd.read_csv("census_file.csv")
    census.index = census["ons_id"]
    census = census.drop(columns="ons_id")
    return census


def get_clusters(names, labels):
    n_clusters = max(labels)
    clusters = {}
    for cluster in range(n_clusters):
        mask = labels == cluster
        similar = list(names[mask])
        clusters[cluster] = similar
    return clusters


def cluster_constituencies_kmeans(n_clusters, constit_demog, features):
    X = constit_demog[features]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    kmeans_clusters = get_clusters(constit_demog.index, kmeans.labels_)
    return kmeans_clusters, kmeans.labels_


def score_campaigns_difference(election, prev_election):
    lab_change = pd.DataFrame(
        [prev_election["lab"], election["lab"]],
        index=["prev_election", "election_result"],
    ).T
    lab_change["difference"] = (
        lab_change["election_result"] - lab_change["prev_election"]
    )
    return lab_change


def score_campaigns_mrp(election):
    year = election["Election Year"][0]
    assert year in [2017, 2019]
    if year == 2017:
        mrp = pd.read_csv("data/yougov_mrp_2017.csv")
    elif year == 2019:
        mrp = pd.read_csv("data/yougov_mrp_2019.csv")

    mrp = mrp.rename({"code": "ons_id"}, axis=1)
    mrp = mrp.set_index("ons_id")
    mrp["Lab"] = mrp["Lab"].div(100)
    lab_score = pd.DataFrame(
        [election["lab_pc"], mrp["Lab"]],
        index=["election_result", "yougov_mrp"]
    ).T
    lab_score["difference"] = lab_score["election_result"] - lab_score["yougov_mrp"]
    return lab_score


def score_campaigns_uns(election, prev_election):
    # Source ge10: https://electionresults.parliament.uk/election/2010-05-06/Results/Location/Country/Great%20Britain
    # Source ge15: https://electionresults.parliament.uk/election/2015-05-07/Results/Location/Country/Great%20Britain
    # Source ge17: https://electionresults.parliament.uk/election/2017-06-08/results/Location/Country/Great%20Britain

    if election["Election Year"][0] == 2010:
        swing = -0.064
    elif election["Election Year"][0] == 2015:
        swing = 0.015
    elif election["Election Year"][0] == 2017:
        swing = 0.098
    elif election["Election Year"][0] == 2019:
        swing = -0.079
    else:
        raise Exception("No data for that year")

    lab_score = pd.DataFrame(
        [prev_election["lab_pc"], election["lab_pc"]],
        index=["prev_election", "election_result"],
    ).T
    lab_score["uns"] = lab_score["prev_election"].map(lambda result: result + swing)
    lab_score["difference"] = lab_score["election_result"] - lab_score["uns"]
    return lab_score.sort_index()


def gather_data(clusters, constituency_scores, ge17):
    data = pd.DataFrame()

    for cluster in clusters:
        constituencies = clusters[cluster]
        swings = []
        for constit in constituencies:
            swings += [constituency_scores["change"].loc[constit]]
        sorted_swings = np.sort(swings)

        for constit, swing in zip(constituencies, swings):
            if len(swings) == 1:
                sigma_from_mean = 0
                pos_in_cluster = 0.5
            else:
                sigma_from_mean = (swing - np.mean(swings)) / np.std(swings)
                pos_in_cluster = np.where(sorted_swings == swing)[0][0] / (
                    len(swings) - 1
                )
            line = pd.DataFrame(
                {
                    "ons_id": constit,
                    "constituency": name_from_onsid(constit, ge17),
                    "cluster": cluster,
                    "cluster_size": len(swings),
                    "mean": np.mean(swings),
                    "std": np.std(swings),
                    "change": swing,
                    "sigma_from_mean": sigma_from_mean,
                    "dist_from_mean": swing - np.mean(swings),
                    "pos_in_cluster": pos_in_cluster,
                },
                index=[0],
            )
            data = data.append(line, ignore_index=True, sort=False)
    return data
