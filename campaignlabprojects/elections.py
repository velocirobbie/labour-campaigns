#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import os

sns.set(color_codes=True)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def read_in_election_results():
    path = os.path.dirname(__file__)+"/../data/"
    ge10 = readge10(path+"ge2010.csv")
    ge15 = readge15(path+"ge2015.csv")
    ge17 = readge17(path+"ge2017.csv")
    ge19 = readge19(path+"ge2019.csv")
    return {10: ge10, 15: ge15, 17: ge17, 19: ge19}


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
        "pc": "Plaid Cymru"
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
        "ukip": "BXP",
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



