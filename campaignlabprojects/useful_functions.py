def onsid_from_name(name, ge):
    return ge.index[ge["Constituency"] == name][0]


def name_from_onsid(onsid, ge):
    return ge["Constituency"].loc[onsid]


def calc_marginal_within(margin, ge):
    opposition = ["con", "ld", "ukip", "grn", "snp"]
    parties = ["lab"] + opposition

    ge["winner"] = ge[parties].T.apply(lambda x: x.nlargest(1).idxmin())
    ge["second"] = ge[parties].T.apply(lambda x: x.nlargest(2).idxmin())
    ge["third"] = ge[parties].T.apply(lambda x: x.nlargest(3).idxmin())
    ge["marginal"] = False

    for row in ge.T:
        win_party = ge.loc[row, "winner"]
        win_vote = ge.loc[row, win_party + "_pc"]
        sec_party = ge.loc[row, "second"]
        sec_vote = ge.loc[row, sec_party + "_pc"]
        thi_party = ge.loc[row, "third"]
        thi_vote = ge.loc[row, thi_party + "_pc"]
        if win_vote - sec_vote < margin:
            if "lab" in [win_party, sec_party]:
                ge.loc[row, "marginal"] = True
        if win_vote - thi_vote < margin:
            if thi_party == "lab":
                ge.loc[row, "marginal"] = True

    return ge["marginal"]


