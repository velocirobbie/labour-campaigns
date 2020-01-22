import pandas as pd

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

    lab_score = pd.DataFrame(
        [prev_election["lab_pc"], election["lab_pc"]],
        index=["prev_election", "election_result"],
    ).T
    #lab_score["uns"] = lab_score["prev_election"].map(lambda result: result + swing)
    lab_score["difference"] = lab_score["election_result"] - lab_score["prev_election"]
    return lab_score.sort_index()


