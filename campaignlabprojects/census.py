import pandas as pd
import os


def read_in_census():
    path = os.path.dirname(__file__)+"/../data/"
    census = pd.read_csv(path+"census2011.csv")
    census.index = census["ons_id"]
    census = census.drop(columns="ons_id")
    return census
