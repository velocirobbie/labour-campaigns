
def read_in_census():
    census = pd.read_csv("data/census_file.csv")
    census.index = census["ons_id"]
    census = census.drop(columns="ons_id")
    return census


