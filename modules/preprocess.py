import pandas as pd

def load_data(path="data/movies.csv"):
    df = pd.read_csv(path)
    df["genres"] = df["genres"].fillna("Unknown")
    return df
