import pandas as pd
import json


def load_lessons_learned_data(path):
    df = pd.read_csv(path, sep=",", encoding="ISO-8859-1", index_col="Lesson ID")
    df.fillna("", inplace=True)
    return df


def load_project_data(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data