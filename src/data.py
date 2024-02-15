from pathlib import Path
import pandas as pd
import os

DATA_FOLDER = os.path.join("..", "data")


def get_all_names():
    return {Path(file).stem for file in os.listdir(DATA_FOLDER)}


def load_df(name):
    valid_names = get_all_names()
    assert (
        name in valid_names
    ), f"{name} is not a valid name, options are: " + ", ".join(valid_names)
    return pd.read_csv(os.path.join(DATA_FOLDER, name + ".csv"))
