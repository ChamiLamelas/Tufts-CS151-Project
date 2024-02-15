from pathlib import Path
from datetime import timedelta
import math
import time
import data
import os

EDA_FOLDER = os.path.join("..", "results", "eda")


def load_dfs():
    return {name: data.load_df(name) for name in data.get_all_names()}


def show_schemas(dfs):
    with open(os.path.join(EDA_FOLDER, "schemas.txt"), "w+") as f:
        f.write("=== Schemas ===\n\n")
        for name, df in dfs.items():
            f.write(f"Table: {name}\n")
            for k, v in df.dtypes.to_dict().items():
                f.write(f"\t{k} [{v}]\n")
            f.write("\n")


def show_unique(dfs):
    for name, df in dfs.items():
        with open(os.path.join(EDA_FOLDER, f"{name}_unique.txt"), "w+") as f:
            f.write("=== Null Counts ===\n\n")
            f.write(f"{len(df)} rows\n\n")
            for col, null_count in df.isna().sum().to_dict().items():
                f.write(f"{col}: {null_count}\n")
            f.write("\n=== Unique ===\n\n")
            for col in df.columns:
                uniq = df[col].unique()
                if len(uniq) == len(df):
                    f.write(f"{col} is unique in each row\n\n")
                else:
                    f.write(
                        f"{col} ({len(uniq)} unique rows): {', '.join(map(str, uniq))}\n\n"
                    )


def main():
    ti = time.time()
    Path(EDA_FOLDER).mkdir(exist_ok=True, parents=True)
    dfs = load_dfs()
    show_schemas(dfs)
    show_unique(dfs)
    tf = time.time()
    print(f"runtime (hh:mm:ss): {timedelta(seconds=math.ceil(tf - ti))}")


if __name__ == "__main__":
    main()
