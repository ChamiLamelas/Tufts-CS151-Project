# Tufts-CS151-Project

## Setup 

1. Extract `archive.zip`. This is the `.zip` file provided on Kaggle [here](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad). You need an account to download it, so I put it here.
2. Take the contained `.csv` files and put it into a folder called `data`.
3. I have Python `3.10.12`. 
4. To install the libraries I have, run `pip install -r requirements.txt`. 

## Code 

All the code is in `src`.
- `data.py`: utilities for loading tables into pandas DataFrames
- `eda.py`: some preliminary exploratory data analysis (running it will produce the files in `results/eda`)

## Results 

### eda 

In this you will find my preliminary exploratory data analysis results. 
- `schemas.txt`: This shows information about the schemas of each of the tables
- `x_unique.txt` where `x` corresponds to each of the 7 tables: This shows information about the number of rows in the table, the unique values in each column, the number of nulls in each column.