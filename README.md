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
- `scores.py`: preliminary analysis for our first presentation in class (on data analysis)
- `diff_private.py`: implementation of our differentially private system plus analysis of it by adding noise to the 5 queries from our data analysis presentation. Note the percent till deadline plot is slightly modified in how it is presented to more easily add differential privacy (i.e. noise).
- `side_channel.py`: implementation of our potential side channel attack as well as our fix plus analysis of it by comparing runtimes of
the two strategies. Note the side channel attack we conceived was not
one of our original queries, but the fix is.

## Results 

Look in the `results/` folder for these subfolders:

### eda 

In this you will find my preliminary exploratory data analysis results. 
- `schemas.txt`: This shows information about the schemas of each of the tables
- `x_unique.txt` where `x` corresponds to each of the 7 tables: This shows information about the number of rows in the table, the unique values in each column, the number of nulls in each column.

### scores 

These are the results from [our data analysis presentation](https://docs.google.com/presentation/d/1-7pvFT6jqMJo2cAY2o8F-wI5Gx_Tkg7jh_FtgQYFRSI/edit?usp=sharing) in class.

### diff-private 

These are the results from [our differential privacy implementation presentation](https://docs.google.com/presentation/d/1n_1PAOA8kjh45x7iZpw9v_C55PKqzLrmKNaGEtIVG98/edit?usp=sharing) in class.

### side-channel 

These are the results from [our side channel attack presentation](https://docs.google.com/presentation/d/1XSaSjwijittda3CL4leFTflyBl_h1I0EgT-l5YhncWQ/edit#slide=id.p) in class.