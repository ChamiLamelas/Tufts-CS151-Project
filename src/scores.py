#!/usr/bin/env python

import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

PLOTS_FOLDER = os.path.join("..", "results", "scores")


def main():
    Path(PLOTS_FOLDER).mkdir(parents=True, exist_ok=True)

    student_assessment = data.load_df("studentAssessment")
    student_vle = data.load_df("studentVle")
    student_info = data.load_df("studentInfo")

    # Drop rows with nan score
    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    # Consistency of students: groupby on student and then get list of grades across assessments
    consistency = (
        student_assessment.groupby("id_student")["score"].apply(list).to_frame()
    )

    # Mean and std of performance
    consistency["mean"] = consistency["score"].apply(np.mean)
    consistency["std"] = consistency["score"].apply(np.std)

    # Calculates total VLE usage by summing "sum_click"
    student_total_vle_usage = (
        student_vle.groupby("id_student")["sum_click"].sum().to_frame()
    )

    # Joins consistency of performance with VLE usage
    student_scores_and_learning = pd.merge(
        consistency, student_total_vle_usage, how="inner", on="id_student"
    )

    # Just grab the mean and sum_click columns
    student_scores_and_learning = student_scores_and_learning[["mean", "sum_click"]]

    # Scatter those 2 columns to show correlation
    plt.scatter(
        student_scores_and_learning["sum_click"].to_numpy(),
        student_scores_and_learning["mean"].to_numpy(),
    )
    plt.savefig(os.path.join(PLOTS_FOLDER, "student_scores_and_learning.png"))
    plt.clf()

    # Joins all student scores with the region information
    student_scores_and_info = pd.merge(
        student_assessment, student_info, how="inner", on="id_student"
    )
    student_scores_and_regions = student_scores_and_info[["region", "score"]]

    # Consolidate regional score into mean
    region_performance = student_scores_and_regions.groupby("region").mean()

    # Plot -- I strip the region names to 4 letters until we fix the rotation
    # issues of getting the full region names to fit
    x = region_performance.index.to_list()
    x = [e[:4] for e in x]
    height = region_performance["score"].to_numpy()
    plt.bar(x, height)
    plt.savefig(os.path.join(PLOTS_FOLDER, "region_performance.png"))
    plt.clf()

    # Joins all student scores with the region information
    student_scores_and_ages = student_scores_and_info[["age_band", "score"]]

    # Consolidate regional score into mean
    age_performance = student_scores_and_ages.groupby("age_band").mean()

    # Plot it
    x = age_performance.index.to_list()
    height = age_performance["score"].to_numpy()
    plt.bar(x, height)
    plt.savefig(os.path.join(PLOTS_FOLDER, "age_performance.png"))
    plt.clf()

    # Join student learning and info
    student_regions_and_learning = pd.merge(
        student_info[["id_student", "region"]],
        student_vle[["id_student", "sum_click"]],
        how="inner",
        on="id_student",
    )

    # Collects total clicks per region
    region_size = student_info.groupby("region")["id_student"].nunique()
    region_learning = student_regions_and_learning.groupby("region")["sum_click"].sum()

    # Plot it
    x = region_learning.index.to_list()
    x = [e[:4] for e in x]
    height = region_learning.values
    plt.bar(x, height)
    plt.savefig(os.path.join(PLOTS_FOLDER, "region_learning.png"))
    plt.clf()

    x = region_size.index.to_list()
    x = [e[:4] for e in x]
    height = region_size.values
    plt.bar(x, height)
    plt.savefig(os.path.join(PLOTS_FOLDER, "region_size.png"))
    plt.clf()


if __name__ == "__main__":
    main()
