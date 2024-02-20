#!/usr/bin/env python

import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

PLOTS_FOLDER = os.path.join("..", "results", "scores")


def top_level_analysis():
    student_assessment = data.load_df("studentAssessment")
    student_vle = data.load_df("studentVle")
    student_info = data.load_df("studentInfo")
    assessment = data.load_df("assessments")

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
    plt.xlabel("Number of Clicks")
    plt.ylabel("Mean Score")
    plt.title("Student Score vs. Learning")
    plt.savefig(os.path.join(PLOTS_FOLDER, "student_scores_and_learning.png"))
    plt.clf()

    # Joins all student scores with the region information
    student_scores_and_info = pd.merge(
        student_assessment, student_info, how="inner", on="id_student"
    )
    student_scores_and_regions = student_scores_and_info[["region", "score"]]

    # Consolidate regional score into mean
    region_performance = (
        student_scores_and_regions.groupby("region")["score"].apply(list).to_frame()
    )

    region_performance["mean"] = region_performance["score"].apply(np.mean)
    region_performance["std"] = region_performance["score"].apply(np.std)

    # Plot -- I strip the region names to 4 letters until we fix the rotation
    # issues of getting the full region names to fit
    x = region_performance.index.to_list()
    x = [e.replace(" Region", "") for e in x]
    height = region_performance["mean"].to_numpy()
    error = region_performance["std"].to_numpy()
    xticks = np.arange(len(x))
    plt.bar(xticks, height)
    plt.errorbar(
        xticks, y=height, yerr=error, linestyle="none", capsize=5, color="black"
    )
    plt.xticks(xticks, x, rotation=45, horizontalalignment="right")
    plt.xlabel("Region")
    plt.ylabel("Mean Score")
    plt.title("Performance of Different Regions\nAcross all Assessments")
    plt.savefig(
        os.path.join(PLOTS_FOLDER, "region_performance.png"), bbox_inches="tight"
    )
    plt.clf()

    # Joins all student scores with the region information
    student_scores_and_ages = student_scores_and_info[["age_band", "score"]]

    # Consolidate regional score into mean
    age_performance = (
        student_scores_and_ages.groupby("age_band")["score"].apply(list).to_frame()
    )

    # Plot it
    x = age_performance.index.to_list()

    # Consolidate regional score into mean
    age_performance = (
        student_scores_and_ages.groupby("age_band")["score"].apply(list).to_frame()
    )

    # Mean and std of performance
    age_performance["mean"] = age_performance["score"].apply(np.mean)
    age_performance["std"] = age_performance["score"].apply(np.std)

    height = age_performance["mean"].to_numpy()
    error = age_performance["std"].to_numpy()

    plt.bar(x, height)
    plt.errorbar(x, y=height, yerr=error, linestyle="none", capsize=5, color="black")

    plt.xlabel("Age Group")
    plt.ylabel("Mean Score")
    plt.title("Performance of Different Age Groups\nAcross all Assessments")
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
    region_learning_idx = region_learning.index.to_list()
    height = [region_learning[k] / region_size[k] for k in region_learning_idx]
    region_learning_idx = [e.replace(" Region", "") for e in region_learning_idx]
    xticks = np.arange(len(region_learning_idx))
    plt.bar(xticks, height)
    plt.xlabel("Region")
    plt.ylabel("Interactions Per Person")
    plt.xticks(xticks, region_learning_idx, rotation=45, horizontalalignment="right")
    plt.title("Virtual Learning Engagement\nAcross Regions")
    plt.savefig(os.path.join(PLOTS_FOLDER, "region_learning.png"), bbox_inches="tight")
    plt.clf()

    student_assessment_and_assessment_info = pd.merge(
        student_assessment, assessment, how="inner", on="id_assessment"
    )
    student_assessment_and_assessment_info["perc_till_deadline"] = 100.0 * (
        student_assessment_and_assessment_info["date_submitted"]
        / student_assessment_and_assessment_info["date"]
    )

    # drop rows where assessments were submitted before the semester (or module) started
    # could come back to this.. we didn't make sense of it..
    student_assessment_and_assessment_info = student_assessment_and_assessment_info[
        student_assessment_and_assessment_info["date_submitted"] >= 0
    ]

    plt.scatter(
        student_assessment_and_assessment_info["perc_till_deadline"].to_numpy(),
        student_assessment_and_assessment_info["score"].to_numpy(),
    )
    plt.xlabel("Percent Till Deadline")
    plt.ylabel("Score")
    plt.axvline(100.0, linestyle='--', color='black')
    plt.title("Student Scores Compared with\nRelative Submission Time")
    plt.savefig(
        os.path.join(PLOTS_FOLDER, "student_perc_till_deadline_and_performance.png")
    )
    plt.clf()


def lower_level_analysis():
    student_assessment = data.load_df("studentAssessment")
    student_vle = data.load_df("studentVle")
    student_info = data.load_df("studentInfo")

    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    assessment = data.load_df("assessments")
    student_exam_assessment = pd.merge(
        student_assessment,
        assessment[assessment["assessment_type"] == "Exam"],
        how="inner",
        on="id_assessment",
    )

    # Consistency of students: groupby on student and then get list of grades across assessments
    consistency = (
        student_exam_assessment.groupby("id_student")["score"].apply(list).to_frame()
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

    # Just grab the meanage_exam_perfor and sum_click columns
    student_scores_and_learning = student_scores_and_learning[["mean", "sum_click"]]

    # Scatter those 2 columns to show correlation
    plt.scatter(
        student_scores_and_learning["sum_click"].to_numpy(),
        student_scores_and_learning["mean"].to_numpy(),
    )
    plt.xlabel("Number of Clicks")
    plt.ylabel("Mean Score")
    plt.title("Student Exam Score vs. Learning")
    plt.savefig(os.path.join(PLOTS_FOLDER, "student_exam_scores_and_learning.png"))
    plt.clf()

    # Joins all student scores with the region information
    student_scores_and_info = pd.merge(
        student_exam_assessment, student_info, how="inner", on="id_student"
    )
    student_scores_and_regions = student_scores_and_info[["region", "score"]]

    # Consolidate regional score into mean
    region_performance = (
        student_scores_and_regions.groupby("region")["score"].apply(list).to_frame()
    )

    region_performance["mean"] = region_performance["score"].apply(np.mean)
    region_performance["std"] = region_performance["score"].apply(np.std)

    # Plot -- I strip the region names to 4 letters until we fix the rotation
    # issues of getting the full region names to fit
    x = region_performance.index.to_list()
    x = [e.replace(" Region", "") for e in x]
    height = region_performance["mean"].to_numpy()
    error = region_performance["std"].to_numpy()
    xticks = np.arange(len(x))
    plt.bar(xticks, height)
    plt.errorbar(
        xticks, y=height, yerr=error, linestyle="none", capsize=5, color="black"
    )

    plt.xticks(xticks, x, rotation=45, horizontalalignment="right")
    plt.xlabel("Region")
    plt.ylabel("Mean Score")
    plt.title("Performance of Different Regions\nAcross all Exams")
    plt.savefig(
        os.path.join(PLOTS_FOLDER, "region_exam_performance.png"), bbox_inches="tight"
    )
    plt.clf()

    student_scores_and_ages = student_scores_and_info[["age_band", "score"]]

    # Consolidate regional score into mean
    age_performance = (
        student_scores_and_ages.groupby("age_band")["score"].apply(list).to_frame()
    )

    # Mean and std of performance
    age_performance["mean"] = age_performance["score"].apply(np.mean)
    age_performance["std"] = age_performance["score"].apply(np.std)

    # Plot it
    x = age_performance.index.to_list()
    height = age_performance["mean"].to_numpy()
    error = age_performance["std"].to_numpy()

    plt.bar(x, height)
    plt.errorbar(x, y=height, yerr=error, linestyle="none", capsize=5, color="black")
    plt.xlabel("Age Group")
    plt.ylabel("Mean Score")
    plt.title("Performance of Different Age Groups\nAcross all Exams")
    plt.savefig(os.path.join(PLOTS_FOLDER, "age_exam_performance.png"))
    plt.clf()

    # leaving this code here.. but plot not too useful

    student_exam_assessment["perc_till_deadline"] = 100.0 * (
        student_exam_assessment["date_submitted"] / student_exam_assessment["date"]
    )

    # drop rows where assessments were submitted before the semester (or module) started
    # could come back to this.. we didn't make sense of it..
    student_exam_assessment = student_exam_assessment[
        student_exam_assessment["date_submitted"] >= 0
    ]

    plt.scatter(
        student_exam_assessment["perc_till_deadline"].to_numpy(),
        student_exam_assessment["score"].to_numpy(),
    )
    plt.axvline(100.0, linestyle="--")
    plt.xlabel("Percent Till Deadline")
    plt.ylabel("Score")
    plt.savefig(
        os.path.join(
            PLOTS_FOLDER, "student_perc_till_deadline_and_exam_performance.png"
        )
    )
    plt.clf()


def main():
    Path(PLOTS_FOLDER).mkdir(parents=True, exist_ok=True)
    top_level_analysis()
    lower_level_analysis()


if __name__ == "__main__":
    main()
