import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data
import os
import time

PLOTS_FOLDER = os.path.join("..", "results", "scores-diff-private")


def get_added_laplace_noise(global_sensitivty, epsilon):
    return np.random.laplace(scale=global_sensitivty / epsilon)

# `get_dp_mean`
# 
# Within data frame `df`, takes the lists of values within `df[src_key]`, 
# and stores differentially private means of those lists within `df[dest_key]`.
# Also stores error/variance within `df[error_key]` based on global sensitivity 
# and epsilon.
# 
# Uses inputted `epsilon` for the epsilon value.
# 
# `global_sensitivity_func` is a function that takes in an arbitrary list 
# that is an element of `df[src_key]` and returns the global sensitivity
# Example global_sensitivity_func: lambda x : 100 / len(x)
#
# Side effects:
#  Stores intermediate results in `df["gs"]` and `df["noise"]`
def get_dp_mean(df, global_sensitivity_func, epsilon, src_key, dest_key="mean", error_key="error"):
    df[dest_key] = df[src_key].apply(np.mean)
    print(f"Min: {df[dest_key].min()}; Max: {df[dest_key].max()}; Mean: {df[dest_key].mean()}")
    df["gs"] = df[src_key].apply(global_sensitivity_func)
    df["noise"] = df["gs"].apply(
        lambda x: get_added_laplace_noise(x, epsilon)
    )
    df[dest_key] += df["noise"]
    print(f"Min: {df[dest_key].min()}; Max: {df[dest_key].max()}; Mean: {df[dest_key].mean()}")

    df[error_key] = df["gs"].apply(lambda x: (2 * (x / epsilon)) ** 2)

# TODO add timing


def student_scores_vs_learning(epsilon):
    student_assessment = data.load_df("studentAssessment")
    student_vle = data.load_df("studentVle")

    # Drop rows with nan score
    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    # Consistency of students: groupby on student and then get list of grades across assessments
    consistency = (
        student_assessment.groupby("id_student")["score"].apply(list).to_frame()
    )

    # Mean and std of performance
    get_dp_mean(consistency, lambda x : 100 / len(x), epsilon, src_key="score")
    # force within range 0-100
    consistency["mean"] = consistency["mean"].clip(lower=0, upper=100)

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

    return consistency["error"].mean(), 0


def age_exam_performance(epsilon):
    student_assessment = data.load_df("studentAssessment")
    student_info = data.load_df("studentInfo")

    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    assessment = data.load_df("assessments")

    # Filter exam assessments only
    student_exam_assessment = pd.merge(
        student_assessment,
        assessment[assessment["assessment_type"] == "Exam"],
        how="inner",
        on="id_assessment",
    )

    # Join student information with exams
    student_scores_and_info = pd.merge(
        student_exam_assessment, student_info, how="inner", on="id_student"
    )

    student_scores_and_ages = student_scores_and_info[["age_band", "score"]]

    # Consolidate regional score into mean
    age_performance = (
        student_scores_and_ages.groupby("age_band")["score"].apply(list).to_frame()
    )

    # Mean and std of performance
    get_dp_mean(age_performance, lambda x : 100 / len(x), epsilon, src_key="score")
    age_performance["std"] = age_performance["score"].apply(np.std)

    # force mean to be within range 0-100
    age_performance["mean"] = age_performance["mean"].clip(lower=0, upper=100)

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

    return age_performance["error"].mean(), 0


def region_exam_performance(epsilon):
    student_assessment = data.load_df("studentAssessment")
    student_info = data.load_df("studentInfo")

    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    assessment = data.load_df("assessments")

    # Filter exam assessments only
    student_exam_assessment = pd.merge(
        student_assessment,
        assessment[assessment["assessment_type"] == "Exam"],
        how="inner",
        on="id_assessment",
    )

    # Join student information with exams
    student_scores_and_info = pd.merge(
        student_exam_assessment, student_info, how="inner", on="id_student"
    )
    student_scores_and_regions = student_scores_and_info[["region", "score"]]

    # Consolidate regional score into mean
    region_performance = (
        student_scores_and_regions.groupby("region")["score"].apply(list).to_frame()
    )

    get_dp_mean(region_performance, lambda x : 100 / len(x), epsilon, src_key="score")

    # force mean to be between 0-100
    region_performance["mean"] = region_performance["mean"].clip(lower=0, upper=100)

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

    return region_performance["error"].mean(), 0


def region_learning(epsilon):
    student_vle = data.load_df("studentVle")
    student_info = data.load_df("studentInfo")

    # Join student learning and info
    student_regions_and_learning = pd.merge(
        student_info[["id_student", "region"]],
        student_vle[["id_student", "sum_click"]],
        how="inner",
        on="id_student",
    )

    # Collects total clicks per region
    region_size = (
        student_info.groupby("region")["id_student"].nunique().to_frame().reset_index()
    )

    region_learning = (
        student_regions_and_learning.groupby("region")["sum_click"]
        .sum()
        .to_frame()
        .reset_index()
    )
    region_learning = region_learning.rename(columns={"sum_click": "sum"})
    print(region_learning["sum"].min(), region_learning["sum"].max())

    region_learning_variance = (
        student_regions_and_learning.groupby("region")["sum_click"]
        .max()
        .to_frame()
        .reset_index()
    )
    region_learning_variance = region_learning_variance.rename(
        columns={"sum_click": "gs"}
    )

    region_learning = pd.merge(
        region_learning, region_learning_variance, how="inner", on="region"
    )

    region_learning["noise"] = region_learning["gs"].apply(
        lambda x: get_added_laplace_noise(x, epsilon)
    )

    region_learning["sum"] += region_learning["noise"]
    print(region_learning["sum"].min(), region_learning["sum"].max())
    region_learning["sum"] = region_learning["sum"].clip(lower=0)
    region_learning["error"] = region_learning["gs"].apply(
        lambda x: (2 * (x / epsilon)) ** 2
    )

    # TODO fix this plotting based on indices
    # Plot it
    # region_learning_idx = region_learning["region"].to_list()
    # height = [region_learning[k] / region_size[k] for k in region_learning_idx]
    # region_learning_idx = [e.replace(" Region", "") for e in region_learning_idx]
    # xticks = np.arange(len(region_learning_idx))
    # plt.bar(xticks, height)
    # plt.xlabel("Region")
    # plt.ylabel("Interactions Per Person")
    # plt.xticks(xticks, region_learning_idx, rotation=45, horizontalalignment="right")
    # plt.title("Virtual Learning Engagement\nAcross Regions")
    # plt.savefig(os.path.join(PLOTS_FOLDER, "region_learning.png"), bbox_inches="tight")
    # plt.clf()

    return region_learning["error"].mean(), 0


def main():
    # for ep in [5, 50]:
    #     err, runtime = student_scores_vs_learning(epsilon=ep)
    #     print(err)

    # for ep in [5, 0.5]:
    #     err, runtime = age_exam_performance(epsilon=ep)
    #     print(err)

    # for ep in [0.5, 0.2]:
    #     err, runtime = region_exam_performance(epsilon=ep)
    #     print(err)

    for ep in [500]:
        err, runtime = region_learning(epsilon=ep)
        print(err)


if __name__ == "__main__":
    main()
