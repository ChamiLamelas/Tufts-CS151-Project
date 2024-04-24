import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data
import os
import time
from datetime import datetime
import pytz

PLOTS_FOLDER = os.path.join("..", "results", "scores-side-channel")


def curr_time(prefix):
    print(
        prefix + datetime.now(pytz.timezone("US/Eastern")).strftime("%m/%d/%y %H:%M:%S")
    )
    return time.time()


def start_timing():
    return curr_time("Start: ")


def end_timing():
    return curr_time("End: ")


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
def get_dp_mean(
    df,
    global_sensitivity_func,
    epsilon,
    src_key,
    dest_key="mean",
    error_key="error",
    debug=False,
):
    df[dest_key] = df[src_key].apply(np.mean)
    if debug:
        print(
            f"Min: {df[dest_key].min()}; Max: {df[dest_key].max()}; Mean: {df[dest_key].mean()}"
        )
    df["gs"] = df[src_key].apply(global_sensitivity_func)
    df["noise"] = df["gs"].apply(lambda x: get_added_laplace_noise(x, epsilon))
    df[dest_key] += df["noise"]
    if debug:
        print(
            f"Min: {df[dest_key].min()}; Max: {df[dest_key].max()}; Mean: {df[dest_key].mean()}"
        )

    df[error_key] = df["gs"].apply(lambda x: (2 * (x / epsilon)) ** 2)


def single_region_exam_performance(epsilon, region):

    student_assessment = data.load_df("studentAssessment")
    student_info = data.load_df("studentInfo")
    assessment = data.load_df("assessments")

    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    # Filter exam assessments only
    student_exam_assessment = pd.merge(
        student_assessment,
        assessment[assessment["assessment_type"] == "Exam"],
        how="inner",
        on="id_assessment",
    )

    ti = time.time()

    student_info = student_info[student_info["region"] == region]

    # Join student information with exams
    student_scores_and_info = pd.merge(
        student_exam_assessment, student_info, how="inner", on="id_student"
    )
    student_scores_and_regions = student_scores_and_info[["region", "score"]]

    student_scores_and_regions = pd.DataFrame(
        [[student_scores_and_regions["score"].tolist()]],
        index=[region],
        columns=["score"],
    )

    # schema will be region and score
    # SELECT region, AVG(score) FROM student_scores_and_regions GROUP BY region
    get_dp_mean(
        student_scores_and_regions, lambda x: 100 / len(x), epsilon, src_key="score"
    )

    # force mean to be between 0-100
    student_scores_and_regions["mean"] = student_scores_and_regions["mean"].clip(
        lower=0, upper=100
    )

    tf = time.time()

    return tf - ti


def all_region_exam_performance(epsilon):

    student_assessment = data.load_df("studentAssessment")
    student_info = data.load_df("studentInfo")
    assessment = data.load_df("assessments")

    # Filter exam assessments only
    student_exam_assessment = pd.merge(
        student_assessment,
        assessment[assessment["assessment_type"] == "Exam"],
        how="inner",
        on="id_assessment",
    )

    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    ti = time.time()

    # Join student information with exams
    student_scores_and_info = pd.merge(
        student_exam_assessment, student_info, how="inner", on="id_student"
    )
    student_scores_and_regions = student_scores_and_info[["region", "score"]]

    # Consolidate regional score into mean
    region_performance = (
        student_scores_and_regions.groupby("region")["score"].apply(list).to_frame()
    )

    # schema will be region and score
    # SELECT region, AVG(score) FROM student_scores_and_regions GROUP BY region
    get_dp_mean(region_performance, lambda x: 100 / len(x), epsilon, src_key="score")

    # force mean to be between 0-100
    region_performance["mean"] = region_performance["mean"].clip(lower=0, upper=100)

    tf = time.time()

    return tf - ti


# `plot_err_runtime`
#
# Given
def plot_runtime(epsilon, chosen_region):
    single_region_exam_performance(epsilon, chosen_region)
    all_region_exam_performance(epsilon)

    NUM_TRIALS = 10

    single_runtimes = np.zeros(NUM_TRIALS)
    agg_runtimes = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        single_runtimes[i] = single_region_exam_performance(epsilon, chosen_region)
        agg_runtimes[i] = all_region_exam_performance(epsilon)

    mean_single_runtime = np.mean(single_runtimes)
    std_single_runtime = np.std(single_runtimes)
    mean_agg_runtime = np.mean(agg_runtimes)
    std_agg_runtime = np.std(agg_runtimes)

    x_axis = ["All Regions", f"Single Region ({chosen_region})"]
    xticks = np.arange(len(x_axis))
    mean_runtimes = [mean_agg_runtime, mean_single_runtime]
    std_runtimes = [std_agg_runtime, std_single_runtime]

    # Plot: Runtimes
    plt.bar(xticks, mean_runtimes)
    plt.xticks(xticks, x_axis)
    plt.xlabel("Query Type")
    plt.ylabel("Runtime (sec)")
    plt.title(f"Runtime of Querying All Regions \nversus One Region ({chosen_region})")

    plt.errorbar(
        xticks,
        y=mean_runtimes,
        yerr=std_runtimes,
        linestyle="none",
        capsize=5,
        color="black",
    )
    plt.savefig(os.path.join(PLOTS_FOLDER, f"{chosen_region.lower()}_side_channel_runtime.png"))
    plt.clf()


def main():
    start_timing()
    plot_runtime(0.3, "Scotland")
    plot_runtime(0.3, "London Region")
    end_timing()


if __name__ == "__main__":
    main()
