import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data
import os
import time
from datetime import datetime
import pytz

PLOTS_FOLDER = os.path.join("..", "results", "scores-diff-private")


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


def student_scores_vs_learning(epsilon):
    ti = time.time()

    student_assessment = data.load_df("studentAssessment")
    student_vle = data.load_df("studentVle")

    # Drop rows with nan score
    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    # Consistency of students: groupby on student and then get list of grades across assessments
    consistency = (
        student_assessment.groupby("id_student")["score"].apply(list).to_frame()
    )

    # schema will be student_assessment's schema
    # SELECT id_student, AVG(score) FROM student_assessment GROUP BY id_student

    # Mean and std of performance
    get_dp_mean(consistency, lambda x: 100 / len(x), epsilon, src_key="score")
    # force within range 0-100
    consistency["mean"] = consistency["mean"].clip(lower=0, upper=100)

    tf = time.time()

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

    return consistency["error"].mean(), tf - ti


def age_exam_performance(epsilon):
    ti = time.time()

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

    # schema will be age_band and score
    # SELECT age_band, AVG(score) FROM student_scores_and_ages GROUP BY age_band

    # Mean and std of performance
    get_dp_mean(age_performance, lambda x: 100 / len(x), epsilon, src_key="score")

    # force mean to be within range 0-100
    age_performance["mean"] = age_performance["mean"].clip(lower=0, upper=100)

    tf = time.time()

    # Plot it
    x = age_performance.index.to_list()
    height = age_performance["mean"].to_numpy()

    plt.bar(x, height)
    plt.xlabel("Age Group")
    plt.ylabel("Mean Score")
    plt.title("Performance of Different Age Groups\nAcross all Exams")
    plt.savefig(os.path.join(PLOTS_FOLDER, "age_exam_performance.png"))
    plt.clf()

    return age_performance["error"].mean(), tf - ti


def region_exam_performance(epsilon):
    ti = time.time()

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

    # schema will be region and score
    # SELECT region, AVG(score) FROM student_scores_and_regions GROUP BY region
    get_dp_mean(region_performance, lambda x: 100 / len(x), epsilon, src_key="score")

    # force mean to be between 0-100
    region_performance["mean"] = region_performance["mean"].clip(lower=0, upper=100)

    tf = time.time()

    # Plot -- I strip the region names to 4 letters until we fix the rotation
    # issues of getting the full region names to fit
    x = region_performance.index.to_list()
    x = [e.replace(" Region", "") for e in x]
    height = region_performance["mean"].to_numpy()
    xticks = np.arange(len(x))
    plt.bar(xticks, height)

    plt.xticks(xticks, x, rotation=45, horizontalalignment="right")
    plt.xlabel("Region")
    plt.ylabel("Mean Score")
    plt.title("Performance of Different Regions\nAcross all Exams")
    plt.savefig(
        os.path.join(PLOTS_FOLDER, "region_exam_performance.png"), bbox_inches="tight"
    )
    plt.clf()

    return region_performance["error"].mean(), tf - ti


def region_learning(epsilon):
    ti = time.time()

    student_vle = data.load_df("studentVle")
    student_info = data.load_df("studentInfo")

    # Collapse table such that every id_student is unique
    student_vle["student_sum_click"] = student_vle.groupby("id_student")[
        "sum_click"
    ].transform("sum")
    student_vle.drop_duplicates(subset=["id_student"], inplace=True)
    student_vle.drop(columns=["sum_click"], inplace=True)

    # Join student learning and info
    student_regions_and_learning = pd.merge(
        student_info[["id_student", "region"]],
        student_vle[["id_student", "student_sum_click"]],
        how="inner",
        on="id_student",
    )

    maxval = student_regions_and_learning["student_sum_click"].max()

    student_regions_and_learning_grouped = (
        student_regions_and_learning.groupby("region")["student_sum_click"]
        .apply(list)
        .to_frame()
    )

    get_dp_mean(
        student_regions_and_learning_grouped,
        lambda x: maxval / len(x),
        epsilon,
        src_key="student_sum_click",
    )

    tf = time.time()

    # schema: id_student region sum_click
    # SELECT region, SUM(sum_click) / COUNT(DISTINCT id_student) FROM student_regions_and_learning GROUP BY region

    # Plot it
    x = student_regions_and_learning_grouped.index.to_list()
    x = [e.replace(" Region", "") for e in x]
    height = student_regions_and_learning_grouped["mean"].to_numpy()
    xticks = np.arange(len(x))
    plt.bar(xticks, height)

    plt.xticks(xticks, x, rotation=45, horizontalalignment="right")
    plt.xlabel("Region")
    plt.ylabel("Interactions Per Person")
    plt.title("Virtual Learning Environment\nAcross Regions")
    plt.savefig(os.path.join(PLOTS_FOLDER, "region_learning.png"), bbox_inches="tight")
    plt.clf()

    return student_regions_and_learning_grouped["error"].mean(), tf - ti


# SELECT pct_till_deadline_band, AVG(score) FROM df GROUP BY pct_till_deadline_band


def perc_till_deadline(epsilon):
    ti = time.time()

    student_assessment = data.load_df("studentAssessment")

    student_assessment = student_assessment[~student_assessment["score"].isnull()]

    assessment = data.load_df("assessments")
    assessment = assessment[~assessment["date"].isnull()]

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

    student_assessment_and_assessment_info = student_assessment_and_assessment_info[
        ["score", "perc_till_deadline"]
    ]

    # defines index of a particular bin `i` 0 <= i <= 10 such that...
    # if 0 <= i < 10, bin i refers to the interval: [10 * i, 10 * i + 10)
    # if i = 10, bin i refers to the interval: [100, \infty)
    student_assessment_and_assessment_info["bin"] = (
        student_assessment_and_assessment_info["perc_till_deadline"].apply(
            lambda x: min(int(x // 10), 10)
        )
    )

    student_assessment_and_assessment_info_grouped = (
        student_assessment_and_assessment_info.groupby("bin")["score"]
        .apply(list)
        .to_frame()
    )

    get_dp_mean(
        student_assessment_and_assessment_info_grouped,
        lambda x: 100 / len(x),
        epsilon,
        src_key="score",
    )

    tf = time.time()

    # Plot it
    x = student_assessment_and_assessment_info_grouped.index.to_list()
    height = student_assessment_and_assessment_info_grouped["mean"].to_numpy()

    plt.bar(x, height)
    plt.xticks(
        x,
        map(lambda e: f"[{10*e},{10*e+10})" if e < 10 else "$\geq 100$", x),
        rotation=45,
        horizontalalignment="right",
    )
    plt.xlabel("Bin")
    plt.ylabel("Mean Score")
    plt.title("Assessment Performance at\nDifferent Deadlines")
    plt.savefig(
        os.path.join(PLOTS_FOLDER, "perc_till_deadline.png"), bbox_inches="tight"
    )
    plt.clf()

    return student_assessment_and_assessment_info_grouped["error"].mean(), tf - ti


# `plot_err_runtime`
#
# Given
def plot_err_runtime(query_func, epsilons, query_name):
    NUM_TRIALS = 10
    errors = np.zeros((len(epsilons), NUM_TRIALS))
    runtimes = np.zeros((len(epsilons), NUM_TRIALS))
    for idx, ep in enumerate(epsilons):
        for i in range(NUM_TRIALS):
            err, runtime = query_func(epsilon=ep)
            errors[idx][i] = err
            runtimes[idx][i] = runtime

    mean_errors = np.mean(errors, axis=1)
    std_errors = np.std(errors, axis=1)
    mean_runtimes = np.mean(runtimes, axis=1)
    std_runtimes = np.std(runtimes, axis=1)

    # Plot: Error vs. Epsilon
    plt.plot(epsilons, mean_errors)
    plt.errorbar(
        epsilons,
        y=mean_errors,
        yerr=std_errors,
        linestyle="none",
        capsize=5,
        color="black",
    )
    plt.title(f"Average Error by Epsilon Value \nfor Query {query_name}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Error Amount")
    plt.savefig(os.path.join(PLOTS_FOLDER, f"{query_func.__name__}_error.png"))
    plt.clf()

    # Plot: Runtime vs. Epsilon
    plt.plot(epsilons, mean_runtimes)
    plt.errorbar(
        epsilons,
        y=mean_runtimes,
        yerr=std_runtimes,
        linestyle="none",
        capsize=5,
        color="black",
    )
    plt.title(f"Average Runtime by Epsilon Value \nfor Query {query_name}")
    plt.xlabel("Epsilon")
    plt.ylabel("Average Runtime")
    plt.savefig(os.path.join(PLOTS_FOLDER, f"{query_func.__name__}_runtime.png"))
    plt.clf()


def main():
    start_timing()
    plot_err_runtime(
        student_scores_vs_learning, [10, 20, 30, 40, 50], "Student Score vs. Learning"
    )
    plot_err_runtime(
        age_exam_performance, [0.2, 0.3, 0.4, 0.5, 1], "Age Exam Performance"
    )
    plot_err_runtime(
        region_exam_performance, [0.1, 0.3, 0.5, 0.7, 0.9], "Region Exam Performance"
    )
    plot_err_runtime(region_learning, [5, 10, 15, 20, 25], "Region Learning")
    plot_err_runtime(
        perc_till_deadline,
        [0.01, 0.02, 0.03, 0.05, 0.1],
        "Assessment Performance Related To Deadline",
    )

    # we want to call the query functions with the best
    # epsilons at the end for our plots for the slides
    # this will also print the errors 
    print(student_scores_vs_learning(40)[0])
    print(age_exam_performance(0.5)[0])
    print(region_exam_performance(0.3)[0])
    print(region_learning(10)[0])
    print(perc_till_deadline(0.03)[0])

    end_timing()


if __name__ == "__main__":
    main()
