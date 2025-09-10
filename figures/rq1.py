import click
import glob
import matplotlib.pyplot as plt
import pandas as pd
import pymannkendall as mk
import seaborn as sns
import sys
sys.path.append("../")

COLOR_PALETTE = "muted"
LABELFONTSIZE = 16
LEGENDFONTSIZE = 14
TICKFONTSIZE = 12
LEGENDTITLEFONTSIZE = 16
XLABELPAD = 6
YLABELPAD = 6

system_names = {
    "batik": "\\textsc{batik}",
    "dconvert": "\\textsc{dconvert}",
    "h2": "\\textsc{h2}",
    "jump3r": "\\textsc{jump3r}",
    "kanzi": "\\textsc{kanzi}",
    "lrzip": "\\textsc{lrzip}",
    "x264": "\\textsc{x264}",
    "xz": "\\textsc{xz}",
    "z3": "\\textsc{z3}",
    "7z": "\\textsc{7z}",
    "BerkeleyDBC": "\\textsc{BDB-C}",
    "Dune": "\\textsc{Dune}",
    "Hipacc": "\\textsc{HIPA$^{cc}$}",
    "JavaGC": "\\textsc{JavaGC}",
    "LLVM": "\\textsc{LLVM}",
    "LRZip": "\\textsc{lrzip}",
    "Polly": "\\textsc{Polly}",
    "VP9": "\\textsc{VP9}",
}

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": "\\usepackage{mathptmx} \\usepackage{eucal} \\usepackage{newtxtext, newtxmath}",     
    "axes.labelsize": LABELFONTSIZE,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "font.weight": "bold",
    "axes.labelweight": "bold",
})

def load_experiment_results(folder):
    casestudies = glob.glob(f"{folder}/*.csv")
    dfs = [pd.read_csv(casestudy) for casestudy in casestudies]
    for casestudy, df in zip(casestudies, dfs):
        df["casestudy"] = casestudy.split("/")[-1].split("\\")[-1].split(".")[0]
    dfs = [df for df in dfs if not df.empty]
    df = pd.concat(dfs)
    df["casestudy"] = df["casestudy"].replace(system_names)
    if "workload" in folder:
        df = df[df["casestudy"] != "\\textsc{lrzip}"]
    if "distance_based" in folder:
        df = df[df["casestudy"] != "\\textsc{lrzip}"]
        df = df[df["casestudy"] != "\\textsc{x264}"]
    return df

def load_experiment_results_multi_method(experiment_path, methods = ["syflow", "rsd", "beam-kl", "beam-mean", "dfs-mean"]):
    df = pd.DataFrame()
    for method in methods:
        folder = f"results/RQ1/{method}/{experiment_path}"
        df_method = load_experiment_results(folder)
        df_method["method"] = method
        df_method["method"] = df_method["method"].replace({
            "syflow": "\\textsc{Syflow}",
            "rsd": "\\textsc{RSD}",
            "beam-kl": "\\textsc{BS-kl}",
            "beam-mean": "\\textsc{BS-$\\mu$}",
            "dfs-mean": "\\textsc{DFS-$\\mu$}",
            "cart": "\\textsc{CART}",
        })
        df = pd.concat([df, df_method])
    return df

@click.group()
def cli():
    """Command line interface for generating RQ2 statistics tables."""
    global methods, method_names
    methods = ["beam-mean", "beam-kl", "dfs-mean", "rsd", "syflow"]
    method_names = ["\\textsc{BS-$\\mu$}", "\\textsc{BS-kl}", "\\textsc{DFS-$\\mu$}", "\\textsc{RSD}", "\\textsc{Syflow}"]

def plot_scalability(experiment_path, target_column):
    global methods, method_names
    df = load_experiment_results_multi_method(experiment_path, methods)
    df["method"] = pd.Categorical(df["method"], categories=method_names, ordered=True)
    color_palette = sns.color_palette(COLOR_PALETTE, len(methods))

    df_runtime = df[["seed", "method", target_column, "runtime"]].copy()
    # Add a dummy row for each method and target combination
    for method in method_names:
        for target in df_runtime[target_column].unique():
            for seed in df_runtime["seed"].unique():
                if not ((df_runtime["seed"] == seed) & (df_runtime["method"] == method) & (df_runtime[target_column] == target)).any():
                    df_runtime = pd.concat([df_runtime, pd.DataFrame({
                        "seed": [seed],
                        "method": [method],
                        target_column: [target],
                        "runtime": [14400],
                    })], ignore_index=True)

    fig = plt.figure(figsize=(6, 3.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Top-left plot: Runtime vs Number of Configuration Options (0-300 seconds)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, method in enumerate(method_names):
        subset = df_runtime[df_runtime["method"] == method]
        if subset["runtime"].max() > 600:
            sns.lineplot(
                data=subset,
                x=target_column,
                y="runtime",
                label=method,
                color=color_palette[i],
                linewidth=1.5,
                ax=ax1,
            )
    ax1.set_xscale("log")
    ax1.set_xlim(df[target_column].min(), df[target_column].max()*1.1)
    ax1.set_ylim(0, 7300)
    ax1.set_yticks([0, 3600, 7200])
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.7, length=3)
    ax1.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.7, length=3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("black")
    ax1.spines["bottom"].set_color("black")
    ax1.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    ax1.set_xticklabels([])
    
    # Bottom-left plot: Runtime vs Number of Configuration Options (0-3600 seconds)
    ax2 = fig.add_subplot(gs[1, 0])
    for i, method in enumerate(method_names):
        subset = df_runtime[df_runtime["method"] == method]
        if subset["runtime"].max() < 600:
            sns.lineplot(
                data=subset,
                x=target_column,
                y="runtime",
                label=method,
                color=color_palette[i],
                linewidth=1.5,
                ax=ax2,
            )
    ax2.set_xscale("log")
    ax2.set_xlim(df[target_column].min(), df[target_column].max()*1.1)
    ax2.set_ylim(0, 370)
    ax2.set_yticks([0, 180, 360])
    ax2.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.7, length=3)
    ax2.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.7, length=3)
    if target_column == "n_features":
        ax2.set_xlabel("$|\\CMcal{O}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    else:
        ax2.set_xlabel("$|\\CMcal{C}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    ax2.set_ylabel("")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("black")
    ax2.spines["bottom"].set_color("black")
    ax2.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)

    # Right plot: F1 vs Number of Configuration Options
    df_scatter = df[["seed", "method", target_column, "f1"]].copy()
    df_scatter["identified"] = df.groupby(["method", target_column])["f1"].transform(lambda x: (x == 1.0).sum())    
    df_scatter = df_scatter.groupby(["method", target_column]).filter(lambda x: len(x) == 100)
    df_scatter = df_scatter.groupby(["method", target_column]).agg({"identified": "max"}).reset_index()
    ax3 = fig.add_subplot(gs[:, 1])  # Span both rows
    sns.scatterplot(
        data=df_scatter,
        x=target_column,
        y="identified",
        hue="method",
        style="method",
        markers=["s", "s", "D", "o", "X"],
        palette=color_palette,
        alpha=0.7,
        s=35,
        ax=ax3,
        linewidth=0.4,
        edgecolor="black",
    )
    # add dashed lines for each method
    for j, method in enumerate(method_names):
        subset = df_scatter[df_scatter["method"] == method]
        if len(subset) > 0:
            ax3.plot(subset[target_column], subset["identified"], color=color_palette[j], linestyle="--", linewidth=1.2, alpha=0.5)

    if target_column == "n_features":
        ax3.set_xlabel("$|\\CMcal{O}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    else:
        ax3.set_xlabel("$|\\CMcal{C}|$", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
    ax3.set_ylabel(r"\% Identified", fontsize=LABELFONTSIZE, labelpad=YLABELPAD)
    ax3.set_xscale("log")
    ax3.set_xlim(df[target_column].min(), df[target_column].max()*1.1)
    ax3.set_ylim(0, 105)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.7, length=3)
    ax3.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.7, length=3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("black")
    ax3.spines["bottom"].set_color("black")
    ax3.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    # Add a single legend for all plots
    handles, labels = ax2.get_legend_handles_labels()
    handles2, labels2 = ax1.get_legend_handles_labels()
    handles += handles2
    labels += labels2
    fig.legend(
        handles,
        labels,
        title="",
        fontsize=LEGENDFONTSIZE,
        title_fontsize=LEGENDTITLEFONTSIZE,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.05),
        edgecolor="white",
        handlelength=1,
        # decrease whitespace between legend entries
        handletextpad=0.4,
        columnspacing=1.2,
    )
    # add some text on the left side of the plot without scaling any of the axes
    fig.text(
        0, 0.57, "Runtime (s)", ha="center", va="center", rotation=90,
        fontsize=LABELFONTSIZE, color="black"
    )
    
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.87, wspace=0.35, hspace=0.4)
    plt.savefig(f"figures/out/rq1_scalability_{target_column}.pdf", dpi=300, bbox_inches="tight", format="pdf")

@cli.command()
def plot_scalability_features():
    '''Plot scalability for number of features.'''
    plot_scalability("scalability_features/workload", "n_features")

@cli.command()
def plot_scalability_samples():
    '''Plot scalability for number of samples.'''
    plot_scalability("scalability_samples/workload", "n_samples")

@cli.command()
def plot_number_of_subgroups():
    """Plot results for varying the number of subgroups."""
    global methods, method_names
    df_wl = load_experiment_results_multi_method("number_of_subgroups/workload", methods)
    df_db = load_experiment_results_multi_method("number_of_subgroups/distance_based", methods)
    df = pd.concat([df_wl, df_db])
    df["method"] = pd.Categorical(df["method"], categories=method_names, ordered=True)
    
    # Drop case studies that don't have 3 groups
    df = df[df["casestudy"].isin(df[df["n_groups"] == 3]["casestudy"].unique())]
    df["casestudy"] = df["casestudy"].astype("str")

    # Create a figure with subplots for each method
    fig, axes = plt.subplots(len(method_names), 1, figsize=(11, 1.3*len(method_names)), sharex=True, sharey=True)

    for i, method in enumerate(method_names):
        ax = axes[i] if len(method_names) > 1 else axes
        method_df = df[df["method"] == method]
        method_df.loc[:, "mean_f1"] = method_df[["f1_0", "f1_1", "f1_2"]].sum(axis=1) / method_df["n_groups"]
        method_df = method_df[~method_df["casestudy"].isin(["\\textsc{VP9}", "\\textsc{JavaGC}", "\\textsc{HIPA$^{cc}$}"])]

        sns.boxplot(
            data=method_df,
            x="n_groups",
            y="mean_f1",
            hue="casestudy",
            palette=COLOR_PALETTE,
            ax=ax,
            gap=0.4,
            width=0.8,
            linecolor="black",
            linewidth=0.4,
            flierprops=dict(marker="o", markersize=3, markeredgewidth=0.4, alpha=0.7, color="black"),
            boxprops=dict(alpha=0.85),
        )
        ax.set_yticklabels([0, "", "", 1])
        ax.set_yticks([0, 0.333, 0.666, 1.0])
        
        ax.set_xlabel(r"\# Seeded Subspaces", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
        ax.set_ylabel(method, fontsize=LABELFONTSIZE-4, labelpad=YLABELPAD)
        ax.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.7, length=3)
        ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.7, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)

        n_conditions_sorted = sorted(method_df["n_groups"].unique())
        casestudies_sorted = sorted(method_df["casestudy"].unique())
        n_casestudies = len(casestudies_sorted)

        group_width = 0.8
        box_width = group_width / n_casestudies

        for j, n_groups in enumerate(n_conditions_sorted):
            for k, casestudy in enumerate(casestudies_sorted):
                subset = method_df[(method_df["n_groups"] == n_groups) & (method_df["casestudy"] == casestudy)]
                count_at_cap = (subset["mean_f1"] == 1.0).sum()

                if len(subset) > 0:
                    x_pos = j - group_width / 2 + k * box_width + box_width / 2
                    ax.text(
                        x_pos, 1.0 + 0.01, f"\\textbf{{{count_at_cap}}}",
                        ha="center", va="bottom", fontsize=TICKFONTSIZE-1, color=sns.color_palette(COLOR_PALETTE, 15)[k]
                    )
        ax.legend_.remove()

    # Add a single legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="",
        fontsize=LEGENDFONTSIZE,
        title_fontsize=LEGENDTITLEFONTSIZE,
        loc="upper center",
        ncol=len(df["casestudy"].unique()),
        bbox_to_anchor=(0.5, 1.05),
        edgecolor="white",
        # decrease width of boxes in legend
        handletextpad=0.4,
        borderaxespad=0.4,
        handlelength=1,
    )
    fig.text(
        0.05, 0.5, "Mean F1 Score", ha="center", va="center", rotation=90,
        fontsize=LABELFONTSIZE, color="black"
    )
    plt.subplots_adjust(top=0.94, hspace=0.55)
    plt.savefig(f"figures/out/rq1_number_of_subgroups.pdf", dpi=300, bbox_inches="tight", format="pdf")

@cli.command()
def plot_number_of_predicates():
    '''Plot results for varying the number of predicates.'''
    global methods, method_names
    df_wl = load_experiment_results_multi_method("number_of_predicates/workload", methods)
    df_db = load_experiment_results_multi_method("number_of_predicates/distance_based", methods)
    df = pd.concat([df_wl, df_db])
    df["method"] = pd.Categorical(df["method"], categories=method_names, ordered=True)#, "\\textsc{CART}"], ordered=True)#
    df = df[df["casestudy"].isin(["\\textsc{7z}", "\\textsc{BDB-C}", "\\textsc{h2}", "\\textsc{jump3r}", "\\textsc{x264}", "\\textsc{z3}"])]
    casestudies = df["casestudy"].unique()
    color_palette = sns.color_palette(COLOR_PALETTE, len(method_names))

    df_scatter = df[["seed", "casestudy", "method", "n_conditions", "f1"]].copy()
    df_scatter["identified"] = df.groupby(["method", "n_conditions", "casestudy"])["f1"].transform(lambda x: (x == 1.0).sum())    
    df_scatter = df_scatter.groupby(["method", "n_conditions", "casestudy"]).agg({"identified": "max"}).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(11, 4), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, casestudy in enumerate(casestudies):
        ax = axes[i]
        casestudy_df = df_scatter[df_scatter["casestudy"] == casestudy]
        sns.scatterplot(
            data=casestudy_df,
            x="n_conditions",
            y="identified",
            hue="method",
            style="method",
            markers=["s", "s", "D", "o", "X"], #, "P"],
            palette=color_palette,
            alpha=0.7,
            s=30,
            ax=ax,
            linewidth=0.4,
            edgecolor="black",
        )
        # add dashed lines for each method
        for j, method in enumerate(method_names):#, "\\textsc{CART}"]):
            subset = casestudy_df[casestudy_df["method"] == method]
            if len(subset) > 0:
                ax.plot(subset["n_conditions"], subset["identified"], color=color_palette[j], linestyle="--", linewidth=1.2, alpha=0.5)

        ax.set_xlabel("", fontsize=LABELFONTSIZE, labelpad=XLABELPAD)
        ax.set_ylabel("", fontsize=LABELFONTSIZE, labelpad=YLABELPAD)
        # put casestudy name on the bottom left corner of the plot, surrounded by a box
        ax.text(
            1, 8, casestudy,
            ha="left", va="bottom", fontsize=LABELFONTSIZE-4,
            color="black", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", linewidth=0.3)
        )
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.tick_params(axis="x", labelsize=TICKFONTSIZE, bottom=True, width=0.7, length=3)
        ax.tick_params(axis="y", labelsize=TICKFONTSIZE, left=True, width=0.7, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.get_legend().remove()

    fig.text(
        0, 0.5, "\\% Identified", ha="center", va="center", rotation=90,
        fontsize=LABELFONTSIZE, color="black"
    )
    fig.text(
        0.5, 0, "\\# Predicates", ha="center", va="center",
        fontsize=LABELFONTSIZE, color="black"
    )
    # Add a single legend for all plots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="",
        fontsize=LEGENDFONTSIZE,
        title_fontsize=LEGENDTITLEFONTSIZE,
        loc="upper center",
        ncol=len(method_names),
        bbox_to_anchor=(0.5, 1.05),
        edgecolor="white",
        handletextpad=0,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.13)
    plt.savefig(f"figures/out/rq1_number_of_predicates.pdf", dpi=300, bbox_inches="tight", format="pdf")

def statistical_significance_trend(experiment_path="scalability_features/workload", target_column="n_features"):
    """
    For each method, test for a monotonic trend (e.g., decreasing F1) across ordered groups
    using the Mann-Kendall test (nonparametric test for monotonic trend).
    """
    df = load_experiment_results_multi_method(experiment_path, ["rsd", "beam-kl", "beam-mean", "dfs-mean", "syflow"])
    methods = df["method"].unique()
    results = {}

    for method in methods:
        method_df = df[df["method"] == method]
        sorted_df = method_df.sort_values(by=target_column)
        sorted_df["identified"] = sorted_df.groupby([target_column])["f1"].transform(lambda x: (x == 1.0).sum())
        grouped = sorted_df.groupby(target_column).agg({"identified": "max"}).reset_index()
        mk_result = mk.original_test(grouped["identified"], alpha=0.05)

        results[method] = mk_result
        print(f"Method: {method}")
        print(mk_result)
        print("-" * 40)

    return results

@cli.command()
def significance_trend_features():
    """Test for monotonic trend in F1 scores across number of features."""
    statistical_significance_trend("scalability_features/workload", "n_features")

@cli.command()
def significance_trend_samples():
    """Test for monotonic trend in F1 scores across number of samples."""
    statistical_significance_trend("scalability_samples/workload", "n_samples")

if __name__ == "__main__":
    cli()