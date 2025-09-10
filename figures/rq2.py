from matplotlib import gridspec
from matplotlib.colors import to_rgba
from scipy.signal import find_peaks
from scipy.stats import skew
import click
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
sys.path.append("./")

from src.eval.statistics import get_unique_subgroups
from src.utils import reconstruct_subgroups_from_csv
import seaborn as sns
import src.loaders.datasets as loaders
import src.loaders.distance_based as db
import src.loaders.workload as wl

COLOR_PALETTE = "muted"
LABELFONTSIZE = 16
LEGENDFONTSIZE = 12
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
    "font.family": "Times New Roman",
    "text.latex.preamble": "\\usepackage{mathptmx} \\usepackage{eucal} \\usepackage{newtxtext, newtxmath}",     
    "mathtext.fontset": "cm",
    "axes.labelsize": LABELFONTSIZE,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "font.weight": "normal",
    "axes.labelweight": "normal",
})

def load_experiment_results(folder):
    casestudies = glob.glob(f"{folder}/*.csv")
    dfs = [pd.read_csv(casestudy) for casestudy in casestudies]
    for casestudy, df in zip(casestudies, dfs):
        df['casestudy'] = casestudy.split("/")[-1].split("\\")[-1].split(".")[0]
    dfs = [df for df in dfs if not df.empty]
    df = pd.concat(dfs)
    df['casestudy'] = df['casestudy'].replace(system_names)
    return df 

def table(df, output_path):
    df = df[["casestudy", "num_samples", "num_features", "sgs_unique", "rule_num_features", "rule_length_min", "rule_length_max", "rule_length_mean", "sgs_mean_jaccard", "coverage_mean", "coverage_total"]]
    df['casestudy'] = df['casestudy'].replace(system_names)
    for col in df.columns[1:]:
        if df[col].dtype == 'float64' and df[col].apply(lambda x: x.is_integer()).all():
            df[col] = df[col].astype(int)

    df.rename(columns={
        "casestudy": "\\multicolumn{1}{l}{System}",
        "num_samples": "$|\\mathcal{C}|$",
        "num_features": "\\multicolumn{1}{r}{$|\\optionSpace|$}",
        "sgs_unique": "$|\\mathcal{S}|$",
        "rule_num_features": "\\multicolumn{1}{r}{$|\\restOptionSpace|$}",
        "rule_length_min": "$|s|_{\\text{\\tiny min}}$",
        "rule_length_max": "$|s|_{\\text{\\tiny max}}$",
        "rule_length_mean": "\\multicolumn{1}{r}{$\\overline{|s|}$}",
        "sgs_mean_jaccard": "$\\overline{J}$",
        "coverage_mean": "$\\overline{\\text{cov}(s)}$",
        "coverage_total": "${\\text{cov}(\\cup_{i=0}^k s_i)}$",
    }, inplace=True)
    print(type(df))
    df.to_latex(
        output_path,
        index=False,
        float_format=lambda x: f"{x:.2f}",
        column_format="lrrrrrrrrrr",
        escape=False,
        multirow=True,
        multicolumn=True,
    )

@click.group()
def cli():
    """Command line interface for generating RQ2 statistics tables."""
    pass

@cli.command()
@click.option("--method", type=click.Choice(["syflow", "beam-kl", "beam-mean", "cart", "dfs-mean", "rsd"]), default="syflow", help="Method to generate statistics for.")
def stats_tables(method):
    """Generate statistics tables for RQ2."""    
    df = load_experiment_results(f"results/RQ2/{method}/workload/stats")
    table(df, f"figures/out/rq2_stats_{method}_workload.tex")
    df = load_experiment_results(f"results/RQ2/{method}/distance_based/stats")
    table(df, f"figures/out/rq2_stats_{method}_distance_based.tex")
    df = pd.DataFrame()
    for dataset in ["workload", "distance_based"]:
        df_dataset = load_experiment_results(f"results/RQ2/{method}/{dataset}/stats")
        df = pd.concat([df, df_dataset])
    df = df[["num_samples", "num_features", "sgs_unique", "rule_num_features", "rule_length_min", "rule_length_max", "rule_length_mean", "sgs_mean_jaccard", "coverage_mean", "coverage_total"]]
    df = df.mean()
    df['casestudy'] = "Average"
    df = df.to_frame().T
    table(df, f"figures/out/rq2_stats_{method}_avg.tex")

@cli.command()
def plot_intro_example():
    '''PLot the example on the first page of the paper.'''
    results = pd.read_csv(f"results/RQ2/syflow/workload/subgroups/batik.csv")
    data = loaders.load("workload", "batik")
    subgroups, rules = reconstruct_subgroups_from_csv(data, results)
    #rules = ["$P$", "$\\neg V \\wedge \\neg P$", "$V \\wedge \\neg P \\wedge R \\neq 8294400$"]
    for i in range(len(rules)):
        rules[i] = rules[i].replace("validate", "\\textit{validate}")
        rules[i] = rules[i].replace("pdf", "\\textit{pdf}")
        rules[i] = rules[i].replace("resolution", "\\textit{resolution}")
        rules[i] = rules[i].replace("8294400.00", "8294400")
    plt.figure(figsize=(11, 1))
    plt.xlabel("Runtime (s)", fontsize=LABELFONTSIZE-2, labelpad=XLABELPAD-1)
    plt.ylabel("")
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_color("black")
    plt.grid(visible=False)
    # Remove default x-data margins to avoid empty space on both sides
    ax = plt.gca()
    ax.margins(x=0.025)
    
    bins = np.histogram(data["target"], bins=100)[1]
    plt.hist(data["target"], bins, color='#d8d8d8', label="$\\CMcal{C}$")
    which_sgs = [0, 2, 3]
    subgroups = [subgroups[i] for i in which_sgs]
    
    subgroup_histograms = []
    for sg in range(len(subgroups)):
        subgroup = data["target"][subgroups[sg]]
        hist, _ = np.histogram(subgroup, bins)
        subgroup_histograms.append(hist)
    
    bottom = np.zeros_like(subgroup_histograms[0])
    for hist, sg in zip(subgroup_histograms, range(len(subgroups))):
        fill_color = to_rgba(plt.cm.tab10.colors[sg], alpha=0.6)
        edge_color = to_rgba(plt.cm.tab10.colors[sg], alpha=0.9)
        plt.bar(
            bins[:-1], hist, width=np.diff(bins), bottom=bottom, 
            # darker edge color, but same color as fill
            label=rules[sg], align='edge', color=fill_color, edgecolor=edge_color
        )
    plt.tick_params(axis="x", labelsize=TICKFONTSIZE-2, bottom=True, width=0.7, length=3)
    plt.xticks(fontsize=10)
    plt.yticks([])
    # Legend on top, single row
    plt.legend(
        fontsize=LEGENDFONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=4,
        edgecolor="white",
        frameon=False,
        handletextpad=0.6,
        handlelength=1.0,
        columnspacing=2.5,
        borderaxespad=0.2,
    )
    plt.savefig(f"figures/out/intro_example.pdf", bbox_inches='tight', pad_inches=0.025, dpi=300)

def plot_subgroups(target, subgroups, rules, which_sgs, label_x, output_path, bins=200, legend=True, stack=True):  
    plt.figure(figsize=(6, 3))
    plt.xlabel(label_x, fontsize=14, labelpad=6)
    plt.ylabel("")
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_color('black')
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    
    bins = np.histogram(target, bins=bins)[1]
    plt.hist(target, bins, alpha=0.3, color='gray')
    
    subgroup_histograms = []
    for sg in which_sgs:
        subgroup = target[subgroups[sg]]
        hist, _ = np.histogram(subgroup, bins)
        subgroup_histograms.append(hist)
    
    bottom = np.zeros_like(subgroup_histograms[0])
    for hist, sg in zip(subgroup_histograms, which_sgs):
        fill_color = to_rgba(plt.cm.tab10.colors[sg], alpha=0.6)
        edge_color = to_rgba(plt.cm.tab10.colors[sg], alpha=0.9)
        plt.bar(
            bins[:-1], hist, width=np.diff(bins), bottom=bottom, 
            # darker edge color, but same color as fill
            label=rules[sg], align='edge', color=fill_color, edgecolor=edge_color
        )
        if stack:
            bottom += hist
    plt.xticks(fontsize=10)
    plt.yticks([])
    if legend:
        # put legend above the plot
        plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.2), edgecolor="white", ncol=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight', dpi=300)

def plot_result(method, dataset, casestudy):
    data = loaders.load(dataset, casestudy)
    results = pd.read_csv(f"results/RQ2/{method}/{dataset}/subgroups/{casestudy}.csv")
    subgroups, rules = reconstruct_subgroups_from_csv(data, results)
    unique_subgroups = get_unique_subgroups(subgroups)
    # get original index of the subgroups
    unique_rules = []
    for sg in unique_subgroups:
        unique_rules.append(rules[[i for i, s in enumerate(subgroups) if np.array_equal(sg, s)][0]])

    # do a new plot every 5 subgroups if the number of subgroups exceeds 5
    if len(unique_subgroups) > 5:
        for i in range(0, len(unique_subgroups), 5):
            rls = rules[i:i+5]
            sgs = unique_subgroups[i:i+5]
            plot_subgroups(
                data["target"], 
                sgs, 
                rls, 
                range(len(rls)), 
                label_x="Performance", 
                output_path=f"figures/out/rq2_real_world/{method}/{dataset}_{casestudy}_{i+1}-{i+len(rls)}", 
                bins=75, 
                stack=False
            )
    else:
        plot_subgroups(
            data["target"], 
            unique_subgroups, 
            rules, 
            range(len(unique_subgroups)), 
            label_x="Performance", 
            output_path=f"figures/out/rq2_real_world/{method}/{dataset}_{casestudy}", 
            bins=75, 
            stack=False
        )

@cli.command()
@click.option("--method", type=click.Choice(["syflow", "beam-kl", "beam-mean", "cart", "dfs-mean", "rsd"]), default="syflow", help="Method to plot real-world subgroups for.")
def plot_real_world_subgroups(method):
    '''Plot real-world subgroups found by the selected method.'''
    casestudies = db.loaders.keys()
    casestudies = [casestudy for casestudy in casestudies if casestudy not in ["lrzip", "x264"]]
    for casestudy in casestudies:
        plot_result(method, "distance_based", casestudy)

    casestudies = wl.loaders.keys()
    casestudies = [casestudy for casestudy in casestudies if casestudy not in ["lrzip"]]
    for casestudy in casestudies:
        plot_result(method, "workload", casestudy)

@cli.command()
@click.option("--method", type=click.Choice(["syflow", "beam-kl", "beam-mean", "cart", "dfs-mean", "rsd"]), default="syflow", help="Method to plot subgroup statistics for.")
def plot_distributional_characteristics(method):
    '''Plot distributional characteristics for subgroups found by the selected method.'''
    df = pd.DataFrame()
    for dataset in ["distance_based", "workload"]:
        if dataset == "distance_based":
            casestudies = db.loaders.keys()
            casestudies = [casestudy for casestudy in casestudies if casestudy not in ["lrzip", "x264"]]
        else:
            casestudies = wl.loaders.keys()
            casestudies = [casestudy for casestudy in casestudies if casestudy not in ["lrzip"]]
        for casestudy in casestudies:
            data = loaders.load(dataset, casestudy)
            results = pd.read_csv(f"results/RQ2/{method}/{dataset}/subgroups/{casestudy}.csv")
            subgroups, rules = reconstruct_subgroups_from_csv(data, results)
            unique_subgroups = get_unique_subgroups(subgroups)
            for subgroup in unique_subgroups:
                original_index = [i for i, sg in enumerate(subgroups) if np.array_equal(sg, subgroup)][0]
                subgroup_mean = np.mean(data["target"][subgroup])
                subgroup_std = np.std(data["target"][subgroup])
                mean_shift = (subgroup_mean - data["target"].mean()) / (data["target"].max() - data["target"].min())
                relative_standard_deviation = subgroup_std / data["target"].std()
                subgroup_hist, _ = np.histogram(data["target"][subgroup], bins=25, range=(data["target"].min(), data["target"].max()))
                skewness = skew(subgroup_hist)
                peaks, _ = find_peaks(subgroup_hist)
                if len(peaks) <= 1:
                    modality = "Unimodal"
                elif len(peaks) == 2:
                    modality = "Bimodal"
                else:
                    modality = "Multimodal"
                df = pd.concat([df, pd.DataFrame({
                    "dataset": dataset,
                    "casestudy": casestudy,
                    "rule": rules[original_index],
                    "sg_idx": original_index,
                    "mean_shift": mean_shift,
                    "rel_std": relative_standard_deviation,
                    "skewness": skewness,
                    "modality": modality
                }, index=[0])], ignore_index=True)
    df = pd.read_csv("results/RQ2/syflow/distributional_characteristics.csv")

    # Change: 3 subplots side by side
    fig = plt.figure(figsize=(11, 1.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1], wspace=0.2, hspace=0.0)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(
        data=df,
        x="mean_shift",
        ax=ax1,
        color=sns.color_palette(COLOR_PALETTE)[0],
        bins=16,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xlabel(r"Relative Mean Shift", fontsize=LABELFONTSIZE-2, labelpad=XLABELPAD-1)
    ax1.set_ylabel("")
    ax1.set_yticks([])
    ax1.set_xlim(-0.8, 0.8)
    ax1.tick_params(axis="x", labelsize=TICKFONTSIZE-2, bottom=True, width=0.7, length=3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_color("black")
    ax1.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    # Add counts on top of bars
    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.7,
                    f'{int(p.get_height())}', ha='center', va='bottom', fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(
        data=df,
        x="rel_std",
        ax=ax2,
        color=sns.color_palette(COLOR_PALETTE)[0],
        bins=16,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlabel(r"Relative Standard Deviation", fontsize=LABELFONTSIZE-2, labelpad=XLABELPAD-1)
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.tick_params(axis="x", labelsize=TICKFONTSIZE-2, bottom=True, width=0.7, length=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_color("black")
    ax2.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    # Add counts on top of bars
    for p in ax2.patches:
        if p.get_height() > 0:
            ax2.text(p.get_x() + p.get_width() / 2, p.get_height() + p.get_height() * 0.05 + 0.7,
                    f'{int(p.get_height())}', ha='center', va='bottom', fontsize=10)

    ax3 = fig.add_subplot(gs[0, 2])
    sns.countplot(
        data=df,
        x="modality",
        ax=ax3,
        color=sns.color_palette(COLOR_PALETTE)[0],
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_yticks([])
    ax3.tick_params(axis="x", labelsize=TICKFONTSIZE-2, rotation=45, bottom=True, width=0.7, length=3, pad=2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["bottom"].set_color("black")
    ax3.grid(visible=True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
    # Add counts on top of bars
    for p in ax3.patches:
        if p.get_height() > 0:
            ax3.text(p.get_x() + p.get_width() / 2, p.get_height() + p.get_height() * 0.05 + 0.7,
                    f'{int(p.get_height())}', ha='center', va='bottom', fontsize=10)
    plt.savefig("figures/out/rq2_distributional_characteristics.pdf", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    cli()