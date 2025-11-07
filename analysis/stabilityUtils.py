import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as scistats
from tqdm import tqdm
import textwrap

def wrap_title(title, width=25):
    return "\n".join(textwrap.wrap(title, width))

def compare_densities(dataframe,x_name="repetitions_duration",hue_name="version"):
    palette = sns.color_palette("tab10", n_colors=dataframe[hue_name].nunique())
    sns.displot(
    data=dataframe,
    x=x_name,
    hue=hue_name,
    stat="probability",
    binwidth=0.0005,
    palette=palette,
    alpha=0.7,
    kind="hist",
    multiple="layer",   # or "stack"
    row=hue_name,       # <— key: each group gets its own subplot + y-axis
    facet_kws={'sharey': False, 'sharex': True}
)
    plt.title("Distribution of the execution time")

def plot_comparing_two_versions_single_repetitions(dataframe:pd.DataFrame,warmup=5,repetitions=1000,versions=["ReferenceAccuracy","BCAccuracy"]):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    temp_df = dataframe[dataframe["warmups"]==warmup]
    temp_df = dataframe[dataframe["repetitions"]==repetitions]
    temp_df = temp_df[temp_df["version"].isin(versions)]
    temp_df = temp_df.drop_duplicates(subset=["repetitions","version"], keep="first")
    temp_df_exploded = temp_df.explode('repetitions_duration')
    fig.suptitle(f"Comparing versions {', '.join(versions)} with repetitions {repetitions}", fontsize=14)

    plt.sca(axs[0])              # Make this axes the active one
    #compare_densities(temp_df_exploded,x_name="repetitions_duration",hue_name="version")
    plot_multiple_densities(temp_df_exploded,hue="version",x_name="repetitions_duration")
    plt.sca(axs[1])
    compare_two_cdf(temp_df_exploded,x_name="repetitions_duration",hue_name="version")
    

    return fig


def interval_plot(ix,high,low,horizontal_line_width=0.25,color='#2187bb'):
    width =  horizontal_line_width / 2
    plt.plot([ix, ix], [high, low], color=color)
    plt.plot([ix-width, ix+width], [high, high], color=color)
    plt.plot([ix-width, ix+width], [low, low], color=color)
    
    
    
def compare_std_std_precision(dataframe,precision=0.0005):
    lbls = ['Precision']
    lbls.extend(dataframe["repetitions"].astype(str).tolist())
    plt.xticks(ticks=np.arange(len(lbls)), labels=lbls) 
    middle = np.mean(((dataframe["ci_high"] - dataframe["ci_low"]) / 2) + dataframe["ci_low"])
    interval_plot(0,middle + precision/2,middle - precision/2,color="red")
    for i in range(len(dataframe)):
        color_ = "blue"
        if (dataframe.iloc[i].ci_high - dataframe.iloc[i].ci_low) <=precision:
            color_ = "green"
        interval_plot(i+1,dataframe.iloc[i].ci_high, dataframe.iloc[i].ci_low,color=color_)  
        plt.title(wrap_title("Measure precision vs 90% confidence interval of the standard deviation",40))
    plt.leg
        
def plot_compare_repetitions_single_version(dataframe:pd.DataFrame,warmup=5,version="ReferenceAccuracy"):
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), constrained_layout=True)
    temp_df = dataframe[dataframe["warmups"]==warmup]
    temp_df = temp_df[temp_df["version"].str.contains(version)]
    temp_df = temp_df.drop_duplicates(subset=["repetitions","version"], keep="first")
    temp_df_exploded = temp_df.explode('repetitions_duration')
    temp_df_exploded['weight'] = temp_df_exploded.groupby('repetitions')['repetitions_duration'].transform(lambda x: 1/len(x))
    plt.sca(axs[0])              # Make this axes the active one
    plot_multiple_densities(dataframe=temp_df_exploded,hue="repetitions",weights_='weight',title="Distribution of the execution time")
    
    temp_df_exploded_boot = temp_df.explode('boostrap_data')
    plt.sca(axs[1])
    plot_multiple_densities(temp_df_exploded_boot,hue="repetitions",x_name='boostrap_data',title="Distribution of boostrapped standard deviation")

    plt.sca(axs[2])
    compare_std_std_precision(temp_df)
    plt.sca(axs[3])
    plot_evolution_std_error(temp_df)
    fig.suptitle(f"Comparing Kernel execution time with different number of repetitions for Kernel {version}", fontsize=16)
    return fig


def plot_compare_repetitions_multiple_version(dataframe:pd.DataFrame,warmup=5):
    dataframe_w = dataframe[dataframe["warmups"]==warmup]
    versions = dataframe_w["version"].unique()
    nb_rows = len(versions)
    fig, axs = plt.subplots(nb_rows, 4, figsize=(16, nb_rows*3), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for i in range(nb_rows):
        temp_df = dataframe_w[dataframe_w["version"].str.contains(versions[i])].copy()
        temp_df = temp_df.drop_duplicates(subset=["repetitions","version"], keep="first")
        temp_df_exploded = temp_df.explode('repetitions_duration')
        temp_df_exploded['weight'] = temp_df_exploded.groupby('repetitions')['repetitions_duration'].transform(lambda x: 1/len(x))
        plt.sca(axs[i][0])              # Make this axes the active one
        plot_multiple_densities(dataframe=temp_df_exploded,hue="repetitions",weights_='weight',title="Distribution of the execution time")
        
        temp_df_exploded_boot = temp_df.explode('boostrap_data')
        plt.sca(axs[i][1])
        plot_multiple_densities(temp_df_exploded_boot,hue="repetitions",x_name='boostrap_data',title="Distribution of boostrapped standard deviation")

        plt.sca(axs[i][2])
        compare_std_std_precision(temp_df)
        plt.sca(axs[i][3])
        plot_evolution_std_error(temp_df)
        #fig.suptitle(f"Comparing Kernel execution time with different number of repetitions for Kernel {versions[i]}", fontsize=16)
        # --- Adjust axes per column ---
    for col in range(4):
        # Compute global min/max for this column
        ymin, ymax = float('inf'), float('-inf')
        for row in range(nb_rows):
            current_ax = axs[row][col]
            ylims = current_ax.get_ylim()
            ymin = min(ymin, ylims[0])
            ymax = max(ymax, ylims[1])
        # Apply same limits to all axes in this column
        for row in range(nb_rows):
            axs[row][col].set_ylim(ymin, ymax)
    for i in range(nb_rows):
        fig.text(0.02, 1 - (i + 0.05)/nb_rows, f"Kernel {versions[i]}", va='center', ha='left', fontsize=12, fontweight='bold')
    return fig

def plot_compare_repetitions_all(df_experiment_boostrap):
    for version in df_experiment_boostrap["version"].unique():
        fig = plot_compare_repetitions_single_version(df_experiment_boostrap,version=version)
        plt.show()
        plt.close(fig)