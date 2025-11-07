from dataLoader import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
from matplotlib.lines import Line2D

def wrap_title(title, width=25):
    return "\n".join(textwrap.wrap(title, width))

def compare_cdf(exp:Experiment,_filter:Filter,hue_name="version",y_name="repetitions_duration"):
    _filter.unique = hue_name
    exp.y_name =y_name
    df = exp.filter(_filter,explode_y=True)
    hue_ = df[hue_name].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[hue_name] == u_hue]
        sorted_data = np.sort(temp_df[exp.y_name])
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        plt.plot(sorted_data, cdf, marker=None, color=colors[i],label=u_hue)
    plt.xlabel(y_name)
    plt.ylabel('CDF')
    plt.title(f'Empirical CDF of {y_name}, comparing {hue_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def compare_densities(exp:Experiment,_filter:Filter,hue_name,title=""):
    _filter.unique = hue_name
    df = exp.filter(_filter,explode_y=True)
    palette = sns.color_palette("tab20", n_colors=df[hue_name].nunique())
    sns.kdeplot(
        data=df,
        x=exp.y_name,
        hue=hue_name,
        common_norm=False,
        palette=palette,
        fill=True,
        alpha=0.4,
        cut=0,
        bw_method="silverman",
        bw_adjust=0.5
    )
    plt.title(wrap_title(title+f"comparing {hue_name}",35))
    
    
def compare_boxplots(exp:Experiment,_filter:Filter,hue_name,x_name="repetitions"):
    _filter.unique = [hue_name,x_name]
    df = exp.filter(_filter,explode_y=True)
    palette = sns.color_palette("tab20", n_colors=df[hue_name].nunique())
    sns.boxplot(df, x=x_name,y="repetitions_duration",hue=hue_name,palette=palette,fill=False)
    plt.title(f"Benchmarking of repetition, comparing {x_name}")
    plt.xlabel(x_name)
    plt.ylabel("Duration of a kernel (ms)")
   
####
#
# Bootstrap Plots
#
####

def comparison_std_error(exp:Experiment,_filter:Filter,hue_name="version"):
    _filter.unique = [hue_name,"repetitions"]
    df = exp.filter(_filter,explode_y=False)
    hue_ = df[hue_name].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[hue_name] == u_hue]
        plt.plot(temp_df["repetitions"], temp_df["std_error"], marker='o', color=colors[i],label=u_hue)
    plt.xlabel('Nbr of Repetitions')
    plt.ylabel('Standard Error')
    plt.title(f'Evolution of Standard error of standard deviation, comparing {hue_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def interval_plot(ix,high,low,horizontal_line_width=0.25,color='#2187bb',stable=False,label=""):
    width =  horizontal_line_width / 2
    line_st = "-"
    if stable:
        line_st = ":"
    plt.plot([ix, ix], [high, low], color=color,linestyle=line_st)
    plt.plot([ix-width, ix+width], [high, high], color=color,linestyle=line_st)
    plt.plot([ix-width, ix+width], [low, low], color=color,linestyle=line_st)
    
def compare_std_std_precision(exp:Experiment,_filter:Filter,x_name="repetitions",hue_name="version",precision=0.0005):
    x_lbls = ['Precision']
    _filter.unique = [x_name,hue_name]
    df = exp.filter(_filter,explode_y=False)
    rep_arr = df[x_name].unique()
    x_lbls.extend(rep_arr)
    plt.xticks(ticks=np.arange(len(x_lbls)), labels=x_lbls) 
    middle = np.mean(((df["ci_high"] - df["ci_low"]) / 2) + df["ci_low"])
    interval_plot(0,middle + precision/2,middle - precision/2,color="red",stable=True)
    hue_ = df[hue_name].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[hue_name] == u_hue]
        for j in range(len(rep_arr)):
            temp_sq = temp_df[temp_df[x_name]==rep_arr[j]]
            if len(temp_sq) > 1:
                print(temp_sq)
                print("Multiple values where only one is wanted")
                raise ValueError()
            stable_ = False
            if (temp_sq.iloc[0].ci_high - temp_sq.iloc[0].ci_low) <=precision:
                stable_ = True
            interval_plot(j+1,temp_sq.iloc[0].ci_high, temp_sq.iloc[0].ci_low,color=colors[i],stable=stable_)  
    plt.title(wrap_title(f"Measure precision vs 90% confidence interval of the standard deviation, comparing {hue_name}",40))
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=hue_[i]) for i in range(len(hue_))]
    plt.legend(handles=legend_elements)
    
    
    
def line_plot_interest_points(x_line,y_line,x_points,y_points,line_color):
    plt.plot(x_line, y_line, color=line_color)
    plt.scatter(x_points, y_points, c="green")

def line_and_bloc(line_val:float,line_label:str,fill_x:list,fill_val:float,min:bool):
    plt.axhline(y=line_val, color='red', linestyle='--', label=f"{line_label}-min")
    if min:
        plt.fill_between(fill_x, fill_val, line_val, color='green', alpha=0.2)
    else:
        plt.fill_between(fill_x,line_val ,fill_val, color='green', alpha=0.2)

####
#
# Entropy Plots
#
####

def compare_entropy(exp:Experiment,_filter:Filter,hue_name="version"):
    _filter.unique = [hue_name]
    df = exp.filter(_filter,explode_y=False)
    batch_entropy = df.batch_entropy.unique()[0]
    batch_linear = df.batch_linear.unique()[0]
    repetitions = df.repetitions.unique()
    if len(repetitions) > 1:
        print(df)
        raise ValueError("Different repetitions")
    repetitions = repetitions[0]
    x_entropy = range(batch_entropy,repetitions,batch_entropy)
    hue_ = df[hue_name].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[hue_name] == u_hue]
        true_indexes = np.where(temp_df["stable_entropy"].iloc[0])[0]*batch_linear
        y_entropy = temp_df[exp.y_name].iloc[0]
        x_points = np.array(x_entropy)[true_indexes]
        y_points = np.array(y_entropy)[true_indexes]
        line_plot_interest_points(x_entropy,y_entropy,x_points,y_points,colors[i])
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=hue_[i]) for i in range(len(hue_))]
    plt.legend(handles=legend_elements)
    
    
def compare_r2(exp:Experiment,_filter:Filter,hue_name="version"):
    _filter.unique = [hue_name]
    df = exp.filter(_filter,explode_y=False)
    batch_entropy = df.batch_entropy.unique()[0]
    batch_linear = df.batch_linear.unique()[0]
    repetitions = df.repetitions.unique()
    if len(repetitions) > 1:
        print(df)
        raise ValueError("Different repetitions")
    repetitions = repetitions[0]
    x_linear = range(0,repetitions,batch_linear*batch_entropy)
    hue_ = df[hue_name].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    max_r2 = max([np.max(np.array(df.explode(["r2"])["r2"])),DEFAULT_R2]) * 1.10
    line_and_bloc(DEFAULT_R2,"r2",x_linear,max_r2,False)
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[hue_name] == u_hue]
        true_indexes = np.where(temp_df["stable_entropy"].iloc[0])[0]
        y_entropy = temp_df["r2"].iloc[0]
        x_points = np.array(x_linear)[true_indexes]
        y_points = np.array(y_entropy)[true_indexes]
        line_plot_interest_points(x_linear,y_entropy,x_points,y_points,colors[i])
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=hue_[i]) for i in range(len(hue_))]
    plt.legend(handles=legend_elements)
    
    
def compare_slope(exp:Experiment,_filter:Filter,hue_name="version"):
    _filter.unique = [hue_name]
    df = exp.filter(_filter,explode_y=False)
    batch_entropy = df.batch_entropy.unique()[0]
    batch_linear = df.batch_linear.unique()[0]
    repetitions = df.repetitions.unique()
    if len(repetitions) > 1:
        print(df)
        raise ValueError("Different repetitions")
    repetitions = repetitions[0]
    x_linear = range(0,repetitions,batch_linear*batch_entropy)
    hue_ = df[hue_name].unique()
    min_slope = 0
    line_and_bloc(DEFAULT_SLOPE,"slope",x_linear,min_slope,True)
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[hue_name] == u_hue]
        true_indexes = np.where(temp_df["stable_entropy"].iloc[0])[0]
        y_entropy = temp_df["slope"].iloc[0]
        x_points = np.array(x_linear)[true_indexes]
        y_points = np.array(y_entropy)[true_indexes]
        line_plot_interest_points(x_linear,y_entropy,x_points,y_points,colors[i])
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=hue_[i]) for i in range(len(hue_))]
    plt.legend(handles=legend_elements)