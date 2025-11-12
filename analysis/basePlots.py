from enum import Enum
from typing import Tuple
from dataLoader import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
from matplotlib.lines import Line2D

def wrap_title(title, width=25)->Tuple[str,int]:
    title_ = "\n".join(textwrap.wrap(title, width))
    return title_, title_.count('\n')

def subtitle(_filter:Filter)->int:
    title,height = wrap_title(_filter.get_subtitle(),70)
    plt.text(0.5, 1.01, title, ha='center', va='bottom', transform=plt.gca().transAxes, fontsize=8)
    return height

def do_title(title,sub_height):
    title,title_height = wrap_title(title,35)
    plt.title(title,y=1.05 + 0.01*(title_height + sub_height))

## Filter : set hue_name and y_name
def compare_cdf(exp:Experiment,_filter:Filter):
    _filter.set_x(used=False)
    _filter.set_y(used=True)
    df = exp.filter(_filter)
    hue_ = df[_filter.on_hue.axe].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[_filter.on_hue.axe] == u_hue]
        sorted_data = np.sort(temp_df[_filter.on_y.axe])
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        plt.plot(sorted_data, cdf, marker=None, color=colors[i],label=u_hue)
    plt.xlabel(_filter.on_y.axe)
    plt.ylabel('CDF')
    title = f'Empirical CDF of {_filter.on_y.axe}'
    title +=f', comparing {_filter.on_hue.axe}'
    sub_height = subtitle(_filter)
    do_title(title,sub_height)
    if _filter.on_hue.used : 
      plt.legend()
    plt.grid(True)

## Filter : set hue_name and y_name
def compare_densities(exp:Experiment,_filter:Filter,title=""):
    _filter.set_x(used=False)
    _filter.set_y(used=True)
    df = exp.filter(_filter)
    palette = sns.color_palette("tab20", n_colors=df[_filter.on_hue.axe].nunique())
    sns.kdeplot(
        data=df,
        x=_filter.on_y.axe,
        hue=_filter.on_hue.axe,
        common_norm=False,
        palette=palette,
        fill=True,
        alpha=0.4,
        cut=0,
        bw_method="silverman",
        bw_adjust=0.5
    )
    title += f"Density function of {_filter.on_y.axe}"
    if _filter.on_hue.used:
        title+=f", comparing {_filter.on_hue.axe}"
    sub_height = subtitle(_filter)
    do_title(title,sub_height)
    plt.ylabel("Density")
    plt.xlabel(_filter.on_y.axe)

# set x, y and hue
def compare_boxplots(exp:Experiment,_filter:Filter):
    _filter.set_x(used=True)
    _filter.set_y(used=True)
    df = exp.filter(_filter)
    palette = sns.color_palette("tab20", n_colors=df[_filter.on_hue.axe].nunique())
    sns.boxplot(df, x=_filter.on_x.axe,y=_filter.on_y.axe,hue=_filter.on_hue.axe,palette=palette,fill=False)
    title = f"Benchmarking of {_filter.on_x.axe}"
    if _filter.on_hue.used:
        title+=f", comparing {_filter.on_hue.axe}"
    sub_height = subtitle(_filter)
    do_title(title,sub_height)
    plt.xlabel(_filter.on_x.axe)
    plt.ylabel(_filter.on_y.axe)
   
def line_plot(exp:Experiment,_filter:Filter):
    _filter.set_x(used=True)
    _filter.set_y(used=True)
    df = exp.filter(_filter)
    hue_ = df[_filter.on_hue.axe].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[_filter.on_hue.axe] == u_hue]
        plt.plot(temp_df[_filter.on_x.axe], temp_df[_filter.on_y.axe], marker='o', color=colors[i],label=u_hue)
    plt.xlabel(_filter.on_x.axe)
    plt.ylabel(_filter.on_y.axe)
    title = f'Evolution of {_filter.on_y.axe}'
    if _filter.on_hue.used:
        title+=f", comparing {_filter.on_hue.axe}"
    sub_height = subtitle(_filter)
    do_title(title,sub_height)
    if _filter.on_hue.used : 
      plt.legend()
    plt.grid(True)

####
#
# Bootstrap Plots
#
####

#Filter, set hue
def comparison_std_error(exp:Experiment,_filter:Filter):
    in_filter = _filter.copy()
    in_filter.set_x("repetitions")
    in_filter.set_y("std_error",explode=False)
    line_plot(exp,in_filter)

def interval_plot(ix,high,low,horizontal_line_width=0.25,color='#2187bb',stable=False,label=""):
    width =  horizontal_line_width / 2
    line_st = "-"
    if stable:
        line_st = ":"
    plt.plot([ix, ix], [high, low], color=color,linestyle=line_st)
    plt.plot([ix-width, ix+width], [high, high], color=color,linestyle=line_st)
    plt.plot([ix-width, ix+width], [low, low], color=color,linestyle=line_st)

# set x and hue, comparing the std of the std against the measure precision, Y is ci_high and ci_low
def compare_std_std_precision(exp:Experiment,_filter:Filter,precision=0.0005):
    x_lbls = ['Precision']
    _filter.set_x(used=True)
    _filter.set_y(used=False)
    df = exp.filter(_filter)
    rep_arr = df[_filter.on_x.axe].unique()
    x_lbls.extend(rep_arr)
    plt.xticks(ticks=np.arange(len(x_lbls)), labels=x_lbls) 
    middle = np.mean(((df["ci_high"] - df["ci_low"]) / 2) + df["ci_low"])
    interval_plot(0,middle + precision/2,middle - precision/2,color="red",stable=True)
    hue_ = df[_filter.on_hue.axe].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[_filter.on_hue.axe] == u_hue]
        for j in range(len(rep_arr)):
            temp_sq = temp_df[temp_df[_filter.on_x.axe]==rep_arr[j]]
            if len(temp_sq) > 1:
                print(temp_sq)
                print("Multiple values where only one is wanted")
                raise ValueError()
            stable_ = False
            if (temp_sq.iloc[0].ci_high - temp_sq.iloc[0].ci_low) <=precision:
                stable_ = True
            interval_plot(j+1,temp_sq.iloc[0].ci_high, temp_sq.iloc[0].ci_low,color=colors[i],stable=stable_)  
    title = "Measure precision vs 90% confidence interval of the standard deviation"
    if _filter.on_hue.used : 
        title += f", comparing {_filter.on_hue.axe}"
    sub_height = subtitle(_filter)
    do_title(title,sub_height)
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=hue_[i]) for i in range(len(hue_))]
    plt.xlabel(_filter.on_x.axe)
    plt.ylabel("Confidence Interval")
    if _filter.on_hue.used : 
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

#set y to either "slope", "entropy", or "r2"
def compare_entropy_types(exp:Experiment,_filter:Filter):
    if _filter.on_y.axe not in ["entropy","slope","r2"] :
        raise ValueError('Filter on Y axis is not properly set, set to either "entropy","slope","r2"')
    in_filter = _filter.copy()
    in_filter.on_y.explode = False
    in_filter.set_x(used=False)
    in_filter.set_y(used=True)
    df = exp.filter(in_filter)
    batch_entropy = df.batch_entropy.unique()[0]
    batch_linear = df.batch_linear.unique()[0]
    repetitions = df.repetitions.unique()
    if len(repetitions) > 1:
        print(df)
        raise ValueError("Different repetitions")
    repetitions = repetitions[0]
    x = range(batch_entropy,repetitions,batch_entropy)
    if in_filter.on_y.axe in ["slope","r2"]:
            x = range(0,repetitions,batch_linear*batch_entropy)
    hue_ = df[in_filter.on_hue.axe].unique()
    colors = sns.color_palette("tab10", n_colors=len(hue_))
    if in_filter.on_y.axe == "r2" : 
        max_r2 = max([np.max(np.array(df.explode(["r2"])["r2"])),DEFAULT_R2]) * 1.10
        line_and_bloc(DEFAULT_R2,in_filter.on_y.axe,x,max_r2,False)
    elif in_filter.on_y.axe == "slope":
        min_slope = 0
        line_and_bloc(DEFAULT_SLOPE,in_filter.on_y.axe,x,min_slope,True)
    for i, u_hue in enumerate(hue_):
        temp_df = df[df[in_filter.on_hue.axe] == u_hue]
        true_indexes = np.where(temp_df["stable_entropy"].iloc[0])[0]
        if in_filter.on_y.axe not in ["slope","r2"]:
            true_indexes*= batch_linear
        y = temp_df[in_filter.on_y.axe].iloc[0]
        x_points = np.array(x)[true_indexes]
        y_points = np.array(y)[true_indexes]
        line_plot_interest_points(x,y,x_points,y_points,colors[i])
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=hue_[i]) for i in range(len(hue_))]
    title = f'Evolution of {in_filter.on_y.axe}'
    if in_filter.on_hue.used:
        title+=f", comparing {in_filter.on_hue.axe}"
    sub_height = subtitle(_filter)
    do_title(title,sub_height)
    plt.xlabel(in_filter.on_x.axe)
    plt.ylabel(in_filter.on_y.axe)
    if in_filter.on_hue.used : 
      plt.legend(handles=legend_elements)