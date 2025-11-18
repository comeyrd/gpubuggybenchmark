import basePlots as bp
import dataLoader as dl
import matplotlib.pyplot as plt
import numpy as np
    

#needed is y and hue
def combined_density_cdf(exp:dl.Experiment,filter_:dl.Filter,axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    plt.sca(axs[0])              
    bp.compare_densities(exp,filter_)
    plt.sca(axs[1])
    bp.compare_cdf(exp,filter_)

def combined_bootstrap(exp:dl.Experiment,filter_:dl.Filter,axs=None):
    in_filter = filter_.copy()
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    plt.sca(axs[0])
    in_filter.set_y("repetitions_duration",explode=True)
    bp.compare_densities(exp,in_filter) 
    plt.sca(axs[1])
    in_filter.set_y("bootstrap_data",explode=True)
    bp.compare_densities(exp,in_filter) 
    plt.sca(axs[2])
    old_hue = in_filter.on_hue.axe
    if old_hue == "repetitions":
        in_filter.set_hue("version",used=False)
    in_filter.set_x("repetitions")
    bp.compare_std_std_precision(exp,in_filter)
    plt.sca(axs[3])
    bp.comparison_std_error(exp,in_filter)
    in_filter.set_hue(old_hue,used=True)
    return

def combined_entropy(exp:dl.Experiment,filter_:dl.Filter,axs=None):
    in_filter = filter_.copy()
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    plt.sca(axs[0])
    in_filter.set_hue(used=False)
    in_filter.set_x("repetitions")
    in_filter.set_y("entropy",explode=True)
    bp.compare_entropy_types(exp,in_filter)
    plt.sca(axs[1])
    in_filter.set_y("r2",explode=False)
    bp.compare_entropy_types(exp,in_filter)
    plt.sca(axs[2])
    in_filter.set_y("slope",explode=False)
    bp.compare_entropy_types(exp,in_filter)
    return 


def combined_entropy_bootstrap(exp:dl.Experiment,_filter:dl.Filter):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    combined_bootstrap(exp,_filter,axs[0])
    combined_entropy(exp,_filter,axs[1])
    return fig

#subplot needs to be set
def combined_subplots_bootstrap(exp: dl.Experiment, filter_:dl.Filter):
    """
   Put in filter subplot the value you want to use for comparison
    """    
    # Get unique versions
    in_filter = filter_.copy()
    temp_df = exp.filter(in_filter)
    subplots_t = temp_df[in_filter.on_subplot.axe].unique()
    nb_rows = len(subplots_t)
    in_filter.set_subplot(used=False)

    # Create subplot grid
    fig, axs = plt.subplots(nb_rows, 4, figsize=(18, nb_rows*5), constrained_layout=True)
    axs = np.atleast_2d(axs)
    
    for i in range(nb_rows):
        # Create filter for this version
        in_filter.set_subplot(used=False)
        in_filter.set_default(in_filter.on_subplot.axe,subplots_t[i])
        
        combined_bootstrap(exp, in_filter, axs[i])
    
    for col in range(2,4):
        ymin, ymax = float('inf'), float('-inf')
        for row in range(nb_rows):
            current_ax = axs[row][col]
            ylims = current_ax.get_ylim()
            ymin = min(ymin, ylims[0])
            ymax = max(ymax, ylims[1])
        
        for row in range(nb_rows):
            axs[row][col].set_ylim(ymin, ymax)
    
    for i in range(nb_rows):
        # Add text centered above the row
        axs[i][0].text(0.5, 1.3,  # Position above middle of row
                    f"{in_filter.on_subplot.axe}: {subplots_t[i]}", 
                    transform=axs[i][0].transAxes,  # Use middle subplot
                    va='bottom', ha='center', 
                    fontsize=12, fontweight='bold')
    return fig


# Set X and hue
def combined_ecdf_scores(exp:dl.Experiment,filter_:dl.Filter,axs=None):
    in_filter = filter_.copy()
    if axs is None:
        fig, axs = plt.subplots(1, len(dl.EcdfMetrics.METRICS_NAMES), figsize=(len(dl.EcdfMetrics.METRICS_NAMES)*4, 4), constrained_layout=True)
    for i,name in enumerate(dl.EcdfMetrics.METRICS_NAMES):
        plt.sca(axs[i])
        in_filter.set_y(name,explode=False)
        bp.line_plot(exp,in_filter)