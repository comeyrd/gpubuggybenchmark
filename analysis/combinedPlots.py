import basePlots as bp
import dataLoader as dl
import matplotlib.pyplot as plt


    

#needed is y and hue
def combined_density_cdf(exp:dl.Experiment,filter_:dl.Filter,axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    plt.sca(axs[0])              
    bp.compare_densities(exp,filter_)
    plt.sca(axs[1])
    bp.compare_cdf(exp,filter_)

def combined_bootstrap(exp:dl.Experiment,filter_:dl.Filter,axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    plt.sca(axs[0])
    filter_.set_axis(filter_.on_x.axe,"repetitions_duration",filter_.on_hue.axe,True)   
    bp.compare_densities(exp,filter_) 
    plt.sca(axs[1])
    filter_.set_axis(filter_.on_x.axe,"bootstrap_data",filter_.on_hue.axe,True)   
    bp.compare_densities(exp,filter_) 
    plt.sca(axs[2])
    old_hue = filter_.on_hue.axe
    if old_hue == "repetitions":
        filter_.on_hue.axe = "version"
        filter_.on_hue.used = False
    filter_.set_axis("repetitions",filter_.on_y.axe,filter_.on_hue.axe)
    bp.compare_std_std_precision(exp,filter_)
    plt.sca(axs[3])
    bp.comparison_std_error(exp,filter_)
    return

def combined_entropy(exp:dl.Experiment,filter_:dl.Filter,axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    plt.sca(axs[0])
    filter_.set_axis("repetitions","entropy",filter_.on_hue.axe)
    bp.compare_entropy_types(exp,filter_)
    plt.sca(axs[1])
    filter_.set_axis("repetitions","r2",filter_.on_hue.axe,False)
    bp.compare_entropy_types(exp,filter_)
    plt.sca(axs[2])
    filter_.set_axis("repetitions","slope",filter_.on_hue.axe,False)
    bp.compare_entropy_types(exp,filter_)
    return 


def combined_entropy_bootstrap(exp:dl.Experiment,_filter:dl.Filter):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    combined_bootstrap(exp,_filter,axs[0])
    combined_entropy(exp,_filter,axs[1])
    return fig