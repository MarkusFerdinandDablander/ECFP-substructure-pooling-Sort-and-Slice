# import packages

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np



def seaborn_scatterplot_for_qsar_ac_study_results(target,
                                                  task_x,
                                                  metric_x,
                                                  task_y,
                                                  metric_y,
                                                  mol_repr_list,
                                                  regr_type_list,
                                                  A_dict,
                                                  task_name_dict,
                                                  metric_name_dict,
                                                  decimals_mean = 6,
                                                  decimals_std = 6,
                                                  plot_legend = True,
                                                  legend_loc = "lower left",
                                                  plot_title = True,
                                                  plot_x_label = True,
                                                  plot_y_label = True,
                                                  plot_x_ticks = True,
                                                  plot_y_ticks = True,
                                                  plot_error_bars = True,
                                                  x_tick_stepsize = "auto",
                                                  y_tick_stepsize = "auto",
                                                  xlim = None,
                                                  ylim = None,
                                                  size = 12, 
                                                  linear_regression = False):
    
    # preallocate pandas dataframe
    df = pd.DataFrame(columns = ["x_mean", "x_std", "y_mean", "y_std", "mol_repr", "regr_type"])

    # extract means and stds of experimental results
    for mol_repr in mol_repr_list:
        for regr_type in regr_type_list:
            
            A_x = A_dict[(target, mol_repr, regr_type, task_x, metric_x)]
            A_y = A_dict[(target, mol_repr, regr_type, task_y, metric_y)]

            x_mean = np.around(np.nanmean(np.nanmean(A_x, axis = 1), axis = 0), decimals = decimals_mean)
            x_std = np.around(np.nanmean(np.nanstd(A_x, axis = 1), axis = 0), decimals = decimals_std)
            
            y_mean = np.around(np.nanmean(np.nanmean(A_y, axis = 1), axis = 0), decimals = decimals_mean)
            y_std = np.around(np.nanmean(np.nanstd(A_y, axis = 1), axis = 0), decimals = decimals_std)
            
            df.loc[len(df)] = [x_mean, x_std, y_mean, y_std, mol_repr, regr_type]
            
    # drop rows which contain nan values (happens for precision when ecfp + knn has no positive predictions for any of the mk trials)
    df = df.dropna()

    # plot results with seaborn
    sns.set(rc={"xtick.bottom" : True,
                "xtick.major.size": size/3,
                "xtick.major.width": size/12,
                "ytick.left" : True,
                "ytick.major.size": size/3,
                "ytick.major.width": size/12,
                "axes.edgecolor":"black", 
                "axes.linewidth": size/15, 
                "font.family": ["sans-serif"], 
                "grid.linewidth": size/8}, 
            style = "darkgrid")

    
    plt.figure(figsize=(size*(2/3), size*(2/3)))

    mol_repr_colour_dict = {"cfp" : "red", "md" : "blue", "mg_gin" : "violet"}
    regr_type_marker_dict = {"rf": "s", "knn": "d", "mlp": "o"}
    
    mol_repr_name_dict = {"cfp" : "ECFP", "md" : "MD", "mg_gin" : "GIN"}
    regr_type_name_dict = {"rf": "RF", "knn": "kNN", "mlp": "MLP"}
    
    sns.scatterplot(data = df,
                    x = "x_mean", 
                    y = "y_mean",
                    hue = "mol_repr",
                    palette = mol_repr_colour_dict,
                    style = "regr_type", 
                    markers = regr_type_marker_dict, 
                    s = 1.4*size**2,
                    linewidth = 0,
                    legend = plot_legend)
    
    if plot_legend == True:
        
        custom = []
        symbol_name_list = []
        for mol_repr in mol_repr_list:
            for regr_type in regr_type_list:
                custom.append(Line2D([], [], marker = regr_type_marker_dict[regr_type], color = mol_repr_colour_dict[mol_repr], linestyle='None'))
                symbol_name_list.append(mol_repr_name_dict[mol_repr] + " + " + regr_type_name_dict[regr_type])

        plt.legend(custom, 
                   symbol_name_list, 
                   loc = legend_loc, 
                   markerscale = size/6 - 1/4, 
                   scatterpoints = 1, 
                   fontsize = 1.1*size)
    
    if plot_title == True:
    
        plt.title(r"Pearson's $r$ = " + str(np.round(np.corrcoef(df["x_mean"], df["y_mean"])[0,1], decimals = 2)),
                  fontsize = 1.1*size, 
                  pad = size*(2/3))
    
    if plot_x_label == True:
    
        plt.xlabel(metric_name_dict[metric_x] + ": " + task_name_dict[task_x] + " (AC-Classification)", 
                   labelpad = size, 
                   fontsize = 1.1*size)
    else:
        plt.xlabel("")
        
    if plot_y_label == True:
    
        plt.ylabel(metric_name_dict[metric_y] + ": " + task_name_dict[task_y] + " (QSAR-Prediction)", 
                   labelpad = size, 
                   fontsize = 1.1*size)
    else:
        plt.ylabel("")
    
    
    if plot_x_ticks == True:
        if x_tick_stepsize == "auto":
            plt.xticks(fontsize = 1.1*size)
        else:
            plt.xticks(np.arange(0, 1, x_tick_stepsize), fontsize = 1.1*size)
    else:
        if x_tick_stepsize == "auto":
            plt.xticks(fontsize = 0)
        else:
            plt.xticks(np.arange(0, 1, x_tick_stepsize), fontsize = 0)
    
    if plot_y_ticks == True:
        if y_tick_stepsize == "auto":
            plt.yticks(fontsize = 1.1*size)
        else:
            plt.yticks(np.arange(0, 1, y_tick_stepsize), fontsize = 1.1*size)
    else:
        if y_tick_stepsize == "auto":
            plt.yticks(fontsize = 0)
        else:
            plt.yticks(np.arange(0, 1, y_tick_stepsize), fontsize = 0)
    
    if plot_error_bars == True:
        
        plt.errorbar(df["x_mean"], 
                     df["y_mean"], 
                     xerr = df["x_std"], 
                     yerr = df["y_std"], 
                     ls = "none", 
                     ecolor = "black", 
                     lw = size/15, 
                     capsize = size/3, capthick = size/15)
        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.tight_layout()
    
    plt.savefig("scatter.svg")
    
    if linear_regression == True:
    
        # create linear least squares polynomial
        line_coeffs = np.polyfit(df["x_mean"], df["y_mean"], deg = 1)
        line_grid = np.linspace(0, 1, 100)
        line_vals = np.polyval(line_coeffs, line_grid)
        plt.plot(line_grid, line_vals)

        print("line_coeffs = (k, d) = ", line_coeffs)
        print("line(mcc = 1) = ", line_coeffs[0] + line_coeffs[1])
    
    plt.show()
    
    
    
def seaborn_scatterplot_for_qsar_ac_cfp_mlp_var_eps_study_results(target,
                                                                  task_x,
                                                                  metric_x,
                                                                  task_y,
                                                                  metric_y,
                                                                  epoch_list,
                                                                  B_dict,
                                                                  task_name_dict,
                                                                  metric_name_dict,
                                                                  decimals_mean = 6,
                                                                  decimals_std = 6,
                                                                  plot_legend = True,
                                                                  legend_loc = "lower left",
                                                                  plot_title = True,
                                                                  plot_x_label = True,
                                                                  plot_y_label = True,
                                                                  plot_x_ticks = True,
                                                                  plot_y_ticks = True,
                                                                  plot_error_bars = True,
                                                                  x_tick_stepsize = "auto",
                                                                  y_tick_stepsize = "auto",
                                                                  xlim = None,
                                                                  ylim = None,
                                                                  size = 12):
    
    # preallocate pandas dataframe
    df = pd.DataFrame(columns = ["x_mean", "x_std", "y_mean", "y_std", "epoch"])

    # extract means and stds of experimental results
    for epoch in epoch_list:

        B_x = B_dict[(target, task_x, metric_x, epoch)]
        B_y = B_dict[(target, task_y, metric_y, epoch)]

        x_mean = np.around(np.nanmean(np.nanmean(B_x, axis = 1), axis = 0), decimals = decimals_mean)
        x_std = np.around(np.nanmean(np.nanstd(B_x, axis = 1), axis = 0), decimals = decimals_std)

        y_mean = np.around(np.nanmean(np.nanmean(B_y, axis = 1), axis = 0), decimals = decimals_mean)
        y_std = np.around(np.nanmean(np.nanstd(B_y, axis = 1), axis = 0), decimals = decimals_std)

        df.loc[len(df)] = [x_mean, x_std, y_mean, y_std, epoch]

    # plot results with seaborn
    sns.set(rc={"xtick.bottom" : True,
                "xtick.major.size": size/3,
                "xtick.major.width": size/12,
                "ytick.left" : True,
                "ytick.major.size": size/3,
                "ytick.major.width": size/12,
                "axes.edgecolor":"black", 
                "axes.linewidth": size/15, 
                "font.family": ["sans-serif"], 
                "grid.linewidth": size/8}, 
            style = "darkgrid")

    
    plt.figure(figsize=(size*(2/3), size*(2/3)))

    sns.scatterplot(data = df,
                    x = "x_mean", 
                    y = "y_mean",
                    hue = "epoch",
                    s = 1.4*size**2,
                    linewidth = 0, 
                    legend = plot_legend)
    
    if plot_legend == True:
        
        plt.legend(title = "Epochs", 
                   loc = legend_loc,
                   markerscale = size/6 - 1/4, 
                   scatterpoints = 1,
                   fontsize = 1.1*size, 
                   title_fontsize = 1.1*size)
    
    if plot_title == True:
    
        plt.title("1024 Bit ECFP4 + MLP: Performance during Training",
                  fontsize = 1.1*size, 
                  pad = size*(2/3))
    
    if plot_x_label == True:
    
        plt.xlabel(task_name_dict[task_x] + " (" + metric_name_dict[metric_x] + ")", 
                   labelpad = size, 
                   fontsize = 1.1*size)
    else:
        plt.xlabel("")
        
    if plot_y_label == True:
    
        plt.ylabel(task_name_dict[task_y] + " (" + metric_name_dict[metric_y] + ")", 
                   labelpad = size, 
                   fontsize = 1.1*size)
    else:
        plt.ylabel("")
    
    
    if plot_x_ticks == True:
        if x_tick_stepsize == "auto":
            plt.xticks(fontsize = 1.1*size)
        else:
            plt.xticks(np.arange(0, 1, x_tick_stepsize), fontsize = 1.1*size)
    else:
        if x_tick_stepsize == "auto":
            plt.xticks(fontsize = 0)
        else:
            plt.xticks(np.arange(0, 1, x_tick_stepsize), fontsize = 0)
    
    if plot_y_ticks == True:
        if y_tick_stepsize == "auto":
            plt.yticks(fontsize = 1.1*size)
        else:
            plt.yticks(np.arange(0, 1, y_tick_stepsize), fontsize = 1.1*size)
    else:
        if y_tick_stepsize == "auto":
            plt.yticks(fontsize = 0)
        else:
            plt.yticks(np.arange(0, 1, y_tick_stepsize), fontsize = 0)
    
    if plot_error_bars == True:
        
        plt.errorbar(df["x_mean"], 
                     df["y_mean"], 
                     xerr = df["x_std"], 
                     yerr = df["y_std"], 
                     ls = "none", 
                     ecolor = "black", 
                     lw = size/15, 
                     capsize = size/3, capthick = size/15)
        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.tight_layout()
    
    plt.savefig("scatter.svg")
    
    plt.show()