import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



def extract_info_from_filename(filename):
    
    """
    Uses filename to extract information of the form 
    (fingerprint_dimension, used_invariants, fingerprint_diameter, used_pooling_method, used_ml_method)
    """
    
    (fp_info, ml_info) = filename.split("_")[2:4]
    ml_info = ml_info.split(".")[0]

    match = re.match(r"([0-9]+)([a-z]+)([0-9]+)([a-z0-9]+)", fp_info, re.I)

    return (int(match.groups()[0]), str(match.groups()[1]), int(match.groups()[2]), str(match.groups()[3]), str(ml_info))



def visualise_bar_charts(dataset_name, split_type, metric, y_unit = ""):
    
    # load results
    path = "results/" + dataset_name + "/" + split_type + "/"
    results_dict = {}

    for filename in os.listdir(path):
        if filename[-3:] == "csv":
            
            (dim, invs, diam, pool, ml) = extract_info_from_filename(filename)
            df_current = pd.read_csv(path + filename, sep = ",")
            
            mean = df_current[metric].values[-2]
            std = df_current[metric].values[-1]
            results_dict[(dim, invs, diam, pool, ml)] = (mean, std)
    
    # initialise figure
    fig, axes = plt.subplots(2, 1, figsize = (16.0, 10.5))
    axes = axes.flatten()
    
    # create dataset name dict
    dataset_name_dict = {"moleculenet_lipophilicity": "MoleculeNet Lipophilicity", 
                         "aqsoldb_solubility": "AqSolDB Aqueous Solubility",
                         "postera_sars_cov_2_mpro": "COVID Moonshot SARS-CoV-2 Main Protease Inhibition",
                         "lit_pcba_esr_ant": "LIT-PCBA Estrogen Receptor Alpha Antagonism",
                         "lit_pcba_tp53": "LIT-PCBA Cellular Tumor Antigen TP53 Inhibition",
                         "ames_mutagenicity": "Ames Mutagenicity"}
    
    # add title
    fig.suptitle(dataset_name_dict[dataset_name], fontsize = 14, x = 0.52, y = 1.005, fontweight = "bold")
    
    # compute limits for y-axis
    y_max = max([v[0] for v in results_dict.values()])
    y_min = min([v[0] for v in results_dict.values()])
    
    for (i, ml) in enumerate(["rf", "mlp"]):
        
        # extract bar heights, error bar lengths and x_tick_labels
        means = []
        stds = []
        x_tick_labels = []
        dim_list = [512, 1024, 2048, 4096]
        
        for dim in dim_list:
            for invs in ["ecfp", "fcfp"]:
                for diam in [2, 4, 6]:
                    
                    x_tick_labels.append(invs.upper() + str(diam))
                    
                    for pool in ["hashed", "filtered", "mim", "sort_and_slice"]:
                        
                        means.append(results_dict[(dim, invs, diam, pool, ml)][0])
                        stds.append(results_dict[(dim, invs, diam, pool, ml)][1])
        
        # define axis object for current ml model
        ax = axes[i]
        
        # construct positioning of bars and barwidth
        barwidth = 1.3
        small_gaps = 1
        large_gaps = 2.4
        
        x_pos = [0]
        for gap_ind in range(1, len(means)):

            if gap_ind % 4 == 1:
                gap = barwidth
            elif gap_ind % 4 == 0 and gap_ind % 24 != 0:
                gap = barwidth + small_gaps
            elif gap_ind % 4 == 0 and gap_ind % 24 == 0:
                gap = barwidth + large_gaps
            x_pos.append(x_pos[-1] + gap)

        # plot bar charts
        color_1 = "silver"
        color_2 = "violet"
        color_3 = "lightsalmon"
        color_4 = "forestgreen"
        
        ax.bar(x = x_pos, 
               height = means,
               width = barwidth,
               color = [color_1, color_2, color_3, color_4]*int(len(means)/4), 
               edgecolor = "black", 
               linewidth = 1.5, 
               alpha = 1, 
               yerr = stds, 
               capsize = 5)
        
        # set xlim and ylim
        ax.set_xlim(x_pos[0] - 2, x_pos[-1] + 2)
        ax.set_ylim(bottom = y_min - 0.005, top = y_max + 0.005)

        # create and position x-ticks
        x_ticks_positions = [(x_pos[4*k] + x_pos[4*k+3])/2 for k in range(int(len(means)/4))]
        ax.set_xticks(x_ticks_positions)
        ax.set_xticklabels(x_tick_labels, fontsize = 14, rotation = 45, ha = "center")
        
        # create and position fingerprint dim labels on top of figure
        ax_2 = ax.twiny()
        ax_2.set_xlim(x_pos[0] - 1, x_pos[-1] + 1)
        dim_positions = []
        for k in range(4):
            dim_positions.append((x_ticks_positions[2 + 6*k] + x_ticks_positions[3 + 6*k])/2)
        ax_2.set_xticks(dim_positions)
        ax_2.set_xticklabels([str(dim) + "-Bit" for dim in dim_list], fontsize = 14, rotation = 0, ha = "center")
        
        # customise y-ticks
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
        
        # set spine layout
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)

        ax.spines["top"].set_edgecolor("black")
        ax.spines["right"].set_edgecolor("black")
        ax.spines["bottom"].set_edgecolor("black")
        ax.spines["left"].set_edgecolor("black")
        
        # add legend
        if i == 0:
            patch1 = mpatches.Patch(color = color_1, label = "Hash")
            patch2 = mpatches.Patch(color = color_2, label = "Filter")
            patch3 = mpatches.Patch(color = color_3, label = "MIM")
            patch4 = mpatches.Patch(color = color_4, label = "Sort & Slice")
            ax.legend(handles = [patch1, patch2, patch3, patch4], loc = "upper right", fontsize = 14)
            
        # set label for y axis
        ax.set_ylabel(metric + y_unit, fontsize = 14, labelpad = 10)
        
        # create split name dict
        split_name_dict = {"rand": "Random Split", "scaff": "Scaffold Split"}
    
        # set title
        if ml == "rf":
            ax.set_title("Random Forest" + " (" + split_name_dict[split_type] + ")", fontsize = 14, pad = 15)
            
        elif ml == "mlp":
            ax.set_title("\n Multilayer Perceptron" + " (" + split_name_dict[split_type] + ")", fontsize = 14, pad = 15)
        
        # remove vertical gridlines
        ax.yaxis.grid(True, linestyle = '-', linewidth = 0.3)
        ax.set_axisbelow(True)
        
    plt.tight_layout()
    plt.savefig("figures/bar_charts_" + split_type + ".svg", bbox_inches = "tight")



def visualise_box_plots(dataset_name, metric, y_unit = "", show_x_ticks = True, show_legend = True):
    
    # initialise figure
    fig, axes = plt.subplots(1, 4, figsize = (10.5, 14.85/5))
    axes = axes.flatten()
    
    # create dataset name dict
    dataset_name_dict = {"moleculenet_lipophilicity": "MoleculeNet Lipophilicity", 
                         "aqsoldb_solubility": "AqSolDB Aqueous Solubility",
                         "postera_sars_cov_2_mpro": "COVID Moonshot SARS-CoV-2 Main Protease Inhibition",
                         "lit_pcba_esr_ant": "LIT-PCBA Estrogen Receptor Alpha Antagonism",
                         "lit_pcba_tp53": "LIT-PCBA Cellular Tumor Antigen TP53 Inhibition",
                         "ames_mutagenicity": "Ames Mutagenicity"}
    
    # add title
    fig.suptitle(dataset_name_dict[dataset_name], fontsize = 12, x = 0.52, y = 0.95, fontweight = "bold")

    # load results
    results_dict = {}

    for split_type in ["rand", "scaff"]:

        path = "results/" + dataset_name + "/" + split_type + "/"

        for filename in os.listdir(path):
            if filename[-3:] == "csv":

                (dim, invs, diam, pool, ml) = extract_info_from_filename(filename)
                df_current = pd.read_csv(path + filename, sep = ",")

                mean = df_current[metric].values[-2]
                std = df_current[metric].values[-1]

                results_dict[(dim, invs, diam, pool, ml, split_type)] = mean

    boxplot_data_dict = {}

    for split_type in ["rand", "scaff"]:
        for ml in ["rf", "mlp"]:
            for pool in ["hashed", "chi2", "mim", "sorted"]:
                boxplot_data_dict[(pool, ml, split_type)] = [mean for (key, mean) in results_dict.items() if key[3:] == (pool, ml, split_type)]

    # assign axis object
    for (j_1, split_type) in enumerate(["rand", "scaff"]):
        for (j_2, ml) in enumerate(["rf", "mlp"]):

            ax = axes[2*j_1 + j_2]

            # create boxplot
            data = [boxplot_data_dict[(pool, ml, split_type)] for pool in ["hashed", "chi2", "mim", "sorted"]]
            colours = ["silver", "violet", "lightsalmon", "forestgreen"]*4
            sns.boxplot(data = data, orient = "v", fliersize = 4, palette = colours, width = 0.7, linewidth = 2, ax = ax)

            # customise x-ticks
            if show_x_ticks == False:
                ax.set_xticks([])
            else:
                x_ticks_positions = [1.5]
                ax.set_xticks(x_ticks_positions)

                if 2*j_1 + j_2 == 0:
                    ax.set_xticklabels(["Random Forest\n(Random Split)"], fontsize = 10, rotation = 0, ha = "center", fontweight = "normal")
                elif 2*j_1 + j_2 == 1:
                    ax.set_xticklabels(["Multilayer Perceptron\n(Random Split)"], fontsize = 10, rotation = 0, ha = "center", fontweight = "normal")
                elif 2*j_1 + j_2 == 2:
                    ax.set_xticklabels(["Random Forest\n(Scaffold Split)"], fontsize = 10, rotation = 0, ha = "center", fontweight = "normal")
                elif 2*j_1 + j_2 == 3:
                    ax.set_xticklabels(["Multilayer Perceptron\n(Scaffold Split)"], fontsize = 10, rotation = 0, ha = "center", fontweight = "normal")

            # customise y-ticks
            # Set the desired number of decimal places

            for label in ax.get_yticklabels():
                label.set_fontsize(10)
                label.set_fontweight("normal")
 

            # set y-label
            if j_1 + j_2 == 0:
                ax.set_ylabel(metric + y_unit, fontsize = 10, labelpad = 10, fontweight = "normal")
                
            # set spine layout
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["right"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.spines["left"].set_linewidth(1.5)

            ax.spines["top"].set_edgecolor("black")
            ax.spines["right"].set_edgecolor("black")
            ax.spines["bottom"].set_edgecolor("black")
            ax.spines["left"].set_edgecolor("black")

            # remove vertical gridlines and add horizontal ones
            ax.yaxis.grid(True, linestyle = '-', linewidth = 0.3)
            ax.set_axisbelow(True)

            # add legend
            if show_legend == True:
                if j_1 + j_2 == 0:
                    patch1 = mpatches.Patch(color = "silver", label = "Hash")
                    patch2 = mpatches.Patch(color = "violet", label = "Filter")
                    patch3 = mpatches.Patch(color = "lightsalmon", label = "MIM")
                    patch4 = mpatches.Patch(color = "forestgreen", label = "Sort & Slice")
                    ax.legend(handles = [patch1, patch2, patch3, patch4], loc = "upper right", fontsize = 10, ncol = 1, framealpha = 1)
                
    plt.tight_layout()
    plt.savefig("figures/box_plots.svg", bbox_inches = "tight")
    plt.show()
