import pandas as pd 
import matplotlib.pyplot as plt

plt.ion()

PATH = "./exp/ABLATION/CFAblation.csv"

df_ablation_scores = pd.read_csv(PATH)

tmnist_validity_preserving_scores = df_ablation_scores[
    (df_ablation_scores["preserving/changing"] == "preserving") &
    (df_ablation_scores["Dataset"] == "TMNIST")
]
tmnist_validity_preserving_scores = tmnist_validity_preserving_scores.pivot_table(index=["Alpha"],
                                                                                  columns="Metric", 
                                                                                  values="Value",
                                                                                  aggfunc=['mean']) 


credit_validity_preserving_scores = df_ablation_scores[
    (df_ablation_scores["preserving/changing"] == "preserving") &
    (df_ablation_scores["Dataset"] == "GIVECREDIT")
]
credit_validity_preserving_scores = credit_validity_preserving_scores.pivot_table(index=["Alpha"],
                                                                                  columns="Metric",
                                                                                  values="Value",
                                                                                 aggfunc=['mean']) 



tmnist_validity_changing_scores = df_ablation_scores[
    (df_ablation_scores["preserving/changing"] == "changing") &
    (df_ablation_scores["Dataset"] == "TMNIST")
]
tmnist_validity_changing_scores = tmnist_validity_changing_scores.pivot_table(index=["Alpha"],
                                                                                  columns="Metric", 
                                                                                  values="Value",
                                                                                  aggfunc=['mean']) 


credit_validity_changing_scores = df_ablation_scores[
    (df_ablation_scores["preserving/changing"] == "changing") &
    (df_ablation_scores["Dataset"] == "GIVECREDIT")
]
credit_validity_changing_scores = credit_validity_changing_scores.pivot_table(index=["Alpha"],
                                                                                  columns="Metric", 
                                                                                  values="Value",
                                                                                  aggfunc='mean') 

######################################################
# Article figure
######################################################

from matplotlib.lines import Line2D

plt.figure("ablation study",figsize=(12,7)); plt.clf()

scenarios = ["preserving","changing"]
scenarios_labels = ["Context-preserving","Context-changing"]
metrics = ["validity","lof_context"]
metrics_labels = ["Validity $(\\uparrow)$","$\\text{context}-\\text{LOF}_{10}$ $(\\downarrow)$"]
fontsize = 18
linewidht = 2
markersize = 10
markers = [".-b","x-r"]

colors = ["#66c2a5", "#fc8d62"]
markers = ["o", "s"]
linestyles = ["-", "--"]

nr = len(scenarios)
nc = len(metrics)


for ii, scenario in enumerate(scenarios):
    for jj, metric in enumerate(metrics):
        
        # Getting scores from ablation csv
        scores_tmnist =  df_ablation_scores[(df_ablation_scores["preserving/changing"] == scenario) &
                                            (df_ablation_scores["Dataset"] == "TMNIST")]
        scores_tmnist = scores_tmnist.pivot_table(index=["Alpha"],
                                                    columns="Metric", 
                                                    values="Value",
                                                    aggfunc='mean') 

        scores_credit =  df_ablation_scores[(df_ablation_scores["preserving/changing"] == scenario) &
                                            (df_ablation_scores["Dataset"] == "GIVECREDIT")]
        scores_credit = scores_credit.pivot_table(index=["Alpha"],
                                                    columns="Metric", 
                                                    values="Value",
                                                    aggfunc='mean') 


        plt.subplot(nr,nc,ii*nc + jj +1)

        plt.plot(scores_tmnist.index.values, scores_tmnist[metric].values,
         linestyle=linestyles[0], marker=markers[0], color=colors[0],
         linewidth=linewidht, markersize=markersize, label="TMNIST")
        plt.xlim([0,1.02])
        plt.plot(scores_credit.index.values, scores_credit[metric].values,
         linestyle=linestyles[1], marker=markers[1], color=colors[1],
         linewidth=linewidht, markersize=markersize, label="GIVECREDIT")
        plt.xlim([0,1.02])
        plt.grid(True)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 
        ax.tick_params(axis='both', labelsize=fontsize-5)

        if ii ==0:
            plt.title(metrics_labels[jj], fontdict={"fontsize":fontsize})
        if jj == 0:
            plt.ylabel(scenarios_labels[ii], fontdict={"fontsize":fontsize})
        if ii == nr-1:
            plt.xlabel("$\\alpha$", fontdict={"fontsize":fontsize})

# Legend
legend_elements = [
    Line2D([0], [0], linestyle=linestyles[0], marker=markers[0],
           color=colors[0], label='TMNIST'),
    Line2D([0], [0], linestyle=linestyles[1], marker=markers[1],
           color=colors[1], label='GiveMeCredit')
]

plt.figlegend(legend_elements, ["TMNIST", "GiveMeCredit"],
              loc='lower left', ncol=2, fontsize=fontsize-5)

plt.subplots_adjust(left= 0.067, right= 0.97, top=0.93,bottom=0.14)
#plt.savefig("./Figs/ablationMetrics.pdf")