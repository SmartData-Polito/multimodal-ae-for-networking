# type: ignore

import seaborn as sns
import matplotlib.patches as mpatches


def violin_task01(plt, df, features):
    ax = plt.gca()
    sns.violinplot(df, x='f1-score', y='feature', cut=0, 
                   inner='stick', fill=False, ax=ax)
    plt.xlim(.0, 1)
    plt.ylabel('Features')
    plt.xlabel('F1-Score')
    plt.setp(ax.collections, alpha=1, linewidth=1)

    for i in range(4):
        ax.collections[i].set_edgecolor((0, 0, 0, 0))
        ax.collections[i].set_facecolor('white')
        ax.collections[i].set_alpha(.6)
    for i in range(4,5):
        ax.collections[i].set_edgecolor((0, 0, 0, 0))
        ax.collections[i].set_facecolor('y')
        ax.collections[i].set_alpha(.6)
    for i in range(5,6):
        ax.collections[i].set_edgecolor((0, 0, 0, 0))
        ax.collections[i].set_facecolor('g')
        ax.collections[i].set_alpha(.6)

    ax.set_ylim(5.5, -.5)
    ax.hlines(4.5, -.2, 1, color='k', linestyle='-.', linewidth=.5)
    ax.hlines(3.5, -.2, 1, color='k', linestyle='-.', linewidth=.5)
    ax.set_yticklabels(x.capitalize() for x in features)

def violin_shallow(plt, df):
    ax = plt.gca()
    sns.violinplot(df, x='f1-score', y='algorithm', cut=0, 
                   inner='stick', fill=False, ax=ax)
    plt.xlim(.0, 1)
    plt.ylabel('Algorithm')
    plt.xlabel('F1-Score')
    plt.setp(ax.collections, alpha=1, linewidth=1)

    for i in range(3):
        ax.collections[i].set_edgecolor((0, 0, 0, 0))
        ax.collections[i].set_facecolor('g')
        ax.collections[i].set_alpha(.6)

    #ax.set_ylim(5.5, -.5)
    ax.set_yticklabels(x.capitalize() for x in ['Deep', '7-NN', 'RF'])

def violin_task02(plt, df, features):
    ax = plt.gca()
    sns.violinplot(df, x='f1-score', y='feature', cut=0, 
                   inner='stick', fill=False, ax=ax)
    plt.xlim(.0, 1)
    plt.ylabel('Features')
    plt.xlabel('F1-Score')
    plt.setp(ax.collections, alpha=1, linewidth=1)

    for i in range(3):
        ax.collections[i].set_edgecolor((0, 0, 0, 0))
        ax.collections[i].set_facecolor('white')
        ax.collections[i].set_alpha(.6)
    for i in range(3,4):
        ax.collections[i].set_edgecolor((0, 0, 0, 0))
        ax.collections[i].set_facecolor('y')
        ax.collections[i].set_alpha(.6)
    for i in range(4,5):
        ax.collections[i].set_edgecolor((0, 0, 0, 0))
        ax.collections[i].set_facecolor('g')
        ax.collections[i].set_alpha(.6)

    ax.set_ylim(4.5, -.5)
    ax.hlines(3.5, -.2, 1, color='k', linestyle='-.', linewidth=.5)
    ax.hlines(2.5, -.2, 1, color='k', linestyle='-.', linewidth=.5)
    ax.set_yticklabels(x.capitalize() for x in features)

def plot_pc(plt, full):
    colors = {'task01':'r', 'task02':'b', 'task03':'g'}
    lstyle = {'mae':'-', 'rawcat':'--'}
    a = {'task01':'$T_{1}$', 'task02':'$T_{2}$', 'task03':'$T_{3}$'}
    b = {'mae':'MAE', 'rawcat':'Concat'}
    for task, model, df in full:
        plt.plot(df.k, df.pc, 
                 color=colors[task], 
                 linestyle=lstyle[model], 
                 label=f'{a[task]} {b[model]}')
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.xlim(1, 19)
    plt.xticks([1, 5, 10, 15, 19])
    plt.ylim(0, 1)
    plt.grid()
    plt.xlabel('$K$')
    plt.ylabel('Macro avg. $p_{C}$')

def plot_kmeans_ari(plt, df1, df2):
    plt.plot(df1.index, df1.values, label='Concat', marker='o')
    plt.plot(df2.index, df2.values, label='MAE', marker='s')
    plt.legend()
    plt.xlim(df1.index[0], df1.index[-1])
    plt.grid()
    plt.xlabel('$k$ of kMeans')
    plt.ylabel('Adjusted Rand Index')
    
def plot_kmeans_sh(plt, df1, df2):
    plt.plot(df1.index, df1.values, label='Concat', marker='o')
    plt.plot(df2.index, df2.values, label='MAE', marker='s')
    plt.legend()
    plt.xlim(df1.index[0], df1.index[-1])
    plt.grid()
    plt.xlabel('$k$ of kMeans')
    plt.ylabel('Silhouette')

def gs_heatmap(plt, df, vmin=.84,  vmax=.87):
    ax = sns.heatmap(df, annot=True, cmap='Blues',
                     vmin=.84, vmax=.87, 
                     cbar_kws={'label':'Macro avg. F1-Score'}, 
                     square=True)
    ax.invert_yaxis()
    ax.set_xlabel('Bottleneck size $l_4$')
    ax.set_ylabel('Adaptation space size $l_1$')
    ax.set_yticklabels([32, 64, 128, 256], rotation=0)

def gs_params_heatmap(plt, df):
    ax = sns.heatmap(df, annot=True, cmap='Blues',
     cbar_kws={'label':'\# Trainables $ \\times 10^{-4}$'}, square=True, fmt='.1f')
    ax.invert_yaxis()
    ax.set_xlabel('Bottleneck size $l_4$')
    ax.set_ylabel('Adaptation space size $l_1$')
    ax.set_yticklabels([16, 32, 64, 128], rotation=0)