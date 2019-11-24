import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import load_cluster_analysis
sns.set_style("darkgrid")

colors = sns.color_palette('viridis', 7)
workcolor = colors[0]
homecolor = colors[3]


def plot_cluster_analysis(palette=None, show=True, legend_location='lower center', ylim=None,
                          **kwargs):

    if palette is None:
        palette = itertools.cycle(sns.color_palette('viridis', 3))

    ca = load_cluster_analysis(**kwargs)
    fig, ax = plt.subplots()
    for key in ca:
        data = ca[key]
        color = next(palette)
        x = data.idxmax()
        y = data.max()
        ax.axhline(y, linestyle='--', color=color, alpha=0.7)
        ax.plot([x, x], [0, y], '--', c=color, alpha=0.5)
        if key.startswith('e'):
            name = 'Euclidean'
        elif key.startswith('s'):
            name = 'SDTW'
        ax.plot(data.index, data.values, marker='o', c=color, label=f'{name}')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    plt.xlabel(r'$k$', fontsize=18)
    plt.ylabel(r'$\overline{S}$', fontsize=18)

    if ylim is None:
        ymax = np.ceil(ca.max().max() * 10) / 10
        plt.ylim((0, ymax))
    else:
        plt.ylim(ylim)
    plt.xlim([1.5, 10.5])
    plt.legend(loc=legend_location, frameon=False, fontsize=14,  framealpha=1.0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    if show:
        plt.show()
    return fig


def plot_silhouette_values(silhouette=None, labels=None, clusters=2, show=True):
    si = silhouette.loc[clusters]
    cl = labels.loc[clusters]

    palette = itertools.cycle(sns.color_palette('viridis', clusters + 1))
    y_lower = 1
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-0.25, 1])

    height = silhouette.shape[1] + (clusters + 1)

    ax.set_ylim([0, height])

    for i in sorted(cl.unique()):
        color = next(palette)
        si_cli = si[cl == i]

        si_cli.sort_values(inplace=True)

        size_cluster_i = si_cli.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, si_cli,
                         facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 1  # 10 for the 0 samples

        print(f'\n\nSilhouette coefficient statistics of Cluster {i + 1}:\n', si_cli.describe())
        print(f'\t->\t{si_cli[si_cli < 0].count()} values <0.')

    ax.axvline(x=si.mean(), color="k", linestyle="--")
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(fontsize=20)

    if show:
        plt.show()

    return fig


def plot_doughnut(TP, show=True, legend_loc='center'):
    ER = 100 - TP
    sizes = [TP, ER]
    labels = ['SR', 'ER']
    explode = (0.05, 0.05)

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.0f%%', shadow=False, pctdistance=0.75,
                                       startangle=90 - ER, colors=[homecolor, workcolor], wedgeprops={
            'alpha': 0.5, 'width': 0.5}, frame=False, radius=1, textprops={'fontsize': 30}, explode=explode)


    ax.legend(wedges, labels, loc=legend_loc,
              # bbox_to_anchor=(0.8, 0.4, 0.5, 1),
              frameon=False, fontsize=30)
    ax.axis('equal')

    if show:
        plt.show()

    return fig
