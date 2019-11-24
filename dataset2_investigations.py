import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from open_source.swm_sdtw.source.utils import compute_barycenter, load_data, normalize
from open_source.swm_sdtw.source.plot_scripts import plot_cluster_analysis, plot_silhouette_values, plot_doughnut
from matplotlib.dates import DateFormatter
sns.set_style('darkgrid')


def plot_barycenter(data=None, labels=None, meta=None, clusters=2, method='softdtw', show=True, colors=None, norm=True, legend=True):

    if colors is None:
        colors = sns.color_palette('viridis', 7)

    data.index = pd.to_datetime(data.index)

    cl = labels.loc[clusters]
    figs = []
    for c in sorted(cl.unique()):

        fig, ax = plt.subplots()

        idx = cl[cl == c].index

        s = data[idx]
        user = meta.loc[idx]['numuser']

        # print(sorted(user.unique()))

        u1_plot = plt.plot([-100, -99], [-100, -99], c=colors[1], linestyle='-', alpha=0.8, linewidth=2.0)
        u2_plot = plt.plot([-100, -99], [-100, -99], c=colors[2], linestyle='-', alpha=0.8, linewidth=2.0)
        u3_plot = plt.plot([-100, -99], [-100, -99], c=colors[3], linestyle='-', alpha=0.8, linewidth=2.0)
        u4_plot = plt.plot([-100, -99], [-100, -99], c=colors[4], linestyle='-', alpha=0.8, linewidth=2.0)
        u5_plot = plt.plot([-100, -99], [-100, -99], c=colors[5], linestyle='-', alpha=0.8, linewidth=2.0)

        for num_u in sorted(user.unique()):
            u = s[user[user == num_u].index]
            ax.plot(u.index, u.values, c=colors[num_u], alpha=0.1)


        if method == 'euclidean':
            x = s.mean(axis=1)

        elif method == 'softdtw':
            if norm:
                # Scale barycenter if it is produced on the normalised data
                x = compute_barycenter(normalize(s), method='softdtw')
                sm = s.mean(axis=1).mean()
                xm = x.mean()
                x = x * sm / xm
            else:
                x = compute_barycenter(s, method='softdtw')


        bary_plot = ax.plot(x.index, x.values, color='k', linewidth=2, linestyle='-', label=r'$\mathbf{x}^\ast$')

        e = compute_barycenter(s, method='euclidean')
        mu_plot = ax.plot(e.index, e.values, color='k', linewidth=1, linestyle='--', label=r'$\mathbf{\mu}$')
        if legend:
            plt.legend([u1_plot[0], u2_plot[0], u3_plot[0],  u4_plot[0], u5_plot[0], bary_plot[0], mu_plot[0]],
                       [f'1 resident', f'2 residents', f'3 residents', f'4 residents', f'5 residents', r'$\mathbf{x}^\ast$', r'$\mathbf{\mu}$'],
                       loc='upper right',
                       frameon=False,
                       fontsize=12)

            # Just add legend to first subplot
            legend = False

        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((0, 0.03))

        hfmt = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        plt.xticks(rotation=45)
        plt.ylabel(r'$Q \quad [L/s]$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if show:
            plt.show()
        figs.append(fig)

    return figs


def numuser_barplot(meta=None, labels=None, colors=None, show=True, clusters=2, silhouette=None):

    l = labels.loc[clusters]

    if colors is None:
        colors = sns.color_palette('viridis', clusters)

    clist = []
    for c in sorted(l.unique()):
        idx = l[l == c].index
        clist.append(meta['numuser'].loc[idx].values)
    fig, ax = plt.subplots()
    ax.hist(clist, color=colors, bins=np.arange(6) + 0.5, align='mid', rwidth=0.7,  alpha=0.7,
            edgecolor='k', label=[f'$C_{x}$' for x in range(1, clusters+1)])
    ax.hist(meta['numuser'], bins=np.arange(6) + 0.5, edgecolor='k', facecolor='None', linewidth=1, rwidth=0.7,
            alpha=0.8, label='All')

    if silhouette is not None:
        s = silhouette.loc[clusters]
        ls = l.copy()
        s = s[s < 0.0]

        ls = ls[s.index]

        slist = []
        for c in sorted(l.unique()):
            idx = ls[ls == c].index
            slist.append(meta['numuser'].loc[idx].values)
        ax.hist(slist, bins=np.arange(6) + 0.5, align='mid', rwidth=0.7, alpha=0.7,
                edgecolor='k', fill=False, hatch='//', label=r'$S(\mathbf{y_l}) < 0$')

    plt.legend(fontsize=16, frameon=False)
    plt.xlabel(r'Number of residents', fontsize=16)
    plt.ylabel(r'Occurences', fontsize=16)

    if show:
        plt.show()

    return fig


def plot_consumption(data=None, meta=None, show=True, labels=None):

    user = meta['numuser']
    # user[user > 3] = 3
    cons = data.mean() * 3600 * 24

    user = user[cons.index]
    user.name = 'user'
    cons.name = 'cons'
    df = pd.concat([cons, user], axis=1)

    if labels is None:
        violinplot = sns.violinplot(x='user', y='cons', data=df, palette='viridis')

    else:
        labels = labels[cons.index]
        labels = labels + 1
        labels.name = 'Cluster'
        df = pd.concat([df, labels], axis=1)
        violinplot = sns.violinplot(x='user', y='cons', hue='Cluster', data=df, palette='viridis', split=True,
                                    inner="stick")

    plt.xlabel('Number of residents', fontsize=16)
    plt.ylabel('Consumption   (L/day)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if show:
        plt.show()

    fig = violinplot.get_figure()

    return fig


def plot_patterns(data=None, meta=None, show=True):

    data.index = pd.to_datetime(data.index)

    users = meta['numuser']
    uniques = sorted(users.unique())

    colors = sns.color_palette('viridis', len(uniques) + 1)

    fig, ax = plt.subplots()

    u1_plot = ax.plot([-100, -99], [-100, -99], c=colors[0], linestyle='-', alpha=0.8, linewidth=2.0)
    u2_plot = ax.plot([-100, -99], [-100, -99], c=colors[1], linestyle='-', alpha=0.8, linewidth=2.0)
    u3_plot = ax.plot([-100, -99], [-100, -99], c=colors[2], linestyle='-', alpha=0.8, linewidth=2.0)
    u4_plot = ax.plot([-100, -99], [-100, -99], c=colors[3], linestyle='-', alpha=0.8, linewidth=2.0)
    u5_plot = ax.plot([-100, -99], [-100, -99], c=colors[4], linestyle='-', alpha=0.8, linewidth=2.0)

    for user in sorted(users.unique(), reverse=True):
        idx = meta[meta['numuser'] == user].index
        x = data[idx]
        ax.plot(x.index, x.values, color=colors[user - 1], alpha=0.1)

    mu_plot = ax.plot(x.index, x.mean(axis=1).values, color='k', linestyle='--')

    plt.xlim([x.index[0], x.index[-1]])
    plt.ylim((0, 0.04))
    hfmt = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation=45)
    plt.ylabel(r'$Q \quad [L/s]$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.legend([u1_plot[0], u2_plot[0], u3_plot[0], u4_plot[0], u5_plot[0], mu_plot[0]],
               [f'1 resident', f'2 residents', f'3 residents', f'4 residents', f'5 residents',
                r'$\mathbf{\mu}$'],
               loc='upper right',
               frameon=False,
               fontsize=12)

    if show:
        plt.show()

    return fig

def checker(x):

    if (x['house_type'] == 'A') and (x['labels'] == 0):
        return True
    elif (x['house_type'] == 'B') and (x['labels'] == 1):
        return True
    else:
        return False

def get_positives(labels=None, meta=None):
    labels.name = 'labels'

    house_type = meta['house_type'].copy(deep=True)
    house_type[house_type != 'family'] = 'A'
    house_type[house_type == 'family'] = 'B'

    house_type = house_type[labels.index]

    df = pd.concat([labels, house_type], axis=1)

    positives = df.apply(checker, axis=1)

    success_rate = positives.sum() / len(positives) * 100
    error_rate = 100 - success_rate
    print(f'Success Rate={success_rate}; Error Rate={error_rate}')

    return positives, success_rate, error_rate


if __name__ == '__main__':

    # Parameters
    dataset = '2'
    clusters = 2
    norm = False
    methods = ['euclidean', 'softdtw']
    figpath = '/Users/davidsteffelbauer/Documents/DASH/Publications/AGU_WRR_DTW_Clustering/misc/dataset2'

    # Program Start:
    collection = load_data(dataset=dataset, norm=norm)
    file_meta = os.path.join('data', f'meta_dataset{dataset[0]}.csv')
    meta = pd.read_csv(file_meta, index_col=0, header=0)

    data = collection['softdtw']['data']

    # Pattern plot
    fig = plot_patterns(data=data, meta=meta)
    fig.savefig(os.path.join(figpath, f'ds{dataset}_patterns.pdf'), bbox_inches='tight', dpi=300)

    # Conumption plot
    fig = plot_consumption(data=data, meta=meta)
    fig.savefig(os.path.join(figpath, f'ds{dataset}_consumption.pdf'), bbox_inches='tight', dpi=300)

    # Cluster analysis plot
    fig = plot_cluster_analysis(dataset=dataset, norm=norm, legend_location='upper center', ylim=(0, 0.8))
    fig.savefig(os.path.join(figpath, f'ds{dataset}_cluster_analysis.pdf'), bbox_inches='tight', dpi=300)

    for method in methods:

        d = collection[method]

        positives, success_rate, error_rate = get_positives(labels=d['labels'].loc[clusters], meta=meta)

        # Doughnut plots
        fig = plot_doughnut(np.ceil(success_rate))
        fig.savefig(os.path.join(figpath, f'ds{dataset}_doughnut_{method}.pdf'), bbox_inches='tight', dpi=300,
                    transparent=True)

        fig = plot_consumption(data=d['data'], meta=meta, labels=positives)
        fig.savefig(os.path.join(figpath, f'ds{dataset}_positives_{method}.pdf'), bbox_inches='tight', dpi=300)

        # Barycenters:
        figs = plot_barycenter(data=d['data'], labels=d['labels'], meta=meta, norm=True, method=method,
                               clusters=clusters)
        for ind, fig in enumerate(figs):
            fig.savefig(os.path.join(figpath, f'ds{dataset}_barycenter_{method}_{ind+1}.pdf'), bbox_inches='tight',
                        dpi=300)

        # Silhouette plots:
        fig = plot_silhouette_values(silhouette=d['silhouette'], labels=d['labels'], clusters=clusters)
        fig.savefig(os.path.join(figpath, f'ds{dataset}_silhouette_{method}.pdf'), bbox_inches='tight', dpi=300)

        # Link clusters to number of users:
        fig = numuser_barplot(meta=meta, labels=d['labels'], clusters=clusters, silhouette=d['silhouette'])
        fig.savefig(os.path.join(figpath, f'ds{dataset}_numusers_{method}.pdf'), bbox_inches='tight', dpi=300)

        fig = plot_consumption(data=d['data'], labels=d['labels'].loc[clusters], meta=meta)
        fig.savefig(os.path.join(figpath, f'ds{dataset}_consumption_{method}.pdf'), bbox_inches='tight', dpi=300)




    """

    # silhouette = collection['softdtw']['silhouette']
    # labels = collection['softdtw']['labels']

    # print(meta.loc[labels[labels == 0].index]['numuser'].describe())
    # print(meta.loc[labels[labels == 1].index]['numuser'].describe())
    # print(meta.loc[s[s < 0].index]['numuser'].describe())

    # numuser_barplot(meta=meta, labels=labels, clusters=clusters, silhouette=silhouette)


    # clist = []
    # colors = sns.color_palette('viridis', 7)
    # for c in sorted(l.unique()):
    #     idx = l[l == c].index
    #     clist.append(meta['numuser'].loc[idx].values)
    # fig, ax = plt.subplots()
    # ax.hist(clist, color=[colors[3], colors[1]], bins=np.arange(6) + 0.5, align='mid', rwidth=0.7,  alpha=0.7, edgecolor='k', label=[r'$C_1$', r'$C_2$'])
    # ax.hist(meta['numuser'], bins=np.arange(6) + 0.5, edgecolor='k', facecolor='None', linewidth=1, rwidth=0.7, alpha=0.8, label='All')
    # plt.legend(fontsize=16, frameon=False)
    # plt.xlabel(r'Number of inhabitants', fontsize=16)
    # plt.ylabel(r'Occurences', fontsize=16)
    # plt.show()




    """
    # normstr = ('norm_' if norm else '')
    # palette = itertools.cycle(sns.color_palette('viridis', clusters+1))
    #
    # file_data = os.path.join('data', f'dataset{dataset}.csv')
    # file_meta = os.path.join('results', f'dataset{dataset}',
    #                         f'cortana_ds{dataset}_{normstr}{method}_clusters_{clusters}.csv')
    # file_cluster = os.path.join('results', f'dataset{dataset}', f'ds{dataset}_{normstr}cluster_labels_{method}.csv')
    # file_silhouette = os.path.join('results', f'dataset{dataset}', f'ds{dataset}_{normstr}'
    #                                                                f'silhouette_coefficients_{method}.csv')
    #
    # data = pd.read_csv(file_data, index_col=0, header=0)
    # meta = pd.read_csv(file_meta, index_col=0, header=0)
    # labels = pd.read_csv(file_cluster, index_col=0, header=0)
    # silhouette = pd.read_csv(file_silhouette, index_col=0, header=0)
    #
    # if norm:
    #     data = normalize(data)
    """
    # cluster analysis
    # fig = plot_cluster_analysis(norm=norm)

    # silhouette plots
    # fig = plot_silhouette_values(silhouette=silhouette, labels=labels, clusters=clusters)

    # barycenter plots
    # figs = plot_barycenter(data=data, labels=labels, meta=meta, method=method, clusters=clusters)

    # color palette plots
    # current_palette = sns.color_palette('viridis', 7)[1:6]
    # sns.palplot(current_palette, 5)

    # numuser plot:
    """

    """
    cl = labels.loc[2]

    clist = []

    colors = sns.color_palette('viridis', clusters+1)
    for c in sorted(cl.unique()):
        idx = cl[cl == c].index
        clist.append(meta['numuser'].loc[idx].values)

    # sns.distplot(meta['numuser'], bins=np.arange(6) + 0.5, ax=ax,
    #              kde_kws={'bw': 0.5, 'color': 'k', 'linestyle': '--'},
    #              hist_kws={'edgecolor': 'k', 'facecolor': 'None', 'alpha': 1.0})

    fig, ax = plt.subplots()
    ax.hist(clist, color=[colors[2], colors[1]], bins=np.arange(6) + 0.5, align='mid', rwidth=0.7,  alpha=0.7, edgecolor='None')
    ax.hist(meta['numuser'], bins=np.arange(6) + 0.5, edgecolor='k', facecolor='None', linewidth=1, rwidth=0.7, alpha=0.8)
    plt.show()
    
    """

#     fig, ax = plt.subplots()
#     for key in ca:
#         color = next(palette)
#         data = ca[key]
#
#         color = next(palette)
#         x = data.idxmax()
#         y = data.max()
#         ax.axhline(y, linestyle='--', color=color, alpha=0.7)
#         ax.plot([x, x], [0, y], '--', c=color, alpha=0.5)
#
#         ax.plot(data.index, data.values, marker='o', c=color, label=f'{key}')
#     ax.yaxis.set_label_position('right')
#     ax.yaxis.tick_right()
#     plt.xlabel(r'$k$', fontsize=16)
#     plt.ylabel(r'$\overline{S}$', fontsize=16)
#
#     ymax = np.ceil(ca.max().max() * 10) / 10
#
#     plt.ylim((0, ymax))
#     plt.xlim([1.5, 10.5])
#     plt.legend(loc='upper right', frameon=False)
#
#     plt.show()
#
"""

# labels.name = 'target'


# with open('data/metadata_dataset2.json', 'r') as json_file:
#     meta = json.load(json_file)

# result = dict()
# for key in data.keys():
#     typelist = [data[key][username]['type'] for username in data[key].keys()]
#     typecounter = dict(Counter(typelist))
#     result[key] = typecounter
# user = pd.DataFrame.from_dict(result, orient='index').fillna(0).astype(int)
#
# user[user==0] = np.nan

ds_number = 2
methods = ['euclidean', 'softdtw']
norm_texts = ['norm_', '']

fig, ax = plt.subplots()
cnt = 0
for method in methods:
    for norm_text in norm_texts:
        df = pd.read_csv(os.path.join('results', f'ds{ds_number}_{norm_text}silhouette_coefficients_{method}.csv'),
                         header=0,
                         index_col=0)
        df = df.mean(axis=1)
        print(df)
        ax.plot(df.index, df.values, label=f'{norm_text}{method}', c=colors[cnt])

        cnt += 1
plt.legend()
plt.show()


method = 'softdtw'
labels = pd.read_csv(os.path.join('results', f'ds2_cluster_labels_{method}.csv'), index_col=0, header=0, squeeze=True).loc[2]
meta['target'] = labels
meta.to_csv('/Users/davidsteffelbauer/Documents/DASH/Programs/Subgroup Discovery/cortana_test.csv', index=False)

meta.to_csv(os.path.join('results', ))

fig, ax= plt.subplots()
for ind, cl in enumerate(labels.unique()):
    s = data[labels[labels == cl].index]
    bc = compute_barycenter(s, method=method)
    ax.plot(bc.index, bc.values, c=colors[ind*2], label=f'cluster={cl}')
plt.legend()
plt.show()

"""