from .source.utils import daily_pattern
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from .source.utils import compute_barycenter, load_data
from .source.plot_scripts import plot_cluster_analysis, plot_silhouette_values, plot_doughnut
from .source.utils import normalize
from matplotlib.dates import DateFormatter
sns.set_style('darkgrid')
import itertools

lines = ['-', '--', '-.', ':']
linecycler = itertools.cycle(lines)


def plot_barycenter(data=None, labels=None, meta=None, clusters=2, method='softdtw', show=True, colors=None, norm=True,
                    alpha=0.5, legend=True):

    if colors is None:
        colors = sns.color_palette('viridis', 5)

    data.index = pd.to_datetime(data.index)

    cl = labels.loc[clusters]
    figs = []
    for c in sorted(cl.unique()):

        fig, ax = plt.subplots()

        idx = cl[cl == c].index

        s = data[idx]
        user = meta.loc[idx]['inhabitants']


        u1_plot = plt.plot([-100, -99], [-100, -99], c=colors[1], linestyle='-', alpha=0.8, linewidth=2.0)
        u2_plot = plt.plot([-100, -99], [-100, -99], c=colors[2], linestyle='-', alpha=0.8, linewidth=2.0)
        u3_plot = plt.plot([-100, -99], [-100, -99], c=colors[3], linestyle='-', alpha=0.8, linewidth=2.0)
        u4_plot = plt.plot([-100, -99], [-100, -99], c=colors[4], linestyle='-', alpha=0.8, linewidth=2.0)

        for num_u in sorted(user.unique()):
            u = s[user[user == num_u].index]
            ax.plot(u.index, u.values, c=colors[num_u], alpha=alpha)


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
            plt.legend([u1_plot[0], u2_plot[0], u3_plot[0],  u4_plot[0], bary_plot[0], mu_plot[0]],
                       [f'1 resident', f'2 residents', f'3 residents', f'4 residents', r'$\mathbf{x}^\ast$', r'$\mathbf{\mu}$'],
                       loc='upper right',
                       frameon=False,
                       fontsize=12)
            legend = False
        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((0, 0.05))

        hfmt = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        plt.xticks(rotation=45)
        plt.ylabel(r'$Q \quad [L/s]$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if show:
            plt.show()
        figs.append(fig)

    return figs


def plot_patterns(data=None, meta=None, show=True):

    data.index = pd.to_datetime(data.index)

    users = meta['inhabitants']
    uniques = sorted(users.unique())

    colors = sns.color_palette('viridis', len(uniques))

    fig, ax = plt.subplots()

    u1_plot = ax.plot([-100, -99], [-100, -99], c=colors[0], linestyle='-', alpha=0.8, linewidth=2.0)
    u2_plot = ax.plot([-100, -99], [-100, -99], c=colors[1], linestyle='-', alpha=0.8, linewidth=2.0)
    u3_plot = ax.plot([-100, -99], [-100, -99], c=colors[2], linestyle='-', alpha=0.8, linewidth=2.0)
    u4_plot = ax.plot([-100, -99], [-100, -99], c=colors[3], linestyle='-', alpha=0.8, linewidth=2.0)
    # u5_plot = ax.plot([-100, -99], [-100, -99], c=colors[4], linestyle='-', alpha=0.8, linewidth=2.0)

    for user in sorted(users.unique(), reverse=True):
        idx = meta[meta['inhabitants'] == user].index
        x = data[idx]
        ax.plot(x.index, x.values, color=colors[user - 1], alpha=0.4)

    mu_plot = ax.plot(x.index, x.mean(axis=1).values, color='k', linestyle='--')

    plt.xlim([x.index[0], x.index[-1]])
    plt.ylim((0, 0.04))
    hfmt = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation=45)
    plt.ylabel(r'$Q \quad [L/s]$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.legend([u1_plot[0], u2_plot[0], u3_plot[0], u4_plot[0], mu_plot[0]],
               [f'1 resident', f'2 residents', f'3 residents', f'4 residents',
                r'$\mathbf{\mu}$'],
               loc='upper right',
               frameon=False,
               fontsize=12)

    if show:
        plt.show()

    return fig


def plot_simple_patterns(data=None, show=True):

    data.index = pd.to_datetime(data.index)
    colors = sns.color_palette('viridis', len(data.columns)+1)

    fig, ax = plt.subplots()

    for ind, (name, x) in enumerate(data.iteritems()):
        name = name.replace('sm_milford_', 'Home ')
        name = name.replace('_', ' (')
        name += ')'

        ax.plot(x.index, x.values, color=colors[ind], alpha=0.8, label=name, linestyle=next(linecycler), linewidth=2)


    plt.xlim([x.index[0], x.index[-1]])
    plt.ylim((0, 0.04))
    hfmt = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation=45)
    plt.ylabel(r'$Q \quad [L/s]$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.legend(loc='upper right',
               frameon=False,
               fontsize=12)

    if show:
        plt.show()

    return fig


def numuser_barplot(meta=None, labels=None, colors=None, show=True, clusters=2, silhouette=None):

    l = labels.loc[clusters]

    if colors is None:
        colors = sns.color_palette('viridis', clusters)

    clist = []
    for c in sorted(l.unique()):
        idx = l[l == c].index
        clist.append(meta['inhabitants'].loc[idx].values)
    fig, ax = plt.subplots()
    ax.hist(clist, color=colors, bins=np.arange(5) + 0.5, align='mid', rwidth=0.7,  alpha=0.7,
            edgecolor='k', label=[f'$C_{x}$' for x in range(1, clusters+1)])
    ax.hist(meta['inhabitants'], bins=np.arange(5) + 0.5, edgecolor='k', facecolor='None', linewidth=1, rwidth=0.7,
            alpha=0.8, label='All')

    if silhouette is not None:
        s = silhouette.loc[clusters]
        ls = l.copy()
        s = s[s < 0.0]

        ls = ls[s.index]

        slist = []
        for c in sorted(l.unique()):
            idx = ls[ls == c].index
            slist.append(meta['inhabitants'].loc[idx].values)
        ax.hist(slist, bins=np.arange(5) + 0.5, align='mid', rwidth=0.7, alpha=0.7,
                edgecolor='k', fill=False, hatch='//', label=r'$S(\mathbf{y_l}) < 0$')

    plt.legend(fontsize=16, frameon=False)
    plt.xlabel(r'Number of residents', fontsize=16)
    plt.ylabel(r'Occurences', fontsize=16)

    if show:
        plt.show()

    return fig


def type_day_barplot(meta=None, labels=None, colors=None, show=True, clusters=2, silhouette=None):

    l = labels.loc[clusters]

    if colors is None:
        colors = sns.color_palette('viridis', clusters)

    clist = []
    for c in sorted(l.unique()):
        idx = l[l == c].index
        clist.append(meta['type_day'].loc[idx].values)
    fig, ax = plt.subplots()
    ax.hist(clist, color=colors, bins=np.arange(3) - 0.5, align='mid', rwidth=0.7,  alpha=0.7,
            edgecolor='k', label=[f'$C_{x}$' for x in range(1, clusters+1)])
    ax.hist(meta['type_day'], bins=np.arange(3) - 0.5, edgecolor='k', facecolor='None', linewidth=1, rwidth=0.7,
            alpha=0.8, label='All')

    if silhouette is not None:
        s = silhouette.loc[clusters]
        ls = l.copy()
        s = s[s < 0.0]

        ls = ls[s.index]

        slist = []
        for c in sorted(l.unique()):
            idx = ls[ls == c].index
            slist.append(meta['type_day'].loc[idx].values)
        ax.hist(slist, bins=np.arange(3) - 0.5, align='mid', rwidth=0.7, alpha=0.7,
                edgecolor='k', fill=False, hatch='//', label=r'$S(\mathbf{y_l}) < 0$')

    plt.legend(fontsize=16, frameon=False)
    plt.xlabel(r'weekday / weekend', fontsize=16)
    plt.ylabel(r'Occurences', fontsize=16)

    if show:
        plt.show()

    return fig


def combine_barplot(meta=None, labels=None, colors=None, show=True, clusters=2, silhouette=None):

    # x = list(np.arange(4) + 1)
    # y = ['weekday', 'weekend']
    # order = [f'{b} {a}' for a, b in itertools.product(y, x)]

    l = labels.loc[clusters]

    if colors is None:
        colors = sns.color_palette('viridis', clusters)

    clist = []
    for c in sorted(l.unique()):
        idx = l[l == c].index
        clist.append(meta['both'].loc[idx].values)
    fig, ax = plt.subplots()
    ax.hist(clist, color=colors, bins=np.arange(9) - 0.5, align='mid', rwidth=0.7,  alpha=0.7,
            edgecolor='k', label=[f'$C_{x}$' for x in range(1, clusters+1)])
    ax.hist(meta['both'], bins=np.arange(9) - 0.5, edgecolor='k', facecolor='None', linewidth=1, rwidth=0.7,
            alpha=0.8, label='All')

    if silhouette is not None:
        s = silhouette.loc[clusters]
        ls = l.copy()
        s = s[s < 0.0]

        ls = ls[s.index]

        slist = []
        for c in sorted(l.unique()):
            idx = ls[ls == c].index
            slist.append(meta['both'].loc[idx].values)
        ax.hist(slist, bins=np.arange(9) - 0.5, align='mid', rwidth=0.7, alpha=0.7,
                edgecolor='k', fill=False, hatch='//', label=r'$S(\mathbf{y_l}) < 0$')

    plt.legend(fontsize=16, frameon=False)
    plt.xlabel(r'weekday / weekend', fontsize=16)
    plt.ylabel(r'Occurences', fontsize=16)

    if show:
        plt.show()

    return fig


def type_barplot(meta=None, labels=None, colors=None, show=True, clusters=2, silhouette=None):

    l = labels.loc[clusters]

    if colors is None:
        colors = sns.color_palette('viridis', clusters)

    clist = []
    for c in sorted(l.unique()):
        idx = l[l == c].index
        clist.append(meta['type'].loc[idx].values)
    fig, ax = plt.subplots()
    ax.hist(clist, color=colors, bins=np.arange(4) - 0.5, align='mid', rwidth=0.7,  alpha=0.7,
            edgecolor='k', label=[f'$C_{x}$' for x in range(1, clusters+1)])
    ax.hist(meta['type'], bins=np.arange(4) - 0.5, edgecolor='k', facecolor='None', linewidth=1, rwidth=0.7,
            alpha=0.8, label='All')

    if silhouette is not None:
        s = silhouette.loc[clusters]
        ls = l.copy()
        s = s[s < 0.0]

        ls = ls[s.index]

        slist = []
        for c in sorted(l.unique()):
            idx = ls[ls == c].index
            slist.append(meta['type'].loc[idx].values)
        ax.hist(slist, bins=np.arange(4) - 0.5, align='mid', rwidth=0.7, alpha=0.7,
                edgecolor='k', fill=False, hatch='//', label=r'$S(\mathbf{y_l}) < 0$')

    plt.legend(fontsize=16, frameon=False)
    plt.xlabel(r'Pattern type', fontsize=16)
    plt.ylabel(r'Occurences', fontsize=16)

    if show:
        plt.show()

    return fig


def plot_consumption(data=None, meta=None, show=True, labels=None):

    user = meta['inhabitants']
    # user[user > 3] = 3
    cons = data.mean() * 3600 * 24

    user = user[cons.index]
    user.name = 'user'
    cons.name = 'cons'
    df = pd.concat([cons, user], axis=1)

    if labels is None:
        violinplot = sns.violinplot(x='user', y='cons', data=df, palette='viridis', inner='stick')

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


def checker(x):

    if (x['type'] == 'work') and (x['labels'] == 0):
        return True
    elif (x['type'] == 'home') and (x['labels'] == 1):
        return True
    else:
        return False


def get_positives(labels=None, meta=None):
    labels.name = 'labels'

    house_type = meta['type'].copy(deep=True)

    house_type = house_type[labels.index]

    df = pd.concat([labels, house_type], axis=1)

    positives = df.apply(checker, axis=1)

    success_rate = positives.sum() / len(positives) * 100
    error_rate = 100 - success_rate
    print(f'Success Rate={success_rate}; Error Rate={error_rate}')

    return positives, success_rate, error_rate


if __name__ == '__main__':

    # Parameters
    dataset = '3'
    clusters = 2
    norm = True
    methods = ['euclidean', 'softdtw']

    # Program Start:
    collection = load_data(dataset=dataset, norm=norm)
    file_meta = os.path.join('data', f'Milford_metadata.txt')

    meta = pd.read_csv(file_meta, index_col=0, header=0, skiprows=[1])
    meta.index = list(map(lambda x: f'sm_milford_{x}', meta.index))

    idx = collection['euclidean']['data'].columns

    type_day = [x.split('_')[-1] for x in idx]
    num_users = [meta.loc[x[:-8]]['inhabitants'] for x in idx]
    type_household = [meta.loc[x[:-8]]['type'] for x in idx]

    type_day = pd.Series(index=idx, data=type_day, name='type_day')
    num_users = pd.Series(index=idx, data=num_users, name='inhabitants')
    type_household = pd.Series(index=idx, data=type_household, name='type')
    type_household[list(map(lambda x: x.endswith('weekend'), type_household.index))] = 'home'

    new_meta = pd.concat([type_day, num_users, type_household], axis=1)
    new_meta['both'] = [' '.join(i) for i in zip(new_meta["inhabitants"].map(str), new_meta["type_day"])]

    data = collection['softdtw']['data']

    # pattern plots
    fig = plot_patterns(data=data, meta=new_meta)

    # Conumption plot
    fig = plot_consumption(data=data, meta=new_meta)

    # Cluster analysis plot
    fig = plot_cluster_analysis(dataset=dataset, norm=norm, legend_location='lower right', ylim=(0, None))

    for method in methods:

        d = collection[method]

        if (method == 'euclidean') and (clusters == 2):
            d['labels'] = (d['labels'] + 1) % 2

        positives, success_rate, error_rate = get_positives(labels=d['labels'].loc[clusters], meta=new_meta)

        # Doughnut plots
        fig = plot_doughnut(success_rate, legend_loc='best')


        fig = plot_consumption(data=d['data'], meta=new_meta, labels=positives)

        legend = True
        figs = plot_barycenter(data=d['data'], labels=d['labels'], meta=new_meta, norm=True, method=method,
                               clusters=clusters, alpha=0.5, legend=legend)

        # Silhouette plots:
        fig = plot_silhouette_values(silhouette=d['silhouette'], labels=d['labels'], clusters=clusters)

        # Link clusters to number of users:
        fig = numuser_barplot(meta=new_meta, labels=d['labels'], clusters=clusters, silhouette=d['silhouette'])

        # Link clusters to weekday and weekend:
        fig = type_day_barplot(meta=new_meta, labels=d['labels'], clusters=clusters, silhouette=d['silhouette'])

        # Link clusters to both:
        fig = combine_barplot(meta=new_meta, labels=d['labels'], clusters=clusters, silhouette=d['silhouette'])

        fig = type_barplot(meta=new_meta, labels=d['labels'], clusters=clusters, silhouette=d['silhouette'])

        # Conumption plot
        fig = plot_consumption(data=d['data'], meta=new_meta,  labels=d['labels'].loc[clusters])


        keys = ['sm_milford_3_weekend',
                'sm_milford_6_weekend',
                'sm_milford_11_weekday',
                'sm_milford_15_weekday',
                'sm_milford_15_weekend',]

        subset = d['data'][keys]

        # Special pattern plot
        fig = plot_simple_patterns(data=subset)
