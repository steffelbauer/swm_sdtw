import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .source.utils import compute_barycenter, normalize, load_data
from .source.plot_scripts import plot_cluster_analysis, plot_silhouette_values, plot_doughnut
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter
sns.set_style('darkgrid')

colors = sns.color_palette('viridis', 7)
workcolor = colors[0]
homecolor = colors[3]


def plot_barycenter(data=None, labels=None, positives=None, clusters=2, method='softdtw', show=True, colors=None, norm=True):

    if colors is None:
        colors = sns.color_palette('viridis', 7)

    data.index = pd.to_datetime(data.index)

    cl = labels.loc[clusters]
    figs = []
    for c in sorted(cl.unique()):

        fig, ax = plt.subplots()

        idx = cl[cl == c].index

        s = data[idx]

        if positives is not None:

            tp_plot = plt.plot([-100, -99], [-100, -99], c=homecolor, linestyle='-', alpha=0.8, linewidth=2.0)
            fp_plot = plt.plot([-100, -99], [-100, -99], c=workcolor, linestyle='--', alpha=0.8, linewidth=2.0)
            posidx = positives[idx]
            subidx = posidx[posidx == True].index

            TP = len(subidx)
            if not subidx.empty:
                ax.plot(s[subidx].index, s[subidx].values, c=homecolor, alpha=0.2)

            subidx = posidx[posidx == False].index
            FP = len(subidx)
            if not subidx.empty:
                ax.plot(s[subidx].index, s[subidx].values, c=workcolor, alpha=0.2, linestyle='--')
        else:

            ax.plot(s.index, s.values, c=homecolor, alpha=0.2)

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

        plt.legend(loc='upper right', fontsize=16, frameon=False)
        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((0, None))

        hfmt = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        plt.xticks(rotation=45)
        plt.ylabel(r'$Q \quad [L/s]$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.legend([tp_plot[0], fp_plot[0], bary_plot[0], mu_plot[0]],
                   [f'TP ({TP:2.0f})', f'FP ({FP:2.0f})', r'$\mathbf{x}^\ast$', r'$\mathbf{\mu}$'],
                   loc='upper left',
                   frameon=False,
                   fontsize=12)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        xmin = data.index[0]
        xmax = data.index[-1]
        ymin = 0
        ymax = 0.014

        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))

        if show:
            plt.show()

        figs.append(fig)

    return figs


def get_positives(labels=None, method='softdtw'):
    if method == 'softdtw':
        work_label = 0
        home_label = 1
    elif method == 'euclidean':
        work_label = 1
        home_label = 0

    positives = dict()
    for key, value in labels.iteritems():
        if key.lower().startswith('home') and value == home_label:
            positives[key] = True
        elif key.lower().startswith('work') and value == work_label:
            positives[key] = True
        if key.lower().startswith('home') and value == work_label:
            positives[key] = False
        elif key.lower().startswith('work') and value == home_label:
            positives[key] = False

    positives = pd.Series(positives)

    success_rate = positives.sum() / len(positives) * 100
    error_rate = 100 - success_rate
    print(f'Success Rate={success_rate}; Error Rate={error_rate}')
    return positives, success_rate, error_rate


def plot_consumption(data=None, show=True):

    cons = data.mean() * 3600 * 24

    user = pd.Series(data=[x[:4] for x in cons.index], index=cons.index)

    user.name = 'user'
    cons.name = 'cons'
    df = pd.concat([cons, user], axis=1)

    violinplot = sns.violinplot(x='user', y='cons', data=df, palette='viridis', alpha=0.7, inner="stick")
    plt.xlabel('User type', fontsize=16)
    plt.ylabel('Consumption   (L/day)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if show:
        plt.show()

    fig = violinplot.get_figure()

    return fig


def plot_variance(data=None, show=True):

    cons = data.std() ** 2

    user = pd.Series(data=[x[:4] for x in cons.index], index=cons.index)

    user.name = 'user'
    cons.name = 'cons'
    df = pd.concat([cons, user], axis=1)

    violinplot = sns.violinplot(x='user', y='cons', data=df, palette='viridis', alpha=0.7, inner="stick")
    plt.xlabel('User type', fontsize=16)
    plt.ylabel(r'Variance   $(L^2)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    if show:
        plt.show()

    fig = violinplot.get_figure()

    return fig


def plot_patterns(data=None, show=True):

    data.index = pd.to_datetime(data.index)
    users = pd.Series(data=[x[:4] for x in data.columns], index=data.columns)

    uniques = users.unique()

    colors = sns.color_palette('viridis', len(uniques) + 1)

    fig, ax = plt.subplots()

    u1_plot = ax.plot([-100, -99], [-100, -99], c=colors[0], linestyle=':', alpha=0.8, linewidth=2.0)
    u2_plot = ax.plot([-100, -99], [-100, -99], c=colors[1], linestyle='-', alpha=0.8, linewidth=2.0)

    linestyles = [':', '-']
    alphas = [0.7, 0.3]

    for ind, user in enumerate(sorted(users.unique(), reverse=True)):
        idx = users[users == user].index
        x = data[idx]
        ax.plot(x.index, x.values, color=colors[ind], alpha=alphas[ind], linestyle=linestyles[ind])

    mu_plot = ax.plot(x.index, x.mean(axis=1).values, color='k', linestyle='--')

    plt.xlim([x.index[0], x.index[-1]])
    plt.ylim((0, 0.012))
    hfmt = DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(hfmt)
    plt.xticks(rotation=45)
    plt.ylabel(r'$Q \quad [L/s]$', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.legend([u1_plot[0], u2_plot[0], mu_plot[0]],
               [f'Work', f'Home',
                r'$\mathbf{\mu}$'],
               loc='upper right',
               frameon=False,
               fontsize=12)

    if show:
        plt.show()

    return fig


if __name__ == '__main__':

    # Parameters
    dataset = 1
    clusters = 9
    norm = True
    methods = ['euclidean', 'softdtw']

    # Program Start:
    collection = load_data(dataset=dataset, norm=norm)

    fig = plot_consumption(data=collection['euclidean']['data'])
    fig = plot_variance(data=collection['euclidean']['data'])
    fig = plot_patterns(data=collection['euclidean']['data'])
    fig = plot_cluster_analysis(dataset=dataset, norm=norm)

    for method in methods:
        d = collection[method]
        labels = collection[method]['labels'].loc[clusters]
        positives, success_rate, error_rate = get_positives(labels=labels, method=method)

        figs = plot_barycenter(data=d['data'], labels=d['labels'], clusters=clusters, positives=positives, method=method,
                              show=True, colors=sns.color_palette('viridis', 9), norm=norm)

        fig = plot_doughnut(success_rate)

        # Silhouette plots
        labels = d['labels']
        if method == 'euclidean':
             labels = (labels + 1) % 2

        fig = plot_silhouette_values(silhouette=d['silhouette'], labels=labels, clusters=clusters, show=True)

        # Last figure
        l = d['labels'].loc[clusters] + 1
        l.name = 'labels'
        l.index.name = 'id'
        l = pd.DataFrame(l)
        l['depl'] = list(map(lambda x: x[:4], l.index))
        test5 = pd.crosstab(index=l['depl'], columns=l['labels'])
        test5.plot(kind='bar', stacked=True, colors=[homecolor, workcolor], alpha=0.7)
