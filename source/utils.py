import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn import metrics as skl_metrics
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import euclidean_barycenter, softdtw_barycenter, dtw_barycenter_averaging
from tslearn.metrics import soft_dtw
import seaborn as sns
sns.set_style('darkgrid')


def daily_pattern(df, how='mean'):
    """Compute daily pattern from smart meter data.

    Args:
        df: pandas Series or DataFrame with Smart Meter IDs as columns names, datetime index and consumption values
        in L/s.
        how: arbitrary numpy function as string to apply for data aggregation

    Returns:
        Daily consumption pattern as pandas Series or DataFrame with timedelta index (including hours, minutes,
        seconds).

    """
    value = df.groupby(by=df.index.time).apply(getattr(np, how))
    value.index = map(lambda x: pd.to_timedelta(x.strftime('%H:%M:%S')), value.index)

    return value


def simdeum_to_pattern(input_path: str = '', base_demand_file: str = 'SIMDEUM_base_demand_file.txt',
                       pattern_file: str = 'SIMDEUM_pattern_file.txt'):

    bd = pd.read_csv(os.path.join(input_path, base_demand_file), skiprows=6, header=None, squeeze=True, index_col=0, sep=' ',
                     usecols=[0, 1])
    bd.name = 'base_demand'
    bd.index.name = 'id'

    p = pd.read_csv(os.path.join(input_path, pattern_file), skiprows=9, header=None, index_col=0, sep=' ', nrows=1000,
                    usecols=range(100*24*12+1))
    p.index.name = 'id'
    start = pd.to_datetime(pd.datetime.today().date())

    dt_index = pd.DatetimeIndex(start=start, freq='5Min', periods=24 * 12 * 100)

    p.columns = dt_index

    p = p.T
    p.index.name = 'datetime'

    dp = daily_pattern(p)
    dp = dp.resample('30Min').mean()
    dp = dp.rolling('2H').mean()

    return dp


def normalize(df):
    """Normalize time series by rescaling  of the data from the original range so that all values are within the range
    of 0 and 1.

    Normalization can be useful, and even required in some machine learning algorithms when your time series data has
    input values with differing scales.It may be required for algorithms, like k-Nearest neighbors, which uses distance
    calculations and Linear Regression and Artificial Neural Networks that weight input values (taken from
    https://machinelearningmastery.com/normalize-standardize-time-series-data-python/).

    Args:
        df: pandas Series or DataFrame with Smart Meter IDs as columns names, datetime index and consumption values
        in L/s.

    Returns:
        Normalized time series as pandas Series or DataFrame.

    """

    return (df - df.min()) / (df.max() - df.min())


def sdtw(x, y):
    """Normalized version of Soft-DTW used as a metric.
        is defined as `sdtw_(x,y) := sdtw(x,y) - 1/2(sdtw(x,x)+sdtw(y,y))`.

    Args:
        x:
        y:

    Returns:

    """

    return soft_dtw(x, y) - 0.5 * (soft_dtw(x, x) + soft_dtw(y, y))


def clustering(df, n_cluster: int = 2, metric: str = 'softdtw', init='k-means++', random_state=1234, verbose=False,
               n_init=1):

    tsk = TimeSeriesKMeans(n_clusters=n_cluster, metric=metric, init=init, random_state=random_state, verbose=verbose,
                           n_init=n_init)
    df = np.roll(df, -6, axis=0)
    M = to_time_series_dataset(df.T)

    cluster_labels = tsk.fit_predict(M)

    return cluster_labels


def cluster_number_computation(df, max_cluster: int = 5, metric: str = 'softdtw', init: str = 'k-means++',
                               random_state: int = 1234, verbose=False, n_init=1) -> dict:
    result = dict()
    for cluster in range(2, max_cluster+1):
        print(f'Compute for k={cluster}...')
        cluster_labels = clustering(df, n_cluster=cluster, metric=metric, init=init, random_state=random_state,
                                    verbose=verbose, n_init=n_init)
        result[cluster] = cluster_labels

    result = pd.DataFrame.from_dict(result, orient='index')
    result.index.name = 'n_cluster'
    result.columns = df.columns
    return result


def silhouette_coefficients(df, cluster_labels, method: str = 'euclidean'):

    if method == 'euclidean':
        metric = euclidean
    elif method == 'softdtw':
        metric = sdtw
    else:
        raise Exception(f'{method} is an unknown metric! Use either '"euclidean"' or '"softdtw"'...')

    X = df.T.values
    sample_silhouette_values = skl_metrics.silhouette_samples(X, cluster_labels, metric=metric)
    sample_silhouette_values = pd.Series(data=sample_silhouette_values, index=df.columns)
    return sample_silhouette_values


def cluster_analysis(df, cluster_labels, method: str = 'euclidean'):

    res = dict()
    for key, row in cluster_labels.iterrows():
        cl = row
        sc = silhouette_coefficients(df, cl, method=method)
        res[key] = sc
        print(key, sc.mean())
    res = pd.DataFrame.from_dict(res, orient='index')
    return res


def compute_barycenter(s, method='softdtw'):

    v = s.T.values
    if method == 'euclidean':
        barycenter_method = euclidean_barycenter
    elif method == 'softdtw':
        barycenter_method = softdtw_barycenter
    elif method == 'dtw':
        barycenter_method = dtw_barycenter_averaging
    else:
        raise NotImplementedError(f'Method {method} is unknown or not implemented yet [euclidean, softdtw]')

    barycenter = barycenter_method(v)
    # print(barycenter)

    return pd.Series(data=barycenter.flatten(), index=s.index)


def load_cluster_analysis(dataset=1, norm=False, methods=None):

    if methods is None:
        methods = ['euclidean', 'softdtw']

    normstr = ('norm_' if norm else '')
    result = dict()

    for met in methods:
        file = os.path.join('results', f'dataset{dataset}', f'ds{dataset}_{normstr}silhouette_coefficients_{met}.csv')
        silhouette = pd.read_csv(file, index_col=0, header=0)

        result[met] = silhouette.mean(axis=1)

    return pd.DataFrame(result)


def load_data(dataset: int = 1, methods: list = ['euclidean', 'softdtw'], norm: bool = True):

    normstr = ('norm_' if norm else '')

    collection = dict()

    for method in methods:
        file_data = os.path.join('data', f'dataset{dataset}.csv')
        file_cluster = os.path.join('results', f'dataset{dataset}', f'ds{dataset}_{normstr}cluster_labels_{method}.csv')
        file_silhouette = os.path.join('results', f'dataset{dataset}', f'ds{dataset}_{normstr}'
                                                                       f'silhouette_coefficients_{method}.csv')
        data = pd.read_csv(file_data, index_col=0, header=0)
        labels = pd.read_csv(file_cluster, index_col=0, header=0)
        silhouette = pd.read_csv(file_silhouette, index_col=0, header=0)

        # if norm:
        norm_data = normalize(data)

        collection[method] = dict()
        collection[method]['data'] = data
        collection[method]['norm_data'] = norm_data
        collection[method]['labels'] = labels
        collection[method]['silhouette'] = silhouette

    return collection
