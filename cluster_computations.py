from .source.utils import *

# Simulation parameters:
dataset_number = '2'
norm = True
method = 'euclidean'


df = pd.read_csv(os.path.join('data', f'dataset{dataset_number}.csv'), index_col=0, header=0)

if norm:
    df = normalize(df)
    norm_text = 'norm_'
else:
    norm_text = ''

cl = cluster_number_computation(df, metric=method, max_cluster=10, init='k-means++', random_state=1234, n_init=10)
cl.to_csv(os.path.join('results', f'dataset{dataset_number}', f'ds{dataset_number}_{norm_text}cluster_labels_{method}.csv'))
sc = cluster_analysis(df, cluster_labels=cl, method=method)
sc.to_csv(os.path.join('results', f'dataset{dataset_number}', f'ds{dataset_number}_{norm_text}silhouette_coefficients_{method}.csv'))
