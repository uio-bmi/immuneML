INTERNAL_EVAL_METRICS = ['calinski_harabasz_score', 'davies_bouldin_score', 'silhouette_score']
EXTERNAL_EVAL_METRICS = ['rand_score', 'adjusted_rand_score', 'adjusted_mutual_info_score', 'completeness_score',
                         'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score',
                         'normalized_mutual_info_score', 'v_measure_score']


def is_internal(metric: str):
    return metric in INTERNAL_EVAL_METRICS


def is_external(metric: str):
    return metric in EXTERNAL_EVAL_METRICS
