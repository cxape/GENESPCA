import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import time 
import igraph
import scipy
from copy import deepcopy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn import metrics
from matplotlib.patches import Ellipse
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

def get_time_str():
    t = time.localtime()
    time_str = f"{t.tm_year}/{t.tm_mon:02d}/{t.tm_mday:02d} {t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
    return time_str


def clr_normalize_each_cell(adata, inplace=True):
    """Modified from SpatialGlue code"""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)
    if not inplace:
        adata = adata.copy()
    adata.X = np.apply_along_axis(seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X)))
    return adata


def preprocess_adata(adata_list, filter_gene=25, filter_cell=50, hvg=2000):
    adata_rna, adata_adt = adata_list
    sc.pp.filter_genes(adata_rna, min_cells=filter_gene)
    sc.pp.filter_cells(adata_rna, min_genes=filter_cell)
    adata2 = adata_adt[adata_rna.obs_names].copy()
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=hvg)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata1 = adata_rna[:, adata_rna.var['highly_variable']]
    adata2 = clr_normalize_each_cell(adata2)
    sc.pp.scale(adata2)
    pos = np.array(adata1.obsm['spatial'])
    X1, X2 = adata1.X.toarray(), adata2.X
    return X1, X2, pos


def clustering_metric(y, y_pred, round=5):
    ami = np.round(metrics.adjusted_mutual_info_score(y, y_pred), round)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), round)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), round)
    return ami, nmi, ari


def preprocess_hvg(x_list=[], select_list=[], top=1000, verbose=1):
    assert len(x_list) == len(select_list)
    x_selected_list = []
    for i, x in enumerate(x_list):
        if select_list[i]:
            if verbose > 0:
                print("selecting top", top, "hvg for modality", i + 1)
            hvg_ind = geneSelection(x, num_genes=top, verbose=verbose)
            x_hvg = x[:, hvg_ind]
            x_selected_list.append(x_hvg)
        else:
            x_selected_list.append(x)

    if verbose > 0:
        print("normalizing counts")
    x_normalized_list = []
    for i, x_selected in enumerate(x_selected_list):
        adata = sc.AnnData(x_selected)
        adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)
        x_normalized = adata.X
        x_normalized_list.append(x_normalized)

    return tuple(x_normalized_list)


# def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
#     if filter_min_counts:
#         
#         sc.pp.filter_genes(adata, min_counts=1)
#         sc.pp.filter_cells(adata, min_counts=1)
#     if size_factors or normalize_input or logtrans_input:
#         adata.raw = adata.copy()
#     else:
#         adata.raw = adata
#     if size_factors:
#         
#         sc.pp.normalize_per_cell(adata)
#         
#         adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
#     else:
#         adata.obs['size_factors'] = 1.0
#     if logtrans_input:
#         sc.pp.log1p(adata)
#     if normalize_input:
#         sc.pp.scale(adata)
#     return adata

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        #sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        #sc.pp.normalize_per_cell(adata)
        sc.pp.normalize_total(adata)
        #adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def geneSelection(data, threshold=0, at_least=10,
                  y_offset=.02, x_offset=5, decay=1.5, num_genes=1000,
                  plot=False, markers=None, genes=None, figsize=(6, 3.5),
                  marker_offsets=None, num_labels=10, alpha=1, verbose=1):
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (1 - zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:, detected] > threshold
        logs = np.zeros_like(data[:, detected]) * np.nan
        logs[mask] = np.log2(data[:, detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < at_least
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if num_genes is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - x_offset)) + y_offset
            if np.sum(selected) == num_genes:
                break
            elif np.sum(selected) < num_genes:
                up = x_offset
                x_offset = (x_offset + low) / 2
            else:
                low = x_offset
                x_offset = (x_offset + up) / 2
        if verbose > 0:
            print('Chosen offset: {:.2f}'.format(x_offset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - x_offset)) + y_offset
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + .1, .1)
        y = np.exp(-decay * (x - x_offset)) + y_offset
        if decay == 1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected), x_offset, y_offset),
                     color='k', fontsize=num_labels, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected), decay, x_offset, y_offset),
                     color='k', fontsize=num_labels, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:, None], y[:, None]), axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        if markers is not None and genes is not None:
            if marker_offsets is None:
                marker_offsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = marker_offsets[num]
                plt.text(meanExpr[i] + dx + .1, zeroRate[i] + dy, g, color='k', fontsize=num_labels)
    return selected


def plot_cluster(labels: np.ndarray, pos: np.ndarray, colorList: list, pointSize=1, show=True):
    assert len(labels) == pos.shape[0]
    xList = pos[:, 0]
    yList = pos[:, 1]
    for i in range(len(xList)):
        plt.plot(xList[i], yList[i], marker='o', color=colorList[labels[i]], markersize=pointSize)
    plt.gca().set_aspect(1)
    if show:
        plt.show()



def SNN_adj(A):
    Am = A.copy()
    indices = np.split(A.indices, A.indptr)[1:-1]
    for i in range(A.shape[0]):
        for j in indices[i]:
            if A[j, i] == 0:
                Am[i, j] = 0
    Am.eliminate_zeros()
    return Am


def walkTrapCluster(latent: np.ndarray, n_clusters, n_neighbors=400):
    A = kneighbors_graph(latent, n_neighbors=n_neighbors, mode="connectivity", metric="euclidean", include_self=False, n_jobs=-1)
    A = SNN_adj(A)
    g = igraph.Graph.Weighted_Adjacency(A)
    wTrap = g.community_walktrap(weights=g.es["weight"])
    clust = wTrap.as_clustering(n=n_clusters)
    y_pred = np.array(clust.membership)
    return y_pred


def refine_cluster(cluster_labels: np.ndarray, location: np.ndarray, shape="hexagon"):
    assert cluster_labels.shape[0] == location.shape[0]
    dist_vec = pdist(location, metric='euclidean')
    dist_mat = squareform(dist_vec)
    if shape == "hexagon":
        nearby_num = 6
    elif shape == "square":
        nearby_num = 4
    else:
        print("Select shape='hexagon' for Visium data, 'square' for ST data. Using default: hexagon!")
        nearby_num = 6
    refined_labels = deepcopy(cluster_labels)
    for i in range(len(cluster_labels)):
        dist_vec = dist_mat[i, ]  # 当前点的距离向量
        nearby_index = dist_vec.argsort()[0: nearby_num + 1]  # 包括自己在内最近的nearby_num个点的下标
        nearby_labels = refined_labels[nearby_index]  # 包括自己在内最近的num_obs个点的labels
        label_set, label_count = np.unique(nearby_labels, return_counts=True)
        max_count_index = np.argmax(label_count)
        max_count = label_count[max_count_index]
        max_count_label = label_set[max_count_index]
        if (max_count_label != refined_labels[i]) and (max_count > nearby_num // 2):
            refined_labels[i] = max_count_label
    return refined_labels