from torch.autograd import Function
from collections import defaultdict, Counter
import random
import torch
import torch.nn.functional as F
import numpy as np
from IPython import embed
import scipy.sparse as sp
import networkx as nx
import sys
import pickle as pkl
from itertools import combinations
from scipy.stats import wasserstein_distance
import torch


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def generateUnseen(num_class, num_unseen):
    return combinations(range(num_class), num_unseen)


def load_data_dblp(args):
    dataset = args.dataset
    metapaths = args.metapaths
    sc = args.sc

    if dataset == 'acm':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format('dblp_v8_reducedlabel_20'), "rb"))
    label = data['label']
    N = label.shape[0]

    truefeatures = data['feature'].astype(float)

    rownetworks = [data[metapath] + np.eye(N) * sc for metapath in metapaths]
    # embed()
    rownetworks = [sp.csr_matrix(rownetwork) for rownetwork in rownetworks]

    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    truefeatures_list = []
    for _ in range(len(rownetworks)):
        truefeatures_list.append(truefeatures)

    return rownetworks, truefeatures_list, label, idx_train, idx_val, idx_test


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                # if True:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    # embed()
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    # embed()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # embed()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    # embed()
    # idx_train = range(len(y))
    if dataset_str == 'pubmed':
        idx_train = range(10000)
    elif dataset_str == 'cora':
        idx_train = range(1500)
    else:
        idx_train = range(1000)
    idx_val = range(len(y), len(y) + 500)


    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def createTraining(labels, max_train=200, balance=True, new_classes=[]):
    dist = defaultdict(list)
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)

    for idx, l in enumerate(labels.numpy().tolist()[:max_train]):
        dist[l].append(idx)
    print(dist)
    cat = []
    _sum = 0
    if balance:
        for k in dist:
            if k in new_classes:
                continue
            _sum += len(dist[k])
            # cat += random.sample(dist[k], k=15)
            train_mask[random.sample(dist[k], k=3)] = 1
    for k in new_classes:
        train_mask[random.sample(dist[k], k=3)] = 1
        # print(_sum, sum(train_mask))
    return train_mask
    # print(len(set(cat)))


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def ind_normalize_adj(adj):
    # """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


import itertools


def createDBLPTraining(labels, idx_train, idx_val, idx_test, max_train=20, balance=True, new_classes=[], unknown=False):
    labels = [np.where(r == 1)[0][0] if r.sum() > 0 else -1 for r in labels]
    # print(Counter(labels))
    new_mapping = {}
    dist = defaultdict(list)
    new_idx_train, new_idx_val, in_idx_test, out_idx_test, new_idx_test = [], [], [], [], []

    for idx in idx_train:
        dist[labels[idx]].append(idx)

    for k in range(len(dist)):
        if k not in new_classes:
            new_mapping[k] = len(new_mapping)
    # embed()
    if False:
        for idx in idx_train:
            if labels[idx]:
                # unknown label id
                new_idx_train.append(idx)
    else:
        for k in dist:
            # embed()
            if max_train < len(dist[k]):
                new_idx_train += np.random.choice(dist[k], max_train, replace=False).tolist()
            else:
                new_idx_train += dist[k]
            # print(len(set(new_idx_train)))

    for idx in idx_val:
        if labels[idx] in new_mapping:
            # unknown label id
            new_idx_val.append(idx)
        else:
            new_idx_val.append(idx)

    for idx in idx_test:
        if labels[idx] in new_mapping:
            # unknown label id
            new_idx_test.append(idx)
            in_idx_test.append(idx)
        else:
            # unknown class
            if unknown:
                new_idx_test.append(idx)
                out_idx_test.append(idx)

    for idx, label in enumerate(labels):
        if label < 0:
            continue
        if label in new_mapping:
            labels[idx] = new_mapping[label]
        else:
            labels[idx] = len(new_mapping)
    # print('its here')
    # embed()
    return new_idx_train, new_idx_val, in_idx_test, new_idx_test, out_idx_test, labels


def createPPITraining(train_labels, val_labels, test_labels, idx_train, idx_val, idx_test, new_classes=[],
                      unknown=False):
    # labels = [np.where(r==1)[0][0] for r in labels]
    new_mapping = {}
    dist = defaultdict(list)
    new_idx_train, new_idx_val, in_idx_test, out_idx_test, new_idx_test = [], [], [], [], []

    for idx in idx_train:
        dist[train_labels[idx]].append(idx)

    for k in range(len(dist)):
        if k not in new_classes:
            new_mapping[k] = len(new_mapping)

    for idx in idx_train:
        assert train_labels[idx] > -1
        if train_labels[idx] in new_mapping:
            # unknown label id
            new_idx_train.append(idx)
            train_labels[idx] = new_mapping[train_labels[idx]]
        else:
            train_labels[idx] = len(new_mapping)

    for idx in idx_val:
        assert val_labels[idx] > -1
        if val_labels[idx] in new_mapping:
            # unknown label id
            new_idx_val.append(idx)
            val_labels[idx] = new_mapping[val_labels[idx]]
        else:
            new_idx_val.append(idx)
            val_labels[idx] = len(new_mapping)

    for idx in idx_test:
        assert test_labels[idx] > -1
        if test_labels[idx] in new_mapping:
            # unknown label id
            new_idx_test.append(idx)
            in_idx_test.append(idx)
            test_labels[idx] = new_mapping[test_labels[idx]]
        else:
            # unknown class
            test_labels[idx] = len(new_mapping)
            if unknown:
                new_idx_test.append(idx)
                out_idx_test.append(idx)

    print(new_mapping)
    return new_idx_train, new_idx_val, in_idx_test, out_idx_test, new_idx_test


def createClusteringData(labels, idx_train, idx_val, idx_test, max_train=200, balance=True, new_classes=[],
                         unknown=False):
    labels = [np.where(r == 1)[0][0] for r in labels]
    #
    # np.where(labels==1)
    new_mapping = {}

    dist = defaultdict(list)

    for idx in idx_train:
        dist[labels[idx]].append(idx)
        # train_mask[idx] = 1

    for k in range(len(dist)):
        if k not in new_classes:
            new_mapping[k] = len(new_mapping)
    print('new mapping is {}'.format(new_mapping))
    new_idx_train, new_labels = [], []

    # embed()
    for idx in idx_train:
        if labels[idx] not in new_classes:
            new_idx_train.append(idx)

    for idx, label in enumerate(labels):
        if label in new_mapping:
            new_labels.append(new_mapping[label])
        else:
            new_labels.append(len(new_mapping))

    # print(dist)
    # embed()
    return new_idx_train, labels, new_labels


def raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    lam_reg = 0.01
    k_reg = min(5, cal_smx.shape[1])
    disallow_zero_sets = False
    rand = True
    # Create regularization vector
    reg_vec = np.array(k_reg * [0, ] + (cal_smx.shape[1] - k_reg) * [lam_reg, ])[None, :]
    # Sort the calibration softmax scores in descending order and get their indices
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    # Rearrange the calibration softmax scores according to the sorted indices
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    # Add regularization to the sorted calibration scores
    cal_srt_reg = cal_srt + reg_vec
    #print(cal_pi.shape, cal_labels.shape)
    #print(cal_pi, cal_labels)

    # Ensure cal_labels has the correct shape for broadcasting
    cal_labels = np.array(cal_labels)
    if cal_labels.ndim == 1:
        cal_labels = cal_labels[:, None]
    # Find the index of the true label in the sorted array for each sample
    cal_L = np.where(cal_pi == cal_labels)[1]
    # print("cal_L:", cal_L)
    # cal_L = np.where(cal_pi == cal_labels[:, None])[1]
    # print(cal_L)
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,
                                                              1) * val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(
        axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:, 0] = True
    prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets, cov, eff




def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    # print(cal_srt, cal_pi, cal_scores)
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    # print(qhat)
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    # print(val_pi, val_srt, prediction_sets)
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets, cov, eff


def biased_Data(one_hot_labels, ori_idx_train, idx_val, ori_idx_test, DATASET, ppr_dist, labels):
    n_repeats = 5
    max_train = 20
    # an example of biased training data
    for _run in range(n_repeats):
        # biased training data
        # generate biased sample
        if True:
            train_seeds, _, _, _, _, _ = createDBLPTraining(one_hot_labels, ori_idx_train, idx_val,
                                                                  ori_idx_test, max_train=1)
            label_idx = []
            if DATASET == 'pubmed':
                num_pool = 10000
            elif DATASET == 'cora':
                num_pool = 1500
            else:
                num_pool = 1000
            for i in train_seeds:
                label_idx.append(torch.where(labels[:num_pool] == labels[i])[0])
            ppr_init = {}
            for i in train_seeds:
                ppr_init[i] = 1
            # print(train_seeds)
            idx_train = []
            for idx in range(len(train_seeds)):
                idx_train += label_idx[idx][
                    ppr_dist[train_seeds[idx], label_idx[idx]].argsort()[:max_train]].tolist()
    return idx_train


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1)
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def KMM(X, Xtest, _A=None, _sigma=1e1, beta=0.2):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    # H /= 3
    # f /= 3
    # z /= 3
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0], 1)))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A == 0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples, 1))
    #print(torch.linalg.matrix_rank())
    '''
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h),
                     matrix(_A), matrix(b))
    return np.array(sol['x']), MMD_dist.item()
    '''
    return MMD_dist.item()

def MMD(X, Xtest):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    return MMD_dist.item()


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K - 1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1, sx2, i + 2))
        # scms+=moment_diff(sx1,sx2,1)
    return sum(scms)


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    # ss1 = sx1.mean(0)
    # ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)


def kld(X, Xtest, epsilon=1e-10):
    """
    Compute KL divergence directly between two tensors.
    
    Args:
    X (torch.Tensor): First tensor
    Xtest (torch.Tensor): Second tensor
    epsilon (float): Small value to avoid log(0) and division by zero
    
    Returns:
    float: The mean KL divergence
    """
    # Ensure inputs are PyTorch tensors
    X = torch.as_tensor(X, dtype=torch.float32)
    Xtest = torch.as_tensor(Xtest, dtype=torch.float32)
    
    # Clamp values to avoid numerical issues
    X = torch.clamp(X, min=epsilon, max=1.0)
    Xtest = torch.clamp(Xtest, min=epsilon, max=1.0)
    
    # Compute KL divergence
    kl_div = torch.sum(X * torch.log(X / Xtest), dim=1)
    
    # Return mean KL divergence
    return kl_div.mean().item()


def jsd(X, Xtest):
    m = 0.5 * (X + Xtest)
    return 0.5 * (kld(X, m) + kld(Xtest, m))


# def emd(X, Xtest):
#     cdf_X = torch.cumsum(X, dim=-1)
#     cdf_Xtest = torch.cumsum(Xtest, dim=-1)
#     return torch.sum(torch.abs(cdf_X - cdf_Xtest)).mean()




# def emd(X, Xtest):
#     """
#     Compute Earth Mover's Distance (EMD) between two distributions using PyTorch.
    
#     Args:
#     X (torch.Tensor): First tensor
#     Xtest (torch.Tensor): Second tensor
    
#     Returns:
#     float: The EMD
#     """
#     # Ensure inputs are PyTorch tensors
#     X = torch.as_tensor(X, dtype=torch.float32)
#     Xtest = torch.as_tensor(Xtest, dtype=torch.float32)
    
#     # Normalize the distributions
#     X = X / torch.sum(X)
#     Xtest = Xtest / torch.sum(Xtest)
    
#     # Compute the cumulative distributions
#     cdf_X = torch.cumsum(X, dim=-1)
#     cdf_Xtest = torch.cumsum(Xtest, dim=-1)
    
#     # Compute the EMD
#     emd = torch.sum(torch.abs(cdf_X - cdf_Xtest))
    
#     return emd.item()



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



