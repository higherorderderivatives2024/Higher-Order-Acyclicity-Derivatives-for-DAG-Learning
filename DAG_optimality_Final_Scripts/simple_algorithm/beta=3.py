# -*- coding: utf-8 -*-
"""Hessian4_closed_form.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p9K4PhWN9Y5a3l_4AXdxTp4-rbve_OeT
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
import scipy.linalg as slin
from copy import copy
import numpy.linalg as la
import sys
import typing
import random
import igraph as ig
import time
import scipy.stats as st


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d,graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=2*d)
        #G_und = ig.Graph.Erdos_Renyi(n=d, p=p)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)

    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G_und = ig.Graph.Barabasi(n=d, m=2)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)

    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


"""
# def simulate_parameter(B, w_ranges=((-10.0, -0.01), (0.01, 10.0))):
def simulate_parameter(B, w_ranges=(-2, 1)):
    #Simulate SEM parameters for a DAG.

    #Args:
    #    B (np.ndarray): [d, d] binary adj matrix of DAG
    #    w_ranges (tuple): disjoint weight ranges

    #Returns:
    #    W (np.ndarray): [d, d] weighted adj matrix of DAG

    W = np.zeros(B.shape)
    # S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    S = np.random.randint(2, size=B.shape)
    for sign in range(2):
        U = np.power(10, np.random.uniform(w_ranges[0], w_ranges[1], size=B.shape))
        W += B * (S == sign) * U
    return W
"""


def simulate_parameter(B, w_ranges=((-2.0, -0.01), (0.01, 2.0))):
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z

        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X_list = []
    X_list_zeros = []
    for idx, action in enumerate(action_list):
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j] = action[j] * _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])

        action_idx = action_idx_list[idx]
        X_list_zeros.append(X)
        X = np.delete(X, action_idx, axis=1)
        X_list.append(X)
    return ordered_vertices, X_list, X_list_zeros


def produce_action_list(gene_knockouts):
    action_list = []
    P_list = []
    first_terms = []
    second_terms = []
    for knockout in gene_knockouts:
        action = np.ones(d)

        for gene in knockout:
            action[gene] = 0

        action_list.append(action)

        P = np.zeros((d, d - len(knockout)))
        idx = 0
        for gene in range(d):

            if gene not in knockout:
                p = np.zeros(d)
                p[gene] = 1
                P[:, idx] = p
                idx = idx + 1

        P_list.append(P)

    return gene_knockouts, action_list, P_list


def threshold_W(W, threshold=0.3):
    """
    :param W: adjacent matrix
    :param threshold:
    :return: a threshed adjacent matrix
    """
    W_new = np.zeros_like(W)
    W_new[:] = W
    W_new[np.abs(W_new) < threshold] = 0
    return W_new


def create_Z(ordering):
    d = len(ordering)
    Z = np.ones((d, d), dtype=bool)
    for i in range(d - 1):
        Z[ordering[i], ordering[i + 1:]] = False
    return Z


"""
def create_Z_reg(ordering):

    #Create edge absence constraints \mathcal{Z} corresponding to topological ordering
    #:param ordering: topological sort
    #:return: bool matrix

    #create_Z([0,1,2,3])
    #Out:
    #array([[ True, False, False, False],
    #   [ True,  True, False, False],
    #   [ True,  True,  True, False],
    #   [ True,  True,  True,  True]])


    d = len(ordering)
    Z = np.ones((d, d), dtype=bool)
    for i in range(d):
        Z[ordering[i], ordering[i:]] = False
    return Z
"""


def create_new_topo(topo, idx, opt=1):
    '''
    Args:
        topo: topological
        index: (i,j)
        opt: 1: how to swap position of i and j
    Returns:
        a new topological sort
    '''

    topo0 = copy(topo)
    topo0 = list(topo0)
    i, j = idx
    i_pos, j_pos = topo0.index(i), topo0.index(j)
    if opt == 1:
        topo0[i_pos] = idx[1]
        topo0[j_pos] = idx[0]
    elif opt == 2:
        topo0.remove(j)
        topo0.insert(i_pos, j)
    else:
        topo0.remove(i)
        topo0.insert(j_pos, i)
    return topo0


def find_hgrad_index(G_loss, G_h, Z, thres=1e-2):
    """
    Find where {(i.j)| i\neq j, (G_h)_{ij}<thres, Z[i,j] = True }

    :param G_h: gradient of h
    :param Z: edge absence constaints
    :param thres: threshold for gradient of h
    :return: set {(i.j)| i\neq j, (G_h)_{ij}<thres, Z[i,j] = True }
    """
    # G = np.divide(G_loss, G_h)

    # index = np.transpose(np.where(np.logical_and(np.absolute(G_loss0/G_h0) >= thres, Z)))

    # not0grad = np.logical_or(G_loss <= (-thres)*(G_h+10**(-20)), G_loss >= thres*(G_h+10**(-20)))
    # not0grad = np.logical_and(G >= (-thres), G <= thres)
    index = np.transpose(np.where(np.logical_and(G_loss >= thres * G_h, Z)))
    # print("G")
    # print(G[np.where(np.logical_and(not0grad, Z))])
    # index = np.transpose(np.where(np.logical_and(G<= thres, Z)))
    # index = index[index[:, 1] != index[:, 0],]
    return index


def find_Fgrad_index(G_loss, G_h, Z, thres=1e-3):
    """
    Find where {(i,j)| G_loss(i,j) not = 0 and Z(i,j)= True}

    :param G_loss: gradient of Loss function
    :param Z: edge absence constaints
    :param thres:
    :return: set {(i.j)| i\neq j, |(G_F)_{ij}|>=thres, Z[i,j] = True }
    """
    not0grad = np.logical_or(G_loss <= (-thres) * G_h, G_loss >= thres * G_h)
    index = np.transpose(np.where(np.logical_and(not0grad, Z)))
    index0 = index[index[:, 1] != index[:, 0],]
    return index0


def find_common(indx1, indx2):
    """
    find the intersection between indx1 and indx2

    :param indx1: index set A
    :param indx2: index set B
    :return: return A\cap B
    """
    A = list(zip(indx1[:, 0], indx1[:, 1]))
    B = list(zip(indx2[:, 0], indx2[:, 1]))
    return set(A).intersection(B)


"""
def find_idx_set(G_h, G_loss, Z, size_small, size_large):


    gFs = [0]
    # gFs =  [0, 1e-8, 1e-6, 1e-4, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 1,2,3,4]
    # gFs = [0, 1e-8, 1e-6, 1e-4, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50]
    ghs = sorted([5e11,1e11,5e10,1e10,5e9,1e9,5e8,1e8,5e7,1e7,5e6,1e6,5e5,1e5,50000,10000,5000,1000,500,100,70,60,50,40, 30, 20, 10, 5, 2, 1, 0])
    #ghs = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    # gFs = [0, 1e-7, 1e-6, 5e-6,1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1,2,5,10,15,20,40]

    M = np.zeros(len(ghs))
    G = np.divide(G_loss, G_h)
    for count_gh, gh in enumerate(ghs):
        indx1 = find_hgrad_index(G, Z=Z, thres=gh)
        # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}
        M[count_gh] = len(indx1)

    i1= np.unravel_index(np.argmin(np.abs(M - size_small), axis=None), M.shape)[0]
    i2 = np.unravel_index(np.argmin(np.abs(M - size_large), axis=None), M.shape)[0]
    print("threshold")
    print(ghs[i1])
    print(ghs[i2])

    indx1_small= find_hgrad_index(G, Z=Z, thres=5e8)

    index_set_small= set(list(zip(indx1_small[:, 0], indx1_small[:, 1])))

    if len(index_set_small) > size_small + 20 and ghs[i1] == 0:
        size1_th_largest = np.partition(np.abs(G_loss[(indx1_small[:, 0], indx1_small[:, 1])]), -1 * size_small)[
            -1 * size_small]
        indx2_small_v = find_Fgrad_index(G_loss, Z=Z, thres=size1_th_largest)
        index_set_small = find_common(indx1_small, indx2_small_v)


    # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}

    indx2_large= find_hgrad_index(G, Z=Z, thres=5e3)
    index_set_large = set(list(zip(indx2_large[:, 0], indx2_large[:, 1])))

    if len(index_set_large) < (size_large - 100):
        indx2_large = find_Fgrad_index(G_loss, Z=Z, thres=0)
        size2 = min(size_large, len(indx2_large))
        size2_th_smallest = np.partition(G_h[(indx2_large[:, 0], indx2_large[:, 1])], size2 - 2)[size2 - 2]
        indx1_large_v = find_hgrad_index(G, Z=Z, thres=size2_th_smallest)
        index_set_large = find_common(index_set_large, indx1_large_v)

    return index_set_small, index_set_large
"""


def find_idx_set(G_h, G_loss, Z, size_small):
    r"""
    Implement Algorithm 2 in Paper, find

    index_set_small = \mathcal{Y}(W,\tau_*,\xi^*) s.t. |index_set_small| = size1
    index_set_large = \mathcal{Y}(W,\tau^*,\xi_*) s.t. |index_set_large| = size2

    :param G_h: gradient of h
    :param G_loss: gradient of loss
    :param Z: edge absence constraints
    :param size1: size of \mathcal{Y}(W,\tau_*,\xi^*)
    :param size2: size of \mathcal{Y}(W,\tau^*,\xi_*)
    :return: index_set_small, index_set_large
    """
    """
    # gFs =  [0, 1e-8, 1e-6, 1e-4, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 1,2,3,4]
    # gFs = [0, 1e-8, 1e-6, 1e-4, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50]
    #ghs = sorted(
    #    [5e11, 1e11, 5e10, 1e10, 5e9, 1e9, 5e8, 1e8, 5e7, 1e7, 5e6, 1e6, 5e5, 1e5, 50000, 10000, 5000, 1000, 500, 100,
    #     70, 60, 50, 40, 30, 20, 10, 5, 2, 1, 0])
    ghs = [5, 4,3,2.8,2.5,2.2,2, 1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2 ,1.1,1,0.9,0.8, 0.7,0.6,0.5,0.4,0.3,0.2, 0.1, 0]
    # ghs = [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3]
    # gFs = [0, 1e-7, 1e-6, 5e-6,1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1,2,5,10,15,20,40]


    G_loss_abs = np.abs(G_loss)



    # G_norm = np.linalg.norm(np.divide(G_loss, np.sqrt(-G_h)))

    M = np.zeros(len(ghs))

    for count_gh, gh in enumerate(ghs):
        indx1 = find_hgrad_index(G_loss_abs, G_h, Z=Z, thres=gh)
        # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}

        index_set = list(zip(indx1[:, 0], indx1[:, 1]))
        M[count_gh] = len(index_set)

    i1 = np.unravel_index(np.argmin(np.abs(M - size_small), axis=None), M.shape)[0]

    #i2 = np.unravel_index(np.argmin(np.abs(M - size_large), axis=None), M.shape)[0]

    indx1_small = find_hgrad_index(G_loss_abs, G_h, Z=Z, thres=ghs[i1])

    index_set_small = set(list(zip(indx1_small[:, 0], indx1_small[:, 1])))

    #indx1_large = find_hgrad_index(G, Z=Z, thres=ghs[i2])

    #index_set_large = set(list(zip(indx1_large[:, 0], indx1_large[:, 1])))
    """

    # Flatten the array to 1D

    # G_loss_abs = np.abs(G_loss)
    Z0 = np.copy(Z)
    np.fill_diagonal(Z0, False)

    G_loss_abs = np.abs(G_loss)

    G_loss0 = G_loss_abs[np.where(Z0)]
    G_h0 = G_h[np.where(Z0)]

    flattened_array = -G_loss0 * (1.0 / G_h0)
    # flattened_array = G_loss0 / (G_h0 + 10 ** (-20))
    small_thres = np.partition(flattened_array, size_small - 1)[size_small - 1]
    # small_thres = np.partition(flattened_array, -size_small)[-size_small]
    # print(small_thres)
    indx1_small = find_hgrad_index(G_loss_abs, G_h, Z=Z0, thres=-small_thres)
    # print("G_loss")
    # print(G_loss0)

    # print("flattened array")
    # print(flattened_array)
    # print(len(indx1_small))
    # indx1_small = find_hgrad_index(G_loss_abs, G_h, Z=Z, thres=small_thres)
    index_set_small = list(zip(indx1_small[:, 0], indx1_small[:, 1]))
    """
    large_thres = np.partition(flattened_array, size_large-1)[size_large-1]
    indx1_large = find_hgrad_index(G_loss_abs, G_h, Z=Z, thres=-large_thres)

    index_set_large = set(list(zip(indx1_large[:, 0], indx1_large[:, 1])))


    index = np.transpose(np.where(Z))
    indx1_large = index[index[:, 1] != index[:, 0],]
    index_set_large = set(list(zip(indx1_large[:, 0], indx1_large[:, 1])))
    """
    return index_set_small, -small_thres


def assign_negative(i, j, topo):
    succ = False
    if np.size(np.where(topo == i)) and np.size(np.where(topo == j)):
        pos_i = np.where(topo == i)
        pos_j = np.where(topo == j)
        if not np.any(topo[pos_j[0][0]:(pos_i[0][0] + 1)] == -1):
            topo[pos_j[0][0]:(pos_i[0][0] + 1)] = -1
            succ = True

    return topo, succ


def create_new_topo_greedy(topo, loss_collections, idx_set, loss, opt=1):
    topo0 = np.array(copy(topo)).astype(int)
    loss_table = np.concatenate((np.array(list(idx_set)), loss_collections.reshape(-1, 1)), axis=1)
    loss_table_good = loss_table[np.where(loss_collections < loss)]
    sorted_loss_table_good = loss_table_good[loss_table_good[:, 2].argsort()]
    len_loss_table_good = sorted_loss_table_good.shape[0]
    for k in range(len_loss_table_good):
        i, j = sorted_loss_table_good[k, 0:2]
        i, j = int(i), int(j)
        topo0, succ = assign_negative(i, j, topo0)
        if succ:
            topo = create_new_topo(topo=topo, idx=(i, j), opt=opt)
    return topo


def structure_X(action_idx_list):
    A = []
    b = []
    for idx, action in enumerate(action_idx_list):

        X = X_list_zeros[idx]
        for data_idx in range(X.shape[0]):
            for d_idx in range(d):
                if d_idx not in action:
                    data_node = np.zeros((d, d))
                    data_node[d_idx, :] = X[data_idx, :]
                    data_node = data_node.flatten()
                    mu_data = np.zeros(d)
                    mu_data[d_idx] = 1

                    A.append(np.concatenate((data_node, mu_data)))
                    b.append(X[data_idx, d_idx])

    return np.array(A), np.array(b)


class TOPO_linear:
    def __init__(self, score, regress):
        super().__init__()
        self.score = score
        self.regress = regress

    def _init_W_slice(self, idx_y, idx_x):
        y = self.X[:, idx_y]
        x = self.X[:, idx_x]
        w= self.regress(X=x, y=y)
        return w

    def _init_W(self, Z):
        W = np.zeros((self.d, self.d))

        for j in range(self.d):
            if (~Z[:, j]).any():
                W[~Z[:, j], j] = self.regress(X=self.X[:, ~Z[:, j]], y=self.X[:, j])
            else:
                W[:, j] = 0

        return W

    """
    def _init_W_slice(self, idx_y, idx_x):
        y = self.X[:, idx_y]
        x = self.X[:, idx_x]
        # w, mu = self.regress(X=x, y=y)
        A_slice = np.hstack((x, np.tile(1, (x.shape[0], 1))))
        w_mu = np.linalg.inv(A_slice.T @ A_slice) @ (A_slice.T @ y)

        return w_mu[:-1], w_mu[-1]

    def _init_W(self, Z):
        Z_flatten = Z.flatten()

        nonzero_idxs = np.where(Z_flatten)[0]

        A_regress = np.hstack((A[:, nonzero_idxs], A[:, -d:]))

        w = slin.inv(A_regress.T @ A_regress) @ A_regress.T @ b

        B_flat = np.zeros(d ** 2)
        np.put(B_flat, nonzero_idxs, w[:-d])

        B = B_flat.reshape(d, d)

        return B.T, w[-d:]
    """

    def _h(self, W, eps=1e-20):
        """Evaluate value and gradient of acyclicity constraint.
        Option 1: h(W) = Tr(I+|W|/d)^d-d
        """

        """
        h(W) = -log det(sI-W*W) + d log (s)
        nabla h(W) = 2 (sI-W*W)^{-T}*W
        """

        Id = np.eye(self.d)

        # M = Id - np.sqrt(W * W+ eps)
        Wabs = np.abs(W)
        M = Id - Wabs * Wabs*Wabs

        # h = -la.slogdet(M)[1]

        H = slin.inv(M).T


        # G_h = -slin.inv(Id - np.abs(B.T)) # *np.sign(B)
        # G_h_flatten = G_h.flatten('F')

        # H = -np.dot(np.diag(N.flatten('F')),np.diag(((B * B + eps)**(-1/2)).flatten('F'))).diagonal().reshape((d,d), order='F')
        # edge from i to j depends on: YES walk from i to j * no edge from j to i
        # edge from i to j should depend on: NO walk from j to i*no edge from j to i
        # H = (-N.flatten('F')*((B * B + eps)**(-1/2)).flatten('F')).reshape((d,d), order='F')
        # H = -(((N.T)**(-1)).flatten('F')*((B * B + eps)**(-1/2)).flatten('F')).reshape((d,d), order='F')
        # H = N.T*(W * W + eps)**(-1/2)

        # H = 2*N.T

        # H = -(N.T - Id)**(-1)
        # H = ((B * B + eps)**(-1/2))
        return H

    def _update_topo_linear(self, W,topo, idx, opt=1):

        topo0 = copy(topo)
        W0 = np.zeros_like(W)

        i, j = idx
        i_pos, j_pos = topo.index(i), topo.index(j)

        W0[:, topo[:j_pos]] = W[:, topo[:j_pos]]
        W0[:, topo[(i_pos + 1):]] = W[:, topo[(i_pos + 1):]]


        topo0 = create_new_topo(topo=topo0, idx=idx, opt=opt)
        for k in range(j_pos, i_pos + 1):
            if len(topo0[:k]) != 0:
                W0[topo0[:k], topo0[k]] = self._init_W_slice(idx_y=topo0[k], idx_x=topo0[:k])

            else:
                W0[:, topo0[k]]= 0
        return W0,topo0

    def fit(self, X, topo: list, no_large_search, size_small, size_large):
        self.n, self.d = X.shape
        self.X = X
        iter_count = 0
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        else:
            self.topo = topo

        # self.Z_reg = create_Z_reg(self.topo)
        self.Z = create_Z(self.topo)
        self.W= self._init_W(self.Z)
        loss, G_loss = self.score(X=self.X, W=self.W)
        G_h = self._h(W=self.W)
        idx_set_small, small_thres = find_idx_set(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small)
        idx_set = idx_set_small
        idx_set_sizes = []
        while bool(idx_set):
            # print(self.topo)
            print(loss)

            idx_len = len(idx_set)
            idx_set_sizes.append(idx_len)

            # print(iter_count)

            loss_collections = np.zeros(idx_len)

            for i in range(idx_len):
                W_c, topo_c = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx_set[i])
                loss_c, _ = self.score(X=self.X, W=W_c)

                loss_collections[i] = loss_c

            if np.any(loss > loss_collections):

                self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)


            else:

                print("Using larger search space, but we cannot find better loss")
                break

            # self.Z_reg = create_Z_reg(self.topo)
            self.Z = create_Z(self.topo)
            self.W= self._init_W(self.Z)
            loss, G_loss = self.score(X=self.X, W=self.W)
            G_h = self._h(W=self.W)
            idx_set_small, small_thres = find_idx_set(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small)
            idx_set = list(idx_set_small)

            iter_count += 1

        # print("iter_count")
        # print(iter_count)
        return self.W, self.topo, self.Z, idx_set_sizes, loss


## Linear Model
def regress(X, y):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X=X, y=y)
    return reg.coef_


def score(X, W):
    M = X @ W
    R = X - M
    loss = (0.5 / X.shape[0]) * (R ** 2).sum()  # + 0.01*np.linalg.norm(W, ord=1)
    G_loss = - (1.0 / X.shape[0]) * X.T @ R  # + 0.01*np.sum(np.sign(W))

    return loss, G_loss


def _score(W: np.ndarray):
    """
    Evaluate value and gradient of the score function.

    Parameters
    ----------
    B : np.ndarray
        :math:`(d,d)` adjacency matrix

    Returns
    -------
    typing.Tuple[float, np.ndarray]
        loss value, and gradient of the loss function
    """

    # B_flatten = B.flatten()

    B_flatten = (W.T).flatten()
    mu = np.zeros(W.shape[0])
    w = np.concatenate((B_flatten, mu))
    loss = (1 / A.shape[0]) * np.linalg.norm(b - A @ w) ** 2

    w_grad = (1 / A.shape[0]) * 2 * (A.T @ A @ w - A.T @ b)

    B_grad = w_grad[:-d].reshape(d, d)


    return loss, B_grad.T


if __name__ == '__main__':
    scores = []

    seed = int(sys.argv[1])

    set_size = int(sys.argv[2])
    #seed=2
    #set_size=50
    # p=0.2

    # for seed in range(10):

    set_random_seed(seed)

    n, d= 1000, 50  # the ground truth is a DAG of 20 nodes

    action_idx_list, action_list, P_list = produce_action_list([[]])

    diag_indices_1d = np.nonzero(np.eye(d).flatten())[0]

    G_true = simulate_dag(d,'ER')
    W_true = simulate_parameter(G_true)

    ordered_vertices, X_list, X_list_zeros = simulate_linear_sem(W_true, n, 'gauss')

    B_true = W_true.T
    Id = np.eye(B_true.shape[0])
    # print("Ground Truth")
    # print(B_true)
    A, b = structure_X(action_idx_list)

    model = TOPO_linear(regress=regress, score=score)

    np.random.seed(5)
    start_time = time.perf_counter()
    initial_permutation = np.random.permutation(d)
    permutation = list(np.copy(initial_permutation))

    # W_est, mu_est, _, _, _, idx_set_sizes = model.fit(X=X_list_zeros[0], topo=permutation, no_large_search=10,
    #                                                  size_small=100, size_large=1000)
    W_est, _, _, idx_set_sizes, loss = model.fit(X=X_list_zeros[0], topo=permutation, no_large_search=0,
                                                         size_small=set_size, size_large=set_size)
    end_time = time.perf_counter()
    print("time")
    print(end_time - start_time)

    print(_score(W_est))
    scores.append(_score(W_est)[0])
    print(loss)

    retrieve_edges = np.sign(W_est.T) * np.sign(B_true)
    print(np.sum(np.abs(np.sign(B_true))))
    print(np.sum(retrieve_edges[retrieve_edges > 0]))
    false_edge = 0
    for i in range(d):
        for j in range(d):
            if (B_true[i, j] == 0.0) and np.abs(np.sign(W_est.T[i, j]) - np.sign(B_true[i, j])) == 1:
                false_edge = false_edge + 1.
    print(false_edge)

    thresholded_B = np.where(np.abs(W_est.T) >= 0.1, W_est.T, 0)
    retrieve_edges = np.sign(thresholded_B) * np.sign(B_true)
    print(np.sum(np.abs(np.sign(B_true))))
    print(np.sum(retrieve_edges[retrieve_edges > 0]))
    false_edge = 0
    for i in range(d):
        for j in range(d):
            if (B_true[i, j] == 0.0) and np.abs(np.sign(thresholded_B[i, j]) - np.sign(B_true[i, j])) == 1:
                false_edge = false_edge + 1
    print(false_edge)
    print(idx_set_sizes)

    # scores = np.array(scores)
    # print(np.mean(scores))
    # ci = st.t.interval(alpha=0.95, df=len(scores) - 1, loc=np.mean(scores), scale=st.sem(scores))
    # print(ci[1]-np.mean(scores))
