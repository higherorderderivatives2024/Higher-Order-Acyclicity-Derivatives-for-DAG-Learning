import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
import scipy.linalg as slin
from copy import copy
import sys
import typing
import random
import igraph as ig
import time
import scipy.stats as st
import matplotlib.pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, graph_type):
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
        #G_und = ig.Graph.Erdos_Renyi(n=d, p=0.2)
        G_und = ig.Graph.Erdos_Renyi(n=d, m=8*d)
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
    """
    Create edge absence constraints \mathcal{Z} corresponding to topological ordering
    :param ordering: topological sort
    :return: bool matrix

    create_Z([0,1,2,3])
    Out:
    array([[ True, False, False, False],
       [ True,  True, False, False],
       [ True,  True,  True, False],
       [ True,  True,  True,  True]])

    """
    d = len(ordering)
    Z = np.ones((d, d), dtype=bool)
    for i in range(d - 1):
        Z[ordering[i], ordering[i + 1:]] = False
    return Z


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




def find_Fgrad_index(G_loss, Z, thres=1e-3):
    """
    Find where {(i,j)| G_loss(i,j) not = 0 and Z(i,j)= True}

    :param G_loss: gradient of Loss function
    :param Z: edge absence constaints
    :param thres:
    :return: set {(i.j)| i\neq j, |(G_F)_{ij}|>=thres, Z[i,j] = True }
    """
    not0grad = np.logical_or(G_loss <= (-thres), G_loss >= thres)
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
    ghs = sorted([40, 30, 20, 10, 5, 2, 1, 0.5, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04,
                  0.03, 0.025, 0.02, 0.01, 0.005, 0.001, 0.0001, 0.00005, 1e-5, 8e-6, 6e-6, 4e-6, 2e-6, 1e-6, 1e-7, 0])
    # ghs = [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3]
    # gFs = [0, 1e-7, 1e-6, 5e-6,1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1,2,5,10,15,20,40]
    # G_h = np.divide(G_loss, G_h)
    M = np.zeros([len(ghs), len(gFs)])
    for count_gF, gF in enumerate(gFs):
        for count_gh, gh in enumerate(ghs):
            indx1 = find_hgrad_index(G_h, Z=Z, thres=gh)
            # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}
            indx2 = find_Fgrad_index(G_loss, Z=Z, thres=gF)
            index_set = find_common(indx1, indx2)
            M[count_gh, count_gF] = len(index_set)

    i1, j1 = np.unravel_index(np.argmin(np.abs(M - size_small), axis=None), M.shape)
    i2, j2 = np.unravel_index(np.argmin(np.abs(M - size_large), axis=None), M.shape)

    indx1_small = find_hgrad_index(G_h, Z=Z, thres=ghs[i1])
    # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}
    indx2_small = find_Fgrad_index(G_loss, Z=Z, thres=gFs[j1])

    index_set_small = find_common(indx1_small, indx2_small)

    if len(index_set_small) > size_small + 20 and ghs[i1] == 0:
        size1_th_largest = np.partition(np.abs(G_loss[(indx1_small[:, 0], indx1_small[:, 1])]), -1 * size_small)[
            -1 * size_small]
        indx2_small_v = find_Fgrad_index(G_loss, Z=Z, thres=size1_th_largest)
        index_set_small = find_common(indx1_small, indx2_small_v)
 

    indx1_large = find_hgrad_index(G_h,Z=Z, thres=ghs[i2])
    # find where {(i,j)|Z[i,j]=FALSE,[\nabla F(W)]_{ij} not =0}
    indx2_large = find_Fgrad_index(G_loss, Z=Z, thres=gFs[j2])
    index_set_large = find_common(indx1_large, indx2_large)

    if len(index_set_large) < (size_large - 100):
        indx2_large = find_Fgrad_index(G_loss, Z=Z, thres=0)
        size2 = min(size_large, len(indx2_large))
        size2_th_smallest = np.partition(G_h[(indx2_large[:, 0], indx2_large[:, 1])], size2 - 2)[size2 - 2]
        indx1_large_v = find_hgrad_index(G_h, Z=Z, thres=size2_th_smallest)
        index_set_large = find_common(indx1_large_v, indx2_large)
   
    return index_set_small, index_set_large
"""

def find_hgrad_index(G_h, Z, thres=1e-2):
    TRUE_positions = np.where(np.logical_and(G_h<= thres, Z))
    positions_list = list(zip(TRUE_positions[0], TRUE_positions[1]))
    return positions_list

def find_idx_set(G_h,G_loss,Z,size_small,size_large):
    d = Z.shape[0]
    Zc = np.array(Z).copy()
    np.fill_diagonal(Zc,False) # don't consider the diagonal element
    assert size_large <= d*(d-1)/2 , "please set correct size for large search space, it must be less than d(d-1)/2"
    assert size_small>=1, "please set correct size for small search space"
    values = G_h[Zc]
    values.sort()
    g_h_thre_small = values[(size_small-1)]
    g_h_thre_large = values[(size_large-1)]
    index_set_small = find_hgrad_index(G_h,Zc,thres= g_h_thre_small)
    index_set_large = find_hgrad_index(G_h,Zc,thres= g_h_thre_large)
    return index_set_small,index_set_large


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
        w = self.regress(X=x, y=y)
        return w

    def _init_W(self, Z):
        W = np.zeros((self.d, self.d))
        for j in range(self.d):
            if (~Z[:, j]).any():
                W[~Z[:, j], j] = self.regress(X=self.X[:, ~Z[:, j]], y=self.X[:, j])
            else:
                W[:, j] = 0
        return W

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint.
        Option 1: h(W) = Tr(I+|W|/d)^d-d
        """

        """
        h(W) = -log det(sI-W*W) + d log (s)
        nabla h(W) = 2 (sI-W*W)^{-T}*W
        """
        Id = np.eye(self.d)
        s = 1
        M = s * Id - np.abs(W)
        h = - np.linalg.slogdet(M)[1] + self.d * np.log(s)
        G_h = slin.inv(M).T
        #h = np.trace(np.linalg.matrix_power(Id+(1/d)*np.abs(W), self.d))-self.d
        #G_h = np.linalg.matrix_power(Id+(1/d)*np.abs(W.T),self.d-1)

        #A = np.abs(W)
        #E = slin.expm(A)
        #h = np.trace(E) - d
        #G_h = E.T

        return h, G_h

    def _update_topo_linear(self, W, topo, idx, opt=1):

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
                W0[:, topo0[k]] = 0
        return W0, topo0



    def fit(self, X, topo: list, no_large_search, size_small, size_large):
        self.n, self.d = X.shape
        self.X = X
        iter_count = 0
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        else:
            self.topo = topo

        Z = create_Z(self.topo)
        self.Z = Z
        self.W = self._init_W(self.Z)
        loss, G_loss = self.score(X=self.X, W=self.W)
        h, G_h = self._h(W=self.W)
        idx_set_small, idx_set_large = find_idx_set(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small,
                                                    size_large=size_large)
        idx_set = list(idx_set_small)
        idx_set_sizes = []
        while bool(idx_set):
            print(loss)
            idx_len = len(idx_set)
            idx_set_sizes.append(idx_len)
            loss_collections = np.zeros(idx_len)

            for i in range(idx_len):
                W_c, topo_c = self._update_topo_linear(W = self.W,topo = self.topo,idx = idx_set[i])
                loss_c,_ = self.score(X = self.X, W = W_c)
                loss_collections[i] = loss_c

            if np.any(loss > np.min(loss_collections)):
                self.topo = create_new_topo_greedy(self.topo,loss_collections,idx_set,loss)

            else:
                if large_space_used < no_large_search:
                    idx_set = list(set(idx_set_large) - set(idx_set_small))
                    idx_len = len(idx_set)
                    idx_set_sizes.append(len(idx_set_large))
                    loss_collections = np.zeros(idx_len)
                    for i in range(idx_len):
                        W_c, topo_c = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx_set[i])
                        loss_c, _ = self.score(X=self.X, W=W_c)
                        loss_collections[i] = loss_c

                    if np.any(loss > loss_collections):
                        large_space_used += 1
                        self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)
                    else:
                        print("Using larger search space, but we cannot find better loss")
                        break


                else:
                    print("We reach the number of chances to search large space, it is {}".format(
                        no_large_search))
                    break

            self.Z = create_Z(self.topo)
            self.W = self._init_W(self.Z)
            loss, G_loss = self.score(X=self.X, W=self.W)
            h, G_h = self._h(W=self.W)
            idx_set_small, idx_set_large = find_idx_set(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small,
                                                        size_large=size_large)
            idx_set = list(idx_set_small)

            iter_count += 1

        return self.W, self.topo, Z, idx_set_sizes,loss

## Linear Model
def regress(X, y):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X=X, y=y)
    return reg.coef_
"""
def score(X, W):
    M = X @ W
    R = X - M
    loss = (1.0 / (X.shape[0]*X.shape[1])) * (R ** 2).sum()  # + 0.01*np.linalg.norm(W, ord=1)
    G_loss = - (2.0 / X.shape[0]*X.shape[1]) * X.T @ R  # + 0.01*np.sum(np.sign(W))

    return loss, G_loss
"""


def score(X, W):
    M = X @ W
    R = X - M
    loss = 0.5 / X.shape[0] * (R ** 2).sum()
    G_loss = - 1.0 / X.shape[0] * X.T @ R

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
    mu = np.zeros(W.shape[0])
    B_flatten = (W.T).flatten()
    w = np.concatenate((B_flatten, mu))
    loss = (1 / A.shape[0]) * np.linalg.norm(b - A @ w) ** 2

    w_grad = (1 / A.shape[0]) * 2 * (A.T @ A @ w - A.T @ b)

    B_grad = w_grad[:-d].reshape(d, d)
    mu_grad = w_grad[-d:]

    return loss, B_grad.T, mu_grad

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': round(fdr,6), 'tpr': round(tpr,6), 'fpr': round(fpr,6), 'shd': shd, 'nnz': pred_size}


if __name__ == '__main__':
    scores = []

    seed = int(sys.argv[2])
    size = int(sys.argv[1])
    #seed= 5
    #size = 10



    set_random_seed(seed)

    n, d= 1000, size  # the ground truth is a DAG of 20 nodes

    action_idx_list, action_list, P_list = produce_action_list([[]])

    diag_indices_1d = np.nonzero(np.eye(d).flatten())[0]

    G_true = simulate_dag(d, 'SF')
    W_true = simulate_parameter(G_true)

    ordered_vertices, X_list, X_list_zeros = simulate_linear_sem(W_true, n, 'gauss')

    B_true = W_true.T
    Id = np.eye(B_true.shape[0])
    # print("Ground Truth")
    # print(B_true)
    A, b = structure_X(action_idx_list)

    model = TOPO_linear(regress=regress, score=score)

    start_time = time.perf_counter()
    no_large_search = -1
    size_small = -1
    size_large = -1

    if size == 10:
        no_large_search = 1
        size_small =30
        size_large=45
    if size == 50:
        no_large_search = 10
        size_small = 100
        size_large = 1000
    if size == 100:
        no_large_search = 15
        size_small = 150
        size_large = 2500

    np.random.seed(5)

    initial_permutation = np.random.permutation(d)
    permutation = list(np.copy(initial_permutation))

    W_est,_, _, idx_set_sizes,loss = model.fit(X=X_list_zeros[0], topo=permutation, no_large_search=no_large_search,
                                                         size_small=size_small, size_large=size_large)
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

    print(count_accuracy(B_true != 0, W_est.T != 0))



