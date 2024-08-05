import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
import scipy.linalg as slin
from copy import copy
import numpy.linalg as la
import random
import igraph as ig
import time


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, graph_type,k):
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
        G_und = ig.Graph.Erdos_Renyi(n=d, m=k*d)
        #G_und = ig.Graph.Erdos_Renyi(n=d, p=p)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)

    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G_und = ig.Graph.Barabasi(n=d, m=k)
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
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
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
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


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
    :param G_h: gradient of h
    :param Z: edge absence constaints
    :param thres: threshold for gradient of fraction
    :return: set {(i.j)| G_loss/(G_h)_{ij}>=thres, Z[i,j] = True }
    """

    index = np.transpose(np.where(np.logical_and(G_loss >= thres * G_h, Z)))

    return index


def find_idx_set(G_h, G_loss, Z, size_small):
    """
    :param G_h: gradient of h
    :param G_loss: gradient of loss
    :param Z: edge absence constraints
    :param size_small: size of candidate set
    :return: small candidate set, threshold for small candidate set
    """

    # Flatten the array to 1D


    Z0 = np.copy(Z)
    np.fill_diagonal(Z0, False)

    G_loss_abs = np.abs(G_loss)

    G_loss0 = G_loss_abs[np.where(Z0)]
    G_h0 = G_h[np.where(Z0)]

    flattened_array = -G_loss0 * (1.0 / G_h0)

    # flattened_array = G_loss0 / (G_h0 + 10 ** (-20))
    small_thres = np.partition(flattened_array, size_small - 1)[size_small - 1]


    indx1_small = find_hgrad_index(G_loss_abs, G_h, Z=Z0, thres=-small_thres)

    index_set_small = list(zip(indx1_small[:, 0], indx1_small[:, 1]))

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


class TOPO_linear:
    def __init__(self, score, regress, beta):
        super().__init__()
        self.score = score
        self.regress = regress
        self.beta = beta

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


    def _h(self, W, eps=1e-20):
        """Evaluate value and gradient of DAGMA acyclicity constraint.
        """

        Id = np.eye(self.d)


        if self.beta == 1:
            M= Id-np.abs(W)
        elif self.beta == 2:
            M = Id-W*W
        elif self.beta == 3:
            Wabs = np.abs(W)
            M = Id - Wabs * Wabs * Wabs
        else:
            M = Id-W*W*W*W


        # h = -la.slogdet(M)[1]

        H = slin.inv(M).T

        #H= np.linalg.matrix_power(Id + (1 / d) * np.abs(W.T), self.d - 1)


        return H

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
                W0[topo0[:k], topo0[k]]= self._init_W_slice(idx_y=topo0[k], idx_x=topo0[:k])

            else:
                W0[:, topo0[k]] = 0
        return W0,topo0

    def fit(self, X, topo: list, size_small):
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
                W_c, topo_c = self._update_topo_linear(W=self.W,topo=self.topo, idx=idx_set[i])
                loss_c, _ = self.score(X=self.X, W=W_c)

                loss_collections[i] = loss_c

            if np.any(loss > loss_collections):

                self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)


            else:
                print("Using larger search space, but we cannot find better loss")
                break

            # self.Z_reg = create_Z_reg(self.topo)
            self.Z = create_Z(self.topo)
            self.W = self._init_W(self.Z)
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


if __name__ == '__main__':
    scores = []
    set_size = 50
    #seed = int(sys.argv[1])
    seed = 0
    #set_size = int(sys.argv[2])

    set_random_seed(seed)

    n, d, k, beta= 1000, 50, 4,2

    G_true = simulate_dag(d, 'SF',k)
    W_true = simulate_parameter(G_true)
    B_true=W_true.T

    X = simulate_linear_sem(W_true, n, 'gauss')

    model = TOPO_linear(regress=regress, score=score, beta=beta)

    np.random.seed(5)
    start_time = time.perf_counter()
    initial_permutation = np.random.permutation(d)
    permutation = list(np.copy(initial_permutation))

    # W_est, mu_est, _, _, _, idx_set_sizes = model.fit(X=X_list_zeros[0], topo=permutation, no_large_search=10,
    #                                                  size_small=100, size_large=1000)
    W_est, _, _, idx_set_sizes, loss = model.fit(X=X, topo=permutation,size_small=set_size)
    end_time = time.perf_counter()
    print("time")
    print(end_time - start_time)
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


