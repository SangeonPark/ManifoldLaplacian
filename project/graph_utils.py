
from absl import logging

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from absl import logging
from tqdm import tqdm
from copy import deepcopy

class Kernel(object):
    """docstring for Kernel"""
    def __init__(self, arg):
        super(Kernel, self).__init__()
        self.arg = arg
        

class Epa(object):
    """docstring for Epa"""
    def __init__(self, arg):
        super(Epa, self).__init__()
        self.arg = arg
        


#def kernel(x, y, eps, function):
#Redefine as Class




#def get_alignment_degree_matrix(adjacency_matrix):
#    """Here
#    """
#    npoints = adjacency_matrix.shape[0]
#    deg = adjacency_matrix.sum(axis=0)


def get_pca_adjacency(data, eps_pca, parallel=1):
    """Here"""
    npoints = data.shape[0]
    assert npoints % parallel == 0
    adj_matrix = scipy.sparse.lil_matrix((npoints, npoints))
    #adj_graph  = scipy.sparse.lil_matrix((npoints, npoints))
    norm = np.sum(data**2, axis=1)
    #cols = np.meshgrid(parallel, np.ones())


    for i in tqdm(range(0, npoints, parallel)):
        dotprod  = data @ data[i:i+parallel].T
        dist_norm = np.abs(norm[:, None] - 2*dotprod + norm[i:i+parallel][None, :])
        #kernel = np.exp(-dist_norm/eps)

        idx = np.where(dist_norm < eps_pca)
        #print(idx)
        matrix_idx = list(deepcopy(idx))
        matrix_idx[1] += i
        matrix_idx = tuple(matrix_idx)
        #print(matrix_idx)
        #adj_matrix[matrix_idx] = np.exp(-dist_norm[idx]/eps)
        #Epach Kernel
        adj_matrix[matrix_idx] = 1 - dist_norm[idx]/eps_pca
        #pairdist[matrix_idx] = dist_norm[idx]
        #adj_graph[matrix_idx] = 1

    return adj_matrix


def get_alignment_adjacency(data, eps, parallel=1):
    """Here
    """
    npoints = data.shape[0]
    assert npoints % parallel == 0
    adj_matrix = scipy.sparse.lil_matrix((npoints, npoints))
    #adj_graph  = scipy.sparse.lil_matrix((npoints, npoints))
    norm = np.sum(data**2, axis=1)
    #cols = np.meshgrid(parallel, np.ones())


    for i in tqdm(range(0, npoints, parallel)):
        dotprod  = data @ data[i:i+parallel].T
        dist_norm = np.abs(norm[:, None] - 2*dotprod + norm[i:i+parallel][None, :])
        #kernel = np.exp(-dist_norm/eps)

        idx = np.where(dist_norm < eps)
        #print(idx)
        matrix_idx = list(deepcopy(idx))
        matrix_idx[1] += i
        matrix_idx = tuple(matrix_idx)
        #print(matrix_idx)
        adj_matrix[matrix_idx] = np.exp(-dist_norm[idx]/eps)
        #pairdist[matrix_idx] = dist_norm[idx]
        #adj_graph[matrix_idx] = 1

    return adj_matrix



#def build_neighborhood_graph_with_Nneighbors


def build_neighborhood_graph_with_distance(data, k, n=1000):
 
    """Build exact k-nearest neighbors graph from numpy data.
    Args:
      data: Data to compute nearest neighbors of, each column is one point
      k: number of nearest neighbors to compute
      n (optional): number of neighbors to compute simultaneously
    Returns:
      A scipy sparse matrix in LIL format giving the symmetric nn graph.
    """
    shape = data.shape
    assert shape[0] % n == 0
    nbr_graph = scipy.sparse.lil_matrix((shape[0], shape[0]))
    norm = np.sum(data**2, axis=1)
    cols = np.meshgrid(np.arange(n), np.ones(k+1))[0]
    for i in tqdm(range(0, shape[0], n)):
        dot = data @ data[i:i+n].T
        dists = np.sqrt(np.abs(norm[:, None] - 2*dot + norm[i:i+n][None, :]))
        idx = np.argpartition(dists, k, axis=0)[:k+1]
        nbrs = idx[np.argsort(dists[idx, cols], axis=0), cols][1:]
        for j in range(n):
            nbr_graph[i+j, nbrs[:, j]] = 1
    # Symmetrize graph
    for i in tqdm(range(shape[0])):
        for j in nbr_graph.rows[i]:
            if nbr_graph[j, i] == 0:
                nbr_graph[j, i] = nbr_graph[i, j]
    logging.info('Symmetrized neighbor graph')
    return nbr_graph



