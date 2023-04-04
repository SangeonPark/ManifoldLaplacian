# Adapted from https://github.com/deepmind/deepmind-research/blob/master/geomancer/geomancer.py


import itertools

from absl import logging

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from tqdm import tqdm




def make_nearest_neighbors_graph(data, k, n=1000):
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


def make_tangents(data, neighbor_graph, k):
    """Construct all tangent vectors for the dataset."""
    tangents = np.zeros((data.shape[0], k, data.shape[1]), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        diff = data[neighbor_graph.rows[i]] - data[i]
        _, _, u = np.linalg.svd(diff, full_matrices=False)
        tangents[i] = u[:k]
    logging.info('Computed all tangents')
    return tangents


def visualize_tangents():





def make_connection(tangents, neighbor_graph):
    """Make connection matrices for all edges of the neighbor graph."""
    connection = {}
    for i in tqdm(range(tangents.shape[0])):
        for j in neighbor_graph.rows[i]:
            if j > i:
                uy, _, ux = np.linalg.svd(tangents[j] @ tangents[i].T,
                                          full_matrices=False)
                conn = uy @ ux
                connection[(i, j)] = conn
                connection[(j, i)] = conn.T
    logging.info('Constructed all connection matrices')
    return connection

def make_2nd_order_laplacian(connection, neighbor_graph, sym=True, zero_trace=True):
    """Make symmetric zero-trace second-order graph connection Laplacian."""
    n = neighbor_graph.shape[0]
    k = list(connection.values())[0].shape[0]
    bsz = (k*(k+1)//2 - 1 if zero_trace else k*(k+1)//2) if sym else k**2
    data = np.zeros((neighbor_graph.nnz + n, bsz, bsz), dtype=np.float32)
    indptr = []
    indices = np.zeros(neighbor_graph.nnz + n)
    index = 0

    for i in tqdm(range(n)):
        indptr.append(index)
        data[index] = len(neighbor_graph.rows[i]) * np.eye(bsz)
        indices[index] = i
        index += 1
        for j in neighbor_graph.rows[i]:
            if sym:
                kron = sym_op(connection[(j, i)], zero_trace=zero_trace)
            else:
                kron = np.kron(connection[(j, i)], connection[(j, i)])
            data[index] = -kron
            indices[index] = j
            index += 1
    indptr.append(index)
    indptr = np.array(indptr)

  laplacian = scipy.sparse.bsr_matrix((data, indices, indptr),
                                      shape=(n*bsz, n*bsz))
  logging.info('Built 2nd-order graph connection Laplacian.')
  return laplacian
