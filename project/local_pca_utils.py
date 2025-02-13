# Adapted from https://github.com/deepmind/deepmind-research/blob/master/geomancer/geomancer.py


import itertools

from absl import logging

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import math


from tqdm import tqdm




def sym_op(x, zero_trace=False):
  """Given X, makes L(A) = X @ A @ X' for symmetric matrices A.
  If A is not symmetric, L(A) will return X @ (A_L + A_L') @ X' where A_L is
  the lower triangular of A (with the diagonal divided by 2).
  Args:
    x: The matrix from which to construct the operator
    zero_trace (optional): If true, restrict the operator to only act on
      matrices with zero trace, effectively reducing the dimensionality by one.
  Returns:
    A matrix Y such that vec(L(A)) = Y @ vec(A).
  """
  n = x.shape[0]
  # Remember to subtract off the diagonal once
  xx = (np.einsum('ik,jl->ijkl', x, x) +
        np.einsum('il,jk->ijkl', x, x) -
        np.einsum('ik,jl,kl->ijkl', x, x, np.eye(n)))
  xx = xx[np.tril_indices(n)]
  xx = xx.transpose(1, 2, 0)
  xx = xx[np.tril_indices(n)]
  xx = xx.T
  if zero_trace:
    diag_idx = np.cumsum([0]+list(range(2, n)))
    proj_op = np.eye(n*(n+1)//2)[:, :-1]
    proj_op[-1, diag_idx] = -1
    # multiply by operator that completes last element of diagonal
    # for a zero-trace matrix
    xx = xx @  proj_op
    xx = xx[:-1]
  return xx


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


#def visualize_tangents():





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


def make_general_order_laplacian(connection, neighbor_graph, p,  sym=False,antisym=False,zero_trace=False):
    """Make symmetric zero-trace second-order graph connection Laplacian."""
    n = neighbor_graph.shape[0]
    k = list(connection.values())[0].shape[0]
    # p is the tensor oder
    #bsz = (k*(k+1)//2 - 1 if zero_trace else k*(k+1)//2) if sym else k**2
    bsz = k**p
    # find symmetrized dimension
    # bsz = math.comb(k+p-1,p)
    data = np.zeros((neighbor_graph.nnz + n, bsz, bsz), dtype=np.float32)
    indptr = []
    indices = np.zeros(neighbor_graph.nnz + n)
    index = 0
    #print('here')
    #print(p)
    for i in tqdm(range(n)):
        indptr.append(index)
        #data[index] = len(neighbor_graph.rows[i]) * np.eye(bsz)
        data[index] = np.eye(bsz)
        indices[index] = i
        index += 1
        for j in neighbor_graph.rows[i]:
            #if sym:
            #    kron = sym_op(connection[(j, i)], zero_trace=zero_trace)
            #else:
                #print(p)
            if p == 1:
                kron = connection[(j, i)]
                #print(kron)
            elif p== 2:
                kron = np.kron(connection[(j, i)], connection[(j, i)])
                #print("here")
                if sym == True:
                    kron = 0.5*(np.kron(connection[(j, i)], connection[(j, i)])+np.kron(connection[(i, j)], connection[(i, j)]))
                    #print(kron.shape)
            elif p>2:
                kron = np.kron(connection[(j, i)], connection[(j, i)])
                for k in range(p-2):
                    kron = np.kron(kron, connection[(j, i)])
            data[index] = -kron
            indices[index] = j
            index += 1
    indptr.append(index)
    indptr = np.array(indptr)
    for i in range(50):
        print(data[i])
    laplacian = scipy.sparse.bsr_matrix((data, indices, indptr),
                                          shape=(n*bsz, n*bsz))
    logging.info('Built general-order graph connection Laplacian.')
    return laplacian