
from absl import logging
from tqdm import tqdm

import math
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg


#def normalize_laplacian

def compute_scale_factor(man_dim,eps):

    #Euclidean ball of radius one in man_dim

    #V_dim = ((np.pi)**(man_dim/2.0))/(math.gamma((man_dim/2.0) + 1))

    # numerator 2m_0 (n V_n factor is cancelled)
    #numerator = 2 * man_dim * 0.5 * (scipy.special.gamma(0.5*man_dim)*scipy.special.gammainc(0.5*man_dim, 1))
    numerator = 2 * 0.5 * (scipy.special.gamma(0.5*man_dim)*scipy.special.gammainc(0.5*man_dim, 1))
    # denominator m_2 (n V_n factor cancelled)
    denominator =  0.5 * eps * (scipy.special.gamma(0.5*man_dim+1.0)*scipy.special.gammainc(0.5*man_dim+1.0, 1))
    scale = numerator / denominator
    ##### Make a graph of the scale factor
    return scale


def compute_weighted_alignment(tangents, alignment_adjacency):


    npoints = tangents.shape[0]
    connection = {}
    for i in tqdm(range(npoints)):
        for j in alignment_adjacency.rows[i]:
            if j > i:
                uy, _, ux = np.linalg.svd(tangents[i] @ tangents[j].T,
                                          full_matrices=False)
                conn = uy @ ux
                connection[(i, j)] = conn
                connection[(j, i)] = conn.T
    logging.info('Constructed all connection matrices')
    return connection



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

def make_2nd_order_laplacian(connection, neighbor_graph):
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
    laplacian = scipy.sparse.bsr_matrix((data, indices, indptr),shape=(n*bsz, n*bsz))
    logging.info('Built 2nd-order graph connection Laplacian.')
    return laplacian




#def compute_weighted_connection_laplacian(neighbor_graph, alignment, tensor_order):

def make_general_order_laplacian_with_weights(alignment, alignment_adjacency, tensor_order, scale):
    """Here"""
    npoints = alignment_adjacency.shape[0]
    man_dim = list(alignment.values())[0].shape[0]
    deg = np.squeeze(np.asarray(alignment_adjacency.sum(axis=0)))
    #print(deg.shape)
    #print(type(deg))

    block_size = man_dim**tensor_order

    data = np.zeros((alignment_adjacency.nnz + npoints, block_size, block_size), dtype=np.float32)
    indptr = []
    indices = np.zeros(alignment_adjacency.nnz + npoints)
    index = 0
    for i in tqdm(range(npoints)):
        indptr.append(index)
        #print(alignment_adjacency[i,i])
        data[index] = np.eye(block_size) * (1/deg[i])
        indices[index] = i
        index += 1
        for j in alignment_adjacency.rows[i]:
            if i == j:
                continue
            #if sym:
            #    kron = sym_op(connection[(j, i)], zero_trace=zero_trace)
            #else:
                #print(p)
            if tensor_order == 1:
                #kron = alignment[(i, j)].T
                kron = alignment[(i,j)]
                #print(kron)

            else:
                #kron = alignment[(i, j)].T
                kron = alignment[(i,j)]
                for i in range(tensor_order-1):
                    #kron = np.kron(kron, alignment[(i, j)].T)
                    kron = np.kron(kron, alignment[(i, j)])
            #elif p== 2:
            #    kron = np.kron(connection[(i, j)].T, connection[(i, j)].T)
            #elif p>2:
            #    kron = np.kron(connection[(i, j)].T, connection[(i, j)].T)
            #    for k in range(p-2):
            #        kron = np.kron(kron, connection[(i, j)].T)
            data[index] = kron * alignment_adjacency[i,j] * (1/deg[i])
            indices[index] = j
            index += 1
    indptr.append(index)
    indptr = np.array(indptr)
    connection_laplacian = scipy.sparse.bsr_matrix((data, indices, indptr),
                                          shape=(npoints*block_size, npoints*block_size))
    connection_laplacian -= scipy.sparse.identity(npoints*block_size, dtype=np.float32, format='bsr')
    logging.info('Built general-order graph connection Laplacian.')
    connection_laplacian *= scale
    return connection_laplacian


def make_general_order_laplacian(connection, neighbor_graph, p):
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