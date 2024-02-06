from tqdm import tqdm
from absl import logging

import numpy as np

# Rewrite Code with Class OOP construction


def make_tangents(data, neighbor_graph, k):
    """Construct all tangent vectors for the dataset."""
    tangents = np.zeros((data.shape[0], k, data.shape[1]), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        diff = data[neighbor_graph.rows[i]] - data[i]
        _, _, u = np.linalg.svd(diff, full_matrices=False)
        tangents[i] = u[:k]
    logging.info('Computed all tangents')
    return tangents

def make_tangents_and_normals(data, neighbor_graph, k):
    """Construct all tangent vectors and the normals for the dataset."""
    tangents = np.zeros((data.shape[0], k, data.shape[1]), dtype=np.float32)
    normals  = np.zeros((data.shape[0], data.shape[1]-k, data.shape[1]), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        diff = data[neighbor_graph.rows[i]] - data[i]
        _, _, u = np.linalg.svd(diff, full_matrices=False)
        tangents[i] = u[:k]
        normals[i] = u[k:]
    logging.info('Computed all tangents')
    return tangents, normals






def make_normals(data, neighbor_graph, k):
    """Construct all normals for the dataset."""
    normals  = np.zeros((data.shape[0], data.shape[1]-k, data.shape[1]), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        diff = data[neighbor_graph.rows[i]] - data[i]
        _, _, u = np.linalg.svd(diff, full_matrices=False)
        normals[i] = u[k:]
    logging.info('Computed all tangents')
    return normals


def local_pca_with_weights_unknown_dim(data, pca_adjacency, threshold=0.7):
    """build local pca tangent and normal space for manifold of unknown dimension.
    """
    npoints = data.shape[0]
    basis = np.zeros((data.shape[0], data.shape[1], data.shape[1]), dtype=np.float32)
    est_local_dim = np.zeros(npoints)
    for i in tqdm(range(npoints)):
        centered_nbhrs = data[pca_adjacency.rows[i]] - data[i]

        
        #print(centered_nbhrs)
        #print(type(pca_adjacency.data[i]))
        weighted_centered_nbhrs = centered_nbhrs * np.asarray(pca_adjacency.data[i])[:,None]
        #print(weighted_centered_nbhrs.shape)
        _, sigma, u = np.linalg.svd(weighted_centered_nbhrs, full_matrices=False)

        variability = np.cumsum(sigma**2)/np.sum(sigma**2)
        
        #print(variability)

        est_local_dim[i] = np.searchsorted(variability, threshold, side='left')+1
        basis[i] = u


    print(est_local_dim)
    est_dim = int(np.median(est_local_dim))
    print(est_dim)
    tangents = basis[:, :est_dim, :]
    normals  = basis[:, est_dim:, :]

    return est_dim, tangents, normals


#def local_pca_with_weights_known_dim(data, eps_pca, man_dim):