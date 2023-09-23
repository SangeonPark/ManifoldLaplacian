import itertools

from absl import logging

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from tqdm import tqdm
def get_second_fundamental_form(data, neighbor_graph, k):
    """Construct all tangent vectors and the normals for the dataset."""
    # k is the dimension of the manifold
    tangents = np.zeros((data.shape[0], k, data.shape[1]), dtype=np.float32)
    normals  = np.zeros((data.shape[0], data.shape[1]-k, data.shape[1]), dtype=np.float32)
    h_total  = np.zeros((data.shape[0], int(k*(k+1)/2),data.shape[1]-k), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        diff = data[neighbor_graph.rows[i]] - data[i]
        _, _, u = np.linalg.svd(diff, full_matrices=False)
        tangents[i] = u[:k]
        normals[i] = u[k:]
        #each row of local_coords is each point of diff represented in local coordinates
        local_coords = diff @ u.T
        Np = local_coords.shape[0]
        n_amb = local_coords.shape[1]
        normal_coordinates = local_coords[:, k:]
        tangent_coordinates = local_coords[:, :k]
        ind_a, ind_b = np.triu_indices(k)
        #ind_a, ind_b = np.where(np.triu(np.ones((k, k))))
        #ind_a, ind_b = np.where(np.ones((k, k)))
        #print(ind_a, ind_b)
        quadt = np.multiply(tangent_coordinates[:, ind_a], tangent_coordinates[:, ind_b])
        quadt = np.insert(quadt, 0, 1, axis=1)
        #print(quadt.T @ quadt)
        h = np.linalg.pinv(quadt.T @ quadt) @ quadt.T @ normal_coordinates
        #print(h.shape)
        h = h[1:,:]
        #print(h.shape)
        h_total[i] = h

    true_h = np.zeros((data.shape[0], k, k, data.shape[1]-k))
    ind_a, ind_b = np.triu_indices(k)
    true_h[:,ind_a, ind_b,:] = h_total
    true_h[:,ind_b, ind_a,:] = h_total
    logging.info('Computed the second fundamental form')
    return tangents, normals, local_coords, h_total, true_h

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

def second_fundamental_form(local_coords, k):
    # k is the dimension of the manifold
    Np = local_coords.shape[0]
    n_amb = local_coords.shape[1]
    normal_coordinates = local_coords[:, k:]
    tangent_coordinates = local_coords[:, :k]
    ind_a, ind_b = np.where(np.triu(np.ones(k, k)))
    quadt = np.multiply(tangent_coordinates[:, ind_a], tangent_coordinates[:, ind_b])
    quadt = np.insert(quadt, 0, 1, axis=1)
    h = np.linalg.inv(quadt.T @ quadt)@quadt.T @ n_amb
    print(h.shape)
    h = h[:,1:]
    return h