from tqdm import tqdm

from graph_utils import get_pca_adjacency, get_alignment_adjacency
#, get_curvature_adjacency
from connection_laplacian import compute_scale_factor, compute_weighted_alignment, make_general_order_laplacian_with_weights
from local_pca import local_pca_with_weights_unknown_dim
from curvature_utils import get_second_fundamental_form_with_weights
from tensorlaplacian_utils import get_riemannian_ricci_scalar, get_weitzenbock_operator, get_eigvals_of_weitzenbock,weitzenbock_to_bsr
import scipy
import numpy as np

class PointCloudAnalyzer(object):
    """docstring for PointCloudAnalyzer"""
    def __init__(self, data, parallel, eps, eps_pca, eps_curvature, pca_threshold):
        super(PointCloudAnalyzer, self).__init__()
        #make dataset loading be a part of it
        self.data = data
        self._eps_pca = None
        self.pca_threshold = pca_threshold
        self._npoints = data.shape[0]
        self.parallel = parallel
        self.eps_pca = eps_pca
        self.eps = eps
        self.eps_curvature = eps_curvature
                

    @property
    def npoints(self):
        return self._npoints

    @property
    def tangents(self):
        return self._tangents

    @property
    def normals(self):
        return self._normals

    @property
    def eps(self):
        return self._eps

    @property
    def eps_pca(self):
        return self._eps_pca

    @property
    def eps_curvature(self):
        return self._eps_curvature


    @property
    def man_dim(self):
        return self._man_dim

    @property
    def pca_threshold(self):
        return self._pca_threshold

    @pca_threshold.setter
    def pca_threshold(self, pca_threshold):
        self._pca_threshold = pca_threshold
        if self._eps_pca is not None:       
            self.eps_pca = self._eps_pca
        #pca_adjacency = get_pca_adjacency(self.data, self._eps_pca, self.parallel)
        #est_dim, tangents, normals = local_pca_with_weights_unknown_dim(self.data, pca_adjacency, self.pca_threshold)
        #print("new estimate of manifold dimension: ", est_dim)
        #self._man_dim = est_dim
        #self._tangents = tangents
        #self._normals  = normals



    @eps.setter
    def eps(self, eps):
        self._eps = eps
        self.scale = compute_scale_factor(self._man_dim, self._eps)
        self._alignment_adjacency = get_alignment_adjacency(self.data, self._eps, self.parallel)
        self._alignment = compute_weighted_alignment(self._tangents, self._alignment_adjacency)




    @eps_pca.setter
    def eps_pca(self, eps_pca):
        self._eps_pca = eps_pca
        pca_adjacency = get_pca_adjacency(self.data, self._eps_pca, self.parallel)
        est_dim, tangents, normals = local_pca_with_weights_unknown_dim(self.data, pca_adjacency, self.pca_threshold)
        print("new estimate of manifold dimension: ", est_dim)
        self._man_dim = est_dim
        self._tangents = tangents
        self._normals  = normals


    @eps_curvature.setter
    def eps_curvature(self, eps_curvature):
        self._eps_curvature = eps_curvature
        pca_adjacency = get_pca_adjacency(self.data, self._eps_curvature, self.parallel)
        _ , _, _, _, h = get_second_fundamental_form_with_weights(self.data, pca_adjacency, self._man_dim)
        self._riem, self._ric, self._sc = get_riemannian_ricci_scalar(h)




    def compute_lichnerowicz_laplacian(self, tensor_order, weitzenbock_scale):
        lap = make_general_order_laplacian_with_weights(self._alignment, self._alignment_adjacency, tensor_order, self.scale)
        weitzenbock = get_weitzenbock_operator(self._riem, self._man_dim, tensor_order)
        block_weitzenbock = weitzenbock_to_bsr(weitzenbock)
        lichnerowicz = -lap + weitzenbock_scale * block_weitzenbock
        return lichnerowicz

    def get_eigvals_and_eigvecs(self, lichnerowicz, smallest_n):
        eigvals, eigvecs = scipy.sparse.linalg.eigs(lichnerowicz, k=smallest_n, which='SM',return_eigenvectors=True)
        return eigvals, eigvecs


    def tensor_field_in_basis(self, eigvecs, index):
        eigvector = eigvec[:, index]
        eigvector = eigvector.reshape(-1, 2)
        killingfield = np.zeros((npoints, 3))
        for i in range(npoints):
            killingfield[i] = self._tangents[i, 0]* eigvector[i, 0] +  self._tangents[i, 1]* eigvector[i, 1] 