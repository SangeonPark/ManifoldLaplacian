import itertools

from absl import logging

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from tqdm import tqdm


def get_riemannian_ricci_scalar(h):
    Riem = np.einsum('nika,njla->nijkl',h,h) - np.einsum('nila,njka->nijkl',h,h)
    Ric = np.einsum('nikjk->nij', Riem)
    Scalar = np.einsum('nijij->n', Riem)
    return Riem, Ric, Scalar


def get_weitzenbock_operator(Riem, k, p):
    npoints = Riem.shape[0]
    Ric = np.einsum('nikjk->nij', Riem)
    if p==1:
        return Ric
    triplet = []
    for i in range(p+1-2):
        for j in range(p+1-i-2):
            triplet.append([i,j,p-2-(i+j)])
    riemannian_factor = np.zeros((npoints, k**p, k**p))
    for left, middle, right in triplet:
        Ileft, Iright = 1, 1
        if left != 0:
            Ileft = np.eye(k**left)
        if right != 0:
            Iright = np.eye(k**right)

        if middle != 0:
            Imiddle  = np.eye(k**middle)
            Rmiddle = np.einsum('nijkl,ab-> njalibk', Riem, Imiddle)
            Rmiddle = Rmiddle.reshape(npoints, 4*(k**middle),4*(k**middle))
        elif middle == 0:
            Rmiddle = np.einsum('nijkl->njlik', Riem)
            Rmiddle = Rmiddle.reshape(npoints, 4,4)

        riemannian_factor += np.kron(np.kron(Ileft, Rmiddle),Iright)
    ricci_factor = np.zeros((npoints, k**p, k**p))
    for i in range(p):
        left = i 
        right = p-1-i
        Ileft, Iright = 1, 1
        if left != 0 and right==0:
            Ileft = np.eye(k**left)
            mult = np.einsum('ab,nij->naibj',Ileft, Ric).reshape(npoints,k**p, k**p)
        elif right != 0 and left==0:
            Iright = np.eye(k**right)
            mult = np.einsum('nij,cd->nicjd', Ric,Iright).reshape(npoints,k**p, k**p)
        elif left != 0 and right !=0:
            Ileft = np.eye(k**left)
            Iright = np.eye(k**right)
            mult = np.einsum('ab,nij,cd->naicbjd',Ileft, Ric, Iright).reshape(npoints,k**p, k**p)
        ricci_factor += mult

    weitzenbock = ricci_factor - 2* riemannian_factor
    return weitzenbock

def get_eigvals_of_weitzenbock(weitzenbock, k, p):
    eigvals = np.empty((0,(k**p)))
    for row in weitzenbock:
        eigvals = np.vstack([eigvals, np.linalg.eigvals(row)])
    sorted_eig = np.sort(eigvals.real.reshape(1,-1)).flatten()
    return sorted_eig

def weitzenbock_to_bsr(weitzenbock):
    n = weitzenbock.shape[0]
    bsz = weitzenbock.shape[1]
    indptr = np.arange(0, n+1)
    indices = np.arange(0, n)
    block_weitzenbock = scipy.sparse.bsr_matrix((weitzenbock, indices, indptr),
                                          shape=(n*bsz, n*bsz),dtype=np.float32)
    return block_weitzenbock


