'''
    encoding image based on celluar features
    1. bag of words
    2. vlad encoding
'''
import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance

def compute_assignment(cell_feat_all, centers):
    '''
        assign each point in cell_feat_all to its nearest neighbor in centers
        kd tree can be considered later
    '''
    D = distance.cdist(cell_feat_all, centers)
    return np.argmin(D, axis=1)

def normalize_feature(cell_feat_all):
    '''
        normalize feature,  only l2 used
        other normalized shoud also considered
        This depends on how centers are computed
    '''
    norm = LA.norm(cell_feat_all, ord=2, axis=1)
    return cell_feat_all / norm[:,np.newaxis]

def llc_encoding(centers, cell_feat_all, **kwargs):
    
    '''
        centers: k x p
        cell_feat_all: n x p, where p is feature dim 
    '''
    
    pooling = kwargs.get('pooling', 'max')
    knn = kwargs.get('knn', 5)
    beta = kwargs.get('beta', 1e-4)

    if cell_feat_all.size == 0:
        return None
    # step 1: do l2 normalization first
    cell_feat_all = normalize_feature(cell_feat_all)
    # step 2: compute distance to centers
    D = distance.cdist(cell_feat_all, centers)
    # step 3: encoding using reconstruction
    E = np.zeros_like(D)
    D_ind = np.argsort(D, axis=1)
    II = np.eye(knn)
    for i in range(D.shape[0]):
        idx = D_ind[i, :knn]
        # shift ith point to origin
        z = centers[idx, :] - cell_feat_all[i,:]
        C = np.dot(z, np.transpose(z))
        C = C + II*beta*np.trace(C)
        # solve Cw = 1
        w = LA.lstsq(C, np.ones(knn))[0]
        w = w/(np.sum(w) + 1e-10)
        E[i, idx] = w
    # step 4 do pooling here
    if pooling == 'max':
        # max pooling
        img_feat = np.max(E, axis=0)
    else:
        # sum pooling
        img_feat = np.sum(E, axis = 0)
    # step 4: do L2 normalization here
    norm = LA.norm(img_feat, ord=2)
    assert norm > 1e-9
    return img_feat/norm

def bow_encoding(centers, cell_feat_all):
    '''
        centers: k x p
        cell_feat_all: n x p, where p is feature dim 
    '''
    if cell_feat_all.size == 0:
        return None
    # step 1: do l2 normalization first
    cell_feat_all = normalize_feature(cell_feat_all)
    # step 2: compute distance to centers
    assignment = compute_assignment(cell_feat_all, centers)
    # step 3: encoding, bag of words here
    k = centers.shape[0]
    bow = np.zeros(k, dtype=np.float32)
    for nn in assignment:
        bow[nn] += 1.0
    # do normalization here
    bow /= len(assignment)
    # step 4: do L2 normalization here
    norm = LA.norm(bow, ord=2)
    assert norm > 1e-9
    return bow/norm

def vlad_encoding(centers, cell_feat_all):
    '''
        vlad encoding
        intra normalization is used here
    '''
    if cell_feat_all.size == 0:
        return None
    # step 1: do l2 normalization first
    cell_feat_all = normalize_feature(cell_feat_all)
    # step 2: compute distance to centers
    assignment = compute_assignment(cell_feat_all, centers)
    # step 3: encoding, vlad here
    k, p = centers.shape
    vlad = np.zeros(k*p, dtype=np.float32)
    for index, nn in enumerate(assignment):
        vlad[nn*p:(nn+1)*p] += cell_feat_all[index,:] - centers[nn,:]
    # step 4: normalization, use intra-normalization here
    for index in range(k):
        norm = LA.norm(vlad[index*p:(index+1)*p], ord=2)
        if norm > 1e-10:
            vlad[index*p:(index+1)*p] /= norm 
    return vlad