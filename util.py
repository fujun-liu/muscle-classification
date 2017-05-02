'''
util functions
'''
import numpy as np
import csv
import pickle

def load_old_patient_info(feat_path):
    with open(feat_path, 'rb') as f:
        feat_info = pickle.load(f)
    pid_all = feat_info['pid_all']
    Y = feat_info['Y']
    slide_name_all = feat_info['slide_name_all']
    group_ids = feat_info['group_ids']
    proba = feat_info['model_scores']
    deep_feat = feat_info['deep_feat']
    patient_info = {}
    for i, pid in enumerate(pid_all):
        if pid in patient_info.keys():
            assert patient_info[pid]['gid'] == group_ids[i]
            assert patient_info[pid]['label'] == Y[i]
            patient_info[pid]['slides'].append(slide_name_all[i])
            patient_info[pid]['proba'] = np.concatenate((patient_info[pid]['proba'], proba[i]), axis=0)
            patient_info[pid]['feat'] = np.concatenate((patient_info[pid]['feat'], deep_feat[i]), axis=0)
        else:
            patient_info[pid] = {'proba':proba[i], 'gid':group_ids[i], 'label':Y[i], 
                                'slides':[slide_name_all[i]], 'feat':deep_feat[i]} 
    return patient_info

def get_tile_rois(fg, tile_size=512, overlap_sz=256, fg_ratio=0.5):
    '''
    get tiles. Remove those majority on background
    '''
    H, W = fg.shape
    grid_sz = tile_size - overlap_sz
    nH, nW = int(np.ceil(1.0*H/grid_sz)), int(np.ceil(1.0*W/grid_sz))
    yx_coord = []
    R = tile_size // 2
    for i in range(nH):
        top = i * grid_sz
        bottom = top+tile_size if i != nH-1 else H
        gy = int(0.5*(top+bottom))
        for j in range(nW):
            left = j*grid_sz
            right = left+tile_size if j != nW-1 else W
            gx = int(0.5*(left+right))
            area = 1.0*(bottom - top)*(right - left)
            ovlap_ratio = np.sum(fg[top:bottom, left:right])/area
            if ovlap_ratio > fg_ratio:
                yx_coord.append([gy, gx])
    return np.array(yx_coord)

def get_dense_nuclei_rois(nuclei_map, fg, topk_roi, grid_sz=64, crop_size=256, fg_ratio=0.5):
    '''
    get dense nuclei regions
    nuclei_map: nuclei map
    fg: foreground mask, 1 for foreground, 0 for background
    '''
    H, W = nuclei_map.shape
    assert nuclei_map.shape == fg.shape
    nuclei_map *= fg
    nH, nW = int(np.ceil(1.0*H/grid_sz)), int(np.ceil(1.0*W/grid_sz))
    num_grids = nH*nW
    grid_cnt = np.zeros(num_grids, dtype=float)
    yx_coord = np.zeros((num_grids, 2), dtype=int)
    R = crop_size // 2
    for i in range(nH):
        top = i * grid_sz
        bottom = top+grid_sz if i != nH-1 else H
        gy = int(0.5*(top+bottom))
        for j in range(nW):
            left = j*grid_sz
            right = left+grid_sz if j != nW-1 else W
            gx = int(0.5*(left+right))
            lin_ind = i*nW + j
            yx_coord[lin_ind] = (gy,gx)
            # ignore those close to slide bondaries
            w_top, w_bottom = max(0, gy-R), min(H, gy+R)
            w_left, w_right = max(0, gx-R), min(W, gx+R)
            area = 1.0*(w_bottom - w_top)*(w_right - w_left)
            ovlap_ratio = np.sum(fg[w_top:w_bottom, w_left:w_right])/area
            if ovlap_ratio > fg_ratio:
                grid_cnt[lin_ind] = np.sum(nuclei_map[top:bottom, left:right])
    # sort grid_cnt in decreasing order
    topk_roi = min(topk_roi, num_grids)
    sort_ind = np.argsort(grid_cnt)[::-1][:topk_roi]
    return yx_coord[sort_ind], grid_cnt[sort_ind]

    

