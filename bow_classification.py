import os, pickle
import numpy as np
from classification_utils import train_model_with_grid_search
from encoding_image_feature import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
'''
BoW classification
1. pca; 2. kmeans; 3. bow encoding of patients; 4. classification
'''
'''
format
feat_info = {'X_lst': X_lst, 'Y':Y, 'pid_all':pid_all, 'slide_name_all': slide_name_all, 
                'centers_lst':centers_lst, 'group_ids':group_ids, 'model_scores':model_scores, 'deep_feat':DeepFeat}
'''
# used for model trained from nuclei regions
#pca_dim = 32
#num_clusters = 16
pca_dim = -1
# 16 used other than tiles
num_clusters = 16
use_proba_feat = False
# only used for tile based algortithm
topk_sel = -1
save_result = True

def cv_bow(patient_info, pca_dim, num_clusters, num_folds=3):
    '''
    format
    patient_info[pid] = {'proba':proba[i], 'gid':group_ids[i], 'label':Y[i], 
                                'slides':[slide_name_all[i]], 'feat':deep_feat[i]} 
    '''
    avg_voting_acc = .0
    avg_bow_acc = .0
    avg_ensemble_acc = .0
    conf_bow = np.zeros((3,3))
    conf_voting = np.zeros((3,3))
    agg_pred, agg_label = None, None
    bow_feat = None
    agg_pid = []
    for gid in range(num_folds):
        proba_train_lst, proba_test_lst = [], []
        X_train_lst, X_test_lst = [], []
        Y_train, Y_test = [], []
        deep_acc = .0
        for pid in patient_info.keys():
            proba_slide = patient_info[pid]['proba']
            proba_sort = np.sort(proba_slide, axis=1)
            p_diff = (proba_sort[:,-1] - proba_sort[:,-2])
            p_std = np.std(proba_slide, axis=1)[:,np.newaxis]
            proba_feat = np.concatenate((proba_slide, p_diff[:,np.newaxis]), axis=1)
            raw_feat = patient_info[pid]['feat']
            if topk_sel > 0:
                sort_ind = np.argsort(p_diff)[::-1][:min(topk_sel, p_diff.size)]
                proba_feat = proba_feat[sort_ind, :]
                raw_feat = raw_feat[sort_ind, :]
            #feat_m1 = np.mean(proba_slide, axis=0)[np.newaxis,:]
            #feat_m2 = np.median(proba_slide, axis=0)[np.newaxis,:]
            #proba_feat = np.concatenate((proba_slide, feat_m1, feat_m2), axis=0)
            #proba_feat = proba_slide
            #print proba_feat.shape
            if patient_info[pid]['gid'] == gid:
                #p_diff /= np.sum(p_diff)
              #
                #proba_slide *= p_diff[:, np.newaxis]
                agg_pid.append(pid)
                deep_pred = np.argmax(np.mean(proba_slide, axis=0))
                conf_voting[patient_info[pid]['label'], deep_pred] += 1.0
                deep_acc += patient_info[pid]['label'] == deep_pred
                # test data
                X_test_lst.append(raw_feat)
                
                #proba_test_lst.append(np.mean(patient_info[pid]['proba'], axis=0)[np.newaxis,:])
                proba_test_lst.append(proba_feat)
                Y_test.append(patient_info[pid]['label'])
            else:
                # train data
                X_train_lst.append(raw_feat)
                #proba_train_lst.append(np.mean(patient_info[pid]['proba'], axis=0)[np.newaxis,:])
                proba_train_lst.append(proba_feat)
                Y_train.append(patient_info[pid]['label'])
        print len(X_train_lst), len(X_test_lst)
        #prob_train = np.concatenate(proba_train_lst, axis=0)
        #prob_test = np.concatenate(proba_test_lst, axis=0)
        if use_proba_feat:
            X_train_lst = proba_train_lst
            X_test_lst = proba_test_lst

        X_train = np.concatenate(X_train_lst, axis=0)
        Y_train = np.asarray(Y_train)
        Y_test = np.asarray(Y_test)
        # pca dim
        if pca_dim > 0:
            pca_model = PCA(n_components=pca_dim)
            X_train = pca_model.fit_transform(X_train)
        # l2 normalize again
        X_train = normalize_feature(X_train)
        # run kmeans on training
        kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, n_jobs=-1)
        kmeans_model.fit(X_train)
        centers = kmeans_model.cluster_centers_
        num_train, num_test = len(X_train_lst), len(X_test_lst)
        code_train = np.zeros((num_train, num_clusters))
        code_test = np.zeros((num_test, num_clusters))
        # encode train
        for i in range(num_train):
            feat_local = X_train_lst[i]
            if pca_dim > 0:
                feat_local = pca_model.transform(feat_local)
            code_train[i] = bow_encoding(centers, feat_local)
        # encode test
        for i in range(num_test):
            feat_local = X_test_lst[i]
            if pca_dim > 0:
                feat_local = pca_model.transform(feat_local)
            code_test[i] = bow_encoding(centers, feat_local)
        
        bow_feat = code_test if bow_feat is None else np.concatenate((bow_feat, code_test), axis=0)
        model, bow_acc = train_model_with_grid_search(code_train, Y_train, ('rf',), X_test=code_test, y_test=Y_test)
        bow_proba = model['pred_model'].predict_proba(model['norm_model'].transform(code_test))
        bow_pred = np.argmax(bow_proba, axis=1)
        agg_label = Y_test if agg_label is None else np.concatenate((agg_label, Y_test))
        agg_pred = bow_pred if agg_pred is None else np.concatenate((agg_pred, bow_pred))
        for i in range(num_test): conf_bow[Y_test[i], bow_pred[i]] += 1.0
        #ensemble_pred = np.argmax(prob_test+bow_proba, axis=1)
        #ensemble_acc = 1.0*np.sum(ensemble_pred == Y_test)/num_test
        voting_acc = deep_acc/num_test
        avg_bow_acc += bow_acc
        avg_voting_acc += voting_acc
        #avg_ensemble_acc += ensemble_acc
        print 'voting acc: {}'.format(voting_acc)
        #print 'ensemble acc: {}'.format(ensemble_acc)
    
    print 'Avg bow acc: {}'.format(avg_bow_acc/num_folds)
    print conf_bow
    print 'Avg voting acc {}'.format(avg_voting_acc/num_folds)
    print conf_voting
    avg_voting_acc /= num_folds 
    bow_acc = np.sum(agg_label == agg_pred)/agg_label.size
    #print 'Avg ensemble acc {}'.format(avg_ensemble_acc/num_folds)
    return agg_label, agg_pred, agg_pid, avg_voting_acc, bow_feat


if __name__ == "__main__":
    from scipy.io import savemat
    feat_dir = 'deep-feat'
    # num_clusters = 8 used
    #feat_name = 'deep-feat-resnet-10'
    #feat_name = 'deep-feat-inception-10'
    #feat_name = 'deep-feat-vgg-10'
    
    # num_clusters 16 used
    #feat_name = 'deep-feat-resnet-20'
    #feat_name = 'deep-feat-vgg-train20-test20'
    #feat_name = 'deep-feat-inception-train20-test20'

    # num_clusters = 16 used
    feat_name = 'deep-feat-resnet-30'
    #feat_name = 'deep-feat-vgg-train30-test30'
    #feat_name = 'deep-feat-inception-train30-test30'
    
    # last
    #feat_name = 'deep-feat-resnet-train10-test10-last'
    # from scratch
    
    # from tiles
    #feat_name = 'deep-feat-resnet-tile'
    #feat_name = 'deep-feat-resnet-tile-200-em30'
    feat_path = os.path.join(feat_dir, feat_name + '.p')
    print feat_path
    #patient_info = load_old_patient_info(feat_path)
    with open(feat_path, 'rb') as f:
        patient_info = pickle.load(f)
    agg_label, agg_pred, agg_pid, voting_acc, bow_feat = cv_bow(patient_info, pca_dim, num_clusters)
    
    if save_result:
        if use_proba_feat: tmp_name = feat_name + '_feat_prob_{}'.format(num_clusters)
        else: tmp_name = feat_name + '_feat_{}'.format(num_clusters)
        if topk_sel > 0: tmp_name += '_top{}'.format(topk_sel)
        mat_path = os.path.join('results', 'mat', tmp_name + '_conf.mat')
        p_path = os.path.join('results', tmp_name + '_conf.p')
        result = {'label':agg_label, 'pred':agg_pred, 'pid':agg_pid, 'voting_acc':voting_acc}
        savemat(mat_path, result)
        with open(p_path, 'wb') as f:
            pickle.dump(result, f)
        
        bow_feat_path = os.path.join('results', feat_name + '_bow_feat.p')
        print bow_feat
        with open(bow_feat_path, 'wb') as f:
            pickle.dump(bow_feat, f)

