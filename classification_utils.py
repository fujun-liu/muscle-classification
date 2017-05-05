from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from numpy import linalg as LA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
import numpy as np
import pickle
import random


def normalize_feature(X):
    '''
         X: n x p, where n is # of samples, p is dim size
        normalize feature,  only l2 used
        other normalized shoud also considered
        This depends on how centers are computed
    '''
    norm = LA.norm(X, ord=2, axis=1)
    return X / norm[:,np.newaxis]


def post_processing_data(X, pca_components, whiten=False):
    '''
        X: n x p, where n is # of samples, p is dim size
        here X denotes raw features
    '''
    # step 1, l2 
    X = normalize_feature(X)
    # step 2, pca
    pca_model = PCA(n_components=pca_components, whiten=whiten)
    X_pca = pca_model.fit_transform(X)
    # step 3, l2 normalize
    return normalize_feature(X_pca), pca_model
    
def gridsearch_model(est, tuned_parameters, X_train, y_train, X_test=None, y_test=None, show_details=False):
    '''
        grid search on model hyper-parameters
    '''
    std_scaler = preprocessing.StandardScaler()
    X_train_transform = std_scaler.fit_transform(X_train)
    model = GridSearchCV(est, tuned_parameters, cv=5)
    model.fit(X_train_transform, y_train)
    print("Best parameters set found on development set:")
    print(model.best_params_)
    if show_details:
        print("Grid scores on development set:")
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        if X_test is not None:
            y_true, y_pred = y_test, model.predict(std_scaler.transform(X_test))
            print(classification_report(y_true, y_pred))
    train_acc = model.score(X_train_transform, y_train)
    print 'training accuracy: {}'.format(train_acc)
    if X_test is not None:
        test_acc = model.score(std_scaler.transform(X_test), y_test)
        print 'testing accuracy: {}'.format(test_acc)
    else:
        test_acc = train_acc
    trained_model = {'norm_model':std_scaler, 'pred_model':model}
    return test_acc, trained_model

def train_model_with_grid_search(X_train, y_train, model_strs, X_test=None, y_test=None, normalize_feat=False, show_details=False):
    if normalize_feat:
        print 'normalizing data'
        X_train = normalize_feature(X_train)
        if X_test is not None:
            X_test = normalize_feature(X_test)

    if 'logit' in model_strs:
        print '---------- grid search for logistic regression -------'
        est = LogisticRegression(n_jobs=-1)
        tuned_parameters = [{'penalty':['l1', 'l2'], 'C':[0.001, 0.01, 0.1, 1, 10]}]
        test_acc, model = gridsearch_model(est, tuned_parameters, X_train, y_train, X_test, y_test)

    if 'svm' in model_strs:
        print '-------- grid search for svm --------'
        est = svm.SVC()
        tuned_parameters = [{'kernel': ['linear', 'rbf'], 'C': [0.001, 0.01, 0.1, 1, 10]}]
        test_acc, model = gridsearch_model(est, tuned_parameters, X_train, y_train, X_test, y_test)
    
    if 'rf' in model_strs:
        print '-------- grid search for rf --------'
        est = RandomForestClassifier()
        tuned_parameters = [{'n_estimators': [30,60,100], }]
        test_acc, model = gridsearch_model(est, tuned_parameters, X_train, y_train, X_test, y_test)
    

    if 'nn' in model_strs:
        print '------- grid search for nearest neighbor classifier -------'
        est = KNeighborsClassifier(n_jobs=-1)
        tuned_parameters = [{'n_neighbors': [5,10,20]}]
        test_acc, model = gridsearch_model(est, tuned_parameters, X_train, y_train, X_test, y_test)
    return model, test_acc

def train_model_help(X_train, X_test, y_train, y_test, pca_options, model_strs, show_details, normalize_feat=True):
    if normalize_feat:
        print 'normalizing data'
        X_train = normalize_feature(X_train)
        X_test = normalize_feature(X_test)
    if pca_options is not None:
        print 'Before pca, X_train is : {} x {}'.format(X_train.shape[0], X_train.shape[1])
        pca_model = PCA(n_components=pca_options['pca_components'], whiten=pca_options['whiten'])
        X_train = pca_model.fit_transform(X_train)
        X_test = pca_model.transform(X_test)
        # l2 normalize again
        X_train = normalize_feature(X_train)
        X_test = normalize_feature(X_test)
        print 'After pca, X_train is : {} x {}'.format(X_train.shape[0], X_train.shape[1])

    if 'logit' in model_strs:
        print '---------- grid search for logistic regression -------'
        est = LogisticRegression(n_jobs=-1)
        tuned_parameters = [{'penalty':['l1', 'l2'], 'C':[0.01, 0.1, 1]}]
        test_acc, model = gridsearch_model(est, tuned_parameters, X_train, y_train, X_test, y_test)

    if 'svm' in model_strs:
        print '-------- grid search for svm --------'
        est = svm.SVC()
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10]}]
        test_acc, model = gridsearch_model(est, tuned_parameters, X_train, y_train, X_test, y_test)
    
    if 'nn' in model_strs:
        print '------- grid search for nearest neighbor classifier -------'
        est = KNeighborsClassifier(n_jobs=-1)
        tuned_parameters = [{'n_neighbors': [5,10,20]}]
        test_acc, model = gridsearch_model(est, tuned_parameters, X_train, y_train, X_test, y_test)
    return test_acc, model

def train_model_repeat(X, Y, num_repeat, pca_options = None, model_strs = ('logit',), show_details=False):
    test_acc = .0
    for i in range(num_repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=i)
        test_acc_this, _ = train_model_help(X_train, X_test, y_train, y_test, pca_options, model_strs, show_details)
        test_acc += test_acc_this
    return test_acc/num_repeat

def image_classification(model, X, Y, pid_intids):
    Y_pred = model['pred_model'].predict(model['norm_model'].transform(X))
    unique_pids = np.unique(pid_intids)
    test_acc = .0
    num_labels = len(np.unique(Y))
    for pid in unique_pids:
        cell_indice = np.nonzero(pid == pid_intids)[0]
        #print '# of cells {}, cell label {}'.format(len(cell_indice), np.unique(Y[cell_indice]))
        cell_label = Y[cell_indice[0]]
        cell_pred_all = Y_pred[cell_indice]
        pred_dist = np.zeros(num_labels)
        for cell_pred in cell_pred_all:
            pred_dist[cell_pred] += 1.0
        pred_label = np.argmax(pred_dist)
        if cell_label == pred_label:
            test_acc += 1.0
        else:
            pass
            #print pred_dist
    return test_acc/len(unique_pids)

def train_cell_model_repeat(X, Y, pid_intids, num_repeat, normalize_feat=True, model_strs = ('logit',),  pca_options=None, show_details=False):
    unique_pids = np.unique(pid_intids)
    N = unique_pids.size
    test_ratio = 0.2
    test_num = int(N*test_ratio)
    train_acc = .0
    test_acc = .0
    for i in range(num_repeat):
        # split based on pids
        rand_pids = np.random.permutation(unique_pids)
        test_pids = rand_pids[:test_num]
        train_pids = rand_pids[test_num:]
        train_test_split_flag = np.zeros(len(pid_intids))
        for pid in train_pids:
            train_test_split_flag[pid == pid_intids] = 1
        # split based
        train_ind = np.nonzero(train_test_split_flag == 1)[0]
        test_ind = np.nonzero(train_test_split_flag == 0)[0]
        print '# patients for training {}, # patients for testing {}'.format(len(train_pids), len(test_pids))
        X_test, X_train = X[test_ind,:], X[train_ind,:]
        y_test, y_train = Y[test_ind], Y[train_ind]
        _, model = train_model_help(X_train, X_test, y_train, y_test, pca_options, model_strs, show_details, normalize_feat)
        # do image classification based on majority voting
        pid_intids_train = pid_intids[train_ind]
        pid_intids_test = pid_intids[test_ind]
        if normalize_feat:
            X_train = normalize_feature(X_train)
            X_test = normalize_feature(X_test)

        train_acc_curr = image_classification(model, X_train, y_train, pid_intids_train)
        test_acc_curr = image_classification(model, X_test, y_test, pid_intids_test)
        print 'round {}: training accuracy {}'.format(i+1, train_acc_curr)
        print 'round {}: testing accuracy {}'.format(i+1, test_acc_curr)
        train_acc += train_acc_curr
        test_acc += test_acc_curr
    
    train_acc = train_acc/num_repeat
    test_acc = test_acc/num_repeat
    print 'Average training accuracy {}'.format(train_acc)
    print 'Average testing accuracy {}'.format(test_acc)

def train_model(X, Y, train_test_split_flag = None, pca_options = None, model_strs = ('logit',), show_details=False):
    
    if train_test_split_flag is None:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    else:
        train_ind = np.nonzero(train_test_split_flag == 1)[0]
        test_ind = np.nonzero(train_test_split_flag == 0)[0]
        X_test, X_train = X[test_ind,:], X[train_ind,:]
        y_test, y_train = Y[test_ind], Y[train_ind]
        print X.shape
        print 'Train: {} out of {} are positive'.format(np.sum(y_train), y_train.shape)
        print 'Test: {} out of {} are positive'.format(np.sum(y_test), y_test.shape)

    train_model_help(X_train, X_test, y_train, y_test, pca_options, model_strs, show_details)

def train_model_leave_one_out(X, Y, pid_intids, pca_options = None, model_strs = ('logit',), show_details=False):
    '''
        Each time leave one patient out
    '''
    unique_pids = np.unique(pid_intids)
    for pid in unique_pids:
        test_ind = np.nonzero(pid == pid_intids)[0]
        train_ind = np.nonzero(pid != pid_intids)[0]
        X_test, X_train = X[test_ind,:], X[train_ind,:]
        y_test, y_train = Y[test_ind], Y[train_ind]
        train_model_help(X_train, X_test, y_train, y_test, pca_options, model_strs, show_details)

    