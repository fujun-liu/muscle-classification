import os, pickle
import numpy as np
from sklearn.decomposition import PCA
import misvm

class MILClassifier():
    def __init__(self, num_classes=3, max_iters=100):
        self.num_classes = num_classes
        self.classifier_lst = []
        for i in range(num_classes):
            # this is a binary classifier
            mil_classifier = misvm.MISVM(kernel='linear', C=10.0, max_iters=max_iters)
            #mil_classifier = misvm.MICA(regularization='L2', max_iters=max_iters)
            self.classifier_lst.append(mil_classifier)

    def fit(self, X_lst, y):
        y = np.asarray(y)
        N = y.size
        for cid in range(self.num_classes):
            labels = np.ones_like(y)
            labels[cid != y] = -1
            self.classifier_lst[cid].fit(X_lst, labels)
        train_acc = self.score(X_lst, y)
        print 'Train acc: {}'.format(train_acc)

    def _softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        e_x = np.exp(z - s[:, np.newaxis])
        div = np.sum(e_x, axis=1)
        return e_x / div[:, np.newaxis]

    def predict_proba(self, X_lst):
        proba = np.zeros((len(X_lst), self.num_classes))
        for cid in range(self.num_classes):
            proba[:,cid] = self.classifier_lst[cid].predict(X_lst)
        print proba
        return self._softmax(proba)
    
    def predict(self, X_lst):
        proba = self.predict_proba(X_lst)
        return np.argmax(proba, axis=1)
    
    def score(self, X_lst, y):
        y = np.asarray(y)
        pred = self.predict(X_lst)
        return 1.0*np.sum(y==pred)/y.size

def cv_mil(patient_info, pca_dim=64, num_folds=3,):
    '''
    format
    patient_info[pid] = {'proba':proba[i], 'gid':group_ids[i], 'label':Y[i], 
                                'slides':[slide_name_all[i]], 'feat':deep_feat[i]} 
    '''
    avg_voting_acc = .0
    avg_mil_acc = .0
    for gid in range(num_folds):
        proba_train_lst, proba_test_lst = [], []
        X_train_lst, X_test_lst = [], []
        Y_train, Y_test = [], []
        deep_acc = .0
        for pid in patient_info.keys():
            if patient_info[pid]['gid'] == gid:
                deep_acc += patient_info[pid]['label'] == np.argmax(np.mean(patient_info[pid]['proba'], axis=0))
                # test data
                X_test_lst.append(patient_info[pid]['feat'])
                proba_test_lst.append(np.mean(patient_info[pid]['proba'], axis=0)[np.newaxis,:])
                Y_test.append(patient_info[pid]['label'])
            else:
                # train data
                X_train_lst.append(patient_info[pid]['feat'])
                proba_train_lst.append(np.mean(patient_info[pid]['proba'], axis=0)[np.newaxis,:])
                Y_train.append(patient_info[pid]['label'])
        num_train, num_test = len(X_train_lst), len(X_test_lst)
        print num_train, num_test
        Y_train = np.asarray(Y_train)
        Y_test = np.asarray(Y_test)
        # pca dim
        if pca_dim > 0:
            pca_model = PCA(n_components=pca_dim)
            X_train = np.concatenate(X_train_lst, axis=0)
            X_train = pca_model.fit_transform(X_train)
            for i in range(num_train):
                X_train_lst[i] = pca_model.transform(X_train_lst[i])
            for i in range(num_test):
                X_test_lst[i] = pca_model.transform(X_test_lst[i])
        num_classes = 1 + np.max(np.asarray(Y_train))
        mil_model = MILClassifier(num_classes=num_classes)
        mil_model.fit(X_train_lst, Y_train)
        mil_acc = mil_model.score(X_test_lst, Y_test)
        voting_acc = deep_acc/num_test
        print 'mil acc: {}'.format(mil_acc)
        print 'voting acc: {}'.format(voting_acc)
        avg_mil_acc += mil_acc
        avg_voting_acc += voting_acc
    
    print 'Avg mil acc: {}'.format(avg_mil_acc/num_folds)
    print 'Avg voting acc {}'.format(avg_voting_acc/num_folds)


if __name__ == "__main__":
    feat_dir = 'deep-feat'
    feat_name = 'deep-feat-resnet-20'
    feat_path = os.path.join(feat_dir, feat_name + '.p')
    print feat_path
    #patient_info = load_old_patient_info(feat_path)
    with open(feat_path, 'rb') as f:
        patient_info = pickle.load(f)
    #pca_dim = -1
    pca_dim = 32
    cv_mil(patient_info, pca_dim=pca_dim)

