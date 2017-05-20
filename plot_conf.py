import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from sklearn.metrics import confusion_matrix

class_names = ['DM', 'PM', 'IBM']
#conf_file_path = 'results/deep-feat-resnet-30_conf.p'
#with open(conf_file_path, 'rb') as f:
#    conf_m = pickle.load(f)
#    y_test = conf_m['pred']
#    y_pred = conf_m['label']
conf_file_path = 'results/deep-feat-resnet-30_repeat20_feat_prob_16_conf.p'
with open(conf_file_path, 'rb') as f:
    '''
    result = {'label':agg_label, 'pred_all':agg_pred_all, 'pid':agg_pid, 
                    'bow_acc_all': bow_acc_all, 'voting_acc':voting_acc}
    '''
    conf_m = pickle.load(f)
    bow_acc_all = conf_m['bow_acc_all']
    index = np.argmax(bow_acc_all)
    y_pred = conf_m['pred_all'][index]
    y_test = conf_m['label']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    font_prop = {'size':'18'}
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, **font_prop)
    plt.yticks(tick_marks, classes, **font_prop)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", **font_prop)

    plt.tight_layout()
    plt.ylabel('True label', **font_prop)
    plt.xlabel('Predicted label', **font_prop)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, 
                      title='Confusion Matrix')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')

plt.show()