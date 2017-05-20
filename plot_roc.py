'''
plot_roc.py
'''
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import pickle

class_names = ['DM', 'PM', 'IBM']
n_classes = len(class_names)
#conf_file_path = 'results/deep-feat-resnet-30_conf.p'
#with open(conf_file_path, 'rb') as f:
#    conf_m = pickle.load(f)
#    y_test = conf_m['pred']
#    y_pred = conf_m['label']
conf_file_path = 'results/deep-feat-resnet-30_repeat20_feat_prob_16_score_conf.p'
with open(conf_file_path, 'rb') as f:
    '''
    result = {'label':agg_label, 'pred_all':agg_pred_all, 'pid':agg_pid, 
                    'bow_acc_all': bow_acc_all, 'voting_acc':voting_acc}
    '''
    conf_m = pickle.load(f)
    bow_acc_all = conf_m['bow_acc_all']
    index = np.argmax(bow_acc_all)
    y_score = conf_m['agg_bow_score'][index]
    y_pred = conf_m['pred_all'][index]
    y_test = conf_m['label']
    y_test_binary = label_binarize(y_test, classes=[0,1,2])

# compute fpr, tpr
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# plot

plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)
#
#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)

lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

    
