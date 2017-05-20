'''
plot_hist.py
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle

bow_feat_path = 'results/deep-feat-resnet-30_bow_feat.p'
with open(bow_feat_path, 'rb') as f:
    bow_feat = pickle.load(f)

tmp = bow_feat > .0
tmp = np.sum(tmp, axis=1)
index = np.argsort(tmp)[50]
print 
plt.bar(range(bow_feat.shape[1]), bow_feat[index,:], color='rgbcky')
plt.show()
#plt.title("Gaussian Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")