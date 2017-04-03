import caffe
import numpy as np
from PIL import Image
import random
import os
import glob

class PythonDataLayer(caffe.Layer):
    '''
    This is a Python data layer 
    '''
    def setup(self, bottom, top):
        '''
        parse param_str. In prototxt file param_str : '{"phase":"TRAIN", "mean":(r,g,b)}' 
        '''
        params = eval(self.param_str)
        self.use_color = params.get('color', True)
        #self.mean = np.array(params['mean'])
        self.use_self_mean = params.get('use_self_mean', True)
        self.batch_sz = params.get('batch_sz', 1)
        if len(top) != 2:
            raise Exception("Two tops, data and label expected!")
        if len(bottom):
            raise Exception("No bottoms, input, expected")
        data_folder = params['data_folder']
        self.label_suffix = '_seg.png'
        self.file_paths = glob.glob(data_folder + '/*' + self.label_suffix)
        # shuffle here
        random.shuffle(self.file_paths)
        self.idx = 0
    
    def reshape(self, bottom, top):
        '''
            reshape called in forward
        '''
        label_path = self.file_paths[self.idx]
        img_path = label_path[:-len(self.label_suffix)] + '.png'
        self.data = self.load_image(img_path)
        self.label = self.load_label(label_path)
        top[0].reshape(self.batch_sz, *self.data.shape)
        top[1].reshape(self.batch_sz, 1, *self.label.shape)
    
    def forward(self, bottom, top):
        for i in range(self.batch_sz):
            label_path = self.file_paths[self.idx + i]
            img_path = label_path[:-len(self.label_suffix)] + '.png'
            top[0].data[i,...] = self.load_image(img_path)
            top[1].data[i,0,:,:] = self.load_label(label_path)
        self.idx += self.batch_sz
        if self.idx+self.batch_sz > len(self.file_paths):
            random.shuffle(self.file_paths)
            self.idx = 0
    
    def backward(self, top, propagate_down, bottom):
        pass
    
    def load_image(self, img_path):
        #print img_path
        img = np.array(Image.open(img_path), dtype=np.float32)/255.0
        if self.use_color:
            img = img[:,:,::-1]
            if self.use_self_mean: img -= np.mean(img, axis=(0,1))
            else: img -= self.mean
            return np.transpose(img, (2,0,1))
        else:
            img = np.mean(img, axis=2) - 128
            return img[np.newaxis,...]
    
    def load_label(self, label_path):
        #print label_path
        label = np.array(Image.open(label_path), dtype=np.float32)
        if len(label.shape) == 3: label = label[:, :, 0]
        label[label > 0] = 1
        return label
