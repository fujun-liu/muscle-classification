'''
    image sampler
    sample images
'''
import glob, os
import numpy as np
from random import shuffle

def WeightedReservoir(weight, k):
    N = weight.size
    if N <= k:
        return range(N)
    weight = weight/np.sum(weight)
    inidce = range(k)
    w_sum = .0
    for i in range(k):
        w_sum += weight[i]/k
    for i in range(k, N):
        w_sum += weight[i]/k
        p = weight[i]/w_sum
        if np.random.rand() < p:
            inidce[np.random.randint(0,k)] = i
    return inidce


class ImageSampler():
    def __init__(self, img_lst, topk=10):
        slide_ids = np.zeros(len(img_lst))
        slide_id_dict = {}
        slide_label_dict = {}
        cnt = 0
        for i, img_path in enumerate(img_lst):
            slide_name = self._get_slide_name(img_path)
            label = int(img_path[-5])
            if slide_name not in  slide_id_dict:
                slide_id_dict[slide_name] = cnt
                slide_label_dict[cnt] = label
                cnt += 1
            else:
                assert slide_label_dict[slide_id_dict[slide_name]] == label
            slide_ids[i] = slide_id_dict[slide_name]
        self.img_lst = np.asarray(img_lst)
        self.slide_ids = slide_ids
        self.num_slides = len(slide_id_dict)
        self.topk = topk
        self.slide_label_dict = slide_label_dict
        print slide_label_dict

    def _get_slide_name(self, img_path):
        img_name = os.path.split(img_path)[-1]
        parts = img_name.split('_')[:-4]
        return '_'.join(parts)
    
    def get_img_lst(self):
        return self.img_lst

    def _sample_from_slides(self, slide_img_lst, slide_weights):
        N = min(slide_img_lst.shape[0], self.topk)
        if slide_weights is None:
            return slide_img_lst[:N]
        else:
            #indice = np.argsort(slide_weights)[::-1][:N]
            indice = WeightedReservoir(slide_weights, N)
            return slide_img_lst[indice]
    
    def sample(self, weight=None):
        img_samples_all = []
        for slide_id in range(self.num_slides):
            indice = np.nonzero(slide_id == self.slide_ids)[0]
            if weight is None:
                slide_weights = np.ones(len(indice))
            else:
                label = self.slide_label_dict[slide_id]
                slide_weights = weight[indice, label]
            imgs_per_slide = self._sample_from_slides(self.img_lst[indice], slide_weights)
            img_samples_all.extend(list(imgs_per_slide))
        return img_samples_all



if __name__ == "__main__":
    #img_dir = '/media/fujunl/FujunLiu/muscle-classification/tiles'
    img_dir = '/media/fujunl/FujunLiu/muscle-classification/train-patches-30'
    img_lst = glob.glob(img_dir + '/*.png')
    sampler = ImageSampler(img_lst)
    weight = np.ones((len(sampler.get_img_lst()), 3))
    tmp = sampler.sample(weight=weight)
            

