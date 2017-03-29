import numpy as np
import cv2
import sys
# Make sure that caffe is on the python path:
# this file is expected to be in {caffe_root}/examples/hed/
# this is traind by zizhao
caffe_root = '/home/fujun/work/caffe-muscle/'
sys.path.insert(0, caffe_root + 'python')
#import _init_paths
import caffe

class FCNSegNet():
	
    def __init__(self, net_paras, gpu_id=-1):
	    # build network
		self.net_paras = net_paras
		# load model and set gpu
		if gpu_id != -1:
			caffe.set_mode_gpu()
			print 'using gpu {}'.format(gpu_id)
			caffe.set_device(gpu_id)
		else:
			caffe.set_mode_cpu()
		self.net = caffe.Net(net_paras['deployproto_path'], net_paras['modelfile_path'], caffe.TEST)

    def forward_patch(self, im, resize_ratio):
        '''
            do forward computation here
        '''
        des_h = int(np.round(im.shape[0]*resize_ratio))
        des_w = int(np.round(im.shape[1]*resize_ratio))  
        in_ = cv2.resize(im, (des_w, des_h)).astype(float)
        if self.net_paras['use_color']:
            in_ = in_[:, :, ::-1]
            if self.net_paras['use_self_mean']:
                in_ -= np.mean(in_, axis=(0,1))
            else:
                in_ -= np.array(self.net_paras['model_mean'])
            in_ = in_.transpose((2, 0, 1))
        else:
            print 'using gray image'
            in_ = np.mean(in_, axis=2) - 128.0
            in_ = in_.reshape(1, *in_.shape)
        
        self.net.blobs['data'].reshape(1, *in_.shape)
        self.net.blobs['data'].data[...] = in_
        self.net.forward()
    

    def forward_batch(self, im, part=500, batch_sz=10, resize_ratio=1.0):
        '''
            no overlap used
        '''
        result_layer_name = self.net_paras['result_layer_name']
        if self.net_paras['normalize']: 
            im /= 255.0
        # size divisible by part size
        nH = int(np.round(1.0*im.shape[0]*resize_ratio/part))
        nW = int(np.round(1.0*im.shape[1]*resize_ratio/part))
        print 'There are {} parts with {}x{} size'.format(nH*nW, part, part)
        H, W = nH*part, nW*part

        in_ = cv2.resize(im, (W, H)).astype(float)
        if self.net_paras['use_color']:
            nchannels = 3
            in_ = in_[:, :, ::-1]
            in_ -= np.array(self.net_paras['model_mean'])
            in_ = in_.transpose((2, 0, 1))
        else:
            print 'using gray image'
            nchannels = 1
            in_ = np.mean(in_, axis=2) - 0.5
            in_ = in_.reshape(1, *in_.shape)
        # collect all data in memory
        num_parts = nH * nW
        patch_all = np.zeros((num_parts, nchannels, part, part), dtype=np.float32)
        start_pos = np.zeros((num_parts, 2), dtype=np.int)
        cnt  = 0
        for ih in range(nH):
            for iw in range(nW):
                top, left = ih*part, iw*part
                patch_all[cnt] = in_[:,top:top+part, left:left+part]
                start_pos[cnt] = (ih, iw)
                cnt += 1
        # batch processing
        num_rounds = int(np.ceil(1.0*num_parts/batch_sz))
        seg_map = np.zeros((H, W), dtype=np.float32)
        for r in range(num_rounds):
            # hadnle this 
            this_size = num_parts-r*batch_sz if r == num_rounds-1 else batch_sz
            self.net.blobs['data'].reshape(this_size, nchannels, part, part)
            self.net.blobs['data'].data[...] = patch_all[r*batch_sz:r*batch_sz+this_size]
            self.net.forward()
            batch_score = self.net.blobs[result_layer_name].data
            print batch_score.shape
            for i in range(this_size):
                ih, iw = start_pos[i+r*batch_sz]
                seg_map[ih*part:(ih+1)*part, iw*part:(iw+1)*part] = batch_score[i,0]
        
        return seg_map

def load_net():
    net_paras = {
    'deployproto_path': '../models/deploy_fiber_seg_he.prototxt',
    'modelfile_path': '../models/fiber_seg_he.caffemodel',
    'model_mean': (151.150621644, 88.8360644307,177.680687791),
    'result_layer_name': 'sigmoid-fuse-seg',
    'normalize': False, 
    'use_self_mean': False,
    'use_color': True
    }
    gpu_id = 0
    return FCNSegNet(net_paras, gpu_id=gpu_id)

if __name__ == "__main__":
    from openslide import OpenSlide
    import glob
    import os
    import time
    # load net
    fcn_net = load_net()

    # network parameters
    wsi_dir = '/data/fujun/data/muscle-whole-slides/'
    des_dir = '/data/fujun/data/muscle-fiber-seg-result'
    result_suffix = 'fiber'
    wsi_suffix = ('.ndpi', '.tiff', '.svs')
    wsi_lst = []
    wsi_name_lst = []
    for suffix in wsi_suffix:
        tmp_lst = glob.glob(wsi_dir + '/*' + suffix)
        wsi_lst += tmp_lst
        wsi_name_lst += [os.path.split(path)[-1][:-len(suffix)] for path in tmp_lst]
    # the size deep learning can handle
    seg_part = 500
    # read whole slide region by region
    read_size = 5000
    # save size
    save_size = 5000
    # slide_level
    res_level_ref = 1
    # for debug, save patch result
    save_patch = False
    for wsi_path, wsi_name in zip(wsi_lst, wsi_name_lst):
        # open slide 
        osi = OpenSlide(wsi_path)
        res_level = min(res_level_ref, len(osi.level_dimensions)-1)
        W, H = osi.level_dimensions[res_level]
        down_ratio = osi.level_downsamples[res_level]
        print down_ratio, type(down_ratio)
        print '************ processing slide {}, size: {} x {} *****************'.format(wsi_name, H, W)
        nH = int(np.floor(1.0*H/read_size))
        nW = int(np.floor(1.0*W/read_size))
        save_size_ratio = 1.0*save_size/min(W,H)
        # figure out all des size 
        pos_H = np.zeros((nH, 2), dtype=np.int)
        pos_W = np.zeros((nW, 2), dtype=np.int)
        des_H, des_W = 0, 0
        for i in range(nH):
            this_H = read_size if i < nH-1 else H-i*read_size
            this_H_des = this_H if save_size_ratio >= 1.0 else int(np.round(this_H*save_size_ratio))
            pos_H[i] = (des_H, this_H_des)
            des_H += this_H_des
        for i in range(nW):
            this_W = read_size if i < nW-1 else W-i*read_size
            this_W_des = this_W if save_size_ratio >= 1.0 else int(np.round(this_W*save_size_ratio))
            pos_W[i] = (des_W, this_W_des)
            des_W += this_W_des
        
        seg_map_merge = np.zeros((des_H, des_W))
        img_merge = np.zeros((des_H, des_W, 3))
        for i in range(nH):
            this_H = read_size if i < nH-1 else H-i*read_size
            h1, h2 = pos_H[i,0], np.sum(pos_H[i])
            for j in range(nW):
                this_W = read_size if j < nW-1 else W-j*read_size
                w1, w2 = pos_W[j,0], np.sum(pos_W[j])
                t1 = time.time()
                region = np.asarray(osi.read_region((int(j*read_size*down_ratio), int(i*read_size*down_ratio)), res_level, (this_W, this_H)))[:,:,:3]
                seg_part = fcn_net.forward_batch(region)
                print 'It took {} seconds to process image with size {}x{}'.format(time.time()-t1, this_H, this_W)
                
                t1 = time.time()
                img_merge[h1:h2, w1:w2,:] = cv2.resize(region, (w2-w1, h2-h1))
                seg_map_merge[h1:h2, w1:w2] = cv2.resize(seg_part, (w2-w1, h2-h1))
                print 'It took {} sceonds to resize two images from {}x{} to {}x{}'.format(time.time()-t1, this_H, this_W, h2-h1, w2-w1)
                # save patch
                if save_patch:
                    tmp_path = os.path.join(des_dir, wsi_name + '_patch_{}_{}.png'.format(i, j))
                    cv2.imwrite(tmp_path, img_merge[h1:h2, w1:w2,:].astype(np.uint8))
                    tmp_path = os.path.join(des_dir, wsi_name + '_patch_{}_{}_{}_seg.png'.format(i, j, result_suffix))
                    cv2.imwrite(tmp_path, (255*seg_map_merge[h1:h2, w1:w2]).astype(np.uint8))
        
        # save results, save both image and map
        t1 = time.time()
        tmp_path = os.path.join(des_dir, wsi_name + '.png')
        cv2.imwrite(tmp_path, img_merge.astype(np.uint8))
        tmp_path = os.path.join(des_dir, wsi_name + '_{}_seg.png'.format(result_suffix))
        cv2.imwrite(tmp_path, (255*seg_map_merge).astype(np.uint8))
        

                


