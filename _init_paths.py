"""Set up paths"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# Add caffe to PYTHONPATH
#caffe_path = osp.join('/home/fujun/py-faster-rcnn', 'caffe-fast-rcnn', 'python')
caffe_path = osp.join('/home/fujun/work/hed', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
#lib_path = osp.join('/home/fujun/py-faster-rcnn', 'lib')
#add_path(lib_path)
