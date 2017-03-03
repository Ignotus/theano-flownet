FLOWNET_DIR = '/home/minh/Development/dispflownet-release'

import numpy as np

import sys
sys.path.append('%s/python' % FLOWNET_DIR)

import caffe

from math import ceil

if __name__ == '__main__':
    model_type = sys.argv[1]
    height = 384
    width = 512
    divisor = 64.
    adapted_width = ceil(width/divisor) * divisor
    adapted_height = ceil(height/divisor) * divisor
    rescale_coeff_x = width / adapted_width
    rescale_coeff_y = height / adapted_height

    replacement_list = {
        '$ADAPTED_WIDTH': ('%d' % adapted_width),
        '$ADAPTED_HEIGHT': ('%d' % adapted_height),
        '$TARGET_WIDTH': ('%d' % width),
        '$TARGET_HEIGHT': ('%d' % height),
        '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
        '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y)
    }

    proto = ''
    with open("model/deploy.tpl.prototxt", "r") as tfile:
        proto = tfile.read()

    for r in replacement_list:
        proto = proto.replace(r, replacement_list[r])

    with open('tmp/deploy.prototxt', "w") as tfile:
        tfile.write(proto)

    caffe.set_mode_gpu()
    caffe.set_device(0)

    deployFile = "tmp/deploy.prototxt"
    caffemodel = "model/flownet%s.caffemodel" % model_type

    net = caffe.Net(deployFile, caffemodel, caffe.TEST)
   
    weights = dict() 
    for key, val in net.params.items():
        print key,
        for i in range(len(val)):
            print val[i].data.shape,
        print

        weights[key] = [val[i].data for i in range(len(val))]

    np.savez('flownet%s.npz' % model_type, weights)
