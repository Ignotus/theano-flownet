from __future__ import print_function

import theano
import theano.tensor as T
import lasagne

import numpy as np

from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer

import cv2

from FlowNetCommon import *

def build_model(weights):
    weights = np.load(weights)['arr_0'][()]

    net = dict()

    # T.nnet.abstract_conv.bilinear_upsampling doesn't work properly if not to
    # specify a batch size
    batch_size = 1

    net['input_1'] = InputLayer([batch_size, 3, 384, 512])

    net['input_2'] = InputLayer([batch_size, 3, 384, 512])

    net['input'] = ConcatLayer([net['input_1'], net['input_2']])

    net['conv1'] = leaky_conv(net['input'], num_filters=64, filter_size=7, stride=2)
    net['conv2'] = leaky_conv(net['conv1'], num_filters=128, filter_size=5, stride=2)

    net['conv3'] = leaky_conv(net['conv2'], num_filters=256, filter_size=5, stride=2)
    net['conv3_1'] = leaky_conv(net['conv3'], num_filters=256, filter_size=3, stride=1)

    net['conv4'] = leaky_conv(net['conv3_1'], num_filters=512, filter_size=3, stride=2)
    net['conv4_1'] = leaky_conv(net['conv4'], num_filters=512, filter_size=3, stride=1)

    net['conv5'] = leaky_conv(net['conv4_1'], num_filters=512, filter_size=3, stride=2)
    net['conv5_1'] = leaky_conv(net['conv5'], num_filters=512, filter_size=3, stride=1)

    net['conv6'] = leaky_conv(net['conv5_1'], num_filters=1024, filter_size=3, stride=2)
    net['conv6_1'] = leaky_conv(net['conv6'], num_filters=1024, filter_size=3, stride=1)

    for layer_id in ['1', '2', '3', '3_1', '4', '4_1', '5', '5_1', '6', '6_1']:
        layer_name = 'conv' + layer_id
        print(layer_name, net[layer_name].W.shape.eval(), weights[layer_name][0].shape)
        print(layer_name, net[layer_name].b.shape.eval(), weights[layer_name][1].shape)
        net[layer_name].W.set_value(weights[layer_name][0][:,:,::-1,::-1])
        net[layer_name].b.set_value(weights[layer_name][1])

    refine_flow(net, weights)

    return net

if __name__ == '__main__':
    net = build_model('archive/flownets.npz')

    run(net)