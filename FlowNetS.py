from __future__ import print_function

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Deconv2DLayer
from lasagne.layers import ConcatLayer

from lasagne.nonlinearities import LeakyRectify
from lasagne.nonlinearities import linear

leaky_rectify = LeakyRectify(0.1)

def leaky_conv(input_layer, **kwargs):
    return Conv2DLayer(input_layer, nonlinearity=leaky_rectify, pad='same', **kwargs)

def leaky_deconv(input_layer, **kwargs):
    return Deconv2DLayer(
        input_layer, nonlinearity=leaky_rectify,
        filter_size=4, stride=2, crop=1, **kwargs)

def flow(input_layer, **kwargs):
    return Conv2DLayer(
        input_layer, num_filters=2, filter_size=3, stride=1,
        b=None, nonlinearity=linear, pad=1, **kwargs)

def upsample(input_layer, **kwargs):
    return Deconv2DLayer(
        input_layer, num_filters=2, filter_size=4, stride=2,
        crop=1, b=None, nonlinearity=linear, **kwargs)

def build_model():
    net = dict()

    net['input_1'] = InputLayer([None, 3, 384, 512])

    net['input_2'] = InputLayer([None, 3, 384, 512])

    net['input'] = ConcatLayer([net['input_1'], net['input_2']])

    net['conv1'] = leaky_conv(net['input'], num_filters=64, filter_size=7, stride=2)
    net['conv2'] = leaky_conv(net['conv1'], num_filters=128, filter_size=5, stride=2)
    
    net['conv3'] = leaky_conv(net['conv2'], num_filters=256, filter_size=5, stride=2)
    net['conv3_1'] = leaky_conv(net['conv3'], num_filters=256, filter_size=3, stride=1)
    
    net['conv4'] = leaky_conv(net['conv3_1'], num_filters=512, filter_size=1, stride=2)
    net['conv4_1'] = leaky_conv(net['conv4'], num_filters=512, filter_size=1, stride=1)
    
    net['conv5'] = leaky_conv(net['conv4_1'], num_filters=512, filter_size=1, stride=2)
    net['conv5_1'] = leaky_conv(net['conv5'], num_filters=512, filter_size=1, stride=1)

    net['conv6'] = leaky_conv(net['conv5_1'], num_filters=1024, filter_size=1, stride=2)
    net['conv6_1'] = leaky_conv(net['conv6'], num_filters=1024, filter_size=1, stride=1)

    net['flow6'] = flow(net['conv6_1'])
    net['flow6_up'] = upsample(net['flow6'])
    net['deconv5'] = leaky_deconv(net['conv6_1'], num_filters=512)

    net['concat5'] = ConcatLayer([net['conv5_1'], net['deconv5'], net['flow6_up']])
    net['flow5'] = flow(net['concat5'])
    net['flow5_up'] = upsample(net['flow5'])
    net['deconv4'] = leaky_deconv(net['concat5'], num_filters=256)

    net['concat4'] = ConcatLayer([net['conv4_1'], net['deconv4'], net['flow5_up']])
    net['flow4'] = flow(net['concat4'])
    net['flow4_up'] = upsample(net['flow4'])
    net['deconv3'] = leaky_deconv(net['concat4'], num_filters=128)

    net['concat3'] = ConcatLayer([net['conv3_1'], net['deconv3'], net['flow4_up']])
    net['flow3'] = flow(net['concat3'])
    net['flow3_up'] = upsample(net['flow3'])
    net['deconv2'] = leaky_deconv(net['concat3'], num_filters=64)

    net['concat2'] = ConcatLayer([net['conv2'], net['deconv2'], net['flow3_up']])
    net['flow2'] = flow(net['concat2'])

    return net


if __name__ == '__main__':
    net = build_model()

    input_vars = lasagne.layers.get_output([net['input_1'], net['input_2']])

    flow = theano.function(input_vars, lasagne.layers.get_output(net['flow2'], deterministic=True))
