from lasagne.layers import Deconv2DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import ConcatLayer

from lasagne.nonlinearities import LeakyRectify
from lasagne.nonlinearities import linear

leaky_rectify = LeakyRectify(0.1)

def leaky_conv(input_layer, pad='same', **kwargs):
    return Conv2DLayer(input_layer, nonlinearity=leaky_rectify, pad=pad, **kwargs)

def leaky_deconv(input_layer, **kwargs):
    return Deconv2DLayer(
        input_layer, nonlinearity=leaky_rectify,
        filter_size=4, stride=2, crop=1, **kwargs)

def upsample(input_layer, **kwargs):
    return Deconv2DLayer(
        input_layer, num_filters=2, filter_size=4, stride=2,
        crop=1, b=None, nonlinearity=linear, **kwargs)

def flow(input_layer, **kwargs):
    return Conv2DLayer(
        input_layer, num_filters=2, filter_size=3, stride=1,
        nonlinearity=linear, pad=1, **kwargs)

def switch_channels(images):
    return images.transpose(0, 3, 1, 2)

def refine_flow(net, weights):
    net['flow6'] = flow(net['conv6_1'])
    net['flow6_up'] = upsample(net['flow6'])
    net['deconv5'] = leaky_deconv(net['conv6_1'], num_filters=512)

    net['concat5'] = ConcatLayer([net['deconv5'], net['conv5_1'], net['flow6_up']])
    net['flow5'] = flow(net['concat5'])
    net['flow5_up'] = upsample(net['flow5'])
    net['deconv4'] = leaky_deconv(net['concat5'], num_filters=256)

    net['concat4'] = ConcatLayer([net['deconv4'], net['conv4_1'], net['flow5_up']])
    net['flow4'] = flow(net['concat4'])
    net['flow4_up'] = upsample(net['flow4'])
    net['deconv3'] = leaky_deconv(net['concat4'], num_filters=128)

    net['concat3'] = ConcatLayer([net['deconv3'], net['conv3_1'], net['flow4_up']])
    net['flow3'] = flow(net['concat3'])
    net['flow3_up'] = upsample(net['flow3'])
    net['deconv2'] = leaky_deconv(net['concat3'], num_filters=64)

    net['concat2'] = ConcatLayer([net['deconv2'], net['conv2'], net['flow3_up']])
    net['flow2'] = flow(net['concat2'])

    # TODO: Should be upsampled before 'flow1' to 384x512

    net['flow1'] = flow(net['flow2'])

    for layer_name in ['deconv5', 'deconv4', 'deconv3', 'deconv2']:
        net[layer_name].W.set_value(weights[layer_name][0])

    upsample_map = {
        'flow6_up': 'upsample_flow6to5',
        'flow5_up': 'upsample_flow5to4',
        'flow4_up': 'upsample_flow4to3',
        'flow3_up': 'upsample_flow3to2'
    }

    for layer_name in ['flow6_up', 'flow5_up', 'flow4_up', 'flow3_up']:
        net[layer_name].W.set_value(weights[upsample_map[layer_name]][0])

    flow_map = {
        'flow6': 'Convolution1',
        'flow5': 'Convolution2',
        'flow4': 'Convolution3',
        'flow3': 'Convolution4',
        'flow2': 'Convolution5',
        'flow1': 'Convolution6'
    }

    for layer_name in ['flow6', 'flow5', 'flow4', 'flow3', 'flow2', 'flow1']:
        net[layer_name].W.set_value(weights[flow_map[layer_name]][0])
        net[layer_name].b.set_value(weights[flow_map[layer_name]][1])
