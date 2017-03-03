from __future__ import print_function

import theano
import theano.tensor as T
import lasagne

import numpy as np

from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Deconv2DLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import ExpressionLayer

from lasagne.layers import MergeLayer
from lasagne.nonlinearities import LeakyRectify
from lasagne.nonlinearities import linear
import cv2


class CorrelationLayer(MergeLayer):
    def __init__(self, first_layer, second_layer,
                 pad_size=20, kernel_size=1, stride1=1, stride2=2,
                 max_displacement=20, **kwargs):
        super(CorrelationLayer, self).__init__(
            [first_layer, second_layer], **kwargs)

        self.pad_size = 20
        self.kernel_size = 1
        self.stride1 = 1
        self.stride2 = 2
        self.max_displacement = 20

    def get_output_shape_for(self, input_shapes):
        # This fake op is just for inferring shape
        op = CorrelationOp(
            intput_shapes[0],
            pad_size=self.pad_size,
            kernel_size=self.kernel_size,
            stride1=self.stride1,
            stride2=self.stride2,
            max_displacement=self.max_displacement)

        return (intput_shapes[0][0], op.top_channels, op.top_height, op.top_width)

    def get_output_for(self, inputs, **kwargs):
        from correlation_layer import CorrelationOp

        op = CorrelationOp(
            intputs[0].shape,
            pad_size=self.pad_size,
            kernel_size=self.kernel_size,
            stride1=self.stride1,
            stride2=self.stride2,
            max_displacement=self.max_displacement)

        return op(*inputs)[2]


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

def build_model(weights):
    weights = np.load(weights)['arr_0'][()]

    net = dict()

    net['input_1'] = InputLayer([None, 3, 384, 512])
    net['input_2'] = InputLayer([None, 3, 384, 512])

    net['conv1'] = leaky_conv(
        net['input_1'], num_filters=64, filter_size=7, stride=2)
    net['conv1b'] = leaky_conv(
        net['input_2'], num_filters=64, filter_size=7, stride=2,
        W=net['conv1'].W, b=net['conv1'].b)
    
    net['conv2'] = leaky_conv(
        net['conv1'], num_filters=128, filter_size=5, stride=2)
    net['conv2b'] = leaky_conv(
        net['conv1b'], num_filters=128, filter_size=5, stride=2,
        W=net['conv2'].W, b=net['conv2'].b)
    
    net['conv3'] = leaky_conv(
        net['conv2'], num_filters=256, filter_size=5, stride=2)
    net['conv3b'] = leaky_conv(
        net['conv2b'], num_filters=256, filter_size=5, stride=2,
        W=net['conv3'].W, b=net['conv3'].b)

    net['corr'] = CorrelationLayer(net['conv3'], net['conv3b'])
    # Adding leaky relu on top
    net['corr'] = ExpressionLayer(net['corr'], lambda x: T.nnet.relu(x, 0.1))

    net['conv_redir'] = leaky_conv(net['conv3a'], num_filters=32, filter_size=1, stride=1, pad=0)

    net['concat'] = ConcatLayer([net['corr'], net['conv_redir']])

    net['conv3_1'] = leaky_conv(net['concat'], num_filters=256, filter_size=3, stride=1)
    
    net['conv4'] = leaky_conv(net['conv3_1'], num_filters=512, filter_size=3, stride=2)
    net['conv4_1'] = leaky_conv(net['conv4'], num_filters=512, filter_size=3, stride=1)
    
    net['conv5'] = leaky_conv(net['conv4_1'], num_filters=512, filter_size=3, stride=2)
    net['conv5_1'] = leaky_conv(net['conv5'], num_filters=512, filter_size=3, stride=1)

    net['conv6'] = leaky_conv(net['conv5_1'], num_filters=1024, filter_size=3, stride=2)
    net['conv6_1'] = leaky_conv(net['conv6'], num_filters=1024, filter_size=3, stride=1)

    for layer_name in ['conv1', 'conv2', 'conv3', 'conv_redir', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv5_1', 'conv6', 'conv6_1']:
        print(layer_name, net[layer_name].W.shape.eval(), weights[layer_name][0].shape)
        print(layer_name, net[layer_name].b.shape.eval(), weights[layer_name][1].shape)
        net[layer_name].W.set_value(weights[layer_name][0])
        net[layer_name].b.set_value(weights[layer_name][1])

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

    return net

def switch_channels(images):
    return images.transpose(0, 3, 1, 2)


if __name__ == '__main__':
    net = build_model('archive/flownetc.npz')

    input_vars = lasagne.layers.get_output([net['input_1'], net['input_2']])

    flow = theano.function(input_vars, lasagne.layers.get_output(net['flow2'], deterministic=True))

    frame1_path = 'data/frame-000967.color.png'
    frame2_path = 'data/frame-000977.color.png'

    frame1 = cv2.resize(cv2.imread(frame1_path, cv2.IMREAD_COLOR), (384, 512))
    frame2 = cv2.resize(cv2.imread(frame2_path, cv2.IMREAD_COLOR), (384, 512))

    frame1 = switch_channels(frame1.reshape(1, 384, 512, 3)).astype(np.float32)
    frame2 = switch_channels(frame2.reshape(1, 384, 512, 3)).astype(np.float32)

    print(frame1.shape, frame2.shape)
    print(flow(frame1, frame2).shape)
