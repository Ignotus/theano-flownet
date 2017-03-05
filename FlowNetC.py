from __future__ import print_function

import theano
import theano.tensor as T
import lasagne

import numpy as np

from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import MergeLayer
from lasagne.nonlinearities import LeakyRectify

import cv2

from correlation_layer import CorrelationOp

from FlowNetCommon import *


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
        self.bottom_shape = lasagne.layers.get_output_shape(first_layer)

    def get_output_shape_for(self, input_shapes):
        # This fake op is just for inferring shape
        op = CorrelationOp(
            self.bottom_shape,
            pad_size=self.pad_size,
            kernel_size=self.kernel_size,
            stride1=self.stride1,
            stride2=self.stride2,
            max_displacement=self.max_displacement)

        return (input_shapes[0][0], op.top_channels, op.top_height, op.top_width)

    def get_output_for(self, inputs, **kwargs):
        op = CorrelationOp(
            self.bottom_shape,
            pad_size=self.pad_size,
            kernel_size=self.kernel_size,
            stride1=self.stride1,
            stride2=self.stride2,
            max_displacement=self.max_displacement)

        return op(*inputs)[2]

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
    net['corr'] = ExpressionLayer(net['corr'], leaky_rectify)

    net['conv_redir'] = leaky_conv(net['conv3'], num_filters=32, filter_size=1, stride=1, pad=0)

    net['concat'] = ConcatLayer([net['corr'], net['conv_redir']])

    net['conv3_1'] = leaky_conv(net['concat'], num_filters=256, filter_size=3, stride=1)

    net['conv4'] = leaky_conv(net['conv3_1'], num_filters=512, filter_size=3, stride=2)
    net['conv4_1'] = leaky_conv(net['conv4'], num_filters=512, filter_size=3, stride=1)

    net['conv5'] = leaky_conv(net['conv4_1'], num_filters=512, filter_size=3, stride=2)
    net['conv5_1'] = leaky_conv(net['conv5'], num_filters=512, filter_size=3, stride=1)

    net['conv6'] = leaky_conv(net['conv5_1'], num_filters=1024, filter_size=3, stride=2)
    net['conv6_1'] = leaky_conv(net['conv6'], num_filters=1024, filter_size=3, stride=1)

    for layer_id in ['1', '2', '3', '_redir', '3_1', '4', '4_1', '5', '5_1', '6', '6_1']:
        layer_name = 'conv' + layer_id
        print(layer_name, net[layer_name].W.shape.eval(), weights[layer_name][0].shape)
        print(layer_name, net[layer_name].b.shape.eval(), weights[layer_name][1].shape)
        net[layer_name].W.set_value(weights[layer_name][0])
        net[layer_name].b.set_value(weights[layer_name][1])

    refine_flow(net, weights)

    return net

if __name__ == '__main__':
    net = build_model('archive/flownetc.npz')

    input_vars = lasagne.layers.get_output([net['input_1'], net['input_2']])

    flow = theano.function(
        input_vars,
        lasagne.layers.get_output([net['flow2'], net['conv3'], net['conv3b'], net['corr']], deterministic=True))

    frame1_path = 'data/frame-000967.color.png'
    frame2_path = 'data/frame-000977.color.png'

    frame1 = cv2.resize(cv2.imread(frame1_path, cv2.IMREAD_COLOR), (512, 384))
    frame2 = cv2.resize(cv2.imread(frame2_path, cv2.IMREAD_COLOR), (512, 384))

    frame1 = switch_channels(frame1.reshape(1, 384, 512, 3)).astype(np.float32)
    frame2 = switch_channels(frame2.reshape(1, 384, 512, 3)).astype(np.float32)

    # Scale pixels to [0, 1]
    frame1 *= 0.00392156862745
    frame2 *= 0.00392156862745

    mean = np.array([0.378156, 0.394731, 0.400841], dtype=np.float32).reshape(1, 3, 1, 1)

    frame1 -= mean
    frame2 -= mean

    print(frame1.shape, frame2.shape)
    flow_res, conv3, conv3b, corr = flow(frame1, frame2)

    np.save('conv3.npy', conv3)
    np.save('conv3b.npy', conv3b)

    np.save('flow.npy', flow_res)
