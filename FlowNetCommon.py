import cv2

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import Layer, MergeLayer

from lasagne.layers import Deconv2DLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
except:
    from lasagne.layers import Conv2DLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import ExpressionLayer

from lasagne.nonlinearities import LeakyRectify
from lasagne.nonlinearities import linear

class BilinearUpscaleLayer(Layer):
    """
    This layer is taken from https://github.com/TobyPDE/FRRN/blob/master/dltools/layers.py#L5

    This layer upscales the 4D input tensor along the trailing spatial dimensions using bilinear interpolation.
    You have to specify image dimensions in order to use this layer - even if you want to have a fully convolutional
    network.
    """
    def __init__(self, incoming, factor, **kwargs):
        """
        Initializes a new instance of the BilinearUpscaleLayer class.
        :param incoming: The incoming network stream
        :param factor: The factor by which to upscale the input
        """
        super(BilinearUpscaleLayer, self).__init__(incoming, **kwargs)
        self.factor = factor

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of the layer given the input shape.
        :param input_shape: The input shape
        :return: The output shape
        """
        return input_shape[0], input_shape[1], self.factor * input_shape[2], self.factor * input_shape[3]

    def get_output_for(self, input, **kwargs):
        """
        Constructs the Theano graph for this layer
        :param input: Symbolic input variable
        :return: Symbolic output variable
        """
        return T.nnet.abstract_conv.bilinear_upsampling(
            input, self.factor,
            batch_size=self.input_shape[0],
            num_input_channels=self.input_shape[1])


leaky_rectify = LeakyRectify(0.1)

def leaky_conv(input_layer, pad='same', **kwargs):
    return Conv2DLayer(input_layer, nonlinearity=leaky_rectify, pad=pad, **kwargs)

def leaky_deconv(input_layer, **kwargs):
    return Deconv2DLayer(
        input_layer, nonlinearity=leaky_rectify,
        filter_size=4, stride=2, crop=1, b=None, **kwargs)

def upsample(input_layer, **kwargs):
    return Deconv2DLayer(
        input_layer, num_filters=2, filter_size=4, stride=2,
        crop=1, b=None, nonlinearity=linear, **kwargs)

def flow(input_layer, filter_size=3, pad=1,**kwargs):
    return Conv2DLayer(
        input_layer, num_filters=2, filter_size=filter_size, stride=1,
        nonlinearity=linear, pad=pad, **kwargs)

def switch_channels(images):
    return images.transpose(0, 3, 1, 2)

def refine_flow(net, weights):
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

    # TODO: What does this magic number mean? We reduced an image size only 4
    # times, didn't we?
    # https://github.com/liruoteng/FlowNet/blob/master/models/flownet/model_simple/deploy.tpl.prototxt#L869
    net['eltwise4'] = ExpressionLayer(net['flow2'], lambda x: x * 20)

    # Should be upsampled before 'flow1' to 384x512
    net['resample4'] = BilinearUpscaleLayer(net['eltwise4'], 4)

    net['flow1'] = flow(net['resample4'], filter_size=1, pad=0)

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
        net[layer_name].W.set_value(weights[flow_map[layer_name]][0][:,:,::-1,::-1])
        net[layer_name].b.set_value(weights[flow_map[layer_name]][1])


def write_flow(file_name, flow):
    import struct
    print('Writing to %s' % file_name)
    print('Shape', flow.shape)
    with open(file_name, 'wb') as f:
        f.write('PIEH')
        f.write(struct.pack('@i', flow.shape[2]))
        f.write(struct.pack('@i', flow.shape[1]))

        for y in xrange(flow.shape[1]):
            for x in xrange(flow.shape[2]):
                f.write(struct.pack('@f', flow[0, y, x]))
                f.write(struct.pack('@f', flow[1, y, x]))


def run(net):
    input_vars = lasagne.layers.get_output([net['input_1'], net['input_2']])

    flow = theano.function(
        input_vars,
        lasagne.layers.get_output(
            [net['flow1'], net['flow2'], net['flow3'],
             net['flow4'], net['flow5'], net['flow6']], deterministic=True))

    # frame1_path = 'data/frame-000967.color.png'
    # frame2_path = 'data/frame-000977.color.png'

    for idx in xrange(9):
        frame1_path = 'data/000000%d-img0.ppm' % idx
        frame2_path = 'data/000000%d-img1.ppm' % idx

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

        flows = flow(frame1, frame2)

        for i in range(1, 7):
            write_flow('output/%07d_flow%d.flo' % (idx, i), flows[i - 1][0])
            np.save('output/%07d_flow%d.npy' % (idx, i), flows[i - 1])

        print(flows[1])