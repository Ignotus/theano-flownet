import numpy as np
import theano
import theano.tensor as T

from theano import Apply
from theano.gof import COp
from theano.gradient import grad_undefined
from theano.sandbox.cuda import as_cuda_ndarray_variable, GpuOp

class CorrelationOp(GpuOp, COp):
  __props__ = ('top_width', 'top_height', 'top_channels', 'pad_size',
               'kernel_radius', 'kernel_size', 'stride1', 'stride2',
               'max_displacement', 'neighborhood_grid_radius',
               'neighborhood_grid_width')

  func_file = "./correlation_layer.cu"
  func_name = "APPLY_SPECIFIC(Forward_gpu)"

  def __init__(self, **kwargs):
    super(CorrelationOp, self).__init__(self.func_file,
                                        self.func_name)

    self.kwargs = kwargs

    # Default parameters taken from the FlowNetC model
    self.pad_size = kwargs.get('pad_size', 20)
    self.kernel_size = kwargs.get('kernel_size', 1)
    self.stride1 = kwargs.get('stride1', 1)
    self.stride2 = kwargs.get('stride2', 2)
    self.max_displacement = kwargs.get('max_displacement', 20)

    self.bottom_shape = kwargs.get('bottom_shape')

    self.kernel_radius = (self.kernel_size - 1) // 2
    border_size = self.max_displacement + self.kernel_radius

    paddedbottomheight = bottom_shape[2] + 2 * self.pad_size;
    paddedbottomwidth = bottom_shape[3] + 2 * self.pad_size;

    self.top_width = np.ceil(float(paddedbottomwidth - border_size * 2) / float(self.stride1))
    self.top_height = np.ceil((float)(paddedbottomheight - border_size * 2) / float(self.stride1))


    self.neighborhood_grid_radius = self.max_displacement / stride2
    self.neighborhood_grid_width = Self.neighborhood_grid_radius * 2 + 1

  def make_node(self, bottom0, bottom1):
    bottom0 = as_cuda_ndarray_variable(bottom0)
    bottom1 = as_cuda_ndarray_variable(bottom1)

    return Apply(self, [bottom0, bottom1], [bottom0.type(), bottom0.type(), bottom0.type()])

  def get_op_params(self):
    return [('TOP_WIDTH', str(self.top_width)),
            ('TOP_HEIGHT', str(self.top_height)),
            ('TOP_CHANNELS', str(self.top_channels)),
            ('PAD_SIZE', str(self.pad_size)),
            ('KERNEL_RADIUS', str(self.kernel_radius)),
            ('KERNEL_SIZE', str(self.kernel_size)),
            ('STRIDE1', str(self.stride1)),
            ('STRIDE2', str(self.stride2)),
            ('MAX_DISPLACEMENT', str(self.max_displacement)),
            ('NEIGHBORHOOD_GRID_RADIUS', str(self.neighborhood_grid_radius)),
            ('NEIGHBORHOOD_GRID_WIDTH', str(self.neighborhood_grid_width))]

  def infer_shape(self, node, in_shapes):
    bottom0_shape = T.shape(node.inputs[0])
    batch_size = bottom0_shape[0]
    bchannels = bottom0_shape[1]
    pb_height = bottom0_shape[2] + 2 * self.pad_size
    pb_width = bottom0_shape[3] + 2 * self.pad_size

    pb_shape = [batch_size, bchannels, pb_height, pb_width]
    out_shape = [batch_size, self.top_channels, self.top_height, self.top_width]
    return [pb_shape, pb_shape, out_shape]

  def grad(self, inp, grads):
    outs = self(*inp)
    grad_op = CorrelationGradOp(**self.kwargs)
    data_grads = grad_op(*(inp + outs + [grads[0]]))

    return data_grads

  def __eq__(self, other):
    return (type(self) == type(other) and
            self.top_width == other.top_width and
            self.top_height == other.top_height and
            self.top_channels == other.top_channels and
            self.pad_size == other.pad_size and
            self.kernel_radius == other.kernel_radius and
            self.kernel_size == other.kernel_size and
            self.stride1 == other.stride1 and
            self.stride2 == other.stride2 and
            self.max_displacement = other.max_displacement and
            self.neighborhood_grid_radius = other.neighborhood_grid_radius and
            self.neighborhood_grid_width = other.neighborhood_grid_width)

  def __hash__(self):
    return (hash(type(self)) ^ 
            hash(self.top_width) ^
            hash(self.top_height) ^
            hash(self.top_channels) ^
            hash(self.pad_size) ^
            hash(self.kernel_radius) ^
            hash(self.kernel_size) ^
            hash(self.stride1) ^
            hash(self.stride2) ^
            hash(self.max_displacement) ^
            hash(self.neighborhood_grid_radius) ^
            hash(self.neighborhood_grid_width)) 

  def c_code_cache_version(self):
    return (1,)


class CorrelationGradOp(GpuOp, COp):
  __props__ = ('pooled_h', 'pooled_w', 'spatial_scale')

  func_file = "./correlation_layer.cu"
  func_name = "APPLY_SPECIFIC(Backward_gpu)"

  def __init__(self, **kwargs):
    super(CorrelationGradOp, self).__init__(self.func_file,
                                            self.func_name)

    self.kwargs = kwargs

    # Default parameters taken from the FlowNetC model
    self.pad_size = kwargs.get('pad_size', 20)
    self.kernel_size = kwargs.get('kernel_size', 1)
    self.stride1 = kwargs.get('stride1', 1)
    self.stride2 = kwargs.get('stride2', 2)
    self.max_displacement = kwargs.get('max_displacement', 20)

    self.bottom_shape = kwargs.get('bottom_shape')

    self.kernel_radius = (self.kernel_size - 1) // 2
    border_size = self.max_displacement + self.kernel_radius

    paddedbottomheight = bottom_shape[2] + 2 * self.pad_size;
    paddedbottomwidth = bottom_shape[3] + 2 * self.pad_size;

    self.top_width = np.ceil(float(paddedbottomwidth - border_size * 2) / float(self.stride1))
    self.top_height = np.ceil((float)(paddedbottomheight - border_size * 2) / float(self.stride1))


    self.neighborhood_grid_radius = self.max_displacement / stride2
    self.neighborhood_grid_width = Self.neighborhood_grid_radius * 2 + 1

  def make_node(self, bottom0, bottom1, rbot0, rbot1, out_grad):
    bottom0 = as_cuda_ndarray_variable(bottom0)
    bottom1 = as_cuda_ndarray_variable(bottom1)
    rbot0 = as_cuda_ndarray_variable(rbot0)
    rbot1 = as_cuda_ndarray_variable(rbot1)
    out_grad = as_cuda_ndarray_variable(out_grad)

    return Apply(self, [bottom0, bottom1, rbot0, rbot1, out_grad],
                 [bottom0.type(), bottom0.type()])

  def get_op_params(self):
    return [('TOP_WIDTH', str(self.top_width)),
            ('TOP_HEIGHT', str(self.top_height)),
            ('TOP_CHANNELS', str(self.top_channels)),
            ('PAD_SIZE', str(self.pad_size)),
            ('KERNEL_RADIUS', str(self.kernel_radius)),
            ('KERNEL_SIZE', str(self.kernel_size)),
            ('STRIDE1', str(self.stride1)),
            ('STRIDE2', str(self.stride2)),
            ('MAX_DISPLACEMENT', str(self.max_displacement)),
            ('NEIGHBORHOOD_GRID_RADIUS', str(self.neighborhood_grid_radius)),
            ('NEIGHBORHOOD_GRID_WIDTH', str(self.neighborhood_grid_width))]


  def infer_shape(self, node, in_shapes):
    return [in_shapes[0], in_shapes[1]]

  def grad(self, inp, grads):
    return [grad_undefined(self, i, inp[i]) for i in xrange(4)]

  def __eq__(self, other):
    return (type(self) == type(other) and
            self.top_width == other.top_width and
            self.top_height == other.top_height and
            self.top_channels == other.top_channels and
            self.pad_size == other.pad_size and
            self.kernel_radius == other.kernel_radius and
            self.kernel_size == other.kernel_size and
            self.stride1 == other.stride1 and
            self.stride2 == other.stride2 and
            self.max_displacement = other.max_displacement and
            self.neighborhood_grid_radius = other.neighborhood_grid_radius and
            self.neighborhood_grid_width = other.neighborhood_grid_width)

  def __hash__(self):
    return (hash(type(self)) ^ 
            hash(self.top_width) ^
            hash(self.top_height) ^
            hash(self.top_channels) ^
            hash(self.pad_size) ^
            hash(self.kernel_radius) ^
            hash(self.kernel_size) ^
            hash(self.stride1) ^
            hash(self.stride2) ^
            hash(self.max_displacement) ^
            hash(self.neighborhood_grid_radius) ^
            hash(self.neighborhood_grid_width)) 

  def c_code_cache_version(self):
    return (1,)