
import numpy as np
import theano
import theano.tensor as T

from correlation_layer import CorrelationOp

data1 = np.random.rand(1, 2, 32, 32).astype(np.float32)

data2 = np.random.rand(1, 2, 32, 32).astype(np.float32)

op = CorrelationOp(data1.shape)

t_data1 = T.ftensor4()
t_data2 = T.ftensor4()

t_outs = op(t_data1, t_data2)

t_c = t_outs[2].sum()

t_g_data = T.grad(t_c, t_data1)

# f = theano.function([t_data1, t_data2], t_outs)

# rbot0, rbot1, out = f(data1, data2)

# print('rbot0.shape', rbot0.shape)
# print('rbot1.shape', rbot1.shape)
# print('out.shape', out.shape)

f = theano.function([t_data1, t_data2], [t_g_data], on_unused_input='warn')

res = f(data1, data2)

# print(res[0])
# print(len(res))
print(res[0].shape)
print(res[0])