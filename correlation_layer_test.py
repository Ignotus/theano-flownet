
import numpy as np
import theano
import theano.tensor as T

from correlation_layer import CorrelationOp

data1 = np.random.rand(1, 2, 32, 32).astype(np.float32)

data2 = np.random.rand(1, 2, 32, 32).astype(np.float32)

op = CorrelationOp()

t_data1 = T.ftensor4()
t_data2 = T.ftensor4()

t_outs = op(t_data1, t_data2)

t_c = t_outs[0].sum()

t_g_data = T.grad(t_c, t_data)[0]

f = theano.function([t_data1, t_data2], t_outs + [t_g_data])

print(f(data1, data2))