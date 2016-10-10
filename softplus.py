import numpy
import chainer
from chainer import cuda
from chainer import utils
from chainer.functions.activation import softplus

class Softplus(softplus.Softplus):

	def backward_cpu(self, inputs, grads):
		x, = inputs
		g, = grads
		sigmoid = numpy.tanh(-self.beta * x * 0.5) * 0.5 + 0.5
		gx = (1 - sigmoid) * g
		return utils.force_array(gx, x.dtype),

	def backward_gpu(self, inputs, grads):
		x, = inputs
		g, = grads
		gx = cuda.elementwise(
			'T x, T g, T beta', 'T gx',
			'gx = (1 - (tanh(-beta * x * 0.5) * 0.5 + 0.5)) * g',
			'softplus_bwd'
		)(x, g, self.beta)
		return gx,

def softplus(x, beta=1.0):
	return Softplus(beta=beta)(x)
