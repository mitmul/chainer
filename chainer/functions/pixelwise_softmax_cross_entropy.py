from chainer import cuda
from chainer import cudnn
from chainer import function
from chainer.functions import softmax

if cudnn.available:
    from chainer.cudnn import libcudnn
    _algorithm = libcudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']
    _mode = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_CHANNEL']


class PixelwiseSoftmaxCrossEntropy(function.Function):

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def pixelwise_softmax_gpu(self, x):
        y = cuda.empty_like(x[0])
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(x[0], x[0].shape[2], x[0].shape[3])
            libcudnn.cudnnSoftmaxForward(
                handle, _algorithm, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(y))
        else:
            raise NotImplementedError()

        return y,

    def forward_gpu(self, inputs):
        x, t = inputs
        self.y, = self.pixelwise_softmax_gpu((x,))
        n, c, h, w = x.shape
        z = cuda.empty((n, 1, h, w))
        cuda.elementwise(
            '''
            float* z, const float* y, const int* label, int cdim, int rdim
            ''', '''
            int n = i / rdim;
            int t = label[i];
            int y_ind = n * cdim * rdim + t * rdim + i % rdim;
            z[i] = -log(y[y_ind]);
            ''', 'channelwise_fwd')(z, self.y, t, c, h * w)
        z = cuda.gpuarray.sum(z) / n

        return z,

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = cuda.empty_like(self.y)
        n, c, h, w = gx.shape
        coeff = gloss / n
        cuda.elementwise(
            '''
            float* gx, const float* y, const int* label, const float* coeff,
            int cdim, int rdim
            ''', '''
            int n = i / rdim / cdim;
            int t = label[n * rdim + i % rdim];
            int c = i / rdim % cdim;
            gx[i] = *coeff * (y[i] - (t == c));
            ''', 'channelwise_bwd')(gx, self.y, t, coeff, c, h * w)

        return gx, None


def pixelwise_softmax_cross_entropy(x, t, use_cudnn=True):
    return PixelwiseSoftmaxCrossEntropy(use_cudnn)(x, t)
