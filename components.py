import numpy as np


class FullyConnectLayer:
    def __init__(self, in_features, out_features, has_bias=True):
        # 初始化权重和偏置
        self.weight = np.random.normal(loc=0, scale=0.01, size=(out_features, in_features))
        self.bias = np.zeros(out_features) if has_bias else None
        self.has_bias = has_bias

        self.inputs = None
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, inputs):
        # TODO: 根据公式编写全连接层的前向传播过程
        self.inputs = inputs
        outputs = inputs @ self.weight.T + self.bias
        return outputs

    def backward(self, in_grad):
        # TODO: 根据公式编写全连接层的反向传播过程
        self.grad_weight = self.inputs.T @ in_grad
        self.grad_bias = np.sum(in_grad, axis=0)
        out_grad = in_grad @ self.weight
        return out_grad

    def update_params(self, lr):
        # TODO: 根据公式编写全连接层的参数更新过程
        self.weight = self.weight - lr * self.grad_weight.T
        self.bias = self.bias - lr * self.grad_bias

    def load_params(self, weight, bias):
        # 加载权重和偏置
        assert self.weight.shape == weight.shape
        self.weight = weight
        if self.has_bias:
            assert self.bias.shape == bias.shape
            self.bias = bias


class ReluLayer:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grad):
        out_grad = in_grad * (self.inputs >= 0).astype(float)
        return out_grad


class SigmoidLayer:
    # TODO: 请自行完善 Sigmoid 的前向和反向传播过程
    def __init__(self):
        self.inputs = None
        self.sigma_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.sigma_inputs = 1 / (1 + np.exp(-inputs))
        return self.sigma_inputs

    def backward(self, in_grad):
        out_grad = in_grad * (self.sigma_inputs * (1 - self.sigma_inputs))
        return out_grad


class CrossEntropy:
    def __init__(self, dim=1):
        self.softmax_out = None
        self.label_onehot = None
        self.batch_size = None
        self.dim = dim

    def _softmax(self, inputs, dim=1):
        exp_inputs = np.exp(inputs)
        result = exp_inputs / np.sum(exp_inputs, axis=dim, keepdims=True)
        return result
    
    def _to_onehot(self, labels, num_classes):
        onehot = np.zeros((labels.size, num_classes))
        onehot[np.arange(labels.size), labels] = 1
        return onehot

    def forward(self, inputs, labels):
        self.softmax_out = self._softmax(inputs, 1)
        self.label_onehot = self._to_onehot(labels, inputs.shape[1])
        self.batch_size = inputs.shape[0]
        log_softmax_out = np.log(self.softmax_out)
        outputs = -np.sum(self.label_onehot * log_softmax_out) / self.batch_size
        return outputs

    def backward(self, in_grad):
        out_grad = (self.softmax_out - self.label_onehot) / self.batch_size
        return out_grad
