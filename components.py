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
        outputs = None
        return outputs

    def backward(self, in_grad):
        # TODO: 根据公式编写全连接层的反向传播过程
        out_grad = None
        return out_grad

    def update_params(self, lr):
        # TODO: 根据公式编写全连接层的参数更新过程
        ...

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
        # TODO: 根据公式编写激活函数ReLU的前向传播过程
        outputs = None
        return outputs

    def backward(self, in_grad):
        # TODO: 根据公式编写激活函数ReLU的反向传播过程
        out_grad = None
        return out_grad


class SigmoidLayer:
    # TODO: 请自行完善 Sigmoid 的前向和反向传播过程
    ...


class CrossEntropy:
    def __init__(self, dim=1):
        self.softmax_out = None
        self.label_onehot = None
        self.batch_size = None
        self.dim = dim

    def _softmax(self, inputs, dim=1):
        # TODO: 根据公式编写 Softmax 函数的前向传播过程
        result = None
        return result

    def forward(self, inputs, labels):
        # TODO: 根据公式编写交叉熵损失函数的前向传播过程
        outputs = None
        return outputs

    def backward(self, in_grad):
        # TODO: 根据公式编写交叉熵损失函数的反向传播过程
        out_grad = None
        return out_grad
