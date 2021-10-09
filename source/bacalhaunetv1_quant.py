from typing import Callable, Union, List

import torch
from torch import nn
from torch import Tensor

from brevitas import nn as qnn
from brevitas import quant

__all__ = ["bacalhaunetv1_default", "BacalhauNetLayerConfig", "BacalhauNetConfig", "BacalhauNetV1"]


class ConvNormReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, padding: int = None,
                 groups: int = 1, norm_layer: Union[Callable[..., nn.Module], None] = nn.BatchNorm1d, w_bits: int = 8,
                 a_bits: int = 8):
        """
        Convolutional Block that consists of a 1D convolution, a normalization layer and a ReLU activation function.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel: 1D convolution kernel length. Defaults to 3.
        :param stride: 1D convolution stride length. Defaults to 1.
        :param padding: 1D convolution padding length. Defaults to floor(kernel/2).
        :param groups: 1D convolution groups. Defaults to 1. Defaults to 1.
        :param norm_layer: Defines the normalization layer used. Defaults to torch.nn.BatchNorm1d.
        :param w_bits: Defines the number of bits used to represent weights. Defaults to 8.
        :param a_bits: Defines the number of bits used to represent activations. Defaults to 8.
        """
        super().__init__()
        if not padding:
            padding = kernel // 2

        self.conv = qnn.QuantConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel,
                                    stride=stride, padding=padding, groups=groups, bias=False,
                                    weight_bit_width=w_bits)
        self.norm = norm_layer(num_features=out_channels)
        self.relu = qnn.QuantReLU(bit_width=a_bits)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, padding: int = None,
                 w_bits: int = 8, a_bits: int = 8):
        """
        Defines a Depthwise Separable Convolution Block. If the in_channels and out_channels are equal and stride is 1
        then a residual connection is applied.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel: 1D convolution kernel length. Defaults to 3.
        :param stride: 1D convolution stride length. Defaults to 1.
        :param padding: 1D convolution padding length. Defaults to floor(kernel/2).
        :param w_bits: Defines the number of bits used to represent weights. Defaults to 8.
        :param a_bits: Defines the number of bits used to represent activations. Defaults to 8.
        """
        super().__init__()
        self.out_channels = out_channels
        self.padding = padding if padding else kernel // 2
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        # In some implementations I have see padding is done on PW, I think on DW is better since he save computations.
        self.dw = ConvNormReLU(in_channels=in_channels, out_channels=in_channels, kernel=kernel, stride=stride,
                               padding=padding, groups=in_channels,
                               w_bits=w_bits, a_bits=a_bits)
        self.pw = ConvNormReLU(in_channels=in_channels, out_channels=out_channels, kernel=1, stride=1,
                               w_bits=w_bits, a_bits=a_bits)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dw(x)
        y = self.pw(y)
        if self.use_res_connect:
            y += x
        return y


"""
# Doesnt get exported on ONNX since kernel size is dynamic!
class QuantGlobalMaxPool1d(nn.Module):
    def __init__(self, a_bits: int = 8):
        super().__init__()
        self.globalpool = partial(qnn.QuantMaxPool1d, stride=1, padding=0)  # , bit_width=a_bits)

    def forward(self, x: Tensor):
        return self.globalpool(kernel_size=(x.size()[-1],))(x)
"""


class BacalhauNetLayerConfig:
    _default_w_bits = 8

    def __init__(
            self,
            kernel: int,
            stride: int,
            out_channels: int,
            w_bits: int = 8,
            a_bits: int = 8
    ):
        """
        Used to configure convolutional layers of BacalhauNetV1 model in order to ease the user interactivity.
        :param kernel: The kernel length of the layer.
        :param stride: The stride length of the layer.
        :param out_channels: The number of output channels of the layer.
        :param w_bits: Defines the number of bits used to represent weights. Defaults to 8.
        :param a_bits: Defines the number of bits used to represent activations. Defaults to 8.
        """
        assert kernel > 0, "Kernel length must be greater than 0."
        assert stride > 0, "Stride length must be greater than 0."
        assert out_channels > 0, "Number of output channels must be greater than 0."
        assert w_bits > 0, "Weights bit width must be greater than 0."
        assert a_bits > 0, "Activations bit width must be greater than 0."
        self.kernel = kernel
        self.stride = stride
        self.out_channels = out_channels
        self.w_bits = w_bits
        self.a_bits = a_bits


class BacalhauNetConfig:
    _layers: List[BacalhauNetLayerConfig] = []

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value: List[BacalhauNetLayerConfig]):
        if isinstance(value, BacalhauNetLayerConfig):
            value = [BacalhauNetLayerConfig]
        assert isinstance(value, list), "layers should be a list of BacalhauNetLayerConfig."
        for layer_config in value:
            assert isinstance(layer_config, BacalhauNetLayerConfig), \
                "layers should be a list of BacalhauNetLayerConfig."
        self._layers = value

    def append_layer(self, kernel: int, stride: int, out_channels: int, w_bits: int = 8, a_bits: int = 8):
        """
        Appends a BacalhauNet layer.
        :param kernel: The kernel length of the layer.
        :param stride: The stride length of the layer.
        :param out_channels: The number of output channels of the layer.
        :param w_bits: Defines the number of bits used to represent weights. Defaults to 8.
        :param a_bits: Defines the number of bits used to represent activations. Defaults to 8.
        """
        self._layers.append(
            BacalhauNetLayerConfig(
                kernel=kernel,
                stride=stride,
                out_channels=out_channels,
                w_bits=w_bits,
                a_bits=a_bits
            )
        )

    def __init__(
            self,
            in_samples: int,
            in_channels: int,
            num_classes: int,
            hardtanh_bit_width: int = 8,
            layers: Union[List[BacalhauNetLayerConfig], BacalhauNetLayerConfig] = None,
            pool_bit_width: int = 8,
            dropout_prob: float = 0,
            fc_bit_width: int = 8
    ):
        """
        Creates a configuration object that allows the instatiation of a BacalhauNetV1.
        :param in_samples: Number of input samples. This is the last dimensions of the input tensor.
        :param in_channels: Number of input channels. This is the second last dimensions of the input tensor.
        :param num_classes: Number of classes to predict.
        :param hardtanh_bit_width: Bit width of the first layer (hard tanh). Defaults to 8.
        :param layers: List of BacalhauNetLayerConfig or BacalhauNetLayerConfig containing information about
        BacalhauNetV1 convolutional layers. Defaults to BacalhauNetLayerConfig(kernel=41, stride=2,
        out_channels=num_classes).
        :param pool_bit_width: Bit width of the final pooling layer (max pool). Defaults to 8.
        :param dropout_prob: Dropout layer probability. Defaults to 0.
        :param fc_bit_width: Bit width of the last layer (fully connected). Defaults to 8.
        """
        if not layers:
            layers = BacalhauNetLayerConfig(kernel=41, stride=2, out_channels=num_classes)
        assert in_samples > 0, "Input samples must be greater than 0."
        assert in_channels > 0, "Input channels must be greater than 0."
        assert num_classes > 0, "Number of classes must be greater than 0."
        assert hardtanh_bit_width > 0, "HardTanh bit width must be greater than 0."
        assert pool_bit_width > 0, "MaxPool bit width must be greater than 0."
        assert dropout_prob >= 0, "Dropout probability must be equal or greater than 0."
        assert fc_bit_width > 0, "Fully connected bit width must be greater than 0."
        self.in_samples = in_samples
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hardtanh_bit_width = hardtanh_bit_width
        self.layers = layers
        self.pool_bit_width = pool_bit_width
        self.dropout_prob = dropout_prob
        self.fc_bit_width = fc_bit_width


class BacalhauNetV1(nn.Module):
    def __init__(self, config: BacalhauNetConfig):
        """
        BacalhauNetV1 was developed under the "ITU-ML5G-PS-007: Lightning-Fast Modulation Classification with
        Hardware-Efficient Neural Networks" challenge. The objective of this network is to achieve at least 56% of
        classification accuracy while reducing the inference cost.
        :param config: Configuration of the network. Should be instance of BacalhauNetConfig.
        """
        super().__init__()
        # Sets the input samples to a range between -2 and 3.
        self.hardtan = qnn.QuantHardTanh(min_val=-2, max_val=3, bit_width=config.hardtanh_bit_width)

        # Creates the first (and required) convolutional layer.
        conv_layers = [
            DepthwiseSeparableConv(in_channels=config.in_channels, out_channels=config.layers[0].out_channels,
                                   kernel=config.layers[0].kernel, stride=config.layers[0].stride,
                                   w_bits=config.layers[0].w_bits, a_bits=config.layers[0].a_bits)]
        out_samples = (config.in_samples + 2*conv_layers[0].padding - config.layers[0].kernel) // config.layers[0].stride + 1

        # If there are more values in kernels, strides and out_channels iterables then add more convolutional layers.
        for i in range(1, len(config.layers)):
            conv_layers.append(DepthwiseSeparableConv(in_channels=conv_layers[-1].out_channels,
                                                      out_channels=config.layers[i].out_channels,
                                                      kernel=config.layers[i].kernel,
                                                      stride=config.layers[i].stride,
                                                      w_bits=config.layers[i].w_bits,
                                                      a_bits=config.layers[i].a_bits))
            out_samples = (out_samples + 2 * conv_layers[-1].padding - config.layers[i].kernel) // config.layers[i].stride + 1

        # Groups the convolutional layers defined above sequentially.
        self.conv = nn.Sequential(*conv_layers)

        # Implements a global pool. # FIXME MaxPool1d didn't allow bit_width definition!

        self.pool = qnn.QuantMaxPool1d(kernel_size=out_samples, stride=1, padding=0)  #, bit_width=config.pool_bit_width)

        # Flattens the tensor.
        self.flatten = nn.Flatten()

        # Implements a dropout layer.
        self.dropout = qnn.QuantDropout(p=config.dropout_prob) if config.dropout_prob != 0 else None

        # A last fully-connected layer classifies the features.
        self.fc = qnn.QuantLinear(in_features=conv_layers[-1].out_channels, out_features=config.num_classes, bias=True,
                                  weight_bit_width=config.fc_bit_width, bias_quant=quant.IntBias,
                                  input_quant=quant.Int8ActPerTensorFloat)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hardtan(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def inference_cost(self, test_input: Tensor = torch.randn(1, 2, 1024), export_path: str = "inference_cost"):
        """
        Computes and prints a formatted string with the normalized inference cost.
        :param test_input: Test input used. Don't worry about the values just the size.
        :param export_path: The full path where the generated files will be stored.
        """
        from brevitas.export.onnx.generic.manager import BrevitasONNXManager
        from finn.util.inference_cost import inference_cost
        import json

        export_onnx_path = export_path + "_export.onnx"
        final_onnx_path = export_path + "_final.onnx"
        cost_dict_path = export_path + "_cost.json"

        BrevitasONNXManager.export(self.cpu(), input_t=test_input, export_path=export_onnx_path)
        inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path, preprocess=True,
                       discount_sparsity=True)

        with open(cost_dict_path, 'r') as f:
            inference_cost_dict = json.load(f)

        bops = int(inference_cost_dict["total_bops"])
        w_bits = int(inference_cost_dict["total_mem_w_bits"])

        bops_baseline = 807699904
        w_bits_baseline = 1244936

        score = 0.5 * (bops / bops_baseline) + 0.5 * (w_bits / w_bits_baseline)
        print(f"Normalized inference cost score: {score}\n"
              f"Operatons: {(bops / bops_baseline)}\n"
              f"Memory: {w_bits / w_bits_baseline}")


def bacalhaunetv1_default(in_samples: int, in_channels: int, num_classes: int,
                          dropout_prob: float = 0) -> BacalhauNetV1:
    """
    Instatiates and returns a default BacalhauNetV1 model.
    :param in_samples: Number of input samples. This is the last dimensions of the input tensor.
    :param in_channels: Number of input channels. This is the second last dimensions of the input tensor.
    :param num_classes: Number of classes to predict.
    :param dropout_prob: The dropout probability. Defaults to 0.
    :return: BacalhauNetV1 model.
    """
    config = BacalhauNetConfig(
        in_samples=in_samples,
        in_channels=in_channels,
        num_classes=num_classes,
        hardtanh_bit_width=8,
        layers=[
            BacalhauNetLayerConfig(
                kernel=27,
                stride=1,
                out_channels=12,
                w_bits=8,
                a_bits=8
            ),
            BacalhauNetLayerConfig(9, 1, 12, 8, 8),
            BacalhauNetLayerConfig(21, 2, 24, 8, 8),
            BacalhauNetLayerConfig(15, 2, 36, 8, 8)
        ],
        pool_bit_width=8,  # FIXME MaxPool1d didn't allow bit_width definition!
        dropout_prob=dropout_prob,
        fc_bit_width=8
    )
    return BacalhauNetV1(config)


def conv_dim_test(c, l, k, s):
    x = torch.randn(1, c, l)
    padding = k // 2
    conv = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=(k,), stride=(s,), padding=padding)
    y = conv(x)

    out_samples = (l + 2*padding - k)//s + 1

    print(y.size()[-1])
    print(out_samples)
    return y.size()[-1] == out_samples
