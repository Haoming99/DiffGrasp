import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)

class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=True):
        super(FCResBlock, self).__init__()
        layers = [nn.Linear(in_size, out_size)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(out_size, out_size))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(0.5))

        self.fc_block = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(x + self.fc_block(x))

class FCResNet(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(FCResNet, self).__init__()
        module_list = [FCBlock(in_channels, hidden_channels, batchnorm=batchnorm, activation=activation)]
        for i in range(num_layers):
            module_list.append(FCResBlock(hidden_channels, hidden_channels, batchnorm=batchnorm, activation=activation))
        if hidden_channels != out_channels:
            module_list.append(FCBlock(hidden_channels, out_channels, batchnorm=batchnorm, activation=activation))
        self.fc_layers = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_layers(x)

def fcresnet(cfg):
    return FCResNet(cfg.MODEL.BACKBONE.IN_CHANNELS, cfg.MODEL.BACKBONE.HIDDEN_CHANNELS,
                    cfg.MODEL.BACKBONE.OUT_CHANNELS, cfg.MODEL.BACKBONE.NUM_LAYERS)

class GroupLinear(nn.Module):
    r"""Feed forward multiple linears layers at once
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        groups: number of groups in current layer
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(B, H_{in}, N)` where :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(B, H_{out}, N)` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{in\_features}, \text{out\_features}, groups)`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features, groups})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = Group(20, 30, 10)
        >>> input = torch.randn(128, 20, 10)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30, 10])
    """
    __constants__ = ['in_features', 'out_features', 'groups']
    in_features: int
    out_features: int
    groups: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, groups: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GroupLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features, groups), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(1, out_features, groups, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = torch.einsum('bin,ijn->bjn', input, self.weight)
        if self.bias is not None:
            res += self.bias
        return res

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, groups={} bias={}'.format(
            self.in_features, self.out_features, self.groups, self.bias is not None
        )

class GroupFCBlock(nn.Module):
    """Wrapper around GroupLinear that includes batch normalization and activation functions."""
    def __init__(self, in_size, out_size, groups, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(GroupFCBlock, self).__init__()
        module_list = [GroupLinear(in_size, out_size, groups)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)

class GroupFCResBlock(nn.Module):
    """Residual block using fully-connected layers."""
    def __init__(self, in_size, out_size, groups, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(GroupFCResBlock, self).__init__()
        layers = [GroupLinear(in_size, out_size, groups)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(GroupLinear(out_size, out_size, groups))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(0.5))

        self.fc_block = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(x + self.fc_block(x))

class GroupDoubleFCLayer(nn.Module):
    """Double layers as a block"""
    def __init__(self, in_size, out_size, groups, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(GroupDoubleFCLayer, self).__init__()
        layers = [GroupLinear(in_size, out_size, groups)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(GroupLinear(out_size, out_size, groups))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(0.5))

        self.fc_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_block(x)