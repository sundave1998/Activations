import torch
import torch.nn as nn


class ExpActivation(nn.Module):
    """
    Applies elementwise activation based on exponential function
    Inspired by squared relu,
    but with bounded range and gradient for better stability
    """

    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, input):
        # print("using exponential activation")
        return torch.exp(input)


class SquaredReLU(nn.Module):
    """
    Applies elementwise activation based on squared ReLU function
    Inspired by squared relu,
    but with bounded range and gradient for better stability
    """

    def __init__(self):
        super(SquaredReLU, self).__init__()

    def forward(self, input):
        # print("using exponential activation")
        return torch.square(nn.functional.relu(input))


class ExpActivationReg(nn.Module):
    """
    Applies elementwise activation based on exponential function
    Inspired by squared relu,
    but with bounded range and gradient for better stability
    """

    def __init__(self):
        super(ExpActivationReg, self).__init__()

    def forward(self, input):
        # print("using exponential activation")
        return torch.exp(input) / torch.e


class ReExpActivationReg(nn.Module):
    """
    Applies elementwise activation based on exponential function
    Inspired by squared relu,
    but with bounded range and gradient for better stability
    """

    def __init__(self):
        super(ReExpActivationReg, self).__init__()

    def forward(self, input):
        # print("using exponential activation")
        return nn.functional.relu(torch.exp(input) / torch.e - 1)


class ReEUActivation(nn.Module):
    """
    Applies elementwise activation based on rectified exponential unit function
    f(x) = exp(x) if x > 0 else 0
    """

    def __init__(self):
        super(ReEUActivation, self).__init__()

    def forward(self, input):
        # return torch.exp(nn.functional.relu(input)) - 1.0
        return nn.functional.relu(torch.exp((input)) - 1.0)


class SoftmaxActivation(nn.Module):
    """
    Applies the Softmax activation function elementwise
    """

    def __init__(self):
        super(SoftmaxActivation, self).__init__()

    def forward(self, input):
        return nn.functional.softmax(input, dim=-1)


class LinearActivation(nn.Module):
    """
    Applies the Softmax activation function elementwise
    """

    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, input):
        return input


class SparseActivation(nn.Module):
    """
    Applies the Softmax activation function elementwise
    topk
    = -1: no sparsity
    = 0: half sparsity
    > 0: number as sparsity
    """

    def __init__(self, topk=0):
        super(SparseActivation, self).__init__()
        self.topk = topk

    def forward(self, input):
        if self.topk == 0:
            topk = int(input.shape[-1] / 2)
        elif self.topk == -1:
            topk = input.shape[-1]
        else:
            topk = self.topk
        res = torch.zeros_like(input)
        with torch.no_grad():
            indices = torch.topk(input, topk).indices
        res = res.scatter(-1, indices, 1)
        return torch.mul(input, res)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_drop=0.2,
        act_fn: str = "relu",
        use_layernorm: bool = True,
        use_dropout: bool = True,
        use_sparse: bool = False,
        sparse_topk: int = 0,
        hidden_dims=[512, 256],
    ):
        super(MLP, self).__init__()
        activation = None
        if use_sparse or act_fn == "sparse":
            activation = SparseActivation(topk=sparse_topk)
        elif act_fn == "selu":
            activation = nn.SELU()
        elif act_fn == "relu":
            activation = nn.ReLU()
        elif act_fn == "exp":
            activation = ExpActivation()
        elif act_fn == "softmax":
            activation = SoftmaxActivation()
        elif act_fn == "reu":
            activation = ReEUActivation()
        elif act_fn == "elu":
            activation = nn.ELU()
        elif act_fn == "linear" or act_fn == "none" or act_fn is None:
            activation = LinearActivation()
        elif act_fn == "expreg":
            activation = ExpActivationReg()
        elif act_fn == "square":
            activation = SquaredReLU()
        elif act_fn == "reexpreg":
            activation = ReExpActivationReg()            
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")
        # activation = nn.SELU() if act_fn == "selu" else nn.ReLU()
        # dropout = nn.AlphaDropout(p=p_drop) if act_fn == "selu"
        # else nn.Dropout(p=p_drop)
        # use standard dropout for all activations
        dropout = nn.Dropout(p=p_drop) if use_dropout else None
        # sparse = SparseActivation(topk=sparse_topk) if use_sparse else None

        layers = [nn.Flatten()]
        previous_dim = in_features
        for hidden_features in hidden_dims:
            layers.append(nn.Linear(in_features=previous_dim, out_features=hidden_features))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_features, elementwise_affine=False))
            # if use_sparse:
            #     layers.append(sparse)
            layers.append(activation)
            if use_dropout:
                layers.append(dropout)
            previous_dim = hidden_features

        # layers.append(nn.Linear(in_features=in_features, out_features=512))
        # if use_layernorm:
        #     layers.append(nn.LayerNorm(512, elementwise_affine=False))
        # if use_sparse:
        #     layers.append(sparse)
        # layers.append(activation)
        # if use_dropout:
        #     layers.append(dropout)

        # layers.append(nn.Linear(in_features=512, out_features=256))
        # if use_layernorm:
        #     layers.append(nn.LayerNorm(256, elementwise_affine=False))
        # if use_sparse:
        #     layers.append(sparse)
        # layers.append(activation)
        # if use_dropout:
        #     layers.append(dropout)

        layers.append(nn.Linear(in_features=previous_dim, out_features=out_features))
        # if use_layernorm:
        #     layers.append(nn.LayerNorm(out_features, elementwise_affine=False))

        self.net = nn.Sequential(*layers)

        if act_fn == "selu":
            for param in self.net.parameters():
                # biases zero
                if len(param.shape) == 1:
                    nn.init.constant_(param, 0)
                # others using lecun-normal initialization
                else:
                    nn.init.kaiming_normal_(
                        param, mode="fan_in", nonlinearity="linear"
                    )

    def forward(self, x):
        return self.net(x)
