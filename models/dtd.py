import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from fph import FPH
from swins import SwinTransformerV2,trunc_normal_,DropPath
import torch.nn.functional as F


from functools import partial
try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None



def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class SegmentationModel(nn.Module):
    """Base class for all segmentation models."""

    # if model supports shape not divisible by 2 ^ n
    # set to False
    requires_divisible_input_shape = True

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    def check_input_shape(self, x):
        """Check if the input shape is divisible by the output stride.
        If not, raise a RuntimeError.
        """
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not torch.jit.is_tracing() or self.requires_divisible_input_shape:
            self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        ipt = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = ipt + self.drop_path(x)
        return x


class AddCoords(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(
            torch.arange(x_dim, dtype=input_tensor.dtype),
            torch.arange(y_dim, dtype=input_tensor.dtype),
        )
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim)
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class VPH(nn.Module):
    def __init__(self, dims=[96, 192], drop_path_rate=0.4, layer_scale_init_value=1e-6):
        super().__init__()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum([3, 3]))]
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(6, dims[0], kernel_size=4, stride=4),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                ),
                nn.Sequential(
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
                ),
            ]
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ConvBlock(
                            dim=dims[0],
                            drop_path=dp_rates[j],
                            layer_scale_init_value=layer_scale_init_value,
                        )
                        for j in range(3)
                    ]
                ),
                nn.Sequential(
                    *[
                        ConvBlock(
                            dim=dims[1],
                            drop_path=dp_rates[3 + j],
                            layer_scale_init_value=layer_scale_init_value,
                        )
                        for j in range(3)
                    ]
                ),
            ]
        )
        self.apply(self._init_weights)

    def initnorm(self):
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(self.dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        outs = []
        x = self.stages[0](self.downsample_layers[0](x))
        outs = [self.norm0(x)]
        x = self.stages[1](self.downsample_layers[1](x))
        outs.append(self.norm1(x))
        return outs


class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        cin,
        cadd,
        cout,
    ):
        super().__init__()
        self.cin = cin + cadd
        self.cout = cout
        self.conv1 = Conv2dReLU(
            self.cin, self.cout, kernel_size=3, padding=1, use_batchnorm=True
        )
        self.conv2 = Conv2dReLU(
            self.cout, self.cout, kernel_size=3, padding=1, use_batchnorm=True
        )

    def forward(self, x1, x2=None):
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:, : self.cin])
        x1 = self.conv2(x1)
        return x1


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, ks, stride=1, norm=True, res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_c,
                    out_c,
                    kernel_size=ks,
                    padding=ks // 2,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            )
        else:
            self.conv = nn.Conv2d(
                in_c, out_c, kernel_size=ks, padding=ks // 2, stride=stride, bias=False
            )
        self.res = res

    def forward(self, x):
        if self.res:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FUSE1(nn.Module):
    def __init__(self, in_channels_list=(96, 192, 384, 768)):
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 1)
        self.c32 = ConvBNReLU(in_channels_list[3], in_channels_list[2], 1)
        self.c33 = ConvBNReLU(in_channels_list[2], in_channels_list[2], 3)

        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1, x2, x3 = x
        h, w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3), size=(h, w)) + self.c31(x2))
        h, w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2), size=(h, w)) + self.c21(x1))
        h, w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1), size=(h, w)) + self.c11(x))
        return x, x1, x2, x3


class FUSE2(nn.Module):
    def __init__(self, in_channels_list=(96, 192, 384)):
        super(FUSE2, self).__init__()

        self.c21 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 1)
        self.c22 = ConvBNReLU(in_channels_list[2], in_channels_list[1], 1)
        self.c23 = ConvBNReLU(in_channels_list[1], in_channels_list[1], 3)

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1, x2 = x
        h, w = x1.shape[-2:]
        x1 = self.c23(
            F.interpolate(
                self.c22(x2), size=(h, w), mode="bilinear", align_corners=True
            )
            + self.c21(x1)
        )
        h, w = x.shape[-2:]
        x = self.c13(
            F.interpolate(
                self.c12(x1), size=(h, w), mode="bilinear", align_corners=True
            )
            + self.c11(x)
        )
        return x, x1, x2


class FUSE3(nn.Module):
    def __init__(self, in_channels_list=(96, 192)):
        super(FUSE3, self).__init__()

        self.c11 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 1)
        self.c12 = ConvBNReLU(in_channels_list[1], in_channels_list[0], 1)
        self.c13 = ConvBNReLU(in_channels_list[0], in_channels_list[0], 3)

    def forward(self, x):
        x, x1 = x
        h, w = x.shape[-2:]
        x = self.c13(
            F.interpolate(
                self.c12(x1), size=(h, w), mode="bilinear", align_corners=True
            )
            + self.c11(x)
        )
        return x, x1


class MID(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        encoder_channels = encoder_channels[1:][::-1]
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [96]
        self.out_channels = decoder_channels
        self.fuse1 = FUSE1()
        self.fuse2 = FUSE2()
        self.fuse3 = FUSE3()
        decoder_convs = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, skip_ch, out_ch
                )
        decoder_convs[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1]
        )
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, *features):
        decoder_features = {}
        features = self.fuse1(features)[::-1]
        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](
            features[0], features[1]
        )
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](
            features[1], features[2]
        )
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](
            features[2], features[3]
        )
        (
            decoder_features["x_2_2"],
            decoder_features["x_1_1"],
            decoder_features["x_0_0"],
        ) = self.fuse2(
            (
                decoder_features["x_2_2"],
                decoder_features["x_1_1"],
                decoder_features["x_0_0"],
            )
        )
        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](
            decoder_features["x_0_0"],
            torch.cat((decoder_features["x_1_1"], features[2]), 1),
        )
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](
            decoder_features["x_1_1"],
            torch.cat((decoder_features["x_2_2"], features[3]), 1),
        )
        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3(
            (decoder_features["x_1_2"], decoder_features["x_0_1"])
        )
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](
            decoder_features["x_0_1"],
            torch.cat(
                (decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]), 1
            ),
        )
        return self.decoder_convs["x_0_3"](
            torch.cat(
                (
                    decoder_features["x_0_2"],
                    decoder_features["x_1_2"],
                    decoder_features["x_2_2"],
                ),
                1,
            )
        )


class DTD(SegmentationModel):
    def __init__(
        self, encoder_name="resnet18", decoder_channels=(384, 192, 96, 64), classes=1
    ):
        super().__init__()
        self.vph = VPH(
            dims=[96, 192, 384, 768], drop_path_rate=0.2, layer_scale_init_value=1e-6
        )
        self.swin = SwinTransformerV2(
            img_size=128,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=192,
            depths=[
                2,
                18,
                2,
            ],
            num_heads=[6, 12, 24],
            window_size=8,
            pretrained_window_sizes=[8, 8, 6],
        ).layers
        self.fph = FPH()
        self.decoder = MID(
            encoder_channels=(96, 192, 384, 768), decoder_channels=decoder_channels
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes, upsampling=2.0
        )
        self.addcoords = AddCoords()
        self.FU = nn.Sequential(
            SCSEModule(448),
            nn.Conv2d(448, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.classification_head = None
        self.initialize()

    def forward(self, x, dct, qt):
        features = self.vph(self.addcoords(x))
        features[1] = self.FU(torch.cat((features[1], self.fph(dct, qt)), 1))
        rst = self.swin[0](features[1].flatten(2).transpose(1, 2).contiguous())
        N, L, C = rst.shape
        H = W = int(L ** (1 / 2))
        features.append(
            self.vph.norm2(rst.transpose(1, 2).contiguous().view(N, C, H, W))
        )
        features.append(
            self.vph.norm3(
                self.swin[2](self.swin[1](rst))
                .transpose(1, 2)
                .contiguous()
                .view(N, C * 2, H // 2, W // 2)
            )
        )
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)


class seg_dtd(nn.Module):
    def __init__(self, model_name="resnet18", n_class=1):
        super().__init__()
        self.model = DTD(encoder_name=model_name, classes=n_class)

    @autocast()
    def forward(self, x, dct, qt):
        x = self.model(x, dct, qt)
        return x
