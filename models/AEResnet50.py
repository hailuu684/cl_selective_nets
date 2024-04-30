from collections import defaultdict

import loguru
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math


# -------------------------------------------------------------------------------- #

# -------------------- BASIC DECODER RESNET50 ARCHITECTURE ------------------------------ #

# -------------------------------------------------------------------------------- #

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0,
                          bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        elif in_channels != down_channels:
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0,
                          bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x


class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False)
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1,
                          bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        else:
            self.upsample = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels,
                                             upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels,
                                             upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)


class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=in_channels, upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class ResNetDecoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024,
                                                layers=configs[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,
                                                layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512, hidden_channels=128, down_channels=256,
                                                layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256, hidden_channels=64, down_channels=64,
                                                layers=configs[3])


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64, layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64, output_channels=64, layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1,
                               bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x


# -------------------------------------------------------------------------------- #

# -------------------- BASIC AE RESNET50 ARCHITECTURE ------------------------------ #

# -------------------------------------------------------------------------------- #


class ResNet50Encoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ResNet50Encoder, self).__init__()
        # Load a pretrained ResNet-50
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the fully connected layer to use as a feature extractor
        # feature dim = 2048, -2
        # feature dim = 1024, -3
        # feature dim = 512, -4
        # feature dim = 256, -5

        if latent_dim == 2048:
            self.features = nn.Sequential(*list(resnet50.children())[:-2])
        elif latent_dim == 1024:
            self.features = nn.Sequential(*list(resnet50.children())[:-3])
        elif latent_dim == 512:
            self.features = nn.Sequential(*list(resnet50.children())[:-4])
        elif latent_dim == 256:
            self.features = nn.Sequential(*list(resnet50.children())[:-5])
        else:
            raise Exception("Valid latent dim = [2048, 1024, 512, 256]")

        # num_ftrs = resnet50.fc.in_features

        # # Freeze parameters, we do not require gradients
        # loguru.logger.info("Resnet50 encoder is freezing")
        # for param in self.features.parameters():
        #     param.requires_grad = False

        self.adaptive_pooling = resnet50.avgpool
        self.fc = nn.Linear(latent_dim, num_classes)  # Adjusting for 10 classes in STL-10

    def forward(self, x):
        x = self.features(x)
        x_pooling = self.adaptive_pooling(x)  # .flatten()
        x_pooling = x_pooling.view(x_pooling.shape[0], -1)
        x_output = self.fc(x_pooling)
        return x, x_output  # laten variables, logits


# ---------------------------------------------------
# ---------------- IMAGE SIZE = 224 -----------------
# ---------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, scale_factor=None):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        self.scale_factor = scale_factor  # Allow dynamic scaling
        if upsample:
            # Use scale_factor if provided, otherwise default to scale factor of 2
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor if scale_factor else 2, mode='bilinear',
                                              align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet50Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet50Decoder, self).__init__()
        # Start from the smallest feature map size as it corresponds to the largest latent_dim
        feature_sizes = {2048: (7, 7), 1024: (14, 14), 512: (28, 28), 256: (56, 56)}
        if latent_dim not in feature_sizes:
            raise ValueError("Invalid latent dimension. Must be one of [2048, 1024, 512, 256].")

        size = feature_sizes[latent_dim]
        channels = [256, 128, 64, 32, 16, 8]  # Adjusted to include more layers for more gradual upsampling
        initial_channels = [latent_dim] + channels[:-1]

        # Create decoder blocks
        layers = []
        for in_channels, out_channels in zip(initial_channels, channels):
            if out_channels == 8:
                # Final upsample to reach 224x224, calculate necessary scale factor
                target_size = 224
                current_size = size[0] * (2 ** (len(layers)))  # Size after applying current number of upsamples
                scale_factor = target_size / current_size
                layers.append(DecoderBlock(in_channels, out_channels, upsample=True, scale_factor=scale_factor))
            else:
                layers.append(DecoderBlock(in_channels, out_channels, upsample=True))

        # Final layer to produce 3-channel output image
        layers.append(nn.Conv2d(channels[-1], 3, 3, padding=1))
        layers.append(nn.Sigmoid())  # Assuming input images are normalized between -1 and 1

        self.decode_layers = nn.Sequential(*layers)
        self.initial_size = size

    def forward(self, x):
        # Reshape the latent space output from the encoder to the starting size
        x = x.view(x.shape[0], -1, self.initial_size[0], self.initial_size[1])
        x = self.decode_layers(x)
        return x


# ---------------------------------------------------
# ---------------- IMAGE SIZE = 112 -----------------
# ---------------------------------------------------
class DecoderBlock_112(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=None):
        super(DecoderBlock_112, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=False) if output_size else None

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet50Decoder_112(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet50Decoder_112, self).__init__()
        # Define the sizes to which we need to upscale gradually
        upsample_sizes = [(14, 14), (28, 28), (56, 56), (112, 112)]
        # Calculate the number of DecoderBlocks needed based on the latent_dim
        num_blocks = int(math.log2(latent_dim / 64))

        # Determine initial channels
        channels = [latent_dim // (2 ** i) for i in range(num_blocks)] + [64, 64]

        # Create the upsampling layers using the DecoderBlock
        layers = []
        for i in range(len(channels) - 1):
            output_size = upsample_sizes[min(i, len(upsample_sizes) - 1)]
            layers.append(DecoderBlock_112(channels[i], channels[i + 1], output_size=output_size))

        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))  # Conv to 3 channels
        layers.append(nn.Sigmoid())  # Assuming images are normalized between -1 and 1

        self.decode_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decode_layers(x)
        return x


# ---------------------------------------------------
# ---------------- IMAGE SIZE = 128 -----------------
# ---------------------------------------------------
class ResNet50Decoder_128(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet50Decoder_128, self).__init__()
        channels = [latent_dim] + [1024, 512, 256, 128, 64]
        self.layers = nn.ModuleList()

        # Iterate through channels list and create DecoderBlocks
        for i in range(len(channels) - 1):
            self.layers.append(DecoderBlock(channels[i], channels[i + 1]))

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        x = self.gate(x)
        # Ensure the final output is exactly 128x128
        if x.size(2) != 128 or x.size(3) != 128:
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return x


# ----------------------------------------------------------- #
# ---------------- COMPLETE AE ARCHITECTURE ----------------- #
# ----------------------------------------------------------- #

class ResNet50Autoencoder(nn.Module):
    def __init__(self, latent_dim, num_classes, output_size=224, use_pretrained_decoder=False):
        super(ResNet50Autoencoder, self).__init__()
        self.encoder = ResNet50Encoder(latent_dim, num_classes=num_classes)

        if output_size == 224:
            self.decoder = ResNet50Decoder(latent_dim)
        elif output_size == 112:
            self.decoder = ResNet50Decoder_112(latent_dim=latent_dim)
        elif output_size == 128:
            self.decoder = ResNet50Decoder_128(latent_dim=latent_dim)
        elif latent_dim == 2048 and output_size == 128 and use_pretrained_decoder:
            self.decoder = ResNetDecoder(configs=[3, 4, 6, 3], bottleneck=True)
        else:
            raise NotImplemented("Check the valid choices")

    def forward(self, x):
        x, x_output = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)

        return x, x_output  # reconstructed image, logits


# class ResNet50Autoencoder(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet50Autoencoder, self).__init__()
#         self.encoder = ResNet50Encoder(num_classes=num_classes)
#         self.decoder = ResNet50Decoder()
#
#     def forward(self, x):
#         x, x_output = self.encoder(x)
#
#         x = self.decoder(x)
#
#         return x, x_output  # reconstructed image, logits


# ------------------------------------------------------------------------------- #

# ------------------------RESNET50 UNET - LIKE ARCHITECTURE ---------------------- #

# --------------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, num_classes=10):  # Assuming 10 classes for the classification task
        super().__init__()
        resnet = models.resnet.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.input_block = nn.Sequential(*list(resnet.children())[:3])
        self.input_pool = list(resnet.children())[3]

        down_blocks = []
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)

        up_blocks = [
            UpBlockForUNetWithResNet50(2048, 1024),
            UpBlockForUNetWithResNet50(1024, 512),
            UpBlockForUNetWithResNet50(512, 256),
            UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128, up_conv_in_channels=256,
                                       up_conv_out_channels=128),
            UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64, up_conv_in_channels=128,
                                       up_conv_out_channels=64)
        ]
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, 3, kernel_size=1, stride=1)
        self.gate = nn.Sigmoid()

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i != (
                    UNetWithResnet50Encoder.DEPTH - 1):  # Store pre-pool features for skip connections, except the last
                pre_pools[f"layer_{i}"] = x

        x_bridge = self.bridge(x)  # Processed by the bridge

        x = x_bridge
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        output_feature_map = x
        reconstructed_img = self.gate(self.out(x))

        # Process the deepest feature map for classification
        class_features = self.global_avg_pool(x_bridge)  # Using features from the bridge for classification
        class_features = class_features.view(class_features.size(0), -1)
        class_logits = self.fc(class_features)

        del pre_pools

        if with_output_feature_map:
            return reconstructed_img, output_feature_map, class_logits
        else:
            return reconstructed_img, class_logits


# -------------------------------------------------------------------------------- #

# -------------------- BASIC UNET-LIKE ARCHITECTURE ------------------------------ #

# -------------------- FOR IMAGE SIZE OF 32 ------------------------------ #

# -------------------------------------------------------------------------------- #


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Output layer for segmentation or image tasks """
        self.outputs = nn.Conv2d(64, 3, kernel_size=1, padding=0)

        """ Classification Branch """
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Classifier """
        pooled_features = self.global_avg_pool(b)
        class_logits = self.classifier(pooled_features.view(pooled_features.size(0), -1))

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Output processing for tasks like segmentation """
        outputs = self.outputs(d4)

        return outputs, class_logits
