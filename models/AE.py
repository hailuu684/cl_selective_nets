from collections import defaultdict

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys

import urllib.request
from urllib.error import HTTPError
import os
from loguru import logger
from pytorch_msssim import ssim, ms_ssim
sys.path.append("../")

from loss_functions import perceptual_loss
from models.resnet import ResNet18Enc, ResNet18Dec, ResNet18Encoder, CustomDecoder


def get_pretrained_ae_simple_cnn(latent_dim, empty_model):
    CHECKPOINT_PATH = '/home/luu/projects/cl_selective_nets/checkpoints'

    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial9/"

    # Files to download
    pretrained_files = ["cifar10_64.ckpt", "cifar10_128.ckpt", "cifar10_256.ckpt", "cifar10_384.ckpt"]

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print(
                    "Something went wrong. Please try to download the file from the GDrive folder, "
                    "or contact the author with the full output including the following error:\n",
                    e)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")

    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
    else:
        raise NotImplementedError(f"Pretrained model for latent {latent_dim} not found")

    # Adjust this because uses a different key
    load_ckpt = torch.load(pretrained_filename)
    model_state_dict = load_ckpt['state_dict']

    adjusted_state_dict = {key.replace('encoder', 'encoder_class'): value for key, value in model_state_dict.items()}
    adjusted_state_dict = {key.replace('decoder', 'decoder_class'): value for key, value in adjusted_state_dict.items()}

    # Load the adjusted state dictionary into the provided empty model
    empty_model.load_state_dict(adjusted_state_dict, strict=False)

    # Freeze encoder and decoder
    for param in empty_model.encoder_class.parameters():
        param.requires_grad = True
    for param in empty_model.decoder_class.parameters():
        param.requires_grad = True

    return empty_model


# -------------------------------------------------------------------- #
# -------------------------- For IMAGE SIZE = 32 --------------------- #
# -------------------------------------------------------------------- #

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Sigmoid()  # The input images is scaled between 0 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


# -------------------------------------------------------------------- #
# -------------------------- For IMAGE SIZE = 128 --------------------- #
# -------------------------------------------------------------------- #
class Encoder_128(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),
            nn.Linear(2 * c_hid * 4 * 4, latent_dim)  # Adjusted for the final spatial dimensions
        )

    def forward(self, x):
        return self.net(x)


class Decoder_128(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * c_hid * 4 * 4),  # Match to the encoder's last feature map size
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 32x32
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 64x64
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, padding=1, stride=2),  # 64x64 => 128x128

            nn.Sigmoid()  # Assuming the input images were normalized to [0,1]
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)  # Initial spatial dimensions to match encoder's last feature map size
        return self.net(x)
# -------------------------------------------------------------------- #
# -------------------------- For IMAGE SIZE = 150 - INTEL IMAGES DATASET ---------------- #
# -------------------------------------------------------------------- #
class IntelEncoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_input_channels, base_channel_size, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(base_channel_size, base_channel_size, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * base_channel_size, 2 * base_channel_size, kernel_size=3, padding=1, stride=2),
            act_fn(),
        )
        self.flatten = nn.Flatten()
        # Placeholder for the linear layer, will be initialized after the first forward pass
        self.fc = None
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.conv_layers(x)

        if self.fc is None:
            # Dynamically create the linear layer based on the output of conv_layers
            num_features_before_fc = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(num_features_before_fc, self.latent_dim)
            self.fc = self.fc.to(x.device)  # Ensure the linear layer is on the right device

        x = self.flatten(x)
        x = self.fc(x)

        return x


class IntelDecoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()

        # Adjust the initial size to fit the upscale path to 150x150
        # This projection size is based on the reverse calculation from the desired output size
        initial_size = 150 // (2 ** 3)  # Assuming 3 upsampling layers, adjust based on your encoder structure
        self.initial_channels = base_channel_size * 8  # Adjust based on the depth progression in your encoder
        self.fc = nn.Linear(latent_dim, initial_size * initial_size * self.initial_channels)

        # Upscaling path designed to reach 150x150 output
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.initial_channels, base_channel_size * 4, 4, stride=2, padding=1),  # -> ~75x75
            act_fn(),
            nn.ConvTranspose2d(base_channel_size * 4, base_channel_size * 2, 4, stride=2, padding=1),
            # -> ~150x150, adjusted for final size
            act_fn(),
            # Final adjustment layer to ensure the output is exactly 150x150
            # This layer's parameters may need fine-tuning to achieve the precise output size desired
            nn.Conv2d(base_channel_size * 2, num_input_channels, 3, padding=1),
            nn.Sigmoid()  # Assuming input images are normalized to [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        initial_size = 150 // (2 ** 3)  # Matching the calculation in __init__
        x = x.view(-1, self.initial_channels, initial_size, initial_size)
        x = self.deconv_layers(x)
        # Ensure the output is exactly 150x150
        # If the size is slightly off, consider applying F.interpolate here
        x = F.interpolate(x, size=(150, 150), mode='bilinear', align_corners=False)
        return x
# -------------------------- RESNET 18 ------------------------------- #


class AE_Resnet18(nn.Module):

    def __init__(self, z_dim, num_classes=10):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

        self.linear_layer = nn.Linear(2 * z_dim, 120)
        self.classifier = nn.Linear(120, num_classes)

    def forward(self, x):
        latent_variables = self.encoder(x)

        # Logit
        linear_layer = self.linear_layer(latent_variables)
        logits = self.classifier(linear_layer)

        reconstructed_image = self.decoder(latent_variables)
        return reconstructed_image, logits


# ------------------------------------------------------------------------------- #


# -------------------------- PRETRAINED RESNET 18 ------------------------------- #


class AE_Resnet18_pretrained(nn.Module):

    def __init__(self, z_dim, num_classes=10):
        super().__init__()
        self.encoder = ResNet18Encoder(latent_dim=z_dim)
        self.decoder = CustomDecoder(latent_dim=z_dim)

        self.linear_layer = nn.Linear(z_dim, 120)
        self.classifier = nn.Linear(120, num_classes)

    def forward(self, x):
        latent_variables = self.encoder(x)

        # Logit
        linear_layer = self.linear_layer(latent_variables)
        logits = self.classifier(linear_layer)

        reconstructed_image = self.decoder(latent_variables)

        return reconstructed_image, logits


# ------------------------------------------------------------------------------- #


# -------------------------- AE with simple CNN ------------------------------- #

class AEWithClassifier(nn.Module):
    def __init__(self, args, base_channel_size: int,
                 latent_dim: int,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32, num_classes=10):
        super().__init__()

        self.args = args

        if args.size == 150:
            self.encoder = IntelEncoder(num_input_channels, base_channel_size, latent_dim)
            self.decoder = IntelDecoder(num_input_channels, base_channel_size, latent_dim)

        elif args.size == 128:
            self.encoder = Encoder_128(num_input_channels, base_channel_size, latent_dim)
            self.decoder = Decoder_128(num_input_channels, base_channel_size, latent_dim)
        else:
            self.encoder= Encoder(num_input_channels, base_channel_size, latent_dim)
            self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim)

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

        self.linear_layer = nn.Linear(latent_dim, 120)
        self.classifier = nn.Linear(120, num_classes)

    def forward(self, x):
        # latent space
        z = self.encoder(x)

        # Logit
        linear_layer = self.linear_layer(z)
        logits = self.classifier(linear_layer)

        # Reconstruct image
        x_hat = self.decoder(z)

        if self.args.size == 128:
            x_hat = F.interpolate(x_hat, size=(128, 128), mode='bilinear')

        return x_hat, logits


# ------------------------------------------------------------------------------- #


class AEWithSelectiveSubnets(nn.Module):
    def __init__(self, args, base_channel_size: int, latent_dim: int, num_input_channels: int = 3,
                 width: int = 32, height: int = 32, num_classes=10, num_child_models=5,
                 perceptual_loss=None):
        super(AEWithSelectiveSubnets, self).__init__()

        self.args = args

        if args.AE_model == 'ae_simple_cnn':
            self.subnets = nn.ModuleList(
                [AEWithClassifier(args=args, base_channel_size=base_channel_size, latent_dim=latent_dim,
                                  num_input_channels=num_input_channels,
                                  width=width, height=height, num_classes=num_classes)
                 for _ in range(num_child_models)])

        elif args.AE_model == 'ae_simple_cnn_pretrained':
            empty_model = AEWithClassifier(args=args, base_channel_size=base_channel_size, latent_dim=latent_dim,
                                           num_input_channels=num_input_channels,
                                           width=width, height=height, num_classes=num_classes)

            pretrained_model = get_pretrained_ae_simple_cnn(latent_dim=latent_dim, empty_model=empty_model)

            self.subnets = nn.ModuleList([pretrained_model for _ in range(num_child_models)])

        elif args.AE_model == 'ae_resnet18':
            self.subnets = nn.ModuleList(
                [AE_Resnet18(z_dim=latent_dim, num_classes=num_classes)
                 for _ in range(num_child_models)])

        elif args.AE_model == 'ae_resnet18_pretrained':
            self.subnets = nn.ModuleList(
                [AE_Resnet18_pretrained(z_dim=latent_dim, num_classes=num_classes)
                 for _ in range(num_child_models)])

        elif args.AE_model == 'ae_resnet50_pretrained':

            from models.AEResnet50 import ResNet50Autoencoder

            logger.info(f"Using pretrained resnet50 with image size {args.size} has fixed latent dim = {args.latent_dim}")

            # Idea: train on task 0, load pretrained for only task 0. Task 1, 2 are not
            # Or in a concrete way that is the number of child model increases as the new task comes
            self.subnets = nn.ModuleList(
                [ResNet50Autoencoder(latent_dim=latent_dim, num_classes=num_classes,
                                     output_size=args.size,
                                     use_pretrained_decoder=False) for _ in range(num_child_models)])

        elif args.AE_model == 'ae_resnet50_encoder_decoder_pretrained':
            from models.AEResnet50 import ResNet50Autoencoder

            resnet50_decoder_pretrained = ResNet50Autoencoder(latent_dim=latent_dim, num_classes=num_classes,
                                                              output_size=args.size, use_pretrained_decoder=True)

            pretrained_autoencoder_path = '/home/luu/projects/cl_selective_nets/checkpoints/objects365-resnet50.pth'

            checkpoint = torch.load(pretrained_autoencoder_path, map_location=args.device)

            decoder_state_dict = {key[len("module.decoder."):]: value for key, value in checkpoint['state_dict'].items()
                                  if key.startswith("module.decoder.")}

            # Load the state dict into your model's decoder
            logger.info('Loading pretrained decoder')
            resnet50_decoder_pretrained.decoder.load_state_dict(decoder_state_dict, strict=False)

            self.subnets = nn.ModuleList([resnet50_decoder_pretrained for _ in range(num_child_models)])

        elif args.AE_model == 'ae_simple_cnn_unet' and args.size == 32:

            from models.AEResnet50 import build_unet
            logger.info("Using Unet-like architecture")
            self.subnets = nn.ModuleList([build_unet(num_classes=num_classes) for _ in range(num_child_models)])

        else:
            raise NotImplementedError("Check if the size and ae_model are aligned!")

        self.num_child_models = num_child_models

        if self.args.reconstruction_loss == 'perceptual_loss':
            self.perceptual_loss = perceptual_loss

    def forward(self, image, task_id=None):

        if task_id is not None:
            x_hat, logits = self.subnets[task_id](image)

            return x_hat, logits

        elif task_id is None:

            reconstruction_score = defaultdict(list)

            for net_id, net in enumerate(self.subnets):
                x_hat, logits = net(image)

                # reconstruction_loss = F.mse_loss(image, x_hat, reduction="none")
                # reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0]).item()

                if self.args.reconstruction_loss == 'mse':
                    rec_loss = F.mse_loss(image, x_hat, reduction="none")
                    rec_loss = rec_loss.sum(dim=[1, 2, 3]).mean(dim=[0])

                    # rec_loss = ssim(image.float(), x_hat.float(), data_range=1, size_average=True)
                    # rec_loss = 1 - rec_loss
                    # rec_loss = self.perceptual_loss(x_hat, image)

                elif self.args.reconstruction_loss == 'perceptual_loss':
                    rec_loss = self.perceptual_loss(x_hat, image)

                # elif self.args.reconstruction_loss == 'perceptual_loss':
                #     ae_perceptual_loss = perceptual_loss.AE_PerceptualLoss(model=net)
                #     rec_loss = ae_perceptual_loss(x_hat, image)
                else:
                    raise NotImplementedError("Reconstruction image loss function is not implemented!")

                reconstruction_score[net_id].append([rec_loss, logits])

            # Identify the net_id with the lowest reconstruction score.
            if self.args.reconstruction_loss in ['mse', 'perceptual_loss']:
                best_net_id = min(reconstruction_score, key=lambda x: reconstruction_score[x][0][0])
            elif self.args.reconstruction_loss == 'ssim':
                best_net_id = max(reconstruction_score, key=lambda x: reconstruction_score[x][0][0])
            else:
                raise NotImplementedError("Reconstruction image loss function is not implemented!")

            best_logits = reconstruction_score[best_net_id]

            return best_net_id, best_logits[0][1]

    def convert_image_range_torch(self, image):
        """
        Check if a PyTorch tensor image has pixels in the range of -1 to 1.
        If so, convert the image to have pixels in the range of 0 to 1.

        Args:
        - image (torch.Tensor): The input image with shape (1, 3, 32, 32).

        Returns:
        - torch.Tensor: The converted image if needed, or the original image.
        """
        # Check if the image has any pixels less than 0, indicating it's in the range -1 to 1
        if image.min() < 0:
            # Convert image from range -1 to 1 to 0 to 1
            image = (image + 1) / 2
        return image
