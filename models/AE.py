from collections import defaultdict

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim


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

        # self.net = nn.Sequential(
        #     nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
        #     nn.MaxPool2d(2, 2),
        #     act_fn(),
        #     nn.Conv2d(c_hid, c_hid * 2, kernel_size=3, padding=1, stride=2),
        #     nn.MaxPool2d(2, 2),
        #     act_fn(),
        #     nn.Conv2d(c_hid * 2, c_hid * 4, kernel_size=3, padding=1, stride=2),
        #     act_fn(),
        #     nn.Flatten(),
        #     nn.Linear(c_hid * 4, latent_dim)  # Adjust according to the actual output size
        # )

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
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class AEWithClassifier(nn.Module):
    def __init__(self, base_channel_size: int,
                 latent_dim: int,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32, num_classes=10):
        super().__init__()
        self.encoder_class = Encoder(num_input_channels, base_channel_size, latent_dim)
        self.decoder_class = Decoder(num_input_channels, base_channel_size, latent_dim)

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

        self.linear_layer = nn.Linear(latent_dim, 120)
        self.classifier = nn.Linear(120, num_classes)

    def forward(self, x):
        # latent space
        z = self.encoder_class(x)

        # Logit
        linear_layer = self.linear_layer(z)
        logits = self.classifier(linear_layer)

        # Reconstruct image
        x_hat = self.decoder_class(z)

        return x_hat, logits

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat, _ = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        loss = self._get_reconstruction_loss(batch)
        return loss

    def test_step(self, batch):
        loss = self._get_reconstruction_loss(batch)
        return loss


class AEWithSelectiveSubnets(nn.Module):
    def __init__(self, base_channel_size: int, latent_dim: int, num_input_channels: int = 3,
                 width: int = 32, height: int = 32, num_classes=10, num_child_models=5):
        super(AEWithSelectiveSubnets, self).__init__()

        self.subnets = nn.ModuleList(
            [AEWithClassifier(base_channel_size=base_channel_size, latent_dim=latent_dim,
                              num_input_channels=num_input_channels,
                              width=width, height=height, num_classes=num_classes)
             for _ in range(num_child_models)])

        self.num_child_models = num_child_models

    def forward(self, image, task_id=None):

        if task_id is not None:
            x_hat, logits = self.subnets[task_id](image)

            return x_hat, logits

        elif task_id is None:

            reconstruction_score = defaultdict(list)

            for net_id, net in enumerate(self.subnets):
                x_hat, logits = net(image)

                reconstruction_loss = F.mse_loss(image, x_hat, reduction="none")
                reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0]).item()

                reconstruction_score[net_id].append([reconstruction_loss, logits])

            # Identify the net_id with the lowest reconstruction score.
            try:
                best_net_id = min(reconstruction_score, key=lambda x: reconstruction_score[x][0][0])
            except RuntimeError:
                print(reconstruction_score)
                raise Exception("Please check the function to find the minimum reconstruction score!")

            best_logits = reconstruction_score[best_net_id]

            return best_net_id, best_logits[0][1]
