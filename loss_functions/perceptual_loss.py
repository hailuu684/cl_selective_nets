import torch.nn as nn
import torchvision
import torch


# Why using perceptual loss:
# https://sanjivgautamofficial.medium.com/perceptual-loss-well-it-sounds-interesting-after-neural-style-transfer-d09a48b6fb7d
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load a pretrained ResNet18 model

        resnet_pretrained_features = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
        # resnet_pretrained_features = torchvision.models.resnet18(
        #     weights=None).eval()

        # Use the features up to the second residual block for feature extraction
        self.feature_extractor = nn.Sequential(*list(resnet_pretrained_features.children())[:5])

        # Freeze parameters, we do not require gradients
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, reconstructed, original):
        # Ensure input is normalized according to ResNet18's expectations
        # Note: You might need to adjust normalization based on your dataset
        original_features = self.feature_extractor(original)
        reconstructed_features = self.feature_extractor(reconstructed)

        # Calculate the loss as the mean squared error between feature maps
        loss = torch.mean((original_features - reconstructed_features) ** 2)
        return loss


class AE_PerceptualLoss(nn.Module):
    def __init__(self, model):
        """
        Initializes the ModifiedPerceptualLoss with an encoder.

        :param model: The AEWithClassifier model.
        """
        super(AE_PerceptualLoss, self).__init__()
        self.encoder = model.encoder_class
        # Ensure the encoder does not update during the loss computation
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, reconstructed, original):
        """
        Calculates the perceptual loss using the latent space representations of the original
        and reconstructed images.

        :param reconstructed: The reconstructed images.
        :param original: The original images.
        :return: The mean squared error between the latent representations of the original
                 and reconstructed images.
        """
        # Use the encoder to get the latent space representations
        original_latent = self.encoder(original)
        reconstructed_latent = self.encoder(reconstructed)

        # Calculate the loss as the mean squared error between the latent representations
        loss = torch.mean((original_latent - reconstructed_latent) ** 2)
        return loss



