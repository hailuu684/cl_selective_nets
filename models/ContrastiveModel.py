from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        x = torch.flatten(x, 1)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ClassifierHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=10):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class ContrastiveModelWithClassifier(nn.Module):
    def __init__(self, input_channels=3, feature_dim=256, projection_dim=128, num_classes=10):
        super(ContrastiveModelWithClassifier, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, feature_dim=feature_dim)
        self.projection_head = ProjectionHead(input_dim=feature_dim, output_dim=projection_dim)
        self.classifier_head = ClassifierHead(input_dim=feature_dim, num_classes=num_classes)

    def forward(self, x, return_embedding=False, return_projection=False):
        features = self.encoder(x)
        classification_logits = self.classifier_head(features)

        if return_embedding:
            return classification_logits, features

        projections = self.projection_head(features)

        if return_projection:
            return classification_logits, projections

        return classification_logits


class ContrastiveResNet50(nn.Module):
    def __init__(self, pretrained=False):
        super(ContrastiveResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        self.model = torchvision.models.resnet50(pretrained=pretrained)

        # Modify the first convolution layer
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the max pooling layer
        self.model.maxpool = nn.Identity()

        # Get the number of input features to the final fully connected layer
        ch = self.model.fc.in_features

        # Replace the fully connected layer with a new sequence of layers
        self.model.fc = nn.Sequential(
            nn.Linear(ch, ch),
            nn.ReLU(),
            nn.Linear(ch, ch)
        )

    def forward(self, x):
        # Forward pass through the modified ResNet-50 model
        return self.model(x)


class ContrastiveWithSelectiveSubnets(nn.Module):
    def __init__(self, args, latent_dim: int, num_input_channels: int = 3, num_classes=10, num_child_models=5):
        super(ContrastiveWithSelectiveSubnets, self).__init__()

        self.args = args

        self.subnets = nn.ModuleList(
            [ContrastiveModelWithClassifier(input_channels=num_input_channels, feature_dim=latent_dim,
                                            projection_dim=latent_dim, num_classes=num_classes)
             for _ in range(num_child_models)])

        self.num_child_models = num_child_models

    def forward(self, image, task_id=None, return_embedding=False, return_projection=False):

        if task_id is not None:

            if return_embedding and not return_projection:
                classification_logits, features = self.subnets[task_id](image, return_embedding=return_embedding,
                                                                        return_projection=return_projection)

                return classification_logits, features

            if return_projection and not return_embedding:
                classification_logits, projections = self.subnets[task_id](image, return_embedding=return_embedding,
                                                                           return_projection=return_projection)

                return classification_logits, projections

            classification_logits = self.subnets[task_id](image, return_embedding=return_embedding,
                                                          return_projection=return_projection)

            return classification_logits

        elif task_id is None:

            # Task agnostic: evaluate which subnet has the highest confidence score
            logits_list = [net(image, return_embedding=False, return_projection=False) for net in self.subnets]

            # Calculate confidence scores, for example, max logit as proxy for confidence
            confidence_scores = [torch.max(F.softmax(logits, dim=1), dim=1)[0] for logits in logits_list]

            # Identify the index (i.e., subnet) with the highest average confidence score
            avg_confidence_scores = torch.stack(confidence_scores).mean(dim=1)
            best_net_idx = torch.argmax(avg_confidence_scores).item()

            # Return logits from the best subnet
            best_logits = logits_list[best_net_idx]

            return best_logits


def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent_loss(x, temperature):
    x = pair_cosine_similarity(x)
    x = torch.exp(x / temperature)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / temperature)))
    return -torch.log(x.mean())
