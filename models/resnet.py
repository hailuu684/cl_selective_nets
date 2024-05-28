from collections import defaultdict

import loguru
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as torch_models
from torchvision import models
import numpy as np
from models.knn import knn_classify


# ------------------------------  RESNET 18 FOR UPPER BOUND--------------------------------- #


def resnet50(pretrained=True, num_classes=10):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)

    # Freeze parameters, we do not require gradients
    # loguru.logger.info("Baseline resnet50 for upper bound is freeze")
    # for param in model.parameters():
    #     param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Adjusting for 10 classes in STL-10

    return model


class Resnet50WithSelectiveSubnets(nn.Module):
    def __init__(self, args, model_dinov2, distilled_feature_dict, task_id_mapping,
                 num_classes=10, num_child_models=5):
        super().__init__()

        self.args = args
        self.model_dinov2 = model_dinov2
        self.distilled_feature_dict = distilled_feature_dict
        self.task_id_mapping = task_id_mapping
        self.subnets = nn.ModuleList([resnet50(num_classes=num_classes) for _ in range(num_child_models)])

        self.features = None

        # Our experiments have the following features in each key: 1, 10, 20
        # So if each key has 1 feature, then using k = 1
        self.k = 5 if len(distilled_feature_dict.get('distilled', [])) > 5 else 1

    def forward(self, image, compare_method=None, task_id=None):

        if task_id is not None and compare_method is None:
            logits = self.subnets[task_id](image)

            return logits

        elif task_id is None:

            if compare_method == 'knn':

                predicted_logit = knn_classify(image, embeddings_dict=self.distilled_feature_dict,
                                               model=self.model_dinov2, K=self.k)

                predicted_task_id = self.find_label_index(predicted_logit, self.task_id_mapping)
                # loguru.logger.info("Using Knn for predicting task id")
            else:
                predicted_task_id = self.compare_features(image, compare_method=compare_method)

            logits = self.subnets[predicted_task_id](image)

            return logits

    def compare_features(self, test_image, compare_method='cosine_similarity'):
        """
        Compare features of a test image with each feature set in the feature dictionary using cosine similarity.
        """
        # Get features of the test image
        test_features = self.model_dinov2(test_image)

        # Dictionary to store similarity results
        similarity_results = {}

        # Compare test image features with each feature in the feature_dict
        for key, images in self.distilled_feature_dict.items():

            # if self.features is None:
            # self.features = [self.model_dinov2(image) for image in images]
            self.features = images

            if compare_method == 'cosine_similarity':
                similarities = [self.cosine_similarity(test_features, feature).item() for feature in self.features]
            elif compare_method == 'euclidean_distance':
                similarities = [self.euclidean_distance(test_features, feature).item() for feature in self.features]
            elif compare_method == 'correlation_based_distance':
                similarities = [self.correlation_based_distance(test_features, feature).item() for feature in
                                self.features]
            else:
                raise NotImplemented("Compare method is not implemented")

            similarity_results[key] = np.mean(similarities)

        if compare_method == 'cosine_similarity':
            predicted_label = max(similarity_results, key=similarity_results.get)
        else:
            predicted_label = min(similarity_results, key=similarity_results.get)

        predicted_task_id = self.find_label_index(predicted_label, self.task_id_mapping)
        return predicted_task_id

    def find_label_index(self, label, task_id_mapping):
        """
        Returns the index of the pair in task_id_mapping that contains the label.

        Parameters:
        - label: The target label to search for.
        - task_id_mapping: A list of sets, where each set contains pairs of labels.

        Returns:
        - The index of the pair that contains the label, or -1 if the label is not found.
        """
        for i, label_set in enumerate(task_id_mapping):
            if label in label_set:
                return i
        return -1  # Return -1 if the label is not found in any pair

    def cosine_similarity(self, tensor1, tensor2):
        """
        Compute the cosine similarity between two tensors.
        """
        return F.cosine_similarity(tensor1, tensor2, dim=1)

    def euclidean_distance(self, tensor1, tensor2):
        errors = np.sqrt(np.sum((tensor1.detach().cpu().numpy() - tensor2.detach().cpu().numpy()) ** 2))

        return errors

    def correlation_based_distance(self, tensor1, tensor2):
        errors = 1 - np.corrcoef(tensor1.detach().cpu().numpy(), tensor2.detach().cpu().numpy())[0, 1]
        # errors = 1 - np.corrcoef(tensor1.detach().cpu().numpy(), tensor2)[0, 1]

        return errors


# --------------------------------------------------------------------------------------------- #
# ------------------------------ ENCODER - DECODER OF RESNET 18 --------------------------------- #
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        z = self.linear(x)
        return z


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim * 2, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

        # Add an AdaptiveAvgPool2d layer to resize the output to 32x32
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)

        x = self.adaptive_pool(x)
        return x


# ------------------------------------------------------------------------------------------------ #

# --------------------------- PRETRAINED ENCODER-DECODER OF RESNET 18 ----------------------------

class ResNet18Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18Encoder, self).__init__()
        resnet18 = torch_models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])

        # Adapt first convolutional layer - Optional based on your requirement
        self.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.features[1] = nn.BatchNorm2d(64)
        self.features[2] = nn.ReLU(inplace=True)

        # Additional layers for producing the latent vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class CustomDecoder(nn.Module):
    def __init__(self, latent_dim, num_input_channels=3):
        super(CustomDecoder, self).__init__()
        # self.fc = nn.Linear(latent_dim, 512)
        #
        # # Reverse of ResNet blocks - simplified version
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.ConvTranspose2d(64, num_input_channels, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()  # Assuming the input images were normalized to [-1, 1]
        # )

        c_hid = 32
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            nn.GELU()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        # x = self.fc(x)
        # x = x.view(x.size(0), 512, 1, 1)  # Reshape to (batch_size, 512, 1, 1)
        # x = self.upsample(x)
        # return x

        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


# ------------------------------------------------------------------------------------------------ #

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes, bias=False)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
