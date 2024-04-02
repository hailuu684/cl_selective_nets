import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
root = '/home/luu/DistilledDataset_ContinualLearning/data'
train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

# # Known classes (by index): Airplane (0), Automobile (1), Bird (2), Cat (3), Deer (4)
# known_classes = [0, 1, 2, 3, 4]


class CIFAR10ModifiedCE(Dataset):
    def __init__(self, dataset, known_classes):

        self.dataset = dataset
        self.known_classes = known_classes
        # Filter dataset to include only known classes
        self.indices = [i for i, (img, label) in enumerate(dataset) if label in self.known_classes]

    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        # Convert label to new class index (0 to len(known_classes)-1)
        label = self.known_classes.index(label)
        return img, label

    def __len__(self):
        return len(self.indices)


def get_known_class_data(batch_size, known_classes=None):

    # dataset use ce loss
    if known_classes is None:
        known_classes = [0, 1, 2, 3, 4]

    train_dataset_modified_ce = CIFAR10ModifiedCE(train_dataset, known_classes)
    test_dataset_modified_ce = CIFAR10ModifiedCE(test_dataset, known_classes)

    # dataloader use ce
    train_loader_ce = DataLoader(train_dataset_modified_ce, batch_size=batch_size, shuffle=True)
    test_loader_ce = DataLoader(test_dataset_modified_ce, batch_size=batch_size, shuffle=False)

    test_all_classes_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader_ce, test_loader_ce, test_all_classes_loader





