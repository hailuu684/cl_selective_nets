import loguru
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

# Define transformations
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                           (0.24703233, 0.24348505, 0.26158768))
                                      ])

bigan_transform = transforms.Compose([
    transforms.ToTensor()
])


# # Known classes (by index): Airplane (0), Automobile (1), Bird (2), Cat (3), Deer (4)
# known_classes = [0, 1, 2, 3, 4]

class CustomDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Retrieve data and target for a given index

        if self.transform:
            x = self.transform(self.images[index])
        else:
            x = self.images[index]

        y = self.targets[index]

        return x, y


def get_cifar10(args):
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    # Load CIFAR-10 dataset
    root = '/home/luu/DistilledDataset_ContinualLearning/data'
    loguru.logger.info("Cifar 10 is loading with Norm of ImageNet")
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths=[0.7, 0.3])

    return train_dataset, val_dataset, test_dataset


def get_distilled_cifar10(args):

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size), antialias=True)
    ])

    root = '/home/luu/DistilledDataset_ContinualLearning/author_distilled_dataset' \
           '/DatasetCondensation/result/cifar10/res_DM_CIFAR10_ConvNet_100ipc.pt'

    loguru.logger.info("Loading distilled cifar10")
    distilled_dataset = torch.load(root, map_location='cuda')

    images = distilled_dataset['data'][0][0]
    labels = distilled_dataset['data'][0][1]

    distilled_dataset = CustomDataset(images, labels, transform=transform)

    return distilled_dataset


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
    # Load CIFAR-10 dataset
    root = '/home/luu/DistilledDataset_ContinualLearning/data'
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

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


class ContrastivePairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img, label, task_id = self.dataset[index]

        img = Image.fromarray(img).convert('RGB')

        imgs = [self.transform(img), self.transform(img)]

        return torch.stack(imgs), label, task_id

    def __len__(self):
        return len(self.dataset)


def get_contrastive_pairs_loader(train_dataset, batch_size=32):
    contrastive_dataset = ContrastivePairDataset(train_dataset)
    loader = DataLoader(contrastive_dataset, batch_size=batch_size, shuffle=True)
    return loader
