import torch
from avalanche.benchmarks import nc_benchmark
from torch.utils.data import DataLoader, Dataset
import os
import torchvision
from sklearn.utils import shuffle
from torchvision.io import read_image
from torchvision import datasets, transforms


class IntelImageDataset(Dataset):
    def __init__(self, directory, transform):

        image_paths = []
        targets = []

        self.transforms = transform

        for dir in os.listdir(directory):
            label = -1
            if dir == "buildings":
                label = 0
            if dir == "forest":
                label = 1
            if dir == "glacier":
                label = 2
            if dir == "mountain":
                label = 3
            if dir == "sea":
                label = 4
            if dir == "street":
                label = 5

            for image_dir in os.listdir(directory + "/" + dir):
                image_paths.append(directory + "/" + dir + "/" + image_dir)
                targets.append(label)

        self.image_paths, self.targets = shuffle(image_paths, targets)

    def __getitem__(self, item):

        image = read_image(self.image_paths[item], torchvision.io.ImageReadMode.RGB)
        label = self.targets[item]

        if self.transforms:
            image = image.float() / 255.0
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.targets)


def get_intel_images(args, train_path, val_path):

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size), antialias=False),
        # transforms.ToTensor(),
        # transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))

    ])

    dataset = IntelImageDataset(directory=train_path, transform=transform)
    test_dataset = IntelImageDataset(directory=val_path, transform=transform)
    # test_dataset = IntelImageDataset(directory=test_path, transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[0.7, 0.3])

    # # Getting the targets for the training subset
    # train_targets = [dataset.targets[i] for i in train_dataset.indices]
    #
    # # Getting the targets for the test subset
    # val_targets = [dataset.targets[i] for i in val_dataset.indices]
    #
    # train_dataset.targets = train_targets
    # val_dataset.targets = val_targets

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train/seg_train'
    test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test/seg_test'

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))

    ])

    train_dataset, val_dataset, test_dataset = get_intel_images(train_path=train_path, val_path=test_path)

    real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                     n_experiences=3, shuffle=True,
                                     seed=6,
                                     task_labels=True)

