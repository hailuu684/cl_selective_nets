import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
#     transforms.ToTensor(),
#     transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
# ])


class CustomSTL10(datasets.STL10):
    def __getitem__(self, index):
        image, target = super(CustomSTL10, self).__getitem__(index)
        return image, target, index

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = self.labels.tolist()


def get_stl_10(args):
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),  # Resize to a slightly larger size before cropping
        # transforms.RandomCrop((args.size, args.size)),  # Randomly crop to 32x32 pixels
        # transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        # transforms.RandomRotation(10),  # Randomly rotate the images within a 10-degree range
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # # Randomly alter the color properties
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load stl-10 dataset
    root = '/home/luu/projects/datasets/stl-10'
    train_dataset = datasets.STL10(root=root, split='train', download=True, transform=transform)
    test_dataset = datasets.STL10(root=root, split='test', download=True, transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths=[0.7, 0.3])

    # from avalanche.benchmarks.utils import concat_datasets
    # train_dataset = concat_datasets([train_dataset, test_dataset])
    #
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, lengths=[0.7, 0.2, 0.1])

    # Function to extract targets
    def extract_targets(dataset):
        return [sample[1] for sample in dataset]

    # Getting targets for train and val datasets
    train_targets = extract_targets(train_dataset)
    val_targets = extract_targets(val_dataset)

    train_dataset.targets = train_targets
    val_dataset.targets = val_targets

    return train_dataset, val_dataset, test_dataset
