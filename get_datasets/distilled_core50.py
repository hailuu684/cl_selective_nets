import torch
import os
from avalanche.benchmarks.utils import concat_datasets
from torch.utils.data import Dataset
from avalanche.benchmarks.utils import make_classification_dataset


class CustomDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Retrieve data and target for a given index
        x = self.images[index]
        y = self.targets[index]

        return x, y


def get_distilled_core50():
    root = '/home/luu/DistilledDataset_ContinualLearning/author_distilled_dataset/DatasetCondensation/result'

    datasets = []
    for i in range(0, 7):  # numbers from 1 to 7
        # filename = f"res_DSA_core50_32_ConvNet_100ipc_{i}exp.pt"
        filename = f"cl_DM_core50_32_ConvNetBN_100ipc_niscenario_{i}batch_32img_size.pt"
        filepath = os.path.join(root, filename)
        if os.path.exists(filepath):
            data = torch.load(filepath)['data'][0]
            images, labels = data

            dataset = CustomDataset(images, labels)
            dataset = make_classification_dataset(dataset)
            datasets.append(dataset)

    datasets = concat_datasets(datasets)
    return datasets


if __name__ == '__main__':
    datasets = get_distilled_core50()
    print(len(datasets))

