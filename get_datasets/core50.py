from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.utils import concat_datasets
import torch


def load_core50(args, scenario='nc', mini=True, obj_lvl=False, dataset_root=''):

    # create benchmark
    core50_stream = CORe50(scenario=scenario, mini=mini, object_lvl=obj_lvl, dataset_root=dataset_root, run=0)

    # recovering the train and test streams
    train_stream = core50_stream.train_stream
    test_stream = core50_stream.test_stream

    full_dataset = []
    # iterating over the train stream
    for experience_id, experience in enumerate(train_stream):

        # The current Pytorch training set can be easily recovered through the
        # experience
        current_training_set = experience.dataset

        # Get full dataset
        full_dataset.append(current_training_set)

    # Get test dataset
    test_dataset = test_stream[0].dataset
    # test_dataset_1, test_dataset2 = torch.utils.data.random_split(test_dataset, lengths=[0.6, 0.4])

    # Concat to get a complete dataset
    full_dataset = concat_datasets(full_dataset)

    # A need of huge gpus to resize image -> 224, then use pretrained model to reach to acc = 82%
    full_dataset = concat_datasets([full_dataset, test_dataset])

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, lengths=[0.65, 0.2, 0.15])

    return train_dataset, val_dataset, test_dataset


