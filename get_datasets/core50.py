import loguru
from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.utils import concat_datasets
import torch
import torchvision.transforms as transforms


def load_core50(args, scenario='nc', mini=True, obj_lvl=False, dataset_root=''):
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loguru.logger.info('Core50 is loading with Norm of ImageNet')

    # create benchmark
    core50_stream = CORe50(scenario=scenario, mini=mini, object_lvl=obj_lvl, dataset_root=dataset_root, run=0,
                           train_transform=transform, eval_transform=transform)

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

    # --------------------------------------------- #
    # # A need of huge gpus to resize image -> 224, then use pretrained model to reach to acc = 82%
    # full_dataset = concat_datasets([full_dataset, test_dataset])
    #
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, lengths=[0.65, 0.2, 0.15])
    # --------------------------------------------- #

    # A need of huge gpus to resize image -> 224, then use pretrained model to reach to acc = 82%
    full_dataset = concat_datasets([full_dataset])

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, lengths=[0.7, 0.3])

    # NOTE!: The test is only good with high image size (224) and with pretrained model like mobilenetv2
    # If train with only simple cnn or even with pretrained resnet50 but small size as 32 or 128, the highest accuracy
    # can get is around 50%

    # Mobilenetv2 pretrained: 87%
    # Resnet50 pretrained: 93% - non pretrained: ~50%
    # SimpleCNN non pretrained: ~45%
    # val_dataset
    return train_dataset, test_dataset, test_dataset


