# https://github.com/lywang3081/CAF/blob/v1.0.0/networks/conv_net_caf.py
# AdaptCL: https://arxiv.org/pdf/2207.11005.pdf
import copy

# Task prediction
# https://github.com/brjathu/iTAML/blob/master/learner_task_itaml.py#L268
from avalanche.benchmarks.utils import concat_datasets
import loguru
import torch.nn as nn
from typing import Optional, Union, List, Callable
import torch
from avalanche.training.templates import SupervisedTemplate
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Subset
from typing import Iterable
import torch.optim as optim
import torch.nn.functional as F
import wandb
import torchvision
import random

try:
    from timm.models.helpers import (
        adapt_input_conv,
        checkpoint_seq,
        resolve_pretrained_cfg,
        build_model_with_cfg,
    )
    from timm.models.layers import PatchEmbed

except ImportError:
    raise ModuleNotFoundError(
        "timm is not found, "
        "if you want to use L2P "
        "please install avalanche with the "
        "detection dependencies: "
        "pip install avalanche-lib[l2p]"
    )

import sys
from pytorch_msssim import ssim

sys.path.append("../")
from models import resnet, dinov2, model_utils, radio
from train_epoch import train, test
from randomaug import RandAugment
from utils import progress_bar

from get_datasets import intel_image


def _experiences_parameter_as_iterable(experiences):
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]


class CL_reminder(SupervisedTemplate):
    def __init__(self, args, model, optimizer, memories, test_entire_dataset, eval_stream, wandb, k):
        super(CL_reminder, self).__init__(model=model, optimizer=optimizer, train_mb_size=32, eval_mb_size=32)

        # Memory parameters
        # todo: can be modify here to add up features into memories over each experience
        self.memories = memories

        # Number of images in each label
        self.k = k

        self.compare_method = args.compare_method  # correlation_based_distance, knn, cosine_similarity
        loguru.logger.info(f"Compare method using: {self.compare_method}")

        # Loss function
        self.cross_entropy = nn.CrossEntropyLoss()

        # Distilled dataset
        self.adapted_distilled_dataset = None
        self.distilled_dataloader = None
        self.distilled_experience = None
        self.task_id = 0
        self.args = args

        # Validation entire dataset for cifar10
        self.val_dataset = test_entire_dataset
        self.val_dataloader = None

        # Test each experience
        self.eval_all_stream_dataset = eval_stream
        self.eval_each_stream_dataloader = None

        # Training
        # use scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        # Use Aug
        if args.aug:
            loguru.logger.info("Use Random Aug")
            self.aug = RandAugment(n=0, m=29)
        else:
            self.aug = None

        # use gpu
        self.model.to(args.device)

        # Loop tasks
        self.acc = np.zeros((args.n_experience, args.n_experience), dtype=np.float32)
        self.lss = np.zeros((args.n_experience, args.n_experience), dtype=np.float32)

        # use wandb
        self.wandb = wandb
        self.save_results_path = f"/home/luu/projects/cl_selective_nets/results/{self.args.dataset}_" \
                                 f"{self.args.cl_strategy}_" \
                                 f"{self.args.size}-img-size_{self.args.compare_method}-compare-method_{self.args.k}-k_" \
                                 f"{self.args.get_mem_method}-mem-method_" \
                                 f"{self.args.latent_dim}-latent-dim.txt"

        # temporary store loss and acc
        self.ce_loss = 0
        self.val_loss = 0

        self.train_acc = 0
        self.val_acc = 0

    def train(self, experiences, task_id=0, **kwargs):
        """
        train phase
        :param task_id:
        :param experiences: tuple of real_exp and distilled_exp
        :param kwargs: keyword arguments inherit from template
        :return:
        """

        self.experience = experiences
        # self.distilled_experience = experiences[1]
        self.task_id = task_id
        if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            self._check_distributed_training_compatibility()
            self._distributed_check = True

        self._before_training(**kwargs)  # Just print << Start of Training Phase >>

        # Validation dataset
        self.make_eval_dataloader()

        self.train_dataset_adaptation()
        self.make_train_dataloader()

        self.training_epoch()  # use original model to get features, prompt model to train incremental tasks
        self._after_training_epoch(**kwargs)

    def training_epoch(self, **kwargs):

        print(f"<< Start of training task {self.task_id} >>")

        # Early stopping
        early_stopper = model_utils.EarlyStopper(patience=10)

        for epoch in range(0, self.args.n_epochs):

            self.train_current_task(self.args, self.model, self.dataloader, epoch)

            print("<< Evaluating the current training >> ")
            val_loss, val_acc, val_rec_loss = self.eval_current_task(self.args, self.model,
                                                                     self.eval_each_stream_dataloader)

            if self.wandb:
                # Use a unique tag for each experience and epoch combination
                tag_loss_val = f'Loss/validation_loss'
                tag_acc_val = f'Accuracy/validation_accuracy'
                tag_ce_loss_train = f'Loss/ce_loss'
                tag_acc_train = f'Accuracy/train_accuracy'

                self.wandb.log({tag_loss_val: val_loss, tag_acc_val: val_acc,
                                tag_ce_loss_train: self.ce_loss, tag_acc_train: self.train_acc,
                                "epoch": epoch})

            # Early stopping, val_rec_loss
            if early_stopper.early_stop(val_acc):
                break

        print(f"<< Finish training on task {self.task_id} >>")
        print(" ")

    def eval_agnostic_core50(self):
        self.model.eval()

        # For core50
        all_test_stream = self.eval_all_stream_dataset

        all_eval_dataset = []

        for test_stream in all_test_stream:
            all_eval_dataset.append(test_stream.dataset)

        test_dataset_all_tasks = concat_datasets(all_eval_dataset)
        stream_dataloader = DataLoader(test_dataset_all_tasks,
                                       batch_size=1, shuffle=True,  # Evaluate 1 image
                                       pin_memory=True, num_workers=6, collate_fn=self.collate_fn)

        # # Test again cifar
        # stream_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)

        test_loss = 0
        correct = 0
        total = 0

        total_loss = []
        total_acc = []

        for batch_idx, (images, targets, _) in enumerate(stream_dataloader):  # , _
            # self.model.to(self.args.device)
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # Given task_id=u for debugging
            outputs = self.model(images, task_id=None, compare_method=self.compare_method)

            loss = self.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_loss.append(test_loss / (batch_idx + 1))
            total_acc.append(100. * correct / total)

            progress_bar(batch_idx, len(stream_dataloader), 'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_loss, test_acc = np.average(total_loss), np.average(total_acc)
        self.acc[self.task_id, self.task_id] = test_acc
        self.lss[self.task_id, self.task_id] = test_loss

        print('Average accuracy={:5.1f}%'.format(test_acc))

        print('Save at ' + self.save_results_path)
        np.savetxt(self.save_results_path, self.acc, '%.4f')

        if self.wandb:
            tag_val_task_agnostic_loss = 'Agnostic-Loss'
            tag_val_task_agnostic_acc = 'Agnostic-Acc'

            self.wandb.log({tag_val_task_agnostic_loss: test_loss,
                            tag_val_task_agnostic_acc: test_acc})

    def eval_agnostic_cifar10(self):
        self.model.eval()

        # Test again cifar
        stream_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)

        test_loss = 0
        correct = 0
        total = 0

        total_loss = []
        total_acc = []

        for batch_idx, (images, targets) in enumerate(stream_dataloader):  # , _
            # self.model.to(self.args.device)
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # Given task_id=u for debugging
            outputs = self.model(images, task_id=None, compare_method=self.compare_method)

            loss = self.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_loss.append(test_loss / (batch_idx + 1))
            total_acc.append(100. * correct / total)

            progress_bar(batch_idx, len(stream_dataloader), 'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_loss, test_acc = np.average(total_loss), np.average(total_acc)
        self.acc[self.task_id, self.task_id] = test_acc
        self.lss[self.task_id, self.task_id] = test_loss

        print('Average accuracy={:5.1f}%'.format(test_acc))

        print('Save at ' + self.save_results_path)
        np.savetxt(self.save_results_path, self.acc, '%.4f')

        if self.wandb:
            tag_val_task_agnostic_loss = 'Agnostic-Loss'
            tag_val_task_agnostic_acc = 'Agnostic-Acc'

            self.wandb.log({tag_val_task_agnostic_loss: test_loss,
                            tag_val_task_agnostic_acc: test_acc})

    def train_current_task(self, args, model, trainloader, epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_ce_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            inputs, targets, taskid_targets = data

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # Forward current model
            # print(taskid_targets) taskid_targets[0]
            outputs = self.model(inputs, task_id=self.task_id)

            ce_loss = self.cross_entropy(outputs, targets)

            self.optimizer.zero_grad()
            ce_loss.backward()
            self.optimizer.step()

            train_ce_loss += ce_loss.item()

            # Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_ce_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.ce_loss = train_ce_loss / (batch_idx + 1)
        self.train_acc = 100. * correct / total

    def eval_current_task(self, args, model, val_loader, test_task_id=None):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        val_reconstruction_loss = 0
        total_loss = []
        total_acc = []
        total_rec_loss = []
        total_ssim = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):

                try:
                    inputs, targets, _ = data
                except:
                    inputs, targets = data

                inputs, targets = inputs.to(args.device), targets.to(args.device)

                # Forward current model
                if test_task_id is not None:
                    current_task = test_task_id
                else:
                    current_task = self.task_id

                task = torch.autograd.Variable(torch.LongTensor([current_task]).cuda())

                # Original code, not have return task id
                outputs = self.model(inputs, task_id=task)

                # # Trying to make task ID predictions
                # pred_taskid, outputs = self.model(inputs, task, return_experts=False, return_task_pred=True)
                # print(torch.argmax(pred_taskid).item())
                loss = self.cross_entropy(outputs, targets)

                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_loss.append(test_loss / (batch_idx + 1))
                total_acc.append(100. * correct / total)
                total_rec_loss.append(val_reconstruction_loss / (batch_idx + 1))

                progress_bar(batch_idx, len(val_loader),
                             'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return np.average(total_loss), np.average(total_acc), np.average(total_rec_loss)

    def _after_training_epoch(self, **kwargs):
        pass

    def train_dataset_adaptation(self, **kwargs):
        self.adapted_dataset = self.experience.dataset
        # self.adapted_distilled_dataset = self.distilled_experience.dataset

    def make_train_dataloader(
            self,
            num_workers=0,
            shuffle=True,
            pin_memory=None,
            persistent_workers=False,
            drop_last=False,
            **kwargs
    ):
        self.dataloader = DataLoader(self.adapted_dataset, batch_size=self.train_mb_size, shuffle=shuffle,
                                     pin_memory=pin_memory, persistent_workers=persistent_workers,
                                     sampler=None, num_workers=num_workers, collate_fn=self.collate_fn)

        # self.distilled_dataloader = DataLoader(self.adapted_distilled_dataset, batch_size=self.train_mb_size,
        #                                        shuffle=shuffle, pin_memory=pin_memory,
        #                                        persistent_workers=persistent_workers, sampler=None,
        #                                        num_workers=num_workers, collate_fn=self.collate_fn, drop_last=True)

    def make_eval_dataloader(
            self,
            num_workers=0,
            shuffle=False,
            pin_memory=None,
            persistent_workers=False,
            **kwargs
    ):
        self.eval_each_stream_dataloader = DataLoader(self.eval_all_stream_dataset[self.task_id].dataset,
                                                      batch_size=self.train_mb_size, shuffle=shuffle,
                                                      pin_memory=pin_memory, persistent_workers=persistent_workers,
                                                      sampler=None, num_workers=num_workers, collate_fn=self.collate_fn)

    def tuple_of_tensors_to_tensor(self, tuple_of_tensors):
        return torch.stack(list(tuple_of_tensors), dim=0)

    def collate_fn(self, batch):
        try:
            images, labels, task_id = zip(*batch)
        except ValueError:
            images, labels, _, task_id = zip(*batch)

        # print(type(images), len(images))
        images = self.tuple_of_tensors_to_tensor(images)
        labels = torch.tensor(labels)  # Convert labels to tensor

        return images, labels, task_id

    def convert_image_range_torch(self, image):
        """
        Check if a PyTorch tensor image has pixels in the range of -1 to 1.
        If so, convert the image to have pixels in the range of 0 to 1.

        Args:
        - image (torch.Tensor): The input image with shape (1, 3, 32, 32).

        Returns:
        - torch.Tensor: The converted image if needed, or the original image.
        """
        # Check if the image has any pixels less than 0, indicating it's in the range -1 to 1
        if image.min() < 0:
            # Convert image from range -1 to 1 to 0 to 1
            image = (image + 1) / 2
        return image


def cl_reminder(args, real_dataset, distilled_dataset):
    if args.use_wandb is True:
        wandb.login(key='1bed216d1f9c32afa692155d2e0911cd750f41dd')

        config = dict(
            num_experience=args.n_experience, num_epoch=args.n_epochs, dataset=args.dataset,
            strategy=args.cl_strategy, scenario=args.scenario, latent_dim=args.latent_dim
        )

        wandb.init(
            # set the wandb project where this run will be logged
            project="CL-Subnets-Predictive-Task",

            # track hyperparameters and run metadata
            config=config,

            name=f"{args.dataset}-dataset_scenario-{args.cl_strategy}" +
                 f'_{args.latent_dim}-latent-dim' + f'_{args.n_epochs}-epochs_{args.nb_classes}classes' +
                 f'_{args.reconstruction_loss}loss',

            entity='luutunghai'
        )

    # dataset intel_images
    # todo: dont understand why loading this dataset from main_cl doesn't work
    if args.dataset == 'intel_images':
        train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train/seg_train'
        test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test/seg_test'

        train_dataset, val_dataset, test_dataset = intel_image.get_intel_images(args=args,
                                                                                train_path=train_path,
                                                                                val_path=test_path)

        loguru.logger.info("Number of experience is 3 due to the number of classes are 6")
        args.n_experience = 3

    else:
        # Get real train, val, and test dataset
        train_dataset, val_dataset, test_dataset = real_dataset

    from avalanche.benchmarks import nc_benchmark, ni_benchmark

    seed = 5
    if args.scenario == 'nc':
        real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=seed,
                                         task_labels=True)

        # distilled_dataset_exps = nc_benchmark(distilled_datasets_train, distilled_datasets_test,
        #                                       n_experiences=args.n_experience,
        #                                       shuffle=True,
        #                                       seed=seed,
        #                                       task_labels=True)

    elif args.scenario == 'ni':
        real_dataset_exps = ni_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=seed,
                                         task_labels=True)

        # distilled_dataset_exps = ni_benchmark(distilled_datasets_train, distilled_datasets_test,
        #                                       n_experiences=args.n_experience,
        #                                       shuffle=True,
        #                                       seed=seed,
        #                                       task_labels=True)

    else:
        raise NotImplementedError(f"This scenario {args.scenario} is not implemented")

    # Get memory
    k = args.k
    loguru.logger.info(f"Loading memories: {k} images / class")

    if args.get_mem_method == 'kmeans':
        path = f"/home/luu/projects/cl_selective_nets/checkpoints/{args.dataset}-dataset_{k}-features.pt"
        memories = torch.load(path, map_location=args.device)

        model_size = 's'
        with_register = False
        extract_feature_model = dinov2.dinov2(model_size=model_size, with_register=with_register).to(args.device)

    else:

        # ## Pre-run to get memories to experience more cases since the limitations in GPUs
        path = f"/home/luu/projects/cl_selective_nets/checkpoints/{args.dataset}_{args.cl_strategy}_" \
               f"{args.size}-img-size_{args.compare_method}-compare-method_{args.k}-k_{args.get_mem_method}-mem-method_" \
               f"{args.latent_dim}-latent-dim"

        memories = torch.load(path, map_location=args.device)

        if args.latent_dim == 384:
            model_size = 's'
        elif args.latent_dim == 784:
            model_size = 'b'
        elif args.latent_dim == 1024:
            model_size = 'l'
        else:
            raise NotImplementedError("Latent dim is not correct!")

        with_register = False

        extract_feature_model = dinov2.dinov2(model_size=model_size, with_register=with_register)

        # --------------------------------------------------------------------
        # ## Not pre-run
        # # train_dataset, distilled_dataset
        # memories, extract_feature_model = get_memories(args, train_dataset, k=k,
        #                                                model=args.compare_model)  # radio, dinov2

    # latent_dim = 256, 384 (best), 512
    task_id_mapping = real_dataset_exps.original_classes_in_exp
    model = resnet.Resnet50WithSelectiveSubnets(args=args,
                                                model_dinov2=extract_feature_model,
                                                distilled_feature_dict=memories, num_classes=args.nb_classes,
                                                num_child_models=args.n_experience,
                                                task_id_mapping=task_id_mapping)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Get strategy
    strategy = CL_reminder(args=args, model=model, optimizer=optimizer, memories=memories,
                           test_entire_dataset=test_dataset,
                           eval_stream=real_dataset_exps.test_stream,
                           wandb=wandb if args.use_wandb is True else None,
                           k=k)

    # for task_id, real_experience in enumerate(real_dataset_exps.train_stream):
    #     print('classes in train stream', real_experience.classes_in_this_experience)
    #     print('classes in val stream', real_dataset_exps.test_stream[task_id].classes_in_this_experience)

    # print(len(real_dataset_exps.test_stream[0].dataset))
    for task_id, real_experience in enumerate(real_dataset_exps.train_stream):
        strategy.train(experiences=real_experience, task_id=task_id)

        if args.dataset == 'core50':
            strategy.eval_agnostic_core50()
        elif args.dataset == 'cifar10':
            strategy.eval_agnostic_cifar10()
        else:
            strategy.eval_agnostic_core50()
    #
    #     # if task_id == 4:
    #     #     torch.save(strategy.stored_features, '/home/luu/projects/cl_selective_nets/results/features.pt'
    #
    torch.save(model.state_dict(), f'/home/luu/projects/cl_selective_nets/results'
                                   f'/{args.dataset}_DISN_task-agnostic_reconstruction_score.pt')

    if args.use_wandb is True:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


def run_get_memories(args, real_dataset, k, model):
    loguru.logger.info(f"Running getting memories on {args.dataset} with {k} features")

    if args.dataset == 'intel_images':
        train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train/seg_train'
        test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test/seg_test'

        train_dataset, val_dataset, test_dataset = intel_image.get_intel_images(args=args,
                                                                                train_path=train_path,
                                                                                val_path=test_path)

        loguru.logger.info("Number of experience is 3 due to the number of classes are 6")
        args.n_experience = 3

    else:
        # Get real train, val, and test dataset
        train_dataset, val_dataset, test_dataset = real_dataset

    memories, extract_feature_model = get_memories(args, train_dataset, k=k, model=model)  # radio, dinov2

    path = f"/home/luu/projects/cl_selective_nets/checkpoints/{args.dataset}_{args.cl_strategy}_" \
           f"{args.size}-img-size_{args.compare_method}-compare-method_{args.k}-k_{args.get_mem_method}-mem-method_" \
           f"{args.latent_dim}-latent-dim"

    # flash_memory = {'memories': memories, 'compare_model': extract_feature_model}

    torch.save(memories, path)

    print(f"saved {path}")


def get_memories(args, distilled_dataset, k, model='dinov2', return_dim_compare_model=False):
    if model == 'dinov2':
        assert args.size // 14, 'Dinov2 requires image size that can be divided with patch size = 14'

        if args.latent_dim == 384:
            model_size = 's'
        elif args.latent_dim == 784:
            model_size = 'b'
        elif args.latent_dim == 1024:
            model_size = 'l'
        else:
            raise NotImplementedError("Latent dim is not correct!")

        with_register = False
        extract_feature_model = dinov2.dinov2(model_size=model_size, with_register=with_register)

        extract_feature_model.to(args.device)

        loguru.logger.info(f"Compare model: Dino_v2 - Model size: {model_size} - With register: {with_register}")

    elif model == 'radio':

        valid_radio_model = ['radio_v2', 'e-radio_v2']

        chosen_model = valid_radio_model[1]
        latent_dim = 512

        extract_feature_model = radio.radio_v2(args, desire_latent_dim=latent_dim, model_version=chosen_model)

        loguru.logger.info(f"Compare model: Radio_v2 - version: {chosen_model}")

    else:
        raise NotImplemented("The extraction model is not defined")

    # Check the image shape constraints
    distilled_image_shape = distilled_dataset[0][0].shape[1]

    assert args.size == distilled_image_shape, 'The training image size is different to distilled image'

    # Ensure the dataset is in a mutable structure like a list
    if isinstance(distilled_dataset, list):
        dataset_list = distilled_dataset
    else:
        dataset_list = list(distilled_dataset)  # This assumes the dataset can be iterated into a list

    dataset_list = dataset_list[:(k * args.nb_classes * 5)]
    # Shuffle the dataset
    random.shuffle(dataset_list)

    class_images = {}

    if args.dataset == 'core50':
        # Collect up to k images per class
        for image, label, task_id in dataset_list:
            image = image.unsqueeze(0).to(args.device)
            # label = label.item()
            if label not in class_images:
                class_images[label] = [extract_feature_model(image)]  # Start a new list for this class
                # class_images[label] = [image]
            elif len(class_images[label]) < k:
                class_images[label].append(
                    extract_feature_model(image))  # Add image to the list if it doesn't reach k yet
                # class_images[label].append(image)
            # Check if we have collected the required number of classes
            enough_k = all(len(images) == k for images in class_images.values())
            if len(class_images) == args.nb_classes and enough_k:
                break

    # Core50 dataset has task_id
    else:
        # Collect up to k images per class
        for image, label in dataset_list:
            image = image.unsqueeze(0).to(args.device)
            # label = label.item()
            if label not in class_images:
                class_images[label] = [extract_feature_model(image)]  # Start a new list for this class
                # class_images[label] = [image]
            elif len(class_images[label]) < k:
                class_images[label].append(
                    extract_feature_model(image))  # Add image to the list if it doesn't reach k yet
                # class_images[label].append(image)
            # Check if we have collected the required number of classes
            enough_k = all(len(images) == k for images in class_images.values())
            if len(class_images) == args.nb_classes and enough_k:
                break

    if return_dim_compare_model:
        return class_images, extract_feature_model, latent_dim

    return class_images, extract_feature_model


def extract_features(model, dataset, device, dataset_name):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        if dataset_name == 'core50':
            for images, target, _ in dataloader:
                images = images.to(device)
                output = model(images)
                features.append(output.cpu())
                labels.append(target)
        else:
            for images, target in dataloader:
                images = images.to(device)
                output = model(images)
                features.append(output.cpu())
                labels.append(target)
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


def kmeans_clustering(features, num_clusters, k):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    closest_images = {}
    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        cluster_features = features[indices]
        centroid = centroids[i].reshape(1, -1)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_indices = indices[np.argsort(distances)[:k]]
        closest_images[i] = closest_indices
    return closest_images


def get_memories_kmeans(args, real_dataset, k):
    loguru.logger.info(f"Running K-means to get most closest features to "
                       f"its centroid on {args.dataset} with {k} closest features")
    # Get real train, val, and test dataset
    train_dataset, val_dataset, test_dataset = real_dataset

    run_kmeans(args, train_dataset, k=k, model='dinov2')


def run_kmeans(args, distilled_dataset, k, model='dinov2'):
    # Model setup
    if model == 'dinov2':
        assert args.size % 14 == 0, 'Dinov2 requires image size that can be divided with patch size = 14'
        model_size = 's'
        with_register = False
        extract_feature_model = dinov2.dinov2(model_size=model_size, with_register=with_register).to(args.device)
        loguru.logger.info(f"Compare model: Dino_v2 - Model size: {model_size} - With register: {with_register}")
    elif model == 'radio':
        valid_radio_model = ['radio_v2', 'e-radio_v2']
        chosen_model = valid_radio_model[1]
        extract_feature_model = radio.radio_v2(args, desire_latent_dim=512, model_version=chosen_model)
        loguru.logger.info(f"Compare model: Radio_v2 - version: {chosen_model}")
    else:
        raise NotImplementedError("The extraction model is not defined")

    features, labels = extract_features(extract_feature_model, distilled_dataset, device=args.device,
                                        dataset_name=args.dataset)
    features_np = features.numpy()

    closest_images = kmeans_clustering(features_np, num_clusters=args.nb_classes, k=k)

    closest_features = {}
    for class_idx, indices in closest_images.items():
        closest_features[class_idx] = [extract_feature_model(distilled_dataset[i][0].unsqueeze(0).to(args.device)) for i
                                       in indices]

    # Save the closest features
    torch.save(closest_features,
               f"/home/luu/projects/cl_selective_nets/checkpoints/{args.dataset}-dataset_{k}-features.pt")

    return closest_features


"""
python3 main_cl.py --dataset core50 --size 224 --cl_strategy cl_reminder  

 python3 main_cl.py --dataset core50 --size 140 --cl_strategy cl_reminder --compare_method knn --compare_model dinov2 --k 10 --get_mem_method else

"""
