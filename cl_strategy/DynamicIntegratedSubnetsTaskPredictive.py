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
from models import vit, vit_prompt, resnet, simple_cnn, AE, model_utils
from train_epoch import train, test
from randomaug import RandAugment
from utils import progress_bar
from loss_functions import perceptual_loss
from get_datasets import intel_image

def _experiences_parameter_as_iterable(experiences):
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]


class DynamicIntegratedContinualLearningWithSubnets(SupervisedTemplate):
    def __init__(self, args, model, optimizer, test_entire_dataset, eval_stream, wandb, perceptual_loss_func, **kwargs):
        super(DynamicIntegratedContinualLearningWithSubnets, self).__init__(model=model, optimizer=optimizer,
                                                                            train_mb_size=32, eval_mb_size=32, **kwargs)
        self.fisher_matrices = None
        self.prev_task_params = {}
        self.adapt_af = True
        self.adap_kld = True

        self.cross_entropy = nn.CrossEntropyLoss()

        self.adapted_distilled_dataset = None
        self.distilled_dataloader = None
        self.distilled_experience = None
        self.task_id = 0
        self.args = args

        # Validation entire dataset for cifar10
        self.val_dataset = test_entire_dataset
        self.val_dataloader = None

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

        # Loss function
        self.criterion = nn.CrossEntropyLoss().to(args.device)

        # Loop tasks
        self.acc = np.zeros((args.n_experience, args.n_experience), dtype=np.float32)
        self.lss = np.zeros((args.n_experience, args.n_experience), dtype=np.float32)

        # Meta model
        from collections import defaultdict
        self.stored_features = defaultdict(list)

        # use wandb
        self.wandb = wandb
        self.ce_loss = 0
        self.reconstruction_loss = 0
        self.val_loss = 0

        self.train_acc = 0
        self.val_acc = 0

        # perceptual loss
        self.perceptual_loss = perceptual_loss_func

        # self.perceptual_loss = perceptual_loss.AE_PerceptualLoss(self.model).to('cuda') if args.device else \
        #     perceptual_loss.AE_PerceptualLoss(self.model)

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
        #
        # loguru.logger.info(f"Optimizing classification loss on task {self.task_id}")
        # early_stopper_classification = model_utils.EarlyStopper(patience=10)
        # for epoch in range(0, 20):
        #     self.train_current_task(self.args, self.model, self.dataloader, epoch, optimize='classification')
        #     print("<< Evaluating the current training >> ")
        #     val_loss, val_acc, val_rec_loss = self.eval_current_task(self.args, self.model,
        #                                                              self.eval_each_stream_dataloader)
        #     print(" ")
        #     if early_stopper.early_stop(val_acc):
        #         break
        #
        # # Phase 2: Freeze the encoder
        # for param in self.model.subnets[self.task_id].encoder.parameters():
        #     param.requires_grad = False
        #
        # loguru.logger.info(f"Optimizing on reconstruction loss on task {self.task_id}")
        for epoch in range(0, self.args.n_epochs):

            self.train_current_task(self.args, self.model, self.dataloader, epoch) # , optimize='reconstruction'

            print("<< Evaluating the current training >> ")
            val_loss, val_acc, val_rec_loss = self.eval_current_task(self.args, self.model,
                                                                     self.eval_each_stream_dataloader)

            if self.wandb:
                # Use a unique tag for each experience and epoch combination
                tag_loss_val = f'Loss/validation_loss'
                tag_acc_val = f'Accuracy/validation_accuracy'
                tag_ce_loss_train = f'Loss/ce_loss'
                tag_reconstruction_loss_train = f'Loss/reconstruction_loss'
                tag_acc_train = f'Accuracy/train_accuracy'

                self.wandb.log({tag_loss_val: val_loss, tag_acc_val: val_acc,
                                tag_ce_loss_train: self.ce_loss, tag_acc_train: self.train_acc,
                                tag_reconstruction_loss_train: self.reconstruction_loss,
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
            best_net_id, outputs = self.model(images, task_id=None)

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

        output_path = f'/home/luu/projects/cl_selective_nets/results/{self.args.dataset}_' \
                      f'DISN_task-agnostic_{self.args.reconstruction_loss}reconstruction_score_' \
                      f'{self.args.latent_dim}-latent-dim_{self.args.size}-image-size.txt'
        print('Save at ' + output_path)
        np.savetxt(output_path, self.acc, '%.4f')

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

        for batch_idx, (images, targets) in enumerate(stream_dataloader): # , _
            # self.model.to(self.args.device)
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # Given task_id=u for debugging
            best_net_id, outputs = self.model(images, task_id=None)

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

        output_path = f'/home/luu/projects/cl_selective_nets/results/{self.args.dataset}_' \
                      f'DISN_task-agnostic_{self.args.reconstruction_loss}reconstruction_score_' \
                      f'{self.args.latent_dim}-latent-dim_{self.args.size}-image-size.txt'
        print('Save at ' + output_path)
        np.savetxt(output_path, self.acc, '%.4f')

        if self.wandb:
            tag_val_task_agnostic_loss = 'Agnostic-Loss'
            tag_val_task_agnostic_acc = 'Agnostic-Acc'

            self.wandb.log({tag_val_task_agnostic_loss: test_loss,
                            tag_val_task_agnostic_acc: test_acc})

    def train_current_task(self, args, model, trainloader, epoch, optimize='classification'):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_ce_loss = 0
        train_reconstruction_loss = 0
        correct = 0
        total = 0
        total_ssim = 0
        for batch_idx, data in enumerate(trainloader):
            inputs, targets, taskid_targets = data

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # Forward current model
            # print(taskid_targets) taskid_targets[0]
            reconstructed_img, outputs = self.model(inputs, task_id=self.task_id)

            ce_loss = self.cross_entropy(outputs, targets)

            if self.args.reconstruction_loss == 'mse':
                reconstruction_loss = F.mse_loss(inputs, reconstructed_img, reduction="none")
                reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0])

            elif self.args.reconstruction_loss == 'perceptual_loss':
                reconstruction_loss = self.perceptual_loss(reconstructed_img, inputs)
            else:
                raise NotImplementedError("Reconstruction image loss function is not implemented!")

            ssim_score = ssim(inputs.float(), reconstructed_img.float(), data_range=1, size_average=True)

            lamda = 0.5
            total_loss = (1 - lamda) * reconstruction_loss + lamda * ce_loss

            # if optimize == 'classification':
            #     total_loss = ce_loss
            # elif optimize == 'reconstruction':
            #     total_loss = reconstruction_loss
            # else:
            #     raise Exception("Unknown optimization")

            # print('loss = ', modified_ce_loss)
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_ce_loss += ce_loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            total_ssim += ssim_score.item()

            # Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'CE Loss: %.3f | Reconstruction Loss: %.3f '
                                                      '| SSIM: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_ce_loss / (batch_idx + 1), train_reconstruction_loss / (batch_idx + 1),
                            total_ssim/(batch_idx+1),
                            100. * correct / total, correct, total))

        self.ce_loss = train_ce_loss / (batch_idx + 1)
        self.reconstruction_loss = train_reconstruction_loss / (batch_idx + 1)
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
                reconstructed_img, outputs = self.model(inputs, task)

                # # Trying to make task ID predictions
                # pred_taskid, outputs = self.model(inputs, task, return_experts=False, return_task_pred=True)
                # print(torch.argmax(pred_taskid).item())
                loss = self.cross_entropy(outputs, targets)

                if self.args.reconstruction_loss == 'mse':
                    reconstruction_loss = F.mse_loss(inputs, reconstructed_img, reduction="none")
                    reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0])

                elif self.args.reconstruction_loss == 'perceptual_loss':
                    reconstruction_loss = self.perceptual_loss(reconstructed_img, inputs)

                ssim_score = ssim(inputs.float(), reconstructed_img.float(), data_range=1, size_average=True)

                val_reconstruction_loss += reconstruction_loss.item()
                test_loss += loss.item()
                total_ssim += ssim_score.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_loss.append(test_loss / (batch_idx + 1))
                total_acc.append(100. * correct / total)
                total_rec_loss.append(val_reconstruction_loss/(batch_idx+1))

                progress_bar(batch_idx, len(val_loader), 'CE Loss: %.3f | Reconstruction Loss: %.3f | SSIM: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1),
                                val_reconstruction_loss / (batch_idx + 1),
                                total_ssim/(batch_idx+1),
                                100. * correct / total, correct, total))

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


def DIWSN_task_predictive(args, real_dataset, distilled_dataset):

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

    # Get real train, val, and test dataset
    train_dataset, val_dataset, test_dataset = real_dataset

    # dataset intel_images
    # todo: dont understand why loading this dataset from main_cl doesn't work
    if args.dataset == 'intel_images':
        train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train/seg_train'
        test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test/seg_test'

        train_dataset, val_dataset, test_dataset = intel_image.get_intel_images(args=args,
                                                                                train_path=train_path,
                                                                                val_path=test_path)

    # Get distilled dataset
    distilled_datasets_train, distilled_datasets_test = torch.utils.data.random_split(distilled_dataset,
                                                                                      lengths=[0.9, 0.1])

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

    # # Get balanced subset
    # balanced_subset = sample_subset(train_dataset, images_per_class=10)

    if args.reconstruction_loss == 'perceptual_loss':
        perceptual_loss_func = perceptual_loss.PerceptualLoss().to('cuda') if args.device else \
            perceptual_loss.PerceptualLoss()

    else:
        perceptual_loss_func = None

    # latent_dim = 256, 384 (best), 512
    model = AE.AEWithSelectiveSubnets(args=args, base_channel_size=args.size, latent_dim=args.latent_dim,
                                      num_child_models=args.n_experience,
                                      num_classes=args.nb_classes, width=args.size, height=args.size,
                                      perceptual_loss=perceptual_loss_func)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Get strategy
    strategy = DynamicIntegratedContinualLearningWithSubnets(args=args, model=model, optimizer=optimizer,
                                                             test_entire_dataset=test_dataset,
                                                             eval_stream=real_dataset_exps.test_stream,
                                                             wandb=wandb if args.use_wandb is True else None,
                                                             perceptual_loss_func=perceptual_loss_func)

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

        # if task_id == 4:
        #     torch.save(strategy.stored_features, '/home/luu/projects/cl_selective_nets/results/features.pt'

    torch.save(model.state_dict(), f'/home/luu/projects/cl_selective_nets/results'
                                   f'/{args.dataset}_DISN_task-agnostic_reconstruction_score.pt')

    if args.use_wandb is True:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


def test_pretrained_model(args, real_dataset):

    # Get real train, val, and test dataset
    train_dataset, val_dataset, test_dataset = real_dataset

    model = AE.AEWithSelectiveSubnets(args=args, base_channel_size=args.size, latent_dim=args.latent_dim,
                                      num_child_models=args.n_experience,
                                      num_classes=args.nb_classes, width=args.size, height=args.size,
                                      perceptual_loss=None)

    pretrained_path = f'/home/luu/projects/cl_selective_nets/results' \
                      f'/{args.dataset}_DISN_task-agnostic_reconstruction_score.pt'

    model.load_state_dict(torch.load(pretrained_path))

    rec_loss_range = {0: 3171, 1: 2729, 2: 2457, 3: 3016, 4: 2517}
    classes_in_exps = {0: [1, 4], 1: [9, 6], 2: [3, 5], 3: [0, 2], 4: [8, 7]}

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_images = 0
    correct_predictions = 0

    for data in test_loader:
        image, label, _ = data
        label = label.item()  # Assuming label is a tensor with a single value
        closest_model = None
        smallest_diff = float('inf')

        for idx, submodel in enumerate(model.subnets):
            reconstructed_img, _ = submodel(image)

            reconstruction_loss = F.mse_loss(image, reconstructed_img, reduction="none")
            reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0]).item()

            loss_diff = abs(rec_loss_range[idx] - reconstruction_loss)

            if loss_diff < smallest_diff:
                smallest_diff = loss_diff
                closest_model = idx

        threshold = 100  # This is adjustable
        if smallest_diff <= threshold and label in classes_in_exps[closest_model]:
            correct_predictions += 1
        total_images += 1

        if smallest_diff <= threshold:
            print(
                f"Image from batch closely matches model {closest_model} which includes classes {classes_in_exps[closest_model]}")
        else:
            print(f"No close reconstruction model found. Loss: {reconstruction_loss}, True label: {label}")

    # Calculate accuracy
    accuracy = (correct_predictions / total_images) * 100
    print(f"Accuracy of correctly predicting experience based on reconstruction loss is {accuracy:.2f}%")


"""
python3 main_cl.py --n_epochs 5 --cl_strategy DI_with_task_predictive

seed = 5
python3 main_cl.py --dataset core50 --n_epochs 200 --latent_dim 128 --cl_strategy DI_with_task_predictive --lr 0.001 --use_wandb True

seed = 6
python3 main_cl.py --dataset cifar10 --n_epochs 30 --latent_dim 256 --cl_strategy DI_with_task_predictive --lr 0.001 --use_wandb false --reconstruction_loss perceptual_loss --AE_model ae_resnet18


# Core50 50 classes - perceptual loss
python3 main_cl.py --dataset core50 --n_epochs 200 --latent_dim 384 --n_experience 5 --obj_lvl true --nb_classes 50 --reconstruction_loss perceptual_loss --cl_strategy DI_with_task_predictive --lr 0.001 --use_wandb false


# Intel images - perceptual loss - image size 224
python3 main_cl.py --dataset intel_images --n_epochs 50 --latent_dim 128 --cl_strategy DI_with_task_predictive --lr 0.001 --use_wandb false --AE_model ae_resnet50_pretrained --size 224 --reconstruction_loss perceptual_loss --n_experience 3 --nb_classes 6

# stl-10 dataset
python3 main_cl.py --dataset stl_10 --n_epochs 90 --latent_dim 256 --cl_strategy DI_with_task_predictive --lr 0.001 --use_wandb false --reconstruction_loss perceptual_loss --AE_model ae_simple_cnn_pretrained --n_experience 5 --nb_classes 10 --size 32

"""
