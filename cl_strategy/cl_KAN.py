import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training.templates import SupervisedTemplate
from typing import Iterable
import numpy as np
from torch.utils.data import DataLoader
# from kan import *
import sys

sys.path.append("../")

from utils import progress_bar
from models.CustomeKAN import KAN as MyKAN


class CL_KAN_IMG(nn.Module):
    def __init__(self, args):
        super(CL_KAN_IMG, self).__init__()  # This initializes the nn.Module

        # setting bias_trainable=False, sp_trainable=False, sb_trainable=False is important.
        # otherwise KAN will have random scaling and shift for samples in previous stages

        self.kan_model = MyKAN(width=[512, 5, 10], grid=50, k=3, noise_scale=0.1,
                               bias_trainable=False, sp_trainable=False, sb_trainable=False,
                               device=args.device)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1])  # Remove the fully connected layer, keep conv layers

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, image):
        x = self.feature_extractor(image).squeeze(2).squeeze(2)
        # print(x.shape)
        x = self.kan_model(x)

        return x


def _experiences_parameter_as_iterable(experiences):
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]


class CL_KAN(SupervisedTemplate):
    def __init__(self, args, model, optimizer, eval_stream, test_entire_dataset, wandb):
        super(CL_KAN, self).__init__(model=model, optimizer=optimizer, train_mb_size=32, eval_mb_size=32)

        # Loop tasks
        self.ce_loss = None
        self.acc = np.zeros((args.n_experience, args.n_experience), dtype=np.float32)
        self.lss = np.zeros((args.n_experience, args.n_experience), dtype=np.float32)

        self.train_acc = None
        self.eval_each_stream_dataloader = None
        self.task_id = 0
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.eval_all_stream_dataset = eval_stream
        self.wandb = wandb

        # Loss function
        self.cross_entropy = nn.CrossEntropyLoss()

        self.val_dataset = test_entire_dataset

        # For core50

        if args.dataset == 'core50':
            all_test_stream = self.eval_all_stream_dataset

            all_eval_dataset = []

            for test_stream in all_test_stream:
                all_eval_dataset.append(test_stream.dataset)

            test_dataset_all_tasks = concat_datasets(all_eval_dataset)
            self.stream_dataloader = DataLoader(test_dataset_all_tasks,
                                                batch_size=64, shuffle=True,  # Evaluate 1 image
                                                pin_memory=True, num_workers=6, collate_fn=self.collate_fn)
        else:
            # Test again cifar
            self.stream_dataloader = DataLoader(self.val_dataset, batch_size=64, shuffle=False, pin_memory=True,
                                                num_workers=6)

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

        for epoch in range(0, self.args.n_epochs):

            # self.train_current_task(self.args, self.model, self.dataloader, epoch)

            self.train_KAN_style(self.args, self.model, epoch, trainloader=self.dataloader, lamb=0.2)
            print("<< Evaluating the current training >> ")

            if self.args.dataset == 'core50':
                self.eval_agnostic_core50()
            else:
                # self.eval_agnostic_cifar10()
                self.eval_KAN_style(lamb=0.)

            # if self.wandb:
            #     # Use a unique tag for each experience and epoch combination
            #     tag_loss_val = f'Loss/validation_loss'
            #     tag_acc_val = f'Accuracy/validation_accuracy'
            #     tag_ce_loss_train = f'Loss/ce_loss'
            #     tag_acc_train = f'Accuracy/train_accuracy'
            #
            #     self.wandb.log({tag_loss_val: val_loss, tag_acc_val: val_acc,
            #                     tag_ce_loss_train: self.ce_loss, tag_acc_train: self.train_acc,
            #                     "epoch": epoch})

        print(f"<< Finish training on task {self.task_id} >>")
        print(" ")

    def train_current_task(self, args, model, trainloader, epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        # model.training = True
        train_ce_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            inputs, targets, taskid_targets = data

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # Forward current model
            # print(taskid_targets) taskid_targets[0]
            outputs = self.model(inputs)

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
        # model.eval()
        model.training = False
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
                outputs = self.model(inputs)

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

    def eval_agnostic_core50(self):

        print(" ")
        print(f"<< Agnostic Testing >>")

        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0

        total_loss = []
        total_acc = []

        for batch_idx, (images, targets, _) in enumerate(self.stream_dataloader):  # , _
            # self.model.to(self.args.device)
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # Given task_id=u for debugging
            outputs = self.model(images)

            loss = self.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_loss.append(test_loss / (batch_idx + 1))
            total_acc.append(100. * correct / total)

            progress_bar(batch_idx, len(self.stream_dataloader), 'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_loss, test_acc = np.average(total_loss), np.average(total_acc)
        self.acc[self.task_id, self.task_id] = test_acc
        self.lss[self.task_id, self.task_id] = test_loss

        print('Average accuracy={:5.1f}%'.format(test_acc))

        # output_path = f'/home/luu/projects/cl_selective_nets/results/{self.args.dataset}_' \
        #               f'{self.args.cl_strategy}_{self.args.size}-image-size.txt'
        #
        # print('Save at ' + output_path)
        # np.savetxt(output_path, self.acc, '%.4f')

        if self.wandb:
            tag_val_task_agnostic_loss = 'Agnostic-Loss'
            tag_val_task_agnostic_acc = 'Agnostic-Acc'

            self.wandb.log({tag_val_task_agnostic_loss: test_loss,
                            tag_val_task_agnostic_acc: test_acc})

    def eval_agnostic_cifar10(self):

        print(" ")
        print(f"<< Agnostic Testing >>")

        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0

        total_loss = []
        total_acc = []

        for batch_idx, (images, targets) in enumerate(self.stream_dataloader):  # , _
            # self.model.to(self.args.device)
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # Given task_id=u for debugging
            outputs = self.model(images)

            loss = self.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_loss.append(test_loss / (batch_idx + 1))
            total_acc.append(100. * correct / total)

            progress_bar(batch_idx, len(self.stream_dataloader), 'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        test_loss, test_acc = np.average(total_loss), np.average(total_acc)
        self.acc[self.task_id, self.task_id] = test_acc
        self.lss[self.task_id, self.task_id] = test_loss

        print('Average accuracy={:5.1f}%'.format(test_acc))

        # output_path = f'/home/luu/projects/cl_selective_nets/results/{self.args.dataset}_' \
        #               f'{self.args.cl_strategy}_{self.args.size}-image-size.txt'
        #
        # print('Save at ' + output_path)
        # np.savetxt(output_path, self.acc, '%.4f')

        if self.wandb:
            tag_val_task_agnostic_loss = 'Agnostic-Loss'
            tag_val_task_agnostic_acc = 'Agnostic-Acc'

            self.wandb.log({tag_val_task_agnostic_loss: test_loss,
                            tag_val_task_agnostic_acc: test_acc})

    def train_KAN_style(self, args, model, epoch, trainloader, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0.,
                        lamb_coefdiff=0.):

        print('\nEpoch: %d' % epoch)
        model.train()
        # model.training = True # This caused a big drop in accuracy
        train_ce_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            inputs, targets, taskid_targets = data

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            def closure():
                self.optimizer.zero_grad()
                outputs_ = model(inputs)
                train_loss = self.cross_entropy(outputs_, targets)
                reg_val = model.kan_model.calculate_regularization(model.kan_model.acts_scale,
                                                                   lamb_l1=lamb_l1, lamb_entropy=lamb_entropy,
                                                                   lamb_coef=lamb_coef,
                                                                   lamb_coefdiff=lamb_coefdiff,
                                                                   small_mag_threshold=1e-16,
                                                                   small_reg_factor=1.)
                loss_ = train_loss + lamb * reg_val
                loss_.backward()

                return loss_, outputs_

            loss, outputs = closure()
            self.optimizer.step()

            train_ce_loss += loss.item()

            # Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_ce_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.ce_loss = train_ce_loss / (batch_idx + 1)
        self.train_acc = 100. * correct / total

    def eval_KAN_style(self, lamb=0.):
        print(" ")
        print(f"<< Agnostic Testing >>")

        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0

        total_loss = []
        total_acc = []

        for batch_idx, (images, targets) in enumerate(self.stream_dataloader):  # , _
            # self.model.to(self.args.device)
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            def closure():
                self.optimizer.zero_grad()
                outputs_ = self.model(images)
                train_loss = self.cross_entropy(outputs_, targets)
                reg_val = self.model.kan_model.calculate_regularization(self.model.kan_model.acts_scale)
                loss_ = train_loss + lamb * reg_val
                return loss_, outputs_

            loss, outputs = closure()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_loss.append(test_loss / (batch_idx + 1))
            total_acc.append(100. * correct / total)

            progress_bar(batch_idx, len(self.stream_dataloader), 'CE Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def train_dataset_adaptation(self, **kwargs):
        self.adapted_dataset = self.experience.dataset

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


def cl_kan(args, real_dataset):
    if args.use_wandb is True:
        wandb.login(key='1bed216d1f9c32afa692155d2e0911cd750f41dd')

        config = dict(
            num_experience=args.n_experience, num_epoch=args.n_epochs, dataset=args.dataset,
            strategy=args.cl_strategy, scenario=args.scenario, latent_dim=args.latent_dim
        )

        wandb.init(
            # set the wandb project where this run will be logged
            project="CL-KAN",

            # track hyperparameters and run metadata
            config=config,

            name=f"{args.dataset}-dataset_scenario-{args.cl_strategy}" +
                 f'_{args.latent_dim}-latent-dim' + f'_{args.n_epochs}-epochs_{args.nb_classes}classes' +
                 f'_{args.reconstruction_loss}loss',

            entity='luutunghai'
        )

    # Get real train, val, and test dataset
    train_dataset, val_dataset, test_dataset = real_dataset

    from avalanche.benchmarks import nc_benchmark, ni_benchmark

    seed = 5
    if args.scenario == 'nc':
        real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=seed,
                                         task_labels=True)

    elif args.scenario == 'ni':
        real_dataset_exps = ni_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=seed,
                                         task_labels=True)

    else:
        raise NotImplementedError(f"This scenario {args.scenario} is not implemented")

    # Define KAN network
    model = CL_KAN_IMG(args).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Get strategy
    strategy = CL_KAN(args=args, model=model, optimizer=optimizer,
                      eval_stream=real_dataset_exps.test_stream,
                      wandb=wandb if args.use_wandb is True else None,
                      test_entire_dataset=test_dataset)

    # for task_id, real_experience in enumerate(real_dataset_exps.train_stream):
    #     print('classes in train stream', real_experience.classes_in_this_experience)
    #     print('classes in val stream', real_dataset_exps.test_stream[task_id].classes_in_this_experience)

    # print(len(real_dataset_exps.test_stream[0].dataset))
    for task_id, real_experience in enumerate(real_dataset_exps.train_stream):
        strategy.train(experiences=real_experience, task_id=task_id)

        # if args.dataset == 'core50':
        #     strategy.eval_agnostic_core50()
        # elif args.dataset == 'cifar10':
        #     strategy.eval_agnostic_cifar10()
        # else:
        #     strategy.eval_agnostic_core50()

    torch.save(model.state_dict(), f'/home/luu/projects/cl_selective_nets/results'
                                   f'/{args.dataset}_DISN_task-agnostic_reconstruction_score.pt')

    if args.use_wandb is True:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()
