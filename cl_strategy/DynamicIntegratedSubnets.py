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
from fast_pytorch_kmeans import KMeans

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

sys.path.append("../")
from models import vit, vit_prompt, resnet
from train_epoch import train, test
from randomaug import RandAugment
from utils import progress_bar


def _experiences_parameter_as_iterable(experiences):
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]


class DynamicIntegratedContinualLearningWithSubnets(SupervisedTemplate):
    def __init__(self, args, model, optimizer, test_entire_dataset, eval_stream, **kwargs):
        super(DynamicIntegratedContinualLearningWithSubnets, self).__init__(model=model, optimizer=optimizer,
                                                                            train_mb_size=32, eval_mb_size=32, **kwargs)
        self.fisher_matrices = None
        self.prev_task_params = {}

        self.kld = KLD()
        self.adapt_af = True
        self.adap_kld = True

        self.cross_entropy = nn.CrossEntropyLoss()

        self.adapted_distilled_dataset = None
        self.distilled_dataloader = None
        self.distilled_experience = None
        self.task_id = 0
        self.args = args

        # Validation
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
        self.acc = np.zeros((5, 5), dtype=np.float32)
        self.lss = np.zeros((5, 5), dtype=np.float32)

        # Meta model
        from collections import defaultdict
        self.stored_features = defaultdict(list)

    def train(self, experiences, task_id=0, **kwargs):
        """
        train phase
        :param task_id:
        :param experiences: tuple of real_exp and distilled_exp
        :param kwargs: keyword arguments inherit from template
        :return:
        """

        self.experience = experiences[0]
        self.distilled_experience = experiences[1]
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
            # train_loss = train(self.args, self.dataloader, epoch, self.model, self.criterion,
            #                    self.optimizer, self.scaler, aug=self.aug)
            # val_loss, acc = test(self.args, self.val_dataloader, self.model, self.criterion)

            self.train_current_task(self.args, self.model, self.dataloader, epoch)

            print("<< Evaluating the current training >> ")
            self.eval_current_task(self.args, self.model, self.eval_each_stream_dataloader)

        print(f"<< Finish training on task {self.task_id} >>")
        print(" ")

    def eval_agnostic(self):
        # todo: evaluation with all tasks at once
        pass

    def eval(self, exp_list, **kwargs):
        self.model.eval()
        all_stream_dataset = self.eval_all_stream_dataset[:self.task_id + 1]

        for u, test_stream in enumerate(all_stream_dataset):
            print(f"<< Testing on task {u} >>")
            stream_dataloader = DataLoader(test_stream.dataset,
                                           batch_size=self.train_mb_size, shuffle=False,
                                           pin_memory=True, num_workers=6, collate_fn=self.collate_fn)

            test_loss = 0
            correct = 0
            total = 0

            total_loss = []
            total_acc = []

            for batch_idx, (images, targets) in enumerate(stream_dataloader):

                # self.model.to(self.args.device)
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                # Given task_id=u for debugging
                outputs = self.model(images, task_id=None, mode='eval')

                loss = self.cross_entropy(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_loss.append(test_loss / (batch_idx + 1))
                total_acc.append(100. * correct / total)

                progress_bar(batch_idx, len(stream_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            test_loss, test_acc = np.average(total_loss), np.average(total_acc)
            self.acc[self.task_id, u] = test_acc
            self.lss[self.task_id, u] = test_loss

        acc = np.mean(self.acc[self.task_id, :self.task_id + 1])
        print('Average accuracy={:5.1f}%'.format(acc))

        output_path = '/home/luu/projects/cl_selective_nets/results/core50_DISN_Kmeans.txt'
        print('Save at ' + output_path)
        np.savetxt(output_path, self.acc, '%.4f')

    def train_k_mean(self):
        pass
        # self.model.eval()

        # for batch, (distilled_images, distilled_labels) in enumerate(self.distilled_dataloader):
        #
        #     task = torch.autograd.Variable(torch.LongTensor([self.task_id]).cuda())
        #
        #     distilled_images = distilled_images.to(self.args.device)
        #
        #     distilled_features = self.model(distilled_images, task)
        #
        #     self.stored_features[self.task_id].append(distilled_features)

        # # Concatenate all feature tensors and create a task ID list
        # all_features = []
        # for task_id, features in self.stored_features.items():
        #
        #     features = torch.stack(features, dim=0)
        #     all_features.append(features)
        #
        # # Convert to a single tensor
        # all_features_tensor = torch.cat(all_features, dim=0).view(-1, 10).to(self.args.device)
        #
        # # Number of clusters
        # k = self.task_id + 1
        #
        # # Perform k-means clustering
        # self.kmeans_model = KMeans(n_clusters=k, mode='euclidean', verbose=1)
        # self.kmeans_model.fit(all_features_tensor)

    def train_current_task(self, args, model, trainloader, epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            try:
                inputs, targets, _ = data
            except:
                inputs, targets = data

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # Forward current model
            task = torch.autograd.Variable(torch.LongTensor([self.task_id]).cuda())
            outputs = self.model(inputs, task, mode='train')

            # Store the features from the current task, only get the features from the last epoch
            # todo: get features from distilled images
            if len(self.stored_features[self.task_id]) <= 100 and epoch == self.args.n_epochs - 1:
                self.stored_features[self.task_id].append(outputs)

            loss = self.cross_entropy(outputs, targets)
            # print('loss = ', modified_ce_loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def eval_current_task(self, args, model, val_loader, test_task_id=None):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        total_loss = []
        total_acc = []

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
                outputs = self.model(inputs, task)

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

                progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return np.average(total_loss), np.average(total_acc)

    def _after_training_epoch(self, **kwargs):
        pass

    def train_dataset_adaptation(self, **kwargs):
        self.adapted_dataset = self.experience.dataset
        self.adapted_distilled_dataset = self.distilled_experience.dataset

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

        self.distilled_dataloader = DataLoader(self.adapted_distilled_dataset, batch_size=self.train_mb_size,
                                               shuffle=shuffle, pin_memory=pin_memory,
                                               persistent_workers=persistent_workers, sampler=None,
                                               num_workers=num_workers, collate_fn=self.collate_fn, drop_last=True)

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
            images, labels, task_id, _ = zip(*batch)

        # print(type(images), len(images))
        images = self.tuple_of_tensors_to_tensor(images)
        labels = torch.tensor(labels)  # Convert labels to tensor

        return images, labels


class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        KLD_loss = 0
        for k in range(len(x)):
            for l in range(len(x)):
                if l != k:
                    KLD_loss += self.criterion_KLD(F.log_softmax(x[k], dim=1), F.softmax(x[l], dim=1).detach())

        return KLD_loss


def DIWSN(args, real_dataset, distilled_dataset):
    # Get real train, val, and test dataset
    train_dataset, val_dataset, test_dataset = real_dataset

    # Get distilled dataset
    distilled_datasets_train, distilled_datasets_test = torch.utils.data.random_split(distilled_dataset,
                                                                                      lengths=[0.9, 0.1])

    from avalanche.benchmarks import nc_benchmark, ni_benchmark

    if args.scenario == 'nc':
        real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=5,
                                         task_labels=True)

        distilled_dataset_exps = nc_benchmark(distilled_datasets_train, distilled_datasets_test,
                                              n_experiences=args.n_experience,
                                              shuffle=True,
                                              seed=5,
                                              task_labels=True)

    elif args.scenario == 'ni':
        real_dataset_exps = ni_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=5,
                                         task_labels=True)

        distilled_dataset_exps = ni_benchmark(distilled_datasets_train, distilled_datasets_test,
                                              n_experiences=args.n_experience,
                                              shuffle=True,
                                              seed=5,
                                              task_labels=True)

    else:
        raise NotImplementedError(f"This scenario {args.scenario} is not implemented")

    # # Get balanced subset
    # balanced_subset = sample_subset(train_dataset, images_per_class=10)

    taskcla = []
    for task_id, real_experience in enumerate(real_dataset_exps.train_stream):
        # class_mask.append(real_experience.classes_in_this_experience)
        current_classes = real_experience.classes_in_this_experience
        taskcla.append((task_id, len(current_classes)))

    model = vit.ViTwithSelectiveSubnets(image_size=args.size, patch_size=args.patch, num_classes=10, dim=args.dimhead,
                                        depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1,
                                        num_tasks=args.n_experience)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Get strategy
    strategy = DynamicIntegratedContinualLearningWithSubnets(args=args, model=model, optimizer=optimizer,
                                                             test_entire_dataset=test_dataset,
                                                             eval_stream=real_dataset_exps.test_stream)

    # print(len(real_dataset_exps.test_stream[0].dataset))
    for task_id, (real_experience, distilled_experience) in enumerate(zip(real_dataset_exps.train_stream,
                                                                          distilled_dataset_exps.train_stream)):
        strategy.train(experiences=(real_experience, distilled_experience), task_id=task_id)

        # if task_id > 0:
        strategy.eval(exp_list=distilled_dataset_exps.train_stream)  # already defined, so non sense here

        # if task_id == 4:
        #     torch.save(strategy.stored_features, '/home/luu/projects/cl_selective_nets/results/features.pt')


def sample_subset(dataset, images_per_class=10):
    targets = np.array([target for _, target, _ in dataset])
    classes, class_counts = np.unique(targets, return_counts=True)

    indices_per_class = {class_: np.where(targets == class_)[0] for class_ in classes}
    sampled_indices = []

    for class_, indices in indices_per_class.items():
        if len(indices) > images_per_class:
            sampled_indices.extend(np.random.choice(indices, images_per_class, replace=False))
        else:
            sampled_indices.extend(indices)

    return Subset(dataset, sampled_indices)


"""
python3 main_cl.py --n_epochs 5 --cl_strategy DIWSN

"""
