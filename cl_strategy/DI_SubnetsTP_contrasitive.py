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

sys.path.append("../")
from models import ContrastiveModel, simple_cnn, AE, model_utils
from train_epoch import train, test
from randomaug import RandAugment
from utils import progress_bar
from loss_functions import ssim, perceptual_loss
from get_datasets import cifar10


def _experiences_parameter_as_iterable(experiences):
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]


class DynamicIntegratedContinualLearningWithSubnets(SupervisedTemplate):
    def __init__(self, args, model, optimizer, test_entire_dataset, eval_stream, wandb, **kwargs):
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

        # use wandb
        self.wandb = wandb
        self.ce_loss = 0
        self.reconstruction_loss = 0
        self.val_loss = 0

        self.train_acc = 0
        self.val_acc = 0

        # perceptual loss
        self.perceptual_loss = perceptual_loss.PerceptualLoss().to('cuda') if args.device else \
            perceptual_loss.PerceptualLoss()

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
            val_loss, val_acc = self.eval_current_task(self.args, self.model,
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

        print(f"<< Finish training on task {self.task_id} >>")
        print(" ")

    def eval_agnostic(self):
        self.model.eval()
        all_test_stream = self.eval_all_stream_dataset

        all_eval_dataset = []

        for test_stream in all_test_stream:
            all_eval_dataset.append(test_stream.dataset)

        test_dataset_all_tasks = concat_datasets(all_eval_dataset)
        stream_dataloader = DataLoader(test_dataset_all_tasks,
                                       batch_size=1, shuffle=True,  # Evaluate 1 image
                                       pin_memory=True, num_workers=6, collate_fn=self.collate_fn)

        test_loss = 0
        correct = 0
        total = 0

        total_loss = []
        total_acc = []

        for batch_idx, (images, targets, _) in enumerate(stream_dataloader):
            # self.model.to(self.args.device)
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)

            # Given task_id=u for debugging
            outputs = self.model(images, task_id=None)

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

        output_path = f'/home/luu/projects/cl_selective_nets/results/{self.args.dataset}_DISN_task-agnostic_reconstruction_score.txt'
        print('Save at ' + output_path)
        np.savetxt(output_path, self.acc, '%.4f')

        if self.wandb:
            tag_val_task_agnostic_loss = 'Agnostic-Loss'
            tag_val_task_agnostic_acc = 'Agnostic-Acc'

            self.wandb.log({tag_val_task_agnostic_loss: test_loss,
                            tag_val_task_agnostic_acc: test_acc})

    def calculate_accuracy(self, logits, labels):
        _, predicted = torch.max(logits, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def train_current_task(self, args, model, trainloader, epoch):
        model.train()
        epoch_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_contrastive_loss = 0.0
        correct = 0
        total = 0
        temperature = 0.5
        c, s = 0, 0

        for batch_idx, data in enumerate(trainloader):
            stacked_image, label, _ = data

            d = stacked_image.size()
            x = stacked_image.view(d[0] * 2, d[2], d[3], d[4]).to(self.args.device)
            label = label.to(self.args.device)

            self.optimizer.zero_grad()

            classification_logits, p = model(x, task_id=self.task_id, return_projection=True)
            cont_loss = ContrastiveModel.nt_xent_loss(p, temperature=temperature)

            # Calculate Classification Loss
            class_loss_1 = self.cross_entropy(classification_logits[:self.train_mb_size], label)
            class_loss_2 = self.cross_entropy(classification_logits[self.train_mb_size:], label)

            class_loss = (class_loss_1 + class_loss_2) * 0.5
            total_loss = class_loss + cont_loss

            s = ((s * c) + (float(cont_loss) * len(p))) / (c + len(p))
            c += len(p)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss += total_loss.item()
            epoch_classification_loss += class_loss.item()
            epoch_contrastive_loss += cont_loss.item()

            _, predicted = torch.max(classification_logits.data, 1)

            concat_label = torch.cat((label, label), dim=0)

            total += concat_label.size(0)
            correct += (predicted == concat_label).sum().item()

        epoch_acc = 100 * correct / total
        print(
            f'Epoch {epoch + 1}: Total Loss: {epoch_loss / len(trainloader):.4f}, '
            f'Classification Loss: {epoch_classification_loss / len(trainloader):.4f}, '
            f'Contrastive Loss: {epoch_contrastive_loss / len(trainloader):.4f}, '
            f'Acc: {epoch_acc:.2f}%')

        self.ce_loss = epoch_classification_loss / len(trainloader)
        self.train_acc = epoch_acc

    def eval_current_task(self, args, model, val_loader, test_task_id=None):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        val_reconstruction_loss = 0
        total_loss = []
        total_acc = []
        total_rec_loss = []

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
                classification_logits = model(inputs, return_projection=False, task_id=task)

                # # Trying to make task ID predictions
                # pred_taskid, outputs = self.model(inputs, task, return_experts=False, return_task_pred=True)
                # print(torch.argmax(pred_taskid).item())
                loss = self.cross_entropy(classification_logits, targets)

                test_loss += loss.item()
                _, predicted = classification_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                total_loss.append(test_loss / (batch_idx + 1))
                total_acc.append(100. * correct / total)

                progress_bar(batch_idx, len(val_loader),
                             'CE Loss: %.3f| Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print(" ")
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
                                     sampler=None, num_workers=num_workers, collate_fn=self.collate_fn,
                                     drop_last=True)

        # self.dataloader = cifar10.get_contrastive_pairs_loader(self.adapted_dataset, batch_size=256)

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

        # self.eval_each_stream_dataloader = model_utils.get_contrastive_pairs_loader(
        #     self.eval_all_stream_dataset[self.task_id].dataset, batch_size=32)

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

    def contrastive_loss(self, embedding1, embedding2, label, margin=1.0):
        distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          (label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
        return loss


tf_tr = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

tf_de = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

tf_te = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

from PIL import Image


class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def __getitem__(self, idx):
        if not self.train:
            # Directly return the result of the parent class's __getitem__ if not in training mode
            return super().__getitem__(idx)

        # Access the image data
        img = self.data[idx]  # Image data at index idx
        img = Image.fromarray(img).convert(
            'RGB')  # Convert the numpy array to a PIL Image and ensure it's in RGB format

        # Access the label
        label = self.targets[idx]  # Access the label using self.targets, assuming the labels are stored there

        # Apply the transform to the image twice, if a transform is specified
        if self.transform is not None:
            imgs = [self.transform(img), self.transform(img)]
        else:
            imgs = [img, img]  # Just duplicate the original image if no transform is specified

        # Stack the images and return them with the label
        return torch.stack(imgs), label


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

            name=f"{args.dataset}-dataset_scenario-{args.cl_strategy}" + f'_{args.latent_dim}-latent-dim' + f'_{args.n_epochs}-epochs',

            entity='luutunghai'
        )

    # # Get real train, val, and test dataset
    # train_dataset, val_dataset, test_dataset = real_dataset

    root = '/home/luu/DistilledDataset_ContinualLearning/data'

    train_dataset = CustomCIFAR10(root=root, train=True, transform=tf_tr, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=tf_de, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=tf_te, download=True)

    # Get distilled dataset
    distilled_datasets_train, distilled_datasets_test = torch.utils.data.random_split(distilled_dataset,
                                                                                      lengths=[0.9, 0.1])

    from avalanche.benchmarks import nc_benchmark, ni_benchmark

    seed = 6
    if args.scenario == 'nc':
        real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=seed,
                                         task_labels=True)

        distilled_dataset_exps = nc_benchmark(distilled_datasets_train, distilled_datasets_test,
                                              n_experiences=args.n_experience,
                                              shuffle=True,
                                              seed=seed,
                                              task_labels=True)

    elif args.scenario == 'ni':
        real_dataset_exps = ni_benchmark(train_dataset, val_dataset,
                                         n_experiences=args.n_experience, shuffle=True,
                                         seed=seed,
                                         task_labels=True)

        distilled_dataset_exps = ni_benchmark(distilled_datasets_train, distilled_datasets_test,
                                              n_experiences=args.n_experience,
                                              shuffle=True,
                                              seed=seed,
                                              task_labels=True)

    else:
        raise NotImplementedError(f"This scenario {args.scenario} is not implemented")

    # latent_dim = 256, 384 (best), 512
    model = ContrastiveModel.ContrastiveWithSelectiveSubnets(args=args, num_input_channels=3,
                                                             latent_dim=args.latent_dim, num_classes=args.nb_classes,
                                                             num_child_models=args.n_experience)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Get strategy
    strategy = DynamicIntegratedContinualLearningWithSubnets(args=args, model=model, optimizer=optimizer,
                                                             test_entire_dataset=test_dataset,
                                                             eval_stream=real_dataset_exps.test_stream,
                                                             wandb=wandb if args.use_wandb is True else None)

    # print(len(real_dataset_exps.test_stream[0].dataset))
    for task_id, (real_experience, distilled_experience) in enumerate(zip(real_dataset_exps.train_stream,
                                                                          distilled_dataset_exps.train_stream)):
        strategy.train(experiences=(real_experience, distilled_experience), task_id=task_id)

        # if task_id > 0:
        # strategy.eval(exp_list=distilled_dataset_exps.train_stream)  # already defined, so non sense here
        strategy.eval_agnostic()

        # if task_id == 4:
        #     torch.save(strategy.stored_features, '/home/luu/projects/cl_selective_nets/results/features.pt'

    torch.save(model.state_dict(), f'/home/luu/projects/cl_selective_nets/results'
                                   f'/{args.dataset}_DISN_task-agnostic_reconstruction_score.pt')

    if args.use_wandb is True:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


"""
python3 main_cl.py --n_epochs 5 --cl_strategy DI_with_task_predictive

seed = 5
python3 main_cl.py --dataset core50 --n_epochs 200 --latent_dim 128 --cl_strategy DI_with_task_predictive --lr 0.001 --use_wandb True

seed = 6
python3 main_cl.py --dataset cifar10 --n_epochs 30 --latent_dim 256 --cl_strategy DI_with_task_predictive --lr 0.001 --use_wandb false --reconstruction_loss perceptual_loss --AE_model ae_resnet18

"""
