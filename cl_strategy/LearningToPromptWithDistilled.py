# https://avalanche-api.continualai.org/en/v0.5.0/generated/avalanche.training.LearningToPrompt.html#avalanche.training.LearningToPrompt
# https://avalanche-api.continualai.org/en/v0.5.0/_modules/avalanche/training/supervised/l2p.html#LearningToPrompt
# https://github.com/JH-LEE-KR/l2p-pytorch/blob/main/prompt.py
# https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
# https://github.com/google-research/l2p/blob/main/configs/core50_l2p.py
# https://blog.research.google/2022/04/learning-to-prompt-for-continual.html
import loguru
from avalanche.models.vit import simpleMLP, vit_tiny_patch16_224
from avalanche.training import LearningToPrompt
import torch.nn as nn
from typing import Optional, Union, List, Callable
import torch
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from timm.models.registry import register_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from torch.utils.data import DataLoader
import numpy as np
from timm.utils import accuracy
from typing import Iterable
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
from models import vit, vit_prompt
from train_epoch import train, test
from randomaug import RandAugment
from utils import progress_bar


def _experiences_parameter_as_iterable(experiences):
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]


class L2PWithDistilledData(SupervisedTemplate):
    def __init__(self, *, args, original_model, original_opt, criterion: nn.Module = nn.CrossEntropyLoss(),
                 train_mb_size: int = 32,
                 train_epochs: int = 10, eval_mb_size: Optional[int] = 32, device: Union[str, torch.device] = "cuda",
                 plugins: Optional[List["SupervisedPlugin"]] = None,
                 evaluator: Union[
                     EvaluationPlugin, Callable[[], EvaluationPlugin]
                 ] = default_evaluator,
                 eval_every: int = -1, peval_mode: str = "epoch", prompt_pool: bool = True, pool_size: int = 20,
                 prompt_length: int = 5, top_k: int = 2, lr: float = 0.01, sim_coefficient: float = 0.1,
                 prompt_key: bool = True, pretrained: bool = True, num_classes: int = 10, drop_rate: float = 0.1,
                 drop_path_rate: float = 0.0, embedding_key: str = "cls", prompt_init: str = "uniform",
                 batchwise_prompt: bool = False, head_type: str = "prompt", use_prompt_mask: bool = False,
                 train_prompt_mask: bool = False, use_cls_features: bool = True, use_mask: bool = True,
                 use_vit: bool = True, **kwargs):
        super().__init__(model=original_model, optimizer=original_opt, train_epochs=args.n_epochs,
                         train_mb_size=train_mb_size, eval_mb_size=train_mb_size, device=device, plugins=plugins,
                         evaluator=evaluator, eval_every=eval_every, **kwargs)
        self.clip_grad = 1.0
        self.class_mask = None
        self.distilled_experience = None
        self.adapted_distilled_dataset = None
        self.distilled_dataloader = None

        # Validation
        self.val_dataset = None
        self.val_dataloader = None

        # Training
        # use scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        # Use Aug
        if args.aug:
            loguru.logger.info("Use Random Aug")
            self.aug = RandAugment(n=0, m=29)
        else:
            self.aug = None

        # Prompts
        self.prompt_pool = prompt_pool
        self.shared_prompt_pool = args.shared_prompt_pool

        self.args = args
        self.num_classes = num_classes
        self.lr = lr
        self.sim_coefficient = sim_coefficient

        self.original_model = original_model
        self.prompt_model = vit_prompt.get_prompt_model(img_size=args.size, patch_size=args.patch, num_classes=10,
                                                        embed_dim=args.dimhead, num_heads=8, depth=6, drop_rate=0.1,
                                                        prompt_pool=prompt_pool, prompt_length=prompt_length,
                                                        prompt_key=prompt_key, top_k=top_k, prompt_init=prompt_init,
                                                        use_prompt_mask=use_prompt_mask, attn_drop_rate=0.1,
                                                        batchwise_prompt=batchwise_prompt, head_type=head_type,
                                                        pool_size=pool_size, embedding_key=embedding_key)

        self.prompt_model.to(args.device)
        self.original_model.to(args.device)

        # Optimizer
        self.prompt_optimizer = create_optimizer(args=args, model=self.prompt_model)
        self.original_optimizer = original_opt

        # Scheduler
        self.prompt_lr_scheduler, _ = create_scheduler(args, self.prompt_optimizer)
        self.original_lr_scheduler, _ = create_scheduler(args, self.prompt_optimizer)

        # Loss function
        self.criterion = criterion.to(device)

    def train(self, experiences, task_id=0, val_dataset=None, class_mask=None, **kwargs):
        """
        train phase
        :param class_mask:
        :param task_id: task id of each experience
        :param experiences: tuple of real_exp and distilled_exp
        :param val_dataset: dataset for evaluation
        :param kwargs: keyword arguments inherit from template
        :return:
        """

        self.experience = experiences[0]
        self.distilled_experience = experiences[1]
        self.mb_task_id = task_id
        self.class_mask = class_mask

        if not self._distributed_check:
            # Checks if the strategy elements are compatible with
            # distributed training
            self._check_distributed_training_compatibility()
            self._distributed_check = True

        self._before_training(**kwargs)  # Just print << Start of Training Phase >>

        # Validation dataset
        self.val_dataset = val_dataset
        self.make_eval_dataloader()

        self.train_dataset_adaptation()
        self.make_train_dataloader()
        self._before_training_exp(**kwargs)  # Transfer the prompt model's weights from the past to the current
        self._before_training_epoch(**kwargs)  # original model trains the current task
        self.training_epoch()  # use original model to get features, prompt model to train incremental tasks
        self._after_training_exp(**kwargs)  # wrap up
        # self.eval()  # Evaluate prompt model

    def _before_training_epoch(self, **kwargs):
        print(f"<< Start of training task {self.mb_task_id} >>")
        self.original_model.train()
        for epoch in range(0, self.args.n_epochs):
            train_loss = train(self.args, self.dataloader, epoch, self.original_model, self.criterion,
                               self.original_optimizer, self.scaler, aug=self.aug)
            val_loss, acc = test(self.args, self.val_dataloader, self.original_model, self.criterion)

        print(f"<< Finish training on task {self.mb_task_id} >>")
        print(" ")
        # For now I haven't reset the original model after finishing learning 1 task because it will forget anw

    def training_epoch(self, **kwargs):
        self.prompt_model.train()
        self.original_model.eval()
        train_loss = 0
        correct = 0
        total = 0
        print(f"<< Training the prompt model >>")
        for epoch in range(0, self.args.n_epochs):
            for batch_idx, (data_real, data_distilled) in enumerate(zip(self.dataloader, self.distilled_dataloader)):
                inputs_real, targets_real = data_real
                inputs_distilled, targets_distilled = data_distilled

                # Real data
                inputs_real = inputs_real.to(self.args.device, non_blocking=True)
                targets_real = targets_real.to(self.args.device, non_blocking=True)

                # Distilled data
                inputs_distilled = inputs_distilled.to(self.args.device, non_blocking=True)
                targets_distilled = targets_distilled.to(self.args.device, non_blocking=True)

                if self.args.choose_inputs == 'real':
                    inputs = inputs_real
                    targets = targets_real
                elif self.args.choose_inputs == 'distilled':
                    inputs = inputs_distilled
                    targets = targets_distilled
                else:
                    raise ValueError("Choose the correct args of choose_inputs --> valid choice [real, distilled]")

                # targets = targets_real

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(inputs)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None

                output = self.prompt_model(inputs, task_id=self.mb_task_id, cls_features=cls_features,
                                           train=True)
                logits = output['logits']

                # here is the trick to mask out classes of non-current tasks
                if self.args.train_mask and self.class_mask is not None:
                    mask = self.class_mask[self.mb_task_id]
                    not_mask = np.setdiff1d(np.arange(self.args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.args.device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                # todo: may use contrastive loss here
                loss = self.criterion(logits, targets)  # base criterion (CrossEntropyLoss)
                if self.args.pull_constraint and 'reduce_sim' in output:
                    loss = loss - self.args.pull_constraint_coeff * output['reduce_sim']

                train_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

                self.prompt_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), self.clip_grad)
                self.prompt_optimizer.step()

                progress_bar(batch_idx, len(self.dataloader), 'Loss: %.3f | Acc: %.3f%%'
                             % (train_loss/(batch_idx+1), 100.*correct/total))

    def _before_training_exp(self, **kwargs):

        # Transfer previous learned prompt params to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_pool:
            if self.mb_task_id > 0:
                prev_start = (self.mb_task_id - 1) * self.args.top_k
                prev_end = self.mb_task_id * self.args.top_k

                cur_start = prev_end
                cur_end = (self.mb_task_id + 1) * self.args.top_k

                if (prev_end > self.args.prompt_size) or (cur_end > self.args.prompt_size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        self.prompt_model.prompt.prompt.grad.zero_()
                        self.prompt_model.prompt.prompt[cur_idx] = self.prompt_model.prompt.prompt[prev_idx]
                        self.prompt_optimizer.param_groups[0]['params'] = self.prompt_model.parameters()
                        print(f"Prompt pool size = {self.prompt_model.prompt.prompt.shape}")
                        print(" ---- Prompts now are shared ---- ")
                        print(" ")

        # Transfer previous learned prompt params to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_key:
            if self.mb_task_id > 0:
                prev_start = (self.mb_task_id - 1) * self.args.top_k
                prev_end = self.mb_task_id * self.args.top_k

                cur_start = prev_end
                cur_end = (self.mb_task_id + 1) * self.args.top_k

                if (prev_end > self.args.prompt_size) or (cur_end > self.args.prompt_size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        self.prompt_model.prompt.prompt_key.grad.zero_()
                        self.prompt_model.prompt.prompt_key[cur_idx] = self.prompt_model.prompt.prompt_key[prev_idx]
                        self.prompt_optimizer.param_groups[0]['params'] = self.prompt_model.parameters()

        if self.mb_task_id > 0 and self.args.reinit_optimizer:
            self.prompt_optimizer = create_optimizer(args=self.args, model=self.prompt_model)

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
                                               num_workers=num_workers, collate_fn=self.collate_fn)

    def make_eval_dataloader(
            self,
            num_workers=0,
            shuffle=False,
            pin_memory=None,
            persistent_workers=False,
            **kwargs
    ):
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.train_mb_size, shuffle=shuffle,
                                         pin_memory=pin_memory, persistent_workers=persistent_workers,
                                         sampler=None, num_workers=num_workers, collate_fn=self.collate_fn)

    def eval(self, exp_list=None, **kwargs):
        if exp_list is None:  # Assuming exp_list contains test datasets for each task
            raise ValueError("exp_list containing test datasets for each task must be provided.")

        real_experiences_list = _experiences_parameter_as_iterable(exp_list)

        self.prompt_model.eval()
        self.original_model.eval()

        acc_matrix = np.zeros((len(real_experiences_list), 2))  # Assuming we're interested in Acc@1 and Acc@5
        print(f"<< Evaluating each task individually >>")

        for task_id, exp in enumerate(real_experiences_list[:self.mb_task_id+1]):
            test_dataloader = DataLoader(exp.dataset, batch_size=self.eval_mb_size, shuffle=False,
                                         pin_memory=True, num_workers=4, collate_fn=self.collate_fn)
            total = 0
            correct = 0
            test_loss = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_dataloader):
                    inputs = inputs.to(self.args.device, non_blocking=True)
                    targets = targets.to(self.args.device, non_blocking=True)

                    if self.original_model is not None:
                        output = self.original_model(inputs)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None

                    output = self.prompt_model(inputs, task_id=task_id, cls_features=cls_features,
                                               train=False)  # Ensure the model knows the task ID
                    logits = output['logits']

                    if self.args.scenario == 'nc' and self.class_mask is not None:
                        # adding mask to output logits
                        mask = self.class_mask[task_id]
                        mask = torch.tensor(mask, dtype=torch.int64).to(self.args.device)
                        logits_mask = torch.ones_like(logits, device=self.args.device) * float('-inf')
                        logits_mask = logits_mask.index_fill(1, mask, 0.0)
                        # print(logits_mask)
                        logits = logits + logits_mask

                    loss = self.criterion(logits, targets)
                    test_loss += loss.item()
                    _, predicted = logits.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc1 = 100. * correct / total
            acc_matrix[task_id] = [acc1, test_loss / len(test_dataloader)]  # Store Acc@1 and Test Loss for each task

            print(f"Task {task_id}: Loss: {test_loss / len(test_dataloader):.3f} | Acc@1: {acc1:.3f}%")

        # print("Evaluation completed for all tasks.")
        return acc_matrix

    def mb_task_id(self):
        return self.mb_task_id

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


def L2PWithDistilled(args, real_dataset, distilled_dataset):
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

    # Get original model
    original_model = vit_prompt.get_prompt_model(img_size=args.size, patch_size=args.patch,
                                                 num_classes=10, embed_dim=args.dimhead, num_heads=8, depth=6,
                                                 drop_rate=0.1)

    original_optimizer = create_optimizer(args=args, model=original_model)

    # Get strategy
    strategy = L2PWithDistilledData(args=args, original_model=original_model, original_opt=original_optimizer)

    # Create class mask
    class_mask = list() if args.scenario == 'nc' or args.train_mask else None
    args.nb_classes = 10  # todo: change later for number of class = 50 of obj_level=True

    for task_id, real_experience in enumerate(real_dataset_exps.train_stream):
        class_mask.append(real_experience.classes_in_this_experience)

    # print(class_mask)

    for task_id, (real_experience, distilled_experience) in enumerate(zip(real_dataset_exps.train_stream,
                                                                          distilled_dataset_exps.train_stream)):
        strategy.train(experiences=(real_experience, distilled_experience), val_dataset=test_dataset,
                       task_id=task_id, class_mask=class_mask)

        acc_matrices = strategy.eval(exp_list=real_dataset_exps.test_stream)

        # save_acc_matrix_to_file(acc_matrices,
        #                         file_path=f'/home/luu/projects/cl_selective_nets/results/'
        #                                   f'{args.choose_inputs}_evaluate_all_tasks.txt',
        #                         task_id=task_id)


def save_acc_matrix_to_file(acc_matrix, file_path, task_id):
    """
    Append the accuracy matrix to a text file for each task.

    Parameters:
    - acc_matrix: A NumPy array containing the accuracy for each task.
    - file_path: The path of the file where the matrix should be saved.
    - task_id: The ID of the current task.
    """
    import os
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    matrix_str = np.array2string(acc_matrix, precision=3, separator=', ')

    with open(file_path, 'a') as file:
        file.write(f"\nAccuracy Matrix until Task {task_id}:\n")
        file.write(matrix_str)


"""
python3 main_cl.py --n_epochs 1 --cl_strategy l2p_distilled --model prompt_vit

"""
