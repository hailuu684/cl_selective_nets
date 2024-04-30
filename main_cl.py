import argparse

import avalanche.training
import loguru
from avalanche.benchmarks.utils import SupervisedClassificationDataset

from utils import str2bool
from models import vit, vit_prompt
from get_datasets import core50, distilled_core50, cifar10, intel_image, stl_10
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import time
from train_epoch import train, test
from randomaug import RandAugment
from cl_strategy import LearningToPromptWithDistilled, DynamicIntegrated, DynamicIntegratedSubnets, \
    DynamicIntegratedSubnetsTaskPredictive, DI_SubnetsTP_contrasitive, cl_reminder


def run_cl_strategy():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')

    # Dataset settings
    parser.add_argument('--dataset', default='core50', type=str, help='choose dataset [core50, cifar10]')
    parser.add_argument('--obj_lvl', default=False, type=str2bool, help='10 or 50 classes for core50')

    # Loss functions
    # SSIM is not good as mse
    parser.add_argument('--reconstruction_loss', default='mse', type=str, help='Loss function',
                        choices=['mse', 'ssim', 'perceptual_loss'])

    # Model
    # core50: (ae_simple_cnn, mse loss (200 epochs) or perceptual_loss (30 epochs)) - 384 latent
    # cifar10: (ae_simple_cnn, perceptual_loss (30 epochs)) - 256 latent
    parser.add_argument('--AE_model', default='ae_simple_cnn', type=str, help='Architectures',
                        choices=['ae_simple_cnn', 'ae_resnet18',
                                 'ae_resnet18_pretrained', 'ae_simple_cnn_pretrained',
                                 'ae_resnet50_pretrained', 'ae_simple_cnn_unet',
                                 'ae_resnet50_encoder_decoder_pretrained',
                                 'cl_reminder'])

    # Continual learning parameters
    parser.add_argument('--cl_strategy', default='DI_with_task_predictive', type=str, help='choose cl strategy')
    parser.add_argument('--scenario', default='nc', type=str, help='nc or ni')
    parser.add_argument('--n_experience', default=5, type=int, help='number of experiences')
    parser.add_argument('--n_epochs', default=5, type=int, help='number of experiences')
    parser.add_argument('--size', default=128, type=int, help='size of image')
    parser.add_argument('--nb_classes', default=10, type=int, help='number of classes')

    # Model parameters
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--aug', default=True, type=bool, help='use randomaug')
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "constant"')

    # Image reconstruction
    parser.add_argument('--latent_dim', default=384, type=int, help='latent dimension: 128, 256, 384, 512')

    # Wandb
    parser.add_argument('--use_wandb', default=False, type=str2bool, help='Use wandb or not')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_amp = not args.noamp

    # Load real dataset
    # let the scenario='ni' is because we will split dataset into nc or ni later in the strategy code
    # avalanche doesn't support to split into how many experiences we want, so we need to separate later

    if args.dataset == 'core50':

        if args.size == 32:
            mini = True
        else:
            mini = False

        if args.obj_lvl:
            loguru.logger.info("Core50 dataset, object level is enable, number of classes is set to 50")
            args.nb_classes = 50
        else:
            loguru.logger.info("Core50 dataset, object level is disable, number of classes is set to 10")
            args.nb_classes = 10

        dataset_cl = core50.load_core50(args=args, scenario='ni', mini=mini, obj_lvl=args.obj_lvl,
                                        dataset_root='/home/luu/DistilledDataset_ContinualLearning'
                                                     '/core50/avalanche_core50')

        # Load distilled dataset
        distilled_datasets = distilled_core50.get_distilled_core50(args, use_transform=True)

    elif args.dataset == 'cifar10':
        dataset_cl = cifar10.get_cifar10(args)

        # Load distilled dataset, temporarily load distilled core50, because currently no use distilled images
        distilled_datasets = cifar10.get_distilled_cifar10(args)

    elif args.dataset == 'intel_images':

        # train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train'
        # test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test'
        #
        # # Pred folder is only for prediction, dont have labels
        # # pred_path = '/home/luu/projects/datasets/intel-image-classification/seg_pred'
        #
        # train_dataset, val_dataset, test_dataset = intel_image.get_intel_images(train_path=train_path, val_path=test_path)
        #
        dataset_cl = None, None, None

        distilled_datasets = distilled_core50.get_distilled_core50(args)

    elif args.dataset == 'stl_10':
        dataset_cl = stl_10.get_stl_10(args)
        distilled_datasets = distilled_core50.get_distilled_core50(args)

    else:
        raise NotImplemented(f"The dataset {args.dataset} is not implemented")

    # Load cl strategy
    if args.cl_strategy == 'l2p_distilled':
        LearningToPromptWithDistilled.L2PWithDistilled(args=args, real_dataset=dataset_cl,
                                                       distilled_dataset=distilled_datasets)
    elif args.cl_strategy == 'DICL':
        DynamicIntegrated.DICL(args=args, real_dataset=dataset_cl, distilled_dataset=distilled_datasets)
    elif args.cl_strategy == 'DIWSN':
        DynamicIntegratedSubnets.DIWSN(args=args, real_dataset=dataset_cl, distilled_dataset=distilled_datasets)
    elif args.cl_strategy == 'DI_with_task_predictive':
        DynamicIntegratedSubnetsTaskPredictive.DIWSN_task_predictive(args=args, real_dataset=dataset_cl,
                                                                     distilled_dataset=distilled_datasets)
        # DynamicIntegratedSubnetsTaskPredictive.test_pretrained_model(args, dataset_cl)
    elif args.cl_strategy == 'DI_with_task_predictive_contrastive':
        DI_SubnetsTP_contrasitive.DIWSN_task_predictive(args=args, real_dataset=None,
                                                        distilled_dataset=distilled_datasets)
    elif args.cl_strategy == 'cl_reminder':
        cl_reminder.cl_reminder(args=args, real_dataset=dataset_cl, distilled_dataset=distilled_datasets)
    else:
        raise NotImplemented("Strategy is not implemented yet")


if __name__ == '__main__':
    run_cl_strategy()
    # avalanche_vit_prompt()
