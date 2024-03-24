import argparse

import avalanche.training
import loguru

from models import vit, vit_prompt
from get_datasets import core50, distilled_core50
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import time
from train_epoch import train, test
from randomaug import RandAugment
from cl_strategy import LearningToPromptWithDistilled, DynamicIntegrated, DynamicIntegratedSubnets


def run_cl_strategy():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')

    # Continual learning parameters
    parser.add_argument('--cl_strategy', default='l2p_distilled', type=str, help='choose cl strategy')
    parser.add_argument('--scenario', default='nc', type=str, help='nc or ni')
    parser.add_argument('--n_experience', default=5, type=int, help='number of experiences')
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--nb_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--choose_inputs', default='real', type=str, help='real or distilled')

    # EWC parameters
    parser.add_argument('--lamb', default=1000, type=float)
    parser.add_argument('--lamb_kld', default=0.002, type=float)
    parser.add_argument('--lamb_af', default=1e-8, type=float)

    # Model parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--aug', default=True, type=bool, help='use randomaug')
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--model', default='vit_prompt', help='[vit, vit_prompt]')
    parser.add_argument('--dp', action='store_true', help='use data parallel')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--size', type=int, default=32, help='image size')
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default=512, type=int)
    parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # cpp
    parser.add_argument('--add_pos_proto', default=False, type=bool, help='whether add positive prototypes')
    parser.add_argument('--add_neg_proto', default=True, type=bool, help='whether add negative prototypes')
    parser.add_argument('--add_uniformity', default=False, type=bool, help='whether add uniformity')
    parser.add_argument('--temp', default=0.6, type=float, help='temperature')
    parser.add_argument('--neg_temp', default=0.6, type=float, help='temperature for neg components')
    parser.add_argument('--proto_aug_bs', default=256, type=int, help='bs for proto augmentation')
    parser.add_argument('--proto_trans', default=True, type=bool, help='whether use proto transformaton')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "constant"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Prompt parameters
    parser.add_argument('--prompt_pool', default=True, type=bool, )
    parser.add_argument('--prompt_size', default=5, type=int, )
    parser.add_argument('--length', default=5, type=int, )
    parser.add_argument('--top_k', default=2, type=int, )
    parser.add_argument('--shared_prompt_pool', default=True, type=bool)
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--pull_constraint', default=False)
    parser.add_argument('--pull_constraint_coeff', default=0.5, type=float)

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_amp = not args.noamp

    if args.size == 32:
        mini = True
    elif args.size == 128:
        mini = False
    else:
        raise Exception("Choose image size 32 or 128")

    # Load real dataset
    cl_core50 = core50.load_core50(args=args, scenario=args.scenario, mini=mini, obj_lvl=False,
                                   dataset_root='/home/luu/DistilledDataset_ContinualLearning/core50/avalanche_core50')

    # Load distilled dataset
    distilled_datasets = distilled_core50.get_distilled_core50()

    # Load cl strategy
    if args.cl_strategy == 'l2p_distilled':
        LearningToPromptWithDistilled.L2PWithDistilled(args=args, real_dataset=cl_core50,
                                                       distilled_dataset=distilled_datasets)
    elif args.cl_strategy == 'DICL':
        DynamicIntegrated.DICL(args=args, real_dataset=cl_core50, distilled_dataset=distilled_datasets)
    elif args.cl_strategy == 'DIWSN':
        DynamicIntegratedSubnets.DIWSN(args=args, real_dataset=cl_core50, distilled_dataset=distilled_datasets)
    else:
        raise NotImplemented("Strategy is not implemented yet")


def avalanche_vit_prompt():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')

    # Continual learning parameters
    parser.add_argument('--cl_strategy', default='l2p_distilled', type=str, help='choose cl strategy')
    parser.add_argument('--scenario', default='nc', type=str, help='nc or ni')
    parser.add_argument('--n_experience', default=5, type=int, help='number of experiences')
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--nb_classes', default=10, type=int, help='number of classes')

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from avalanche.benchmarks.classic import CORe50
    from avalanche.logging import InteractiveLogger
    from avalanche.training.plugins import EvaluationPlugin
    from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
        loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
    from torchvision.transforms import v2

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=False),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=False),
        forgetting_metrics(experience=True, stream=False),
        confusion_matrix_metrics(num_classes=10, save_image=False, stream=False),
        loggers=[interactive_logger]
    )

    # --- TRANSFORMATIONS
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = v2.Compose(
        [v2.Resize(224), v2.ToTensor(), normalize]
    )
    test_transform = v2.Compose(
        [v2.Resize(224), v2.ToTensor(), normalize]
    )

    # Load real dataset
    dataset_root = '/home/luu/DistilledDataset_ContinualLearning/core50/avalanche_core50'
    scenario = CORe50(scenario='nc', object_lvl=False, mini=False, dataset_root=dataset_root,
                      train_transform=train_transform, eval_transform=test_transform)

    cl_strategy = avalanche.training.LearningToPrompt(model_name='vit_base_patch32_224', train_mb_size=4,
                                                      train_epochs=12, eval_mb_size=4,
                                                      device='cuda', pretrained=False, num_classes=10,
                                                      evaluator=eval_plugin, use_cls_features=False)

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream))


if __name__ == '__main__':
    run_cl_strategy()
    # avalanche_vit_prompt()
