import argparse

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


def baseline_upperbound():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')
    parser.add_argument('--datatype', default='real', type=str, help='choose between real and distilled')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--aug', default=True, type=bool, help='use randomaug')
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--model', default='vit')
    parser.add_argument('--dp', action='store_true', help='use data parallel')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
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

    args = parser.parse_args()

    start_epoch = 0
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_amp = not args.noamp

    if args.size == 32:
        mini = True
    elif args.size == 128:
        mini = False
    else:
        raise Exception("Choose image size 32 or 128")

    # Load real dataset
    cl_core50 = core50.load_core50(scenario='nc', n_experience=5, mini=mini, obj_lvl=False,
                                   dataset_root='/home/luu/DistilledDataset_ContinualLearning/core50/avalanche_core50')

    train_dataset, val_dataset, test_dataset = cl_core50

    # Load distilled dataset
    distilled_datasets = distilled_core50.get_distilled_core50()
    distilled_train_loader = torch.utils.data.DataLoader(distilled_datasets, batch_size=args.bs,
                                                         shuffle=True, num_workers=6)

    # Load real dataloader
    real_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=6)

    if args.model == 'vit':
        model = vit.ViT(image_size=args.size, patch_size=args.patch, num_classes=10, dim=args.dimhead, depth=6,
                        heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    elif args.model == 'vit_prompt':
        model = vit_prompt.get_prompt_model(img_size=args.size, patch_size=args.patch, num_classes=10,
                                            embed_dim=args.dimhead, num_heads=8, depth=6, drop_rate=0.1)
    else:
        raise Exception("Model is not defined")

    # For Multi-GPU
    if 'cuda' in args.device:
        print(args.device)
        if args.dp:
            print("using data parallel")
            model = torch.nn.DataParallel(model)  # make parallel
            cudnn.benchmark = True

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise Exception("Optimizer is not defined")

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    # use scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Use Aug
    if args.aug:
        loguru.logger.info("Use Random Aug")
        aug = RandAugment(n=0, m=29)
    else:
        aug = None

    list_loss = []
    list_acc = []

    model.cuda()

    # Choose datatype to train the baseline
    if args.datatype == 'real':
        train_loader = real_train_loader
    elif args.datatype == 'distilled':
        train_loader = distilled_train_loader
    else:
        raise Exception("Please choose the args.datatype is either real or distilled")

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        train_loss = train(args, train_loader, epoch, model, criterion, optimizer, scaler, aug=aug)
        val_loss, acc = test(args, val_loader, model, criterion)
        # test_loss, test_acc = test(args, test_loader, model, criterion)

        scheduler.step(epoch - 1)  # step cosine scheduling

        # list_loss.append(val_loss)
        # list_acc.append(acc)

    test_loss, test_acc = test(args, test_loader, model, criterion)
    final_acc = test_acc
    loguru.logger.info(f"Test accuracy is {final_acc}")

    if final_acc > 80.0:
        # Save model
        PATH = f"/home/luu/projects/cl_selective_nets/results/{args.model}-model_{final_acc:.1f}-accuracy_{args.datatype}-data.pth"
        torch.save(model, PATH)
    else:
        loguru.logger.warning("Have not saved the model due to low accuracy")


def test_pretrained_distilled_data():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')
    parser.add_argument('--datatype', default='real', type=str, help='choose between real and distilled')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--aug', default=True, type=bool, help='use randomaug')
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')

    args = parser.parse_args()

    # Load real dataset
    cl_core50 = core50.load_core50(scenario='nc', n_experience=5, mini=True, obj_lvl=False,
                                   dataset_root='/home/luu/DistilledDataset_ContinualLearning/core50/avalanche_core50')

    train_dataset, val_dataset, test_dataset = cl_core50

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=6)

    # Load distilled dataset
    distilled_datasets = distilled_core50.get_distilled_core50()
    distilled_train_loader = torch.utils.data.DataLoader(distilled_datasets, batch_size=128,
                                                         shuffle=True, num_workers=6)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_amp = not args.noamp

    # use scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Load model
    pretrained_model = vit.ViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=6,
                               heads=8,
                               mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    path = "/home/luu/projects/cl_selective_nets/results/vit-model_98.5-accuracy_real-data.pt"
    pretrained_model.load_state_dict(torch.load(path))
    # pretrained_model = torch.load(path)

    for param in pretrained_model.parameters():
        param.requires_grad = False

    for param in pretrained_model.mlp_head.parameters():  # Adjust attribute name as necessary
        param.requires_grad = True

    pretrained_model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    # Optimize only the parameters of the final layer
    optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

    pretrained_model.train()  # Set the model to training mode
    for epoch in range(20):  # Loop over the dataset multiple times
        train_loss = train(args, distilled_train_loader, epoch, pretrained_model, criterion, optimizer, scaler,
                           aug=None)
        val_loss, acc = test(args, val_loader, pretrained_model, criterion)
        # test_loss, test_acc = test(args, test_loader, pretrained_model, criterion)


if __name__ == '__main__':
    baseline_upperbound()
    # test_pretrained_distilled_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
