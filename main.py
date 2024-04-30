import argparse
import copy
import loguru
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import concat_datasets
from torch.utils.data import DataLoader

from models import vit, vit_prompt, simple_cnn, resnet, MobileNetV2, AEResnet50, VAE
from get_datasets import core50, distilled_core50, stl_10, intel_image, cifar10
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import time
from train_epoch import train, test, train_img_reconstruction, test_img_reconstruction, \
    train_encoder_classifier, train_decoder, save_and_average_images, compare_features, calculate_cosine_similarity
from randomaug import RandAugment
import numpy as np

def baseline_upperbound():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')
    parser.add_argument('--datatype', default='real', type=str, help='choose between real and distilled')
    parser.add_argument('--dataset', default='core50', type=str, help='choose dataset')
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
    else:
        mini = False

    # Load real dataset

    if args.dataset == 'core50':
        obj_lvl = False
        cl_dataset = core50.load_core50(args=args, scenario='ni', mini=mini, obj_lvl=obj_lvl,
                                        dataset_root='/home/luu/DistilledDataset_ContinualLearning'
                                                     '/core50/avalanche_core50')

        num_classes = 10

        if obj_lvl is True:
            num_classes = 50

        # Load distilled dataset
        distilled_datasets = distilled_core50.get_distilled_core50(args, use_transform=True)
        distilled_train_loader = torch.utils.data.DataLoader(distilled_datasets, batch_size=args.bs,
                                                             shuffle=True, num_workers=6)

    elif args.dataset == 'stl_10':
        cl_dataset = stl_10.get_stl_10(args=args)
        num_classes = 10

        distilled_train_loader = None

    elif args.dataset == 'intel_images':
        train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train/seg_train'
        test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test/seg_test'

        cl_dataset = intel_image.get_intel_images(args=args, train_path=train_path, val_path=test_path)
        num_classes = 6

        distilled_train_loader = None

    elif args.dataset == 'cifar10':
        cl_dataset = cifar10.get_cifar10(args)
        num_classes = 10

        distilled_datasets = cifar10.get_distilled_cifar10(args)
        distilled_train_loader = torch.utils.data.DataLoader(distilled_datasets, batch_size=args.bs,
                                                             shuffle=True, num_workers=0)
    else:
        raise Exception(f"Not implemented {args.dataset} dataset yet")

    train_dataset, val_dataset, test_dataset = cl_dataset

    # Load real dataloader
    real_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=6)

    if args.model == 'vit':
        model = vit.ViT(image_size=args.size, patch_size=args.patch, num_classes=num_classes, dim=args.dimhead, depth=6,
                        heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    elif args.model == 'simple_cnn':
        model = simple_cnn.CNN(num_classes=num_classes)
    elif args.model == 'resnet50':
        model = resnet.resnet50(pretrained=True, num_classes=num_classes)

    elif args.model == 'mobilenetv2':
        model = MobileNetV2.MobileNetV2(n_class=num_classes, input_size=args.size)

        # loaded_dict = torch.load('/home/luu/projects/cl_selective_nets/pretrained_model/model_mobilenetv2.pth')  # add map_location='cpu' if no gpu
        #
        # state_dict = {k: v for k, v in loaded_dict.items() if k in model.state_dict()}
        # state_dict["classifier.1.weight"] = model.state_dict()["classifier.1.weight"]
        # state_dict["classifier.1.bias"] = model.state_dict()["classifier.1.bias"]
        # model.load_state_dict(state_dict)

    else:
        raise Exception("Model is not defined")

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

    # task_id = 1
    # train_loader, val_loader, test_loader_cl = get_single_cl_dataloader(train_dataset, val_dataset, task=task_id)

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        train_loss = train(args, train_loader, epoch, model, criterion, optimizer, scaler, aug=aug)
        val_loss, acc = test(args, val_loader, model, criterion)
        # test_loss, test_acc = test(args, test_loader, model, criterion)

        scheduler.step(epoch - 1)  # step cosine scheduling

        # list_loss.append(val_loss)
        # list_acc.append(acc)

    # test_loss, test_acc = test(args, test_loader, model, criterion)
    # final_acc = test_acc
    # loguru.logger.info(f"Test accuracy is {final_acc}")

    # test_loss, test_acc = test(args, test_loader_cl, model, criterion)
    # loguru.logger.info(f"Test accuracy on task 1 is {final_acc}")

    # torch.save(model.state_dict(), f'/home/luu/projects/cl_selective_nets/results'
    #                                f'/{args.dataset}_{args.model}-model_task-{task_id}.pt')


def baseline_image_reconstruction_upperbound():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')
    parser.add_argument('--datatype', default='real', type=str, help='choose between real and distilled')
    parser.add_argument('--dataset', default='core50', type=str, help='choose dataset')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
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
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.use_amp = not args.noamp

    if args.size == 32:
        mini = True
    else:
        mini = False

    # Load real dataset

    if args.dataset == 'core50':
        obj_lvl = False
        cl_dataset = core50.load_core50(args=args, scenario='ni', mini=mini, obj_lvl=obj_lvl,
                                        dataset_root='/home/luu/DistilledDataset_ContinualLearning'
                                                     '/core50/avalanche_core50')

        num_classes = 10

        if obj_lvl is True:
            num_classes = 50

    elif args.dataset == 'stl_10':
        cl_dataset = stl_10.get_stl_10(args=args)
        num_classes = 10

    elif args.dataset == 'intel_images':
        train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train/seg_train'
        test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test/seg_test'

        cl_dataset = intel_image.get_intel_images(args=args, train_path=train_path, val_path=test_path)
        num_classes = 6
    else:
        raise Exception(f"Not implemented {args.dataset} dataset yet")

    train_dataset, val_dataset, test_dataset = cl_dataset

    # Load distilled dataset
    distilled_datasets = distilled_core50.get_distilled_core50(args)
    distilled_train_loader = torch.utils.data.DataLoader(distilled_datasets, batch_size=args.bs,
                                                         shuffle=True, num_workers=6)

    # Load real dataloader
    real_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=6)

    if args.model == 'pretrained_decoder_resnet50':
        model = AEResnet50.ResNet50Autoencoder(latent_dim=2048, num_classes=num_classes,
                                               output_size=args.size, use_pretrained_decoder=True)

        # Load Pretrained
        pretrained_autoencoder_path = '/home/luu/projects/cl_selective_nets/checkpoints/objects365-resnet50.pth'

        checkpoint = torch.load(pretrained_autoencoder_path, map_location=args.device)

        decoder_state_dict = {key[len("module.decoder."):]: value for key, value in checkpoint['state_dict'].items()
                              if key.startswith("module.decoder.")}

        # Load the state dict into your model's decoder
        model.decoder.load_state_dict(decoder_state_dict, strict=False)

    elif args.model == 'ae_resnet50_pretrained':  # Not working
        model = AEResnet50.ResNet50Autoencoder(latent_dim=2048, num_classes=num_classes, output_size=args.size,
                                               use_pretrained_decoder=False)

    elif args.model == 'pretrained_resnet50_unet':  # got nan
        model = AEResnet50.UNetWithResnet50Encoder(num_classes=num_classes)
    elif args.model == 'vae':
        model = VAE.ResNet_VAE(args=args, num_classes=num_classes, fc_hidden1=1024, fc_hidden2=1024, CNN_embed_dim=384)
    else:
        raise NotImplementedError

    # # For Multi-GPU
    # if 'cuda' in args.device:
    #     print(args.device)
    #     if args.dp:
    #         print("using data parallel")
    #         model = torch.nn.DataParallel(model)  # make parallel
    #         cudnn.benchmark = True

    # Loss is MSE
    criterion = nn.MSELoss()

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

    model.to(args.device)

    # Choose datatype to train the baseline
    if args.datatype == 'real':
        train_loader = real_train_loader
    elif args.datatype == 'distilled':
        train_loader = distilled_train_loader
    else:
        raise Exception("Please choose the args.datatype is either real or distilled")

    # for epoch in range(start_epoch, args.n_epochs):
    #     start = time.time()
    #     train_loss = train_img_reconstruction(args, train_loader, epoch, model, criterion, optimizer, aug=aug)
    #     val_loss = test_img_reconstruction(args, val_loader, model, criterion)
    #     scheduler.step(epoch - 1)  # step cosine scheduling

    task_id = 0
    loguru.logger.info(f'Loading data task id: {task_id}')
    train_loader_0, val_loader_0, test_loader_cl_0, cl_val_dataset_0 = get_single_cl_dataloader(train_dataset, val_dataset, task=task_id)
    #
    # saved_model_path = f'/home/luu/projects/cl_selective_nets/results' \
    #                    f'/{args.dataset}_{args.model}_autoencoder_{task_id}.pt'
    #
    # train_encoder_classifier(args=args, model=model, train_loader=train_loader_0, val_loader=val_loader_0, scaler=scaler)
    # train_decoder(args=args, model=model, train_loader=train_loader_0, val_loader=val_loader_0)

    loguru.logger.info('Loading pretrained model')
    model_0 = copy.deepcopy(model)
    model_0.load_state_dict(torch.load(f'/home/luu/projects/cl_selective_nets/results'
                                       f'/{args.dataset}_{args.model}_autoencoder_0.pt'))

    average_images_0 = save_and_average_images(args=args, model=model_0, train_loader=train_loader_0)
    #
    # # Save model 0
    # torch.save(model.state_dict(), saved_model_path)
    #
    # # Free encoder after initial training
    # for param in model.encoder.parameters():
    #     param.requires_grad = True

    task_id = 1
    loguru.logger.info(f'Loading data task id: {task_id}')
    train_loader_1, val_loader_1, test_loader_cl_1, cl_val_dataset_1 = get_single_cl_dataloader(train_dataset,
                                                                                                val_dataset, task=task_id)
    # train_encoder_classifier(args=args, model=model, train_loader=train_loader_1, val_loader=val_loader, scaler=scaler)
    # train_decoder(args=args, model=model, train_loader=train_loader_1, val_loader=val_loader_1)
    # average_images_1 = save_and_average_images(args=args, model=model, train_loader=train_loader_1)
    #
    # # Save model 1
    # saved_model_path = f'/home/luu/projects/cl_selective_nets/results' \
    #                    f'/{args.dataset}_{args.model}_autoencoder_{task_id}.pt'
    # torch.save(model.state_dict(), saved_model_path)


    model_1 = copy.deepcopy(model)
    model_1.load_state_dict(torch.load(f'/home/luu/projects/cl_selective_nets/results'
                                       f'/{args.dataset}_{args.model}_autoencoder_1.pt'))
    average_images_1 = save_and_average_images(args=args, model=model_1, train_loader=train_loader_1)

    concat_test_dataset = concat_datasets([cl_val_dataset_0, cl_val_dataset_1])
    concat_test_loader = DataLoader(concat_test_dataset, batch_size=1, shuffle=True)

    compare_features(concat_test_loader, average_images_0, average_images_1)

    # test_loss, test_acc = test(args, test_loader, model, criterion)
    # final_acc = test_acc
    # loguru.logger.info(f"Test accuracy is {final_acc}")

    # torch.save(model.state_dict(), f'/home/luu/projects/cl_selective_nets/results'
    #                                f'/{args.dataset}_{args.model}_autoencoder_{task_id}.pt')

    # python3 main.py --dataset core50 --size 128 --model pretrained_resnet50_simple_decoder --n_epochs 40

def get_single_cl_dataloader(train_dataset, val_dataset, task, batch_size=64):

    real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                     n_experiences=5, shuffle=True,
                                     seed=6,
                                     task_labels=True)

    cl_train_dataset = None
    cl_val_dataset = None
    cl_test_dataset = None
    for task_id, real_experience in enumerate(real_dataset_exps.train_stream):

        # Train on task 0, then test on task 1 to see the accuracy
        cl_test_dataset = real_dataset_exps.test_stream[task_id].dataset

        if task_id == task + 1:
            break

        print('classes in train stream', real_experience.classes_in_this_experience)
        print('classes in val stream', real_dataset_exps.test_stream[task_id].classes_in_this_experience)

        cl_train_dataset = real_experience.dataset
        cl_val_dataset = real_dataset_exps.test_stream[task_id].dataset

    cl_train_dataloader = DataLoader(cl_train_dataset, batch_size=batch_size, shuffle=True,
                                     pin_memory=True, num_workers=6, collate_fn=collate_fn)

    cl_val_dataloader = DataLoader(cl_val_dataset, batch_size=batch_size, shuffle=True,
                                   pin_memory=True, num_workers=6, collate_fn=collate_fn)

    cl_test_dataloader = DataLoader(cl_test_dataset, batch_size=batch_size, shuffle=True,
                                    pin_memory=True, num_workers=6, collate_fn=collate_fn)

    return cl_train_dataloader, cl_val_dataloader, cl_test_dataloader, cl_val_dataset


def train_distilled_dataset():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--n_epochs', type=int, default=32)
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    args = parser.parse_args()

    start_epoch = 0
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_amp = not args.noamp

    loguru.logger.info("Loading distilled core50 dataset")
    distilled_datasets = distilled_core50.get_distilled_core50(args, use_transform=False)

    obj_lvl = False
    mini = True
    cl_dataset = core50.load_core50(args=args, scenario='ni', mini=mini, obj_lvl=obj_lvl,
                                    dataset_root='/home/luu/DistilledDataset_ContinualLearning'
                                                 '/core50/avalanche_core50')

    train_dataset, val_dataset, test_dataset = cl_dataset

    # Load distilled dataset
    distilled_train_loader = torch.utils.data.DataLoader(distilled_datasets, batch_size=args.bs,
                                                         shuffle=True, num_workers=6, collate_fn=collate_fn)

    # Load real dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=6)

    num_classes = 10

    model = resnet.resnet50(pretrained=True, num_classes=num_classes)

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

    model.cuda()

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        train_loss = train(args, distilled_train_loader, epoch, model, criterion, optimizer, scaler, aug=None)
        val_loss, acc = test(args, test_loader, model, criterion)
        # test_loss, test_acc = test(args, test_loader, model, criterion)

        scheduler.step(epoch - 1)  # step cosine scheduling


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)

def collate_fn(batch):
    try:
        images, labels, task_id = zip(*batch)
    except ValueError:
        images, labels, _, task_id = zip(*batch)

    # print(type(images), len(images))
    images = tuple_of_tensors_to_tensor(images)
    labels = torch.tensor(labels)  # Convert labels to tensor

    return images, labels

def test_trained_models():
    from pytorch_msssim import ssim

    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')
    parser.add_argument('--datatype', default='real', type=str, help='choose between real and distilled')
    parser.add_argument('--dataset', default='core50', type=str, help='choose dataset')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--aug', default=True, type=bool, help='use randomaug')
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=100)
    args = parser.parse_args()

    start_epoch = 0
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.use_amp = not args.noamp

    if args.size == 32:
        mini = True
    else:
        mini = False

    # Load real dataset

    if args.dataset == 'core50':
        obj_lvl = False
        cl_dataset = core50.load_core50(args=args, scenario='ni', mini=mini, obj_lvl=obj_lvl,
                                        dataset_root='/home/luu/DistilledDataset_ContinualLearning'
                                                     '/core50/avalanche_core50')

        num_classes = 10

        if obj_lvl is True:
            num_classes = 50

    elif args.dataset == 'stl_10':
        cl_dataset = stl_10.get_stl_10(args=args)
        num_classes = 10

    elif args.dataset == 'intel_images':
        train_path = '/home/luu/projects/datasets/intel-image-classification/seg_train/seg_train'
        test_path = '/home/luu/projects/datasets/intel-image-classification/seg_test/seg_test'

        cl_dataset = intel_image.get_intel_images(args=args, train_path=train_path, val_path=test_path)
        num_classes = 6
    else:
        raise Exception(f"Not implemented {args.dataset} dataset yet")

    train_dataset, val_dataset, test_dataset = cl_dataset

    train_loader_0, val_loader_0, _ = get_single_cl_dataloader(train_dataset, val_dataset, task=0, batch_size=1)
    train_loader_1, val_loader_1, _ = get_single_cl_dataloader(train_dataset, val_dataset, task=1, batch_size=1)

    model_0 = resnet.resnet50(pretrained=False, num_classes=num_classes).to(args.device)

    model_1 = resnet.resnet50(pretrained=False, num_classes=num_classes).to(args.device)

    trained_path_0 = f'/home/luu/projects/cl_selective_nets/results/{args.dataset}_resnet50-model_task-0.pt'
    trained_path_1 = f'/home/luu/projects/cl_selective_nets/results/{args.dataset}_resnet50-model_task-1.pt'

    model_0.load_state_dict(torch.load(trained_path_0))
    model_1.load_state_dict(torch.load(trained_path_1))

    # Feature extraction function
    features_0 = {}
    features_1 = {}
    def get_activation(name, features):
        def hook(model, input, output):
            features[name] = output.detach().squeeze().cpu().numpy()

        return hook

    model_0.avgpool.register_forward_hook(get_activation('avgpool', features_0))
    model_1.avgpool.register_forward_hook(get_activation('avgpool', features_1))

    def extract_features(dataloader, model, features):
        model.eval()
        all_features = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(args.device)
                _ = model(inputs)  # Forward pass to trigger the hook
                all_features.append(features['avgpool'])  # Collect features
                labels_list.append(labels.cpu().numpy())

        labels_array = np.array(labels_list)
        return np.concatenate(all_features, axis=0), labels_array.reshape(labels_array.shape[0])

    def calculate_class_mean(features, labels):
        # features = features.squeeze()
        features = features.reshape(labels.shape[0], -1)
        class_means = []
        for label in np.unique(labels):
            # print(labels.shape)
            class_features = features[labels == label]

            # Calculate class means
            class_mean = np.mean(class_features, axis=0)
            class_means.append(class_mean)

        # Compute covariance matrix
        covariance_matrix = np.cov(features, rowvar=False)
        cov_inv = np.linalg.inv(covariance_matrix)

        return class_means, cov_inv

    def mahalanobis_distance(x, means, cov_inv):
        distances = []
        for mean in means:
            diff = x - mean
            dist = np.dot(np.dot(diff.T, cov_inv), diff)
            distances.append(dist)
        return distances

    # Extract features
    # (16814, 2048, 1, 1)
    features_array_0, labels_0 = extract_features(train_loader_0, model_0, features_0)
    features_array_1, labels_1 = extract_features(train_loader_1, model_1, features_1)

    class_means_0, cov_inv_0 = calculate_class_mean(features_array_0, labels_0)
    class_means_1, cov_inv_1 = calculate_class_mean(features_array_1, labels_1)

    # Example usage
    test_feat_0 = {}
    test_feat_1 = {}
    model_0.avgpool.register_forward_hook(get_activation('avgpool', test_feat_0))
    model_1.avgpool.register_forward_hook(get_activation('avgpool', test_feat_1))
    test_features_0, _ = extract_features(val_loader_0, model_0, test_feat_0)  # Extract features for test set
    test_features_1, _ = extract_features(val_loader_1, model_1, test_feat_1)

    acc_model_0 = 0
    acc_model_1 = 0

    for test_feat_0, test_feat_1 in zip(test_features_0, test_features_1):

        distances_00 = mahalanobis_distance(test_feat_0, class_means_0, cov_inv_0)
        distances_11 = mahalanobis_distance(test_feat_1, class_means_1, cov_inv_1)

        distances_01 = mahalanobis_distance(test_feat_0, class_means_1, cov_inv_1)
        distances_10 = mahalanobis_distance(test_feat_1, class_means_0, cov_inv_0)

        if distances_00 < distances_01:
            acc_model_0 += 1

        if distances_11 < distances_10:
            acc_model_1 += 1

    print(f'Accuracy of model 0: {acc_model_0/len(test_features_0)}')
    print(f'Accuracy of model 1: {acc_model_1 / len(test_features_1)}')


if __name__ == '__main__':
    baseline_upperbound()
    # baseline_image_reconstruction_upperbound()
    # test_trained_models()
    # test_pretrained_distilled_data()
    # train_distilled_dataset()


    """
    python3 main.py --dataset core50 --size 128 --model resnet50 --n_epochs 20

    """

