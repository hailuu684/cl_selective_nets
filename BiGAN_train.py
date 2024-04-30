import argparse
from utils import str2bool
from get_datasets import core50, distilled_core50, cifar10
from avalanche.benchmarks import nc_benchmark, ni_benchmark
import torch
from models import BiGAN

# This one maybe better: https://github.com/fmu2/Wasserstein-BiGAN/blob/master/wali_cifar10.py

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


def collate_fn(batch):
    try:
        images, labels = zip(*batch)
    except ValueError:
        images, labels, task_id = zip(*batch)

    # print(type(images), len(images))
    images = tuple_of_tensors_to_tensor(images)
    labels = torch.tensor(labels)  # Convert labels to tensor

    return images, labels


def main():
    parser = argparse.ArgumentParser(description='PyTorch CORe50 Training')

    # Dataset settings
    parser.add_argument('--dataset', default='core50', type=str, help='choose dataset [core50, cifar10]')
    parser.add_argument('--scenario', default='nc', type=str, help='nc or ni')
    parser.add_argument('--n_experience', default=5, type=int, help='number of experiences')
    parser.add_argument('--seed', default=5, type=int, help='number of seed')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--lr_adam', default=1e-4, type=float)
    parser.add_argument('--IMAGE_SIZE', default=32, type=int)
    parser.add_argument('--DIM', default=128, type=int)
    parser.add_argument('--C_ITERS', default=5, type=int)
    parser.add_argument('--EG_ITERS', default=1, type=int)
    parser.add_argument('--LAMBDA', default=10, type=int)
    parser.add_argument('--ITER', default=200000, type=int)

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.betas = (0.5, 0.9)
    args.lr = 1e-4

    dataset_cl = cifar10.get_cifar10(args)

    train_dataset, val_dataset, test_dataset = dataset_cl

    real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                     n_experiences=args.n_experience, shuffle=True,
                                     seed=args.seed,
                                     task_labels=True)

    for i in range(2):
        experience = real_dataset_exps.train_stream[i]

        current_exp = experience.current_experience
        print(f"Training on experience {current_exp}")
        train_loader = torch.utils.data.DataLoader(experience.dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(real_dataset_exps.test_stream[experience.current_experience].dataset,
                                                 batch_size=64, shuffle=False)

        bigan = BiGAN.TrainerBiGAN(args, train_loader, args.device, current_exp=current_exp)
        bigan.train()

        torch.save(bigan.G.state_dict(), f'/home/luu/projects/cl_selective_nets/results/BiGAN/generator_{current_exp}exp')
        torch.save(bigan.C.state_dict(), f'/home/luu/projects/cl_selective_nets/results/BiGAN/discriminator_{current_exp}exp')
        torch.save(bigan.E.state_dict(),
                   f'/home/luu/projects/cl_selective_nets/results/BiGAN/encoder_{current_exp}exp')


if __name__ == '__main__':
    main()


