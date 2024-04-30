from get_datasets import cifar10
from models import dla
from avalanche.benchmarks import nc_benchmark
import torch
from train_epoch import train, test
import torch.nn as nn
import torch.optim as optim


class Args:
    use_amp = True
    device = 'cuda'
    model = 'dla'

def main():

    args = Args()

    train_dataset, val_dataset, test_dataset = cifar10.get_cifar10(args)

    # Design for cifar10
    model = dla.DLA()

    # Get experiences
    real_dataset_exps = nc_benchmark(train_dataset, val_dataset,
                                     n_experiences=5, shuffle=True,
                                     seed=6,
                                     task_labels=True)

    desire_exp = real_dataset_exps.train_stream[4]

    train_loader = torch.utils.data.DataLoader(desire_exp.dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(real_dataset_exps.test_stream[desire_exp.current_experience].dataset,
                                             batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # use scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    model.to(args.device)

    for i in range(50):

        train(args=args, trainloader=train_loader, epoch=i, model=model, criterion=criterion, optimizer=optimizer,
              scaler=scaler, aug=None)

        test(args=args, testloader=val_loader, model=model, criterion=criterion)


if __name__ == '__main__':
    main()

