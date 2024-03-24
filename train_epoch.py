# https://github.com/kentaroy47/vision-transformers-cifar10/tree/main

import torch
from utils import progress_bar


def train(args, trainloader, epoch, model, criterion, optimizer, scaler, aug=None):
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

        # Train with amp
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            if aug is not None:
                inputs = aug(inputs)
            outputs = model(inputs)

            if args.model == 'vit_prompt':
                outputs = outputs['logits']

            # Calculate loss
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1)


def test(args, testloader, model, criterion):

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            try:
                inputs, targets, _ = data
            except:
                inputs, targets = data

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs = model(inputs)

            if args.model == 'vit_prompt':
                outputs = outputs['logits']

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total

    return test_loss, acc