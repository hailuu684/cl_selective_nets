# https://github.com/kentaroy47/vision-transformers-cifar10/tree/main

import torch
from utils import progress_bar
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F


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
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (batch_idx + 1)


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


# ----------------------------------------------------------------------------
# ------------------------------ VAE -----------------------------------------
# ----------------------------------------------------------------------------
# Doesnt work so far
def train_vae_model(model, train_loader, test_loader, known_classes,
                    epochs=10, lr=2.0, weight_decay=1e-5, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()  # Assuming classification task

    epoch_start = 1

    for epoch in range(epoch_start, epochs + 1):

        model.train()

        train_correct = 0
        train_total = 0

        data_stream = tqdm(enumerate(train_loader, 1))
        for batch_index, (x, labels) in data_stream:
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            (mean, logvar), x_reconstructed, logits = model(x)

            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            classification_loss = criterion(logits, labels)

            total_loss = reconstruction_loss + kl_divergence_loss + classification_loss

            total_loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar description
            # Note: .item() should be used to extract scalar values from tensors
            data_stream.set_description((
                'Epoch: {epoch} | '
                'Total Loss: {total_loss:.4f} | '
                'Reconstruction Loss: {reconstruction_loss:.3f} | '
                'KL Divergence Loss: {kl_divergence_loss:.3f} | '
                'Classification Loss: {classification_loss:.3f} | '
                'Accuracy: {accuracy:.2f}%'
            ).format(
                epoch=epoch,
                total_loss=total_loss.item(),
                reconstruction_loss=reconstruction_loss.item(),
                kl_divergence_loss=kl_divergence_loss.item(),
                classification_loss=classification_loss.item(),
                accuracy=100. * train_correct / train_total,
            ))

        centroids = compute_centroid(model, data_loader=train_loader, device=device)

        validate_vae_model(model, test_loader, known_classes, mse_threshold=0.25, centroids=centroids, device=device)
        print(" ")


def validate_vae_model(model, test_loader, known_classes, mse_threshold, centroids, device):
    model.eval()
    correct = 0
    total = 0
    unknown_detected = 0
    unknown_total = 0
    unknown_correct = 0
    mse_criterion = torch.nn.MSELoss(reduction='mean')

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            (mean, logvar), x_reconstructed, logits = model(images)

            # Compute MSE between original and reconstructed images
            mse_errors = mse_criterion(x_reconstructed, images)

            # Compute latent distance
            latent_distance = torch.norm(mean - centroids, dim=1)

            # Prediction logic
            _, predicted = torch.max(logits.data, 1)
            max_mse_errors = mse_errors.data

            # Detecting unknowns based on reconstruction error
            unknown_criterion_1 = (max_mse_errors > mse_threshold)
            unknown_criterion_2 = (latent_distance > 0.2)
            is_unknown = unknown_criterion_1
            unknown_detected += is_unknown.sum().item()

            # # Correctly identified as unknown
            # unknown_correct = ((labels not in known_classes) & is_unknown).sum().item()

            for i in range(len(labels)):
                if is_unknown:  # labels[i] not in known_classes and
                    unknown_correct += 1
                    # print(max_mse_errors)
                if labels[i] not in known_classes:
                    unknown_total += 1

            # Known classes accuracy calculation
            correct_preds = (predicted == labels)
            correct += correct_preds.sum().item()
            total += labels.size(0)

            # Count actual unknowns for accuracy calculation
            # actual_unknowns = (labels not in known_classes)
            # unknown_total += actual_unknowns.sum().item()

    known_accuracy = 100 * correct / total if total > 0 else 0
    unknown_detection_accuracy = 100 * unknown_correct / unknown_total if unknown_total > 0 else 0

    print(f'Validation - Known Accuracy: {known_accuracy:.2f}%, '
          f'Unknown Detection Accuracy: {unknown_detection_accuracy:.2f}%')


def compute_centroid(model, data_loader, device):
    model.eval()
    total_mu = 0
    count = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            (mean, _), _, _ = model(images)
            total_mu += mean.sum(dim=0)
            count += mean.size(0)

    centroid = total_mu / count
    return centroid


# ----------------------------------------------------------------------------
# ------------------------------ AE ------------------------------------------
# ----------------------------------------------------------------------------

def train_ae_model(model, train_loader, test_loader, known_classes,
                   epochs=10, lr=1e-3, weight_decay=1e-5, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()  # Assuming classification task

    epoch_start = 1

    for epoch in range(epoch_start, epochs + 1):

        model.train()

        train_correct = 0
        train_total = 0

        data_stream = tqdm(enumerate(train_loader, 1))
        for batch_index, (x, labels) in data_stream:
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            x_hat, logits = model(x)

            reconstruction_loss = F.mse_loss(x, x_hat, reduction="none")
            reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0])

            classification_loss = criterion(logits, labels)

            total_loss = reconstruction_loss + classification_loss

            total_loss.backward()

            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar description
            # Note: .item() should be used to extract scalar values from tensors
            data_stream.set_description((
                'Epoch: {epoch} | '
                'Total Loss: {total_loss:.4f} | '
                'Reconstruction Loss: {reconstruction_loss:.3f} | '
                'Classification Loss: {classification_loss:.3f} | '
                'Accuracy: {accuracy:.2f}%'
            ).format(
                epoch=epoch,
                total_loss=total_loss.item(),
                reconstruction_loss=reconstruction_loss.item(),
                classification_loss=classification_loss.item(),
                accuracy=100. * train_correct / train_total,
            ))

        validate_ae_model(model, test_loader, criterion, known_classes, mse_threshold=0.1, device=device)


def validate_ae_model(model, test_loader, criterion, known_classes, mse_threshold, device):
    print(" ")
    print('Validation')
    model.eval()
    correct = 0
    total = 0
    unknown_detected = 0
    unknown_total = 0
    unknown_correct = 0

    with torch.no_grad():
        test_stream = tqdm(enumerate(test_loader, 1))
        for batch_index, (images, labels) in test_stream:
            images = images.to(device)
            labels = labels.to(device)

            x_hat, logits = model(images)

            # Reconstruction loss
            reconstruction_loss = F.mse_loss(images, x_hat, reduction="none")
            reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0])

            # Classification loss
            classification_loss = criterion(logits, labels)

            # Total loss
            total_loss = reconstruction_loss + classification_loss

            # Known classes accuracy calculation
            # Prediction logic
            _, predicted = torch.max(logits.data, 1)
            correct_preds = (predicted == labels)
            correct += correct_preds.sum().item()
            total += labels.size(0)

            test_stream.set_description((
                'Total Loss: {total_loss:.4f} | '
                'Reconstruction Loss: {reconstruction_loss:.3f} | '
                'Classification Loss: {classification_loss:.3f} | '
                'Accuracy: {accuracy:.2f}%'
            ).format(
                total_loss=total_loss.item(),
                reconstruction_loss=reconstruction_loss.item(),
                classification_loss=classification_loss.item(),
                accuracy=100. * correct / total,
            ))

    print(" ")
