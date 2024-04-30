# https://github.com/kentaroy47/vision-transformers-cifar10/tree/main
from collections import defaultdict

import torch
from torchvision.utils import save_image

from utils import progress_bar
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

# ----------------------------------------------------------------------------
# ------------------------------ CLASSIFICATION ------------------------------
# ----------------------------------------------------------------------------
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
# ------------------------------ IMAGE RECONSTRUCTION ------------------------
# ----------------------------------------------------------------------------
from pytorch_msssim import ssim, ms_ssim


def vae_loss_function(recon_x, x, mu, logvar):
    # MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train_img_reconstruction(args, trainloader, epoch, model, criterion, optimizer, aug=None):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    total_ssim = 0
    for batch_idx, data in enumerate(trainloader):
        try:
            inputs, targets, _ = data
        except:
            inputs, targets = data

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        if aug is not None:
            inputs = aug(inputs)

        optimizer.zero_grad()

        if args.model == 'vae':
            reconstructed_img, z, mu, logvar, logits = model(inputs)
            mse_loss = vae_loss_function(reconstructed_img, inputs, mu, logvar)
        else:
            reconstructed_img, _ = model(inputs)

            # Calculate loss
            mse_loss = criterion(reconstructed_img, inputs)
        # mse_loss = F.mse_loss(inputs, reconstructed_img, reduction="mean")

        # Total loss
        loss = mse_loss

        # Calculate SSIM for similarity measurement
        current_ssim = ssim(inputs.float(), reconstructed_img.float(), data_range=1, size_average=True)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        total_ssim += current_ssim.item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f, SSIM: %.3f' % (train_loss / (batch_idx + 1), total_ssim / (batch_idx + 1)))

    return train_loss / (batch_idx + 1)


def test_img_reconstruction(args, testloader, model, criterion):
    model.eval()
    test_loss = 0
    total_ssim = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            try:
                inputs, targets, _ = data
            except:
                inputs, targets = data

            inputs, targets = inputs.to(args.device), targets.to(args.device)

            if args.model == 'vae':
                reconstructed_img, z, mu, logvar, logits = model(inputs)
                mse_loss = vae_loss_function(reconstructed_img, inputs, mu, logvar)
            else:
                reconstructed_img, _ = model(inputs)

                # Calculate loss
                mse_loss = criterion(reconstructed_img, inputs)
            # mse_loss = F.mse_loss(inputs, reconstructed_img, reduction="mean")

            # ssim loss
            # ssim_loss = 1 - ssim(inputs.float(), reconstructed_img.float(), data_range=1, size_average=True)

            # Total loss
            loss = mse_loss

            # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
            # loss = criterion(reconstructed_img, inputs)
            # Calculate SSIM for similarity measurement
            current_ssim = ssim(inputs.float(), reconstructed_img.float(), data_range=1, size_average=True)

            test_loss += loss.item()
            total_ssim += current_ssim.item()

            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f, SSIM: %.3f' % (test_loss / (batch_idx + 1), total_ssim / (batch_idx + 1)))

    return test_loss


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

# ------------------------------------------------------------ #
# ---------------- TRAIN BY PARTS ---------------------------- #
# ------------------------------------------------------------ #
import torch.nn as nn
def train_encoder_classifier(args, model, train_loader, val_loader, scaler):
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=0.001)

    for epoch in range(3):  # Train for 3 epochs
        train(args, train_loader, epoch, model.encoder, criterion, encoder_optimizer, scaler, aug=None)
        test(args, val_loader, model.encoder, criterion)

    # Freeze encoder after initial training
    for param in model.encoder.parameters():
        param.requires_grad = False


def train_decoder(args, model, train_loader, val_loader):
    reconstruction_criterion = nn.MSELoss()
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=0.001)
    model.train()
    for epoch in range(args.n_epochs):  # Continue for more epochs

        train_img_reconstruction(args, train_loader, epoch, model,
                                 reconstruction_criterion, decoder_optimizer, aug=None)

        test_img_reconstruction(args, val_loader, model, reconstruction_criterion)

def save_and_average_images(args, model, train_loader):
    from collections import defaultdict
    import os

    reconstructed_images = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            latent, _ = model.encoder(images)
            reconstructed = model.decoder(latent)

            for img, label in zip(reconstructed, labels):
                reconstructed_images[label.item()].append(img.cpu())

    # Calculate average image per class
    average_images = {}
    for label, imgs in reconstructed_images.items():
        average_images[label] = torch.mean(torch.stack(imgs), dim=0)
        # Optionally save or visualize these average images

        # Calculate average image per class and save them
        if not os.path.exists('average_images'):
            os.makedirs('average_images')

    average_images = {}
    for label, imgs in reconstructed_images.items():
        average_images[label] = torch.mean(torch.stack(imgs), dim=0)
        # Save average image
        save_path = os.path.join('average_images', f'class_{label}.png')
        save_image(average_images[label], save_path)
        print(f"Saved average image for class {label} at {save_path}")

    return average_images


def calculate_cosine_similarity(feat1, feat2):
    return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0), dim=1).mean()


def compare_features(test_loader, average_images_0, average_images_1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = nn.Identity()  # Modify the last layer to output the feature vector
    resnet18 = resnet18.to(device)
    resnet18.eval()

    # Prepare average features
    avg_features = defaultdict(torch.Tensor)
    with torch.no_grad():
        for label, avg_img in average_images_0.items():
            avg_img = avg_img.unsqueeze(0).to(device)  # Assuming avg_img is a tensor
            avg_features[label] = resnet18(avg_img)

        for label, avg_img in average_images_1.items():
            avg_img = avg_img.unsqueeze(0).to(device)  # Assuming avg_img is a tensor
            if label in avg_features:
                continue
            else:
                avg_features[label] = resnet18(avg_img)

    correct_count = 0
    total_count = 0

    # Compare each test image's features to each average features
    with torch.no_grad():
        for images, _, task_ids in test_loader:
            images = images.to(device)
            outputs = resnet18(images)

            for output, task_id in zip(outputs, task_ids):
                similarities = {label: calculate_cosine_similarity(output, avg_feature.mean(0)) for label, avg_feature
                                in avg_features.items()}
                predicted_label = max(similarities, key=similarities.get)  # Find the key with the highest similarity

                # Compare predicted label with actual task_id and count correct predictions
                if int(predicted_label) == task_id.item():
                    correct_count += 1
                total_count += 1

    accuracy = correct_count / total_count
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy






